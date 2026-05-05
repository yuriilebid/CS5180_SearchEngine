"""
Browser UI for the search engine. Run from anywhere:
  python web_ui/app.py
Loads index.pkl from the project root (parent of this directory).
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from flask import Flask, jsonify, render_template, request

from evaluator import _open_json_first
from retrieval import (
    adl,
    doc_lengths,
    int_id_lookup,
    inverted,
    score_bim,
    score_bm25,
    score_hybrid_rrf,
    score_lm_dirichlet,
    score_rocchio_prf,
    score_rm3,
    score_vsm,
)
from text_processor import preprocess

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
    static_url_path="/static",
)

PAGE_SIZE_DEFAULT = 5
MAX_RANK_DEPTH = 2000
PAGER_MAX_PAGES = 5

_SCORE_FNS = {
    "bm25": score_bm25,
    "vsm": score_vsm,
    "bim": score_bim,
    "lm_dirichlet": score_lm_dirichlet,
    "hybrid_rrf": score_hybrid_rrf,
    "rocchio_prf": score_rocchio_prf,
    "rm3": score_rm3,
}

_doc_text: dict[str, str] | None = None
_qrels_map: dict[str, set[str]] | None = None
_query_texts_cache: list[str] | None = None

SUGGEST_PREFIX_MIN_LEN = 2
SUGGEST_LIMIT_CAP = 12
SUGGEST_MAX_EXTRA_WORDS = 6
SUGGEST_MAX_QUESTION_WORDS = 48

_QUESTION_LEAD = re.compile(
    r"^\s*(what|why|how|when|where|who|whose|which|whom|can|could|should|would|"
    r"is|are|was|were|am|do|does|did|has|have|had|must|may|might|shall|will|"
    r"aren't|isn't|wasn't|weren't|don't|doesn't|didn't|hasn't|haven't|hadn't|can't|cannot)\b",
    re.I,
)


def _looks_like_question(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s.endswith("?"):
        return True
    return bool(_QUESTION_LEAD.match(s))


def _within_suggestion_word_budget(raw: str, suggestion: str) -> bool:
    """Suggestion must not exceed typed word count + ``SUGGEST_MAX_EXTRA_WORDS`` (hard cap)."""
    typed_n = len(raw.strip().split())
    cap = typed_n + SUGGEST_MAX_EXTRA_WORDS
    w = len(suggestion.strip().split())
    return w <= min(cap, SUGGEST_MAX_QUESTION_WORDS)


def _query_stems(raw: str) -> list[str]:
    return preprocess(raw)


def _stem_match_count(raw: str, suggestion: str) -> int:
    q = _query_stems(raw)
    if not q:
        return 0
    s_set = set(preprocess(suggestion))
    return sum(1 for stem in q if stem in s_set)


def _min_stems_required(raw: str, relaxed: bool) -> int | None:
    """None means skip stem gate (query had no content tokens after preprocessing)."""
    q = _query_stems(raw)
    if not q:
        return None
    if relaxed:
        return max(1, len(q) - 1) if len(q) > 1 else 1
    return len(q)


def _fuzzy_cutoff_for_prefix_len(n: int) -> float:
    """Looser cutoff for longer typed prefixes (short prefixes stay strict to limit noise)."""
    if n <= 2:
        return 0.45
    if n <= 4:
        return 0.30
    return 0.26


def _fuzzy_start_ratio(raw: str, candidate: str) -> float:
    """Similarity between typed prefix and the start of the candidate (case-insensitive)."""
    r = raw.strip().lower()
    c = candidate.strip().lower()
    if not r or not c:
        return 0.0
    nr, nc = len(r), len(c)
    if nc >= nr:
        chunk, sub_r = c[:nr], r
    else:
        chunk, sub_r = c, r[:nc]
    return difflib.SequenceMatcher(None, sub_r, chunk).ratio()


def _best_fuzzy_score(raw: str, candidate: str) -> float:
    """How well the candidate matches what the user typed (prefix-first, plus whole-string hint)."""
    r = raw.strip()
    c = candidate.strip()
    if not r or not c:
        return 0.0
    if c.lower().startswith(r.lower()):
        return 1.0
    start = _fuzzy_start_ratio(r, c)
    whole = difflib.SequenceMatcher(None, r.lower(), c.lower()).ratio()
    return max(start, whole * 0.88)


def _load_doc_text() -> dict[str, str]:
    global _doc_text
    if _doc_text is not None:
        return _doc_text
    with _open_json_first("documents.json") as f:
        documents = json.load(f)
    _doc_text = {doc["doc_id"]: doc.get("text", "") for doc in documents}
    return _doc_text


def _load_qrels() -> dict[str, set[str]]:
    global _qrels_map
    if _qrels_map is not None:
        return _qrels_map
    try:
        with _open_json_first("qrels.json") as f:
            qrels = json.load(f)
    except OSError:
        _qrels_map = {}
        return _qrels_map
    m: dict[str, set[str]] = {}
    for row in qrels:
        qid = str(row["query_id"])
        m.setdefault(qid, set()).add(row["doc_id"])
    _qrels_map = m
    return _qrels_map


def _snippet(text: str, limit: int = 280) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _title_for_doc(doc_id: str, text: str) -> str:
    line = text.strip().split("\n")[0] if text else ""
    line = line.replace("\n", " ").strip() or doc_id
    if len(line) > 88:
        return line[:85] + "..."
    return line


def _load_query_texts() -> list[str]:
    global _query_texts_cache
    if _query_texts_cache is not None:
        return _query_texts_cache
    rows = None
    for path in (
        os.path.join(_ROOT, "dataset", "queries.json"),
        os.path.join(_ROOT, "queries.json"),
    ):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            break
        except OSError:
            continue
    if rows is None:
        try:
            with _open_json_first("queries.json") as f:
                rows = json.load(f)
        except OSError:
            _query_texts_cache = []
            return _query_texts_cache
    _query_texts_cache = sorted(
        {str(row.get("text", "")).strip() for row in rows if row.get("text")},
        key=lambda s: s.lower(),
    )
    return _query_texts_cache


def _continuation_display(typed: str, corpus_text: str) -> str:
    """Show suggestion as typed-prefix + remainder (match user’s casing on the prefix)."""
    t = typed.strip()
    s = corpus_text.strip()
    if not t:
        return s
    if not s.lower().startswith(t.lower()):
        return s
    return t + s[len(t) :]


def _suggestions_for_prefix(prefix: str, limit: int) -> list[dict]:
    """
    Benchmark questions only. Rules:

    - Word budget: suggestion length ≤ ``typed_words + SUGGEST_MAX_EXTRA_WORDS``
      (also capped by ``SUGGEST_MAX_QUESTION_WORDS``).
    - Lexical relevance: every stem from ``preprocess(typed)`` must appear in the
      suggestion (same stemming as retrieval). If that yields nothing, relax once to
      ``max(1, len(stems)-1)`` matches so sparse corpora still return results.
    - Rank by stem overlap count, then fuzzy prefix score, then brevity.
    """
    raw = prefix.strip()
    if len(raw) < SUGGEST_PREFIX_MIN_LEN:
        return []

    cutoff = _fuzzy_cutoff_for_prefix_len(len(raw))
    scored: list[tuple[int, float, float, str, str]] = []

    for relaxed in (False, True):
        scored.clear()
        seen_lower: set[str] = set()
        min_stems = _min_stems_required(raw, relaxed)

        for text in _load_query_texts():
            t = text.strip()
            if (
                not t
                or not _looks_like_question(t)
                or not _within_suggestion_word_budget(raw, t)
            ):
                continue
            tl = t.lower()
            if tl in seen_lower:
                continue
            if min_stems is not None:
                mc = _stem_match_count(raw, t)
                if mc < min_stems:
                    continue
            score = _best_fuzzy_score(raw, t)
            if score < cutoff:
                continue
            seen_lower.add(tl)
            stem_hits = _stem_match_count(raw, t) if min_stems is not None else 0
            tie = float(5000 - len(t))
            if t.lower().startswith(raw.lower()):
                score += 0.5
            scored.append((stem_hits, score, tie, t, "query"))

        if scored:
            break

    scored.sort(
        key=lambda x: (-x[0], -x[1], -x[2], len(x[3]), x[3].lower()),
    )
    out: list[dict] = []
    seen_final: set[str] = set()
    for _hits, _score, _tie, text, source in scored:
        if text.lower().startswith(raw.lower()):
            display = _continuation_display(raw, text)
        else:
            display = text
        key = display.lower()
        if key in seen_final:
            continue
        seen_final.add(key)
        out.append({"text": display, "source": source})
        if len(out) >= limit:
            break
    return out


def _run_search(
    query: str,
    model: str,
    page: int,
    page_size: int,
    query_id: str | None,
):
    query = (query or "").strip()
    if not query:
        return {"error": "empty_query"}, 400

    score_fn = _SCORE_FNS.get(model)
    if score_fn is None:
        return {"error": "unknown_model"}, 400

    # Enough depth for UI pager (e.g. 5 pages × 5 hits = 25), not only current page.
    max_ui_rank_depth = min(PAGER_MAX_PAGES * page_size, MAX_RANK_DEPTH)
    need = max(
        page_size,
        max_ui_rank_depth,
        min(page * page_size, MAX_RANK_DEPTH),
    )
    ranked = score_fn(query, need)

    effective_pages = min(
        PAGER_MAX_PAGES,
        max(1, math.ceil(len(ranked) / page_size)),
    )
    page = max(1, min(page, effective_pages))
    start = (page - 1) * page_size
    window = ranked[start : start + page_size]
    ranked_ids = [d for d, _ in ranked]

    qrels = _load_qrels()
    qid_key = str(query_id).strip() if query_id else None
    rel = qrels.get(qid_key, set()) if qid_key else set()
    qrels_active = bool(qid_key) and qid_key in qrels
    total_rel = len(rel)

    doc_text = _load_doc_text()
    query_terms = set(preprocess(query))

    results = []
    for offset, (doc_id, score) in enumerate(window):
        rank = start + offset + 1
        int_id = int_id_lookup.get(doc_id)
        dl = doc_lengths[int_id] if int_id is not None else 0
        text = doc_text.get(doc_id, "")

        matched_terms = 0
        if int_id is not None and query_terms:
            matched_terms = sum(
                1 for t in query_terms if int_id in inverted.get(t, {})
            )
        term_coverage = (
            matched_terms / len(query_terms) if query_terms else 0.0
        )

        hits = sum(1 for j in range(rank) if ranked_ids[j] in rel)
        is_rel = doc_id in rel if qrels_active else False
        prec_k = hits / rank if rank else 0.0
        recall_k = (hits / total_rel) if total_rel else None
        dcg_inc = (1.0 / math.log2(rank + 1)) if is_rel else 0.0

        results.append(
            {
                "doc_id": doc_id,
                "rank": rank,
                "score": score,
                "title": _title_for_doc(doc_id, text),
                "snippet": _snippet(text),
                "metrics": {
                    "rank": rank,
                    "score": score,
                    "doc_id": doc_id,
                    "doc_tokens": dl,
                    "length_ratio": round((dl / adl), 4) if adl else 0.0,
                    "term_coverage": round(term_coverage, 4),
                    "matched_terms": matched_terms,
                    "query_terms": len(query_terms),
                    "relevant": is_rel if qrels_active else None,
                    "precision_at_rank": round(prec_k, 6) if qrels_active else None,
                    "recall_at_rank": round(recall_k, 6)
                    if qrels_active and recall_k is not None
                    else None,
                    "dcg_increment": round(dcg_inc, 6) if qrels_active else None,
                },
            }
        )

    has_more = page < effective_pages

    out: dict = {
        "query": query,
        "model": model,
        "page": page,
        "page_size": page_size,
        "returned": len(results),
        "depth": len(ranked),
        "has_more": has_more,
        "pager_max_pages": PAGER_MAX_PAGES,
        "pager_total_pages": effective_pages,
        "query_id": qid_key,
        "qrels_active": qrels_active,
        "total_relevant_in_qrels": total_rel if qrels_active else 0,
        "results": results,
    }
    return out, 200


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """Structured breakdown of ranking math for one query / document / model."""
    from explain_ranking import build_explanation

    data = request.get_json(silent=True) or {}
    q = (data.get("q") or "").strip()
    doc_id = (data.get("doc_id") or "").strip()
    model = (data.get("model") or "bm25").strip().lower()
    if not q or not doc_id:
        return jsonify({"error": "missing_q_or_doc"}), 400
    out = build_explanation(q, doc_id, model)
    err = out.get("error")
    if err == "unknown_model":
        return jsonify(out), 400
    if err == "unknown_doc":
        return jsonify(out), 404
    return jsonify(out)


@app.route("/")
def index():
    return render_template("search.html")


@app.route("/api/models", methods=["GET"])
def api_models():
    models = [
        {"id": "bm25", "label": "BM25", "available": True},
        {"id": "vsm", "label": "VSM (TF-IDF cosine)", "available": True},
        {"id": "bim", "label": "BIM", "available": True},
        {"id": "lm_dirichlet", "label": "Language model (Dirichlet)", "available": True},
        {"id": "hybrid_rrf", "label": "Hybrid RRF", "available": True},
        {"id": "rocchio_prf", "label": "Rocchio PRF", "available": True},
        {"id": "rm3", "label": "RM3", "available": True},
    ]
    return jsonify({"models": models, "default_model": "bm25"})


@app.route("/api/suggest", methods=["GET"])
def api_suggest():
    q = (request.args.get("q") or "").strip()
    try:
        limit = int(request.args.get("limit", 8))
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(limit, SUGGEST_LIMIT_CAP))
    suggestions = _suggestions_for_prefix(q, limit)
    return jsonify({"suggestions": suggestions})


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(silent=True) or {}
    query = data.get("q", "")
    model = (data.get("model") or "bm25").strip().lower()
    try:
        page = max(1, int(data.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(data.get("page_size", PAGE_SIZE_DEFAULT))
    except (TypeError, ValueError):
        page_size = PAGE_SIZE_DEFAULT
    page_size = max(1, min(page_size, 50))
    qid = data.get("query_id")
    if qid is not None:
        qid = str(qid).strip() or None
    payload, status = _run_search(query, model, page, page_size, qid)
    return jsonify(payload), status


@app.route("/api/document/<doc_id>")
def api_document(doc_id):
    """Full document text for the article reader (doc_id must exist in corpus)."""
    docs = _load_doc_text()
    if doc_id not in docs:
        return jsonify({"error": "not_found"}), 404
    text = docs[doc_id] or ""
    return jsonify(
        {
            "doc_id": doc_id,
            "title": _title_for_doc(doc_id, text),
            "text": text,
            "char_count": len(text),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
