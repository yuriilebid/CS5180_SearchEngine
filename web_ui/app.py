"""
Browser UI for the search engine. Run from anywhere:
  python web_ui/app.py
Loads index.pkl from the project root (parent of this directory).
"""

from __future__ import annotations

import json
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from flask import Flask, jsonify, render_template, request

from bill_score import score_bill_bm25
from evaluator import _open_json_first
from retrieval import (
    adl,
    doc_ids,
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


def _bill_available() -> bool:
    return os.path.isfile(os.path.join(_ROOT, "bill_index.pkl"))


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

    if model == "bill_bm25":
        score_fn = score_bill_bm25
    else:
        score_fn = _SCORE_FNS.get(model)
    if score_fn is None:
        return {"error": "unknown_model"}, 400

    try:
        need = min(page * page_size, MAX_RANK_DEPTH)
        need = max(need, page_size)
        ranked = score_fn(query, need)
    except FileNotFoundError as e:
        return {"error": "bill_index_missing", "detail": str(e)}, 503

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

    has_more = (start + page_size) < len(ranked)
    if len(ranked) >= MAX_RANK_DEPTH and (start + page_size) >= MAX_RANK_DEPTH:
        has_more = False

    return {
        "query": query,
        "model": model,
        "page": page,
        "page_size": page_size,
        "returned": len(results),
        "depth": len(ranked),
        "has_more": has_more,
        "query_id": qid_key,
        "qrels_active": qrels_active,
        "total_relevant_in_qrels": total_rel if qrels_active else 0,
        "results": results,
    }, 200


@app.route("/")
def index():
    return render_template("search.html")


@app.route("/api/models", methods=["GET"])
def api_models():
    bill_ok = _bill_available()
    models = [
        {"id": "bm25", "label": "BM25", "available": True},
        {"id": "vsm", "label": "VSM (TF-IDF cosine)", "available": True},
        {"id": "bim", "label": "BIM", "available": True},
        {"id": "lm_dirichlet", "label": "Language model (Dirichlet)", "available": True},
        {"id": "hybrid_rrf", "label": "Hybrid RRF", "available": True},
        {"id": "rocchio_prf", "label": "Rocchio PRF", "available": True},
        {"id": "rm3", "label": "RM3", "available": True},
        {
            "id": "bill_bm25",
            "label": "Bill BM25 (finance index)",
            "available": bill_ok,
        },
    ]
    return jsonify({"models": models, "default_model": "bm25"})


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
