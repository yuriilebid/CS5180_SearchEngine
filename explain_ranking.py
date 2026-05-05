"""
Structured ranking explanations for the web UI (per query + document + model).
Mirrors formulas in retrieval.py.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from text_processor import preprocess

# Full document bodies for highlighting (same corpus as UI article viewer).
_documents_map: dict[str, str] | None = None
MAX_EXPLAIN_DOC_CHARS = 120_000


def _load_documents_map() -> dict[str, str]:
    global _documents_map
    if _documents_map is not None:
        return _documents_map
    try:
        from evaluator import _open_json_first

        with _open_json_first("documents.json") as f:
            rows = json.load(f)
        _documents_map = {
            str(r["doc_id"]): str(r.get("text", "") or "") for r in rows
        }
    except OSError:
        _documents_map = {}
    return _documents_map


def _contributing_stems(payload: dict[str, Any]) -> set[str]:
    stems = {
        row["term"]
        for row in payload.get("terms", [])
        if row.get("status") == "matched"
    }
    if stems:
        return stems
    q = payload.get("query") or ""
    doc_id = payload.get("doc_id") or ""
    if q and doc_id and payload.get("model") == "hybrid_rrf":
        aux = explain_bm25(q, doc_id)
        if not aux.get("error"):
            stems_bm = {
                t["term"]
                for t in aux.get("terms", [])
                if t.get("status") == "matched"
            }
            return stems_bm
    return stems


def _segment_query_visual(raw: str, contributing: set[str]) -> list[dict[str, Any]]:
    """Preserve exact query spacing/punctuation; stem each word like retrieval."""
    out: list[dict[str, Any]] = []
    if not raw:
        return out
    for m in re.finditer(r"\w+|\W+", raw):
        chunk = m.group()
        if re.fullmatch(r"\w+", chunk):
            stems = preprocess(chunk)
            stem = stems[0] if stems else None
            out.append(
                {
                    "text": chunk,
                    "word": True,
                    "stem": stem,
                    "indexed": stem is not None,
                    "contributing": stem is not None and stem in contributing,
                }
            )
        else:
            out.append({"text": chunk, "word": False})
    return out


def _segment_document_visual(text: str, contributing: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"\w+|\W+", text):
        chunk = m.group()
        if re.fullmatch(r"\w+", chunk):
            stems = preprocess(chunk)
            stem = stems[0] if stems else None
            hit = stem is not None and stem in contributing
            out.append({"text": chunk, "word": True, "hit": hit})
        else:
            out.append({"text": chunk, "word": False, "hit": False})
    return out


def enrich_visualization(payload: dict[str, Any], raw_query: str, doc_id: str) -> None:
    contributing = _contributing_stems(payload)
    payload["contributing_stems"] = sorted(contributing)
    payload["query_segments"] = _segment_query_visual(raw_query, contributing)

    doc_text = _load_documents_map().get(doc_id, "")
    truncated = len(doc_text) > MAX_EXPLAIN_DOC_CHARS
    if truncated:
        doc_text = doc_text[:MAX_EXPLAIN_DOC_CHARS]
    payload["document_truncated"] = truncated
    payload["document_segments"] = _segment_document_visual(doc_text, contributing)


def _rank_and_score(score_fn, query: str, doc_id: str, depth: int) -> tuple[int | None, float]:
    ranked = score_fn(query, depth)
    for i, (d, s) in enumerate(ranked, start=1):
        if d == doc_id:
            return i, float(s)
    return None, 0.0


def explain_bm25(query: str, doc_id: str) -> dict[str, Any]:
    from retrieval import adl, b, doc_lengths, idf_bm25, int_id_lookup, inverted, k1, k2, N

    int_id = int_id_lookup.get(doc_id)
    if int_id is None:
        return {"error": "unknown_doc"}

    tokens = preprocess(query)
    query_tf = Counter(tokens)
    dl = doc_lengths[int_id]
    term_rows: list[dict[str, Any]] = []
    total = 0.0

    for term in sorted(set(tokens)):
        qf = query_tf[term]
        qc = (qf * (k2 + 1.0)) / (qf + k2)
        if term not in inverted:
            term_rows.append(
                {
                    "term": term,
                    "qf": qf,
                    "status": "oov_index",
                    "query_component": qc,
                    "contribution": 0.0,
                }
            )
            continue
        posting = inverted[term]
        if int_id not in posting:
            term_rows.append(
                {
                    "term": term,
                    "qf": qf,
                    "status": "missing_in_doc",
                    "idf": idf_bm25[term],
                    "query_component": qc,
                    "contribution": 0.0,
                }
            )
            continue
        f = posting[int_id]
        K = k1 * ((1.0 - b) + b * (dl / adl))
        dc = (f * (k1 + 1.0)) / (f + K)
        idf_v = idf_bm25[term]
        contrib = idf_v * dc * qc
        total += contrib
        term_rows.append(
            {
                "term": term,
                "qf": qf,
                "status": "matched",
                "idf": idf_v,
                "tf_doc": f,
                "dl": dl,
                "adl": adl,
                "K": K,
                "k1": k1,
                "b": b,
                "k2": k2,
                "doc_component": dc,
                "query_component": qc,
                "contribution": contrib,
            }
        )

    return {
        "model": "bm25",
        "title": "BM25 (Okapi)",
        "doc_id": doc_id,
        "query": query,
        "query_tokens": tokens,
        "constants": {"k1": k1, "b": b, "k2": k2, "adl": adl, "N": N, "dl": dl},
        "total_score": total,
        "formula": {
            "name": "BM25 sum over query terms",
            "latex": r"s(q,d)=\sum_{t\in q} \mathrm{IDF}(t)\cdot \frac{f_{t,d}(k_1+1)}{f_{t,d}+K}\cdot \frac{qf_t(k_2+1)}{qf_t+k_2},\quad K=k_1\cdot\left((1-b)+b\cdot\frac{\lvert d\rvert}{\mathrm{adl}}\right)",
            "steps": [
                "Preprocess query → stems (Porter); drop English stopwords.",
                "For each stem t in the query: take collection IDF(t), doc frequency f_{t,d}, doc length |d|.",
                "Saturation on the query side uses qf_t (stem frequency in the query) with k₂.",
            ],
        },
        "terms": term_rows,
    }


def explain_vsm(query: str, doc_id: str) -> dict[str, Any]:
    from retrieval import doc_norms, idf_smooth, int_id_lookup, inverted, tfidf_vectors

    int_id = int_id_lookup.get(doc_id)
    if int_id is None:
        return {"error": "unknown_doc"}

    tokens = preprocess(query)
    query_tf = Counter(tokens)
    query_vec: dict[str, float] = {}
    for term in set(tokens):
        qtf = query_tf[term]
        tf_log = 1.0 + math.log(qtf)
        query_vec[term] = tf_log * idf_smooth.get(term, 0.0)

    query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
    dn = doc_norms[int_id]
    term_rows: list[dict[str, Any]] = []
    dot = 0.0

    for term in sorted(query_vec.keys()):
        qv = query_vec[term]
        if term not in inverted or int_id not in inverted[term]:
            dv = 0.0
            term_rows.append(
                {
                    "term": term,
                    "status": "missing_in_doc",
                    "query_weight": qv,
                    "doc_weight": dv,
                    "product": 0.0,
                }
            )
            continue
        dv = tfidf_vectors[int_id].get(term, 0.0)
        p = qv * dv
        dot += p
        term_rows.append(
            {
                "term": term,
                "status": "matched",
                "query_weight": qv,
                "doc_weight": dv,
                "product": p,
            }
        )

    cosine = dot / (query_norm * dn) if query_norm > 0 and dn > 0 else 0.0

    return {
        "model": "vsm",
        "title": "Vector space model (TF-IDF cosine)",
        "doc_id": doc_id,
        "query": query,
        "query_tokens": tokens,
        "constants": {
            "query_l2_norm": query_norm,
            "doc_l2_norm": dn,
            "dot_product": dot,
        },
        "total_score": cosine,
        "formula": {
            "name": "Cosine similarity",
            "latex": r"\mathrm{sim}(q,d)=\frac{\mathbf{q}\cdot\mathbf{d}}{\|\mathbf{q}\|\,\|\mathbf{d}\|}",
            "steps": [
                "Query TF-IDF: (1+log(tf_q(t)))·IDF_smooth(t); OOV terms get zero.",
                "Document vectors are precomputed TF-IDF weights per posting.",
                "Cosine = dot product divided by query and document L2 norms.",
            ],
        },
        "terms": term_rows,
    }


def explain_bim(query: str, doc_id: str) -> dict[str, Any]:
    from retrieval import idf_bim, int_id_lookup, inverted

    int_id = int_id_lookup.get(doc_id)
    if int_id is None:
        return {"error": "unknown_doc"}

    tokens = preprocess(query)
    term_rows: list[dict[str, Any]] = []
    total = 0.0

    for term in sorted(set(tokens)):
        if term not in inverted:
            term_rows.append({"term": term, "status": "oov_index", "weight": None, "contribution": 0.0})
            continue
        if int_id not in inverted[term]:
            term_rows.append(
                {
                    "term": term,
                    "status": "missing_in_doc",
                    "weight": idf_bim[term],
                    "contribution": 0.0,
                }
            )
            continue
        w = idf_bim[term]
        total += w
        term_rows.append({"term": term, "status": "matched", "weight": w, "contribution": w})

    return {
        "model": "bim",
        "title": "Binary independence model",
        "doc_id": doc_id,
        "query": query,
        "query_tokens": tokens,
        "total_score": total,
        "formula": {
            "name": "Sum of log-odds weights",
            "latex": r"s(q,d)=\sum_{t\in q\cap d} w^{BIM}_t",
            "steps": [
                "Binary document: term either present or absent.",
                "Each matching term adds a precomputed BIM weight (log odds vs collection).",
            ],
        },
        "terms": term_rows,
    }


def explain_lm_dirichlet(query: str, doc_id: str) -> dict[str, Any]:
    from retrieval import doc_lengths, int_id_lookup, inverted, mu, p_collection

    int_id = int_id_lookup.get(doc_id)
    if int_id is None:
        return {"error": "unknown_doc"}

    tokens = preprocess(query)
    query_tf = Counter(tokens)
    dl = doc_lengths[int_id]
    term_rows: list[dict[str, Any]] = []
    total = 0.0

    for term in sorted(set(tokens)):
        qfreq = query_tf[term]
        if term not in inverted:
            term_rows.append(
                {
                    "term": term,
                    "qf": qfreq,
                    "status": "oov_index",
                    "contribution": 0.0,
                }
            )
            continue
        posting = inverted[term]
        if int_id not in posting:
            term_rows.append(
                {
                    "term": term,
                    "qf": qfreq,
                    "status": "missing_in_doc",
                    "contribution": 0.0,
                    "note": "Retriever skips documents without postings for this term.",
                }
            )
            continue
        f = posting[int_id]
        p_coll = p_collection.get(term, 1e-10)
        p_td = (f + mu * p_coll) / (dl + mu)
        contrib = qfreq * math.log(p_td)
        total += contrib
        term_rows.append(
            {
                "term": term,
                "qf": qfreq,
                "status": "matched",
                "tf_doc": f,
                "p_coll": p_coll,
                "mu": mu,
                "dl": dl,
                "p_td": p_td,
                "log_p_td": math.log(p_td),
                "contribution": contrib,
            }
        )

    return {
        "model": "lm_dirichlet",
        "title": "Language model (Dirichlet smoothing)",
        "doc_id": doc_id,
        "query": query,
        "query_tokens": tokens,
        "constants": {"mu": mu, "dl": dl},
        "total_score": total,
        "formula": {
            "name": "Query likelihood under smoothed multinomial",
            "latex": r"\log P(q \mid d)=\sum_{t\in q} qf_t\cdot \log\frac{f_{t,d}+\mu\,P(t \mid C)}{\lvert d\rvert+\mu}",
            "steps": [
                "Collection prior P(t|C) comes from the indexed corpus.",
                "μ pulls sparse estimates toward the collection when f_{t,d} is small.",
            ],
        },
        "terms": term_rows,
    }


def explain_rocchio_prf(query: str, doc_id: str) -> dict[str, Any]:
    from retrieval import (
        adl,
        b,
        doc_lengths,
        doc_norms,
        idf_smooth,
        int_id_lookup,
        inverted,
        k1,
        k2,
        tfidf_vectors,
    )
    from retrieval import score_bm25 as score_bm25_fn

    int_id = int_id_lookup.get(doc_id)
    if int_id is None:
        return {"error": "unknown_doc"}

    fb_docs = 15
    fb_terms = 25
    alpha = 0.4
    beta = 0.6

    pass1_results = score_bm25_fn(query, fb_docs)
    tokens = preprocess(query)
    query_tf = Counter(tokens)
    query_term_set = set(tokens)
    q_vec: dict[str, float] = {}
    for term in set(tokens):
        qtf = query_tf[term]
        tf_log = (1.0 + math.log(qtf)) if qtf > 0 else 0.0
        q_vec[term] = tf_log * idf_smooth.get(term, 0.0)

    centroid = defaultdict(float)
    for ddoc_id, _ in pass1_results:
        iid = int_id_lookup[ddoc_id]
        for term, weight in tfidf_vectors[iid].items():
            centroid[term] += weight / fb_docs

    expanded = defaultdict(float)
    for term, w in q_vec.items():
        expanded[term] += alpha * w
    for term, w in centroid.items():
        expanded[term] += beta * w

    expansion_candidates = [
        (term, expanded[term]) for term in expanded if term not in query_term_set
    ]
    expansion_candidates.sort(key=lambda x: x[1], reverse=True)
    top_expansions = expansion_candidates[:fb_terms]

    final_vec = dict(q_vec)
    for term, w in top_expansions:
        final_vec[term] = w

    query_norm = math.sqrt(sum(v * v for v in final_vec.values()))
    dn = doc_norms[int_id]
    term_rows: list[dict[str, Any]] = []
    dot = 0.0

    for term in sorted(final_vec.keys()):
        w = final_vec[term]
        if term not in inverted or int_id not in inverted[term]:
            dv = 0.0
            term_rows.append(
                {
                    "term": term,
                    "status": "missing_in_doc",
                    "expanded_weight": w,
                    "doc_weight": dv,
                    "product": 0.0,
                    "from_feedback": term not in query_term_set,
                }
            )
            continue
        dv = tfidf_vectors[int_id].get(term, 0.0)
        p = w * dv
        dot += p
        term_rows.append(
            {
                "term": term,
                "status": "matched",
                "expanded_weight": w,
                "doc_weight": dv,
                "product": p,
                "from_feedback": term not in query_term_set,
            }
        )

    cosine = dot / (query_norm * dn) if query_norm > 0 and dn > 0 else 0.0

    return {
        "model": "rocchio_prf",
        "title": "Rocchio pseudo-relevance feedback + cosine",
        "doc_id": doc_id,
        "query": query,
        "query_tokens": tokens,
        "constants": {
            "alpha": alpha,
            "beta": beta,
            "fb_docs": fb_docs,
            "fb_terms": fb_terms,
            "query_l2_norm": query_norm,
            "doc_l2_norm": dn,
            "dot_product": dot,
            "pass1_bm25_docs": [d for d, _ in pass1_results],
            "note_k_params_unused": "Rocchio stage uses TF-IDF vectors only (k₁,k₂ shown for corpus context).",
            "index_k1": k1,
            "index_k2": k2,
            "index_b": b,
            "adl": adl,
            "dl": doc_lengths[int_id],
        },
        "total_score": cosine,
        "formula": {
            "name": "Rocchio expansion then cosine",
            "latex": (
                r"\mathbf{q}'=\alpha\mathbf{q}+\beta\cdot\frac{1}{\lvert D_{\mathrm{fb}}\rvert}"
                r"\sum_{d\in D_{\mathrm{fb}}}\mathbf{d};\quad "
                r"\mathrm{sim}(q',d)=\frac{\mathbf{q}'\cdot \mathbf{d}}"
                r"{\left\lVert \mathbf{q}' \right\rVert \, \left\lVert \mathbf{d} \right\rVert}"
            ),
            "steps": [
                "BM25 retrieves pseudo-relevant set D_fb.",
                "Centroid of feedback TF-IDF vectors mixes into the query with β; original scaled by α.",
                "Top expanded dimensions added; cosine vs document vector yields the score.",
            ],
        },
        "terms": term_rows,
        "expanded_terms_preview": [{"term": t, "weight": round(w, 6)} for t, w in top_expansions[:12]],
    }


def explain_rm3(query: str, doc_id: str) -> dict[str, Any]:
    from retrieval import (
        adl,
        b,
        doc_lengths,
        idf_bm25,
        int_id_lookup,
        inverted,
        k1,
        k2,
        mu,
        p_collection,
        tfidf_vectors,
    )
    from retrieval import score_bm25 as score_bm25_fn

    int_id = int_id_lookup.get(doc_id)
    if int_id is None:
        return {"error": "unknown_doc"}

    fb_docs = 10
    fb_terms = 20
    lam = 0.5

    pass1 = score_bm25_fn(query, fb_docs * 2)
    if not pass1:
        return {"error": "no_pass1"}

    pass1_head = pass1[:fb_docs]
    total_mass = sum(s for _, s in pass1_head)
    if total_mass <= 0.0:
        return {"error": "zero_mass"}

    p_d_given_q = {d: s / total_mass for d, s in pass1_head}

    rm1: defaultdict[str, float] = defaultdict(float)
    for did, p_dq in p_d_given_q.items():
        iid = int_id_lookup[did]
        dl_fb = doc_lengths[iid]
        for term in tfidf_vectors[iid]:
            f = inverted[term][iid]
            p_coll = p_collection.get(term, 1e-10)
            p_t_d = (f + mu * p_coll) / (dl_fb + mu)
            rm1[term] += p_dq * p_t_d

    s_rm1 = sum(rm1.values())
    if s_rm1 > 0.0:
        for t in rm1:
            rm1[t] /= s_rm1

    tokens = preprocess(query)
    query_tf = Counter(tokens)
    total_qf = sum(query_tf.values())
    if total_qf <= 0:
        return {"error": "empty_query"}

    p_t_q = {t: query_tf[t] / total_qf for t in query_tf}

    rm3_w = defaultdict(float)
    for term in set(rm1) | set(p_t_q):
        rm3_w[term] = lam * p_t_q.get(term, 0.0) + (1.0 - lam) * rm1.get(term, 0.0)

    selected_terms = sorted(rm3_w.keys(), key=lambda t: rm3_w[t], reverse=True)[:fb_terms]

    dl = doc_lengths[int_id]
    term_rows: list[dict[str, Any]] = []
    total_score = 0.0

    for term in sorted(selected_terms):
        qf_weight = rm3_w[term]
        qc = (qf_weight * (k2 + 1.0)) / (qf_weight + k2)
        if term not in inverted:
            term_rows.append(
                {
                    "term": term,
                    "rm3_weight": qf_weight,
                    "status": "oov_index",
                    "query_component": qc,
                    "contribution": 0.0,
                }
            )
            continue
        posting = inverted[term]
        if int_id not in posting:
            term_rows.append(
                {
                    "term": term,
                    "rm3_weight": qf_weight,
                    "status": "missing_in_doc",
                    "idf": idf_bm25[term],
                    "query_component": qc,
                    "contribution": 0.0,
                }
            )
            continue
        f = posting[int_id]
        K = k1 * ((1.0 - b) + b * (dl / adl))
        dc = (f * (k1 + 1.0)) / (f + K)
        idf_v = idf_bm25[term]
        contrib = idf_v * dc * qc
        total_score += contrib
        term_rows.append(
            {
                "term": term,
                "rm3_weight": qf_weight,
                "status": "matched",
                "idf": idf_v,
                "tf_doc": f,
                "K": K,
                "doc_component": dc,
                "query_component": qc,
                "contribution": contrib,
            }
        )

    return {
        "model": "rm3",
        "title": "RM3 → BM25 rescoring",
        "doc_id": doc_id,
        "query": query,
        "query_tokens": tokens,
        "constants": {
            "lambda": lam,
            "fb_docs": fb_docs,
            "fb_terms": fb_terms,
            "mu_rm1": mu,
            "dl": dl,
            "adl": adl,
            "k1": k1,
            "k2": k2,
            "b": b,
        },
        "total_score": total_score,
        "formula": {
            "name": "RM1 smoothed toward query (RM3), then BM25-style sum",
            "latex": r"P_{RM3}(t)=\lambda P_Q(t)+(1-\lambda)P_{RM1}(t);\quad s=\sum_t \mathrm{IDF}(t)\cdot\frac{f(k_1+1)}{f+K}\cdot\frac{w(k_2+1)}{w+k_2}",
            "steps": [
                "BM25 builds pseudo-relevant top documents and weights them.",
                "RM1 aggregates smoothed multinomial term likelihoods from feedback docs.",
                "Mix with query unigrams using λ; top weighted terms drive second-stage BM25.",
            ],
        },
        "feedback_docs": [{"doc_id": d, "p_feedback": round(p, 8)} for d, p in p_d_given_q.items()],
        "terms": term_rows,
    }


def explain_hybrid_rrf(query: str, doc_id: str, depth: int = 400) -> dict[str, Any]:
    from retrieval import score_bm25, score_rm3, score_vsm

    k_rrf = 60
    fns = [
        ("BM25", score_bm25),
        ("VSM", score_vsm),
        ("RM3", score_rm3),
    ]

    fusion_rows: list[dict[str, Any]] = []
    total_rrf = 0.0
    for label, fn in fns:
        rnk, raw_sc = _rank_and_score(fn, query, doc_id, depth)
        comp = (1.0 / (k_rrf + rnk)) if rnk is not None else 0.0
        total_rrf += comp
        fusion_rows.append(
            {
                "ranker": label,
                "rank": rnk,
                "raw_score": raw_sc,
                "rrf_component": comp,
            }
        )

    return {
        "model": "hybrid_rrf",
        "title": "Hybrid RRF",
        "doc_id": doc_id,
        "query": query,
        "query_tokens": preprocess(query),
        "constants": {"k_rrf": k_rrf, "depth_searched": depth},
        "total_score": total_rrf,
        "formula": {
            "name": "Reciprocal Rank Fusion",
            "latex": r"\mathrm{RRF}(d)=\sum_{r\in\mathcal{R}}\frac{1}{k+\mathrm{rank}_r(d)}",
            "steps": [
                "Each base ranker produces an ordering (BM25, VSM, RM3).",
                "Contributions add inversely with rank plus constant k.",
                "No raw-score normalization across rankers.",
            ],
        },
        "fusion": fusion_rows,
        "terms": [],
    }


def build_explanation(query: str, doc_id: str, model: str) -> dict[str, Any]:
    model_l = model.strip().lower()
    rq = query.strip()
    did = doc_id.strip()

    dispatch: dict[str, Any] = {
        "bm25": explain_bm25,
        "vsm": explain_vsm,
        "bim": explain_bim,
        "lm_dirichlet": explain_lm_dirichlet,
        "rocchio_prf": explain_rocchio_prf,
        "rm3": explain_rm3,
        "hybrid_rrf": explain_hybrid_rrf,
    }
    fn = dispatch.get(model_l)
    if fn is None:
        return {"error": "unknown_model"}
    out = fn(rq, did)
    if out.get("error"):
        return out
    enrich_visualization(out, rq, did)
    return out
