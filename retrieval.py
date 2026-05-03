"""
Online retrieval: only sparse traversals of inverted lists and cheap arithmetic.
All corpus statistics come from index.pkl (loaded once at import).
"""

import math
import pickle
from collections import Counter, defaultdict

from text_processor import preprocess

with open("index.pkl", "rb") as f:
    _idx = pickle.load(f)

inverted = _idx["inverted"]
doc_ids = _idx["doc_ids"]
doc_lengths = _idx["doc_lengths"]
N = _idx["N"]
adl = _idx["adl"]
idf_bm25 = _idx["idf_bm25"]
idf_smooth = _idx["idf_smooth"]
idf_bim = _idx["idf_bim"]
tfidf_vectors = _idx["tfidf_vectors"]
doc_norms = _idx["doc_norms"]
p_collection = _idx["p_collection"]
mu = _idx["mu"]
k1 = _idx["k1"]
b = _idx["b"]
k2 = _idx["k2"]

# Map external doc_id back to dense integer row index for vector / posting lookups
int_id_lookup = {doc_id: i for i, doc_id in enumerate(doc_ids)}


def score_bm25(query, top_k):
    tokens = preprocess(query)
    query_tf = Counter(tokens)
    scores = defaultdict(float)

    for term in set(tokens):
        if term not in inverted:
            continue
        qf = query_tf[term]
        # Query-side BM25 saturation: repeated query terms still add mass, but taper
        query_component = (qf * (k2 + 1.0)) / (qf + k2)
        for int_id, f in inverted[term].items():
            dl = doc_lengths[int_id]
            # Doc-length normalization in denominator K (Okapi BM25)
            K = k1 * ((1.0 - b) + b * (dl / adl))
            doc_component = (f * (k1 + 1.0)) / (f + K)
            scores[int_id] += idf_bm25[term] * doc_component * query_component

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_ids[i], s) for i, s in ranked]


def score_vsm(query, top_k):
    tokens = preprocess(query)
    query_tf = Counter(tokens)

    query_vec = {}
    for term in set(tokens):
        qtf = query_tf[term]
        tf_log = 1.0 + math.log(qtf)
        # OOV query terms get zero smooth-IDF mass (not in corpus vocabulary)
        query_vec[term] = tf_log * idf_smooth.get(term, 0.0)

    query_norm = math.sqrt(sum(v * v for v in query_vec.values()))
    if query_norm == 0.0:
        return []

    dot_products = defaultdict(float)
    for term, qv in query_vec.items():
        if term not in inverted:
            continue
        for int_id in inverted[term]:
            dot_products[int_id] += qv * tfidf_vectors[int_id].get(term, 0.0)

    scores = {}
    for int_id, dot in dot_products.items():
        dn = doc_norms[int_id]
        if dn > 0.0:
            # Cosine similarity = dot product / (||q|| * ||d||); both norms precomputed for d
            scores[int_id] = dot / (query_norm * dn)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_ids[i], s) for i, s in ranked]


def score_bim(query, top_k):
    # Binary independence: binary doc representation; score is sum of precomputed log-odds
    tokens = set(preprocess(query))
    scores = defaultdict(float)

    for term in tokens:
        if term not in inverted:
            continue
        weight = idf_bim[term]
        for int_id in inverted[term]:
            scores[int_id] += weight

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_ids[i], s) for i, s in ranked]


def score_lm_dirichlet(query, top_k):
    tokens = preprocess(query)
    query_tf = Counter(tokens)
    scores = defaultdict(float)

    for term in set(tokens):
        if term not in inverted:
            continue
        p_coll = p_collection.get(term, 1e-10)
        qfreq = query_tf[term]
        for int_id, f in inverted[term].items():
            dl = doc_lengths[int_id]
            # Dirichlet-smoothed multinomial: μ pulls estimate toward collection p(t|C)
            p_td = (f + mu * p_coll) / (dl + mu)
            scores[int_id] += qfreq * math.log(p_td)

    # Drop docs that never matched any query term: defaultdict leaves them at 0.0,
    # which incorrectly outranks true negative log-likelihood scores when sorting descending.
    ranked = sorted(
        ((i, s) for i, s in scores.items() if s != 0.0),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    return [(doc_ids[i], s) for i, s in ranked]


def score_rocchio_prf(query, top_k, fb_docs=15, fb_terms=25, alpha=0.4, beta=0.6):
    # Pseudo-relevance feedback: centroid of top BM25 docs expands the query vector (Rocchio).
    pass1_results = score_bm25(query, fb_docs)

    tokens = preprocess(query)
    query_tf = Counter(tokens)
    query_term_set = set(tokens)
    q_vec = {}
    for term in set(tokens):
        qtf = query_tf[term]
        # Sublinear TF on the original query (same scaling as VSM query branch)
        tf_log = (1.0 + math.log(qtf)) if qtf > 0 else 0.0
        q_vec[term] = tf_log * idf_smooth.get(term, 0.0)

    centroid = defaultdict(float)
    for doc_id, _ in pass1_results:
        int_id = int_id_lookup[doc_id]
        for term, weight in tfidf_vectors[int_id].items():
            # Average feedback document vector (already TF-IDF weighted per dimension)
            centroid[term] += weight / fb_docs

    expanded = defaultdict(float)
    for term, w in q_vec.items():
        expanded[term] += alpha * w
    for term, w in centroid.items():
        expanded[term] += beta * w

    # Highest-weight terms not in the original query become expansion dimensions
    expansion_candidates = [
        (term, expanded[term])
        for term in expanded
        if term not in query_term_set
    ]
    expansion_candidates.sort(key=lambda x: x[1], reverse=True)
    top_expansions = expansion_candidates[:fb_terms]

    final_vec = dict(q_vec)
    for term, w in top_expansions:
        final_vec[term] = w

    query_norm = math.sqrt(sum(v * v for v in final_vec.values()))
    if query_norm == 0.0:
        return []

    dot = defaultdict(float)
    for term, w in final_vec.items():
        if term not in inverted:
            continue
        for int_id in inverted[term]:
            dot[int_id] += w * tfidf_vectors[int_id].get(term, 0.0)

    scores = {}
    for int_id, d in dot.items():
        dn = doc_norms[int_id]
        if dn > 0.0 and query_norm > 0.0:
            scores[int_id] = d / (query_norm * dn)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_ids[i], s) for i, s in ranked]


def score_rm3(query, top_k, fb_docs=10, fb_terms=20, lam=0.5):
    # Lavrenko-style relevance model (RM1) smoothed toward the original query (RM3), then BM25 rescore.
    pass1 = score_bm25(query, fb_docs * 2)
    if not pass1:
        return []
    pass1_head = pass1[:fb_docs]
    total = sum(s for _, s in pass1_head)
    if total <= 0.0:
        return []
    # Feedback document distribution P(d|q) ∝ first-stage BM25 scores (truncated renormalized)
    p_d_given_q = {doc_id: s / total for doc_id, s in pass1_head}

    rm1 = defaultdict(float)
    for doc_id, p_dq in p_d_given_q.items():
        int_id = int_id_lookup[doc_id]
        dl = doc_lengths[int_id]
        for term in tfidf_vectors[int_id]:
            f = inverted[term][int_id]
            p_coll = p_collection.get(term, 1e-10)
            # Smoothed multinomial likelihood of term t in document d (same μ as index LM)
            p_t_d = (f + mu * p_coll) / (dl + mu)
            rm1[term] += p_dq * p_t_d

    s_rm1 = sum(rm1.values())
    if s_rm1 > 0.0:
        for t in rm1:
            rm1[t] /= s_rm1

    tokens = preprocess(query)
    query_tf = Counter(tokens)
    total_qf = sum(query_tf.values())
    if total_qf <= 0:
        return []
    p_t_q = {t: query_tf[t] / total_qf for t in query_tf}

    rm3 = defaultdict(float)
    for term in set(rm1) | set(p_t_q):
        # λ mixes original query unigram prior with RM1 feedback distribution
        rm3[term] = lam * p_t_q.get(term, 0.0) + (1.0 - lam) * rm1.get(term, 0.0)

    selected_terms = sorted(rm3.keys(), key=lambda t: rm3[t], reverse=True)[:fb_terms]

    scores = defaultdict(float)
    for term in selected_terms:
        if term not in inverted:
            continue
        qf_weight = rm3[term]
        query_component = (qf_weight * (k2 + 1.0)) / (qf_weight + k2)
        for int_id, f in inverted[term].items():
            dl = doc_lengths[int_id]
            K = k1 * ((1.0 - b) + b * (dl / adl))
            doc_component = (f * (k1 + 1.0)) / (f + K)
            scores[int_id] += idf_bm25[term] * doc_component * query_component

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_ids[i], s) for i, s in ranked]


def score_hybrid_rrf(query, top_k, score_fns=None, k_rrf=60):
    # Reciprocal Rank Fusion over arbitrary rankers (default: BM25 + VSM + RM3).
    if score_fns is None:
        score_fns = [score_bm25, score_vsm, score_rm3]

    rrf_scores = defaultdict(float)
    for fn in score_fns:
        ranked = fn(query, top_k * 2)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ranked[:top_k]]
