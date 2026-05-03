"""
Offline index construction: tokenize corpus once, precompute IDF variants,
TF-IDF vectors + L2 norms, and Dirichlet LM collection statistics. Query time
only reads the pickled structures.
"""

import json
import math
import pickle
from collections import Counter, defaultdict

from text_processor import preprocess


def _open_json_first(path):
    # Prefer cwd; fall back to dataset/ for common course layouts (no extra deps)
    for candidate in (path, "dataset/" + path):
        try:
            return open(candidate, "r", encoding="utf-8")
        except OSError:
            continue
    raise FileNotFoundError(path)


def _safe_bim_idf(df_term, N):
    # bim_pt = p(t|NR) ≈ fraction of docs containing t; clip so log-odds is finite
    pt = df_term / N
    pt = min(max(pt, 1e-12), 1.0 - 1e-12)
    return math.log((1.0 - pt) / pt), df_term / N


def build_index(documents_path="documents.json", out_path="index.pkl"):
    with _open_json_first(documents_path) as f:
        documents = json.load(f)

    doc_ids = []
    doc_lengths = []
    tf_raw = []
    df = defaultdict(int)
    total_tokens = 0

    # STEP 1 — per-doc token counts and document frequencies
    for int_id, doc in enumerate(documents):
        tokens = preprocess(doc["text"])
        doc_ids.append(doc["doc_id"])
        dl = len(tokens)
        doc_lengths.append(dl)
        total_tokens += dl
        ctr = Counter(tokens)
        tf_raw.append(ctr)
        for term in ctr:
            df[term] += 1

    N = len(documents)
    adl = total_tokens / N if N else 0.0

    # STEP 2 — inverted index: raw TF postings only (no weights here)
    inverted = {}
    for term in df:
        inverted[term] = {}
    for int_id in range(N):
        for term, tf in tf_raw[int_id].items():
            inverted[term][int_id] = tf

    # STEP 3 — corpus-level IDF / log-odds weights (computed once)
    idf_standard = {}
    idf_smooth = {}
    idf_bm25 = {}
    idf_bim = {}
    bim_pt = {}
    for term, df_t in df.items():
        # Standard IDF: rare terms get larger weight; log dampens extremes
        idf_standard[term] = math.log(N / df_t)
        # Smooth IDF: avoids log(0) when df is small; +1 shifts curve (Robertson-style)
        idf_smooth[term] = math.log((N + 1) / (df_t + 1)) + 1.0
        # BM25 Robertson–Jones IDF with 0.5 smoothing constants in numerator/denominator
        idf_bm25[term] = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)
        idf_b_val, pt = _safe_bim_idf(df_t, N)
        idf_bim[term] = idf_b_val
        bim_pt[term] = pt

    # STEP 4 — VSM: sublinear TF × smooth IDF, plus L2 norm for cosine
    tfidf_vectors = []
    doc_norms = []
    for int_id in range(N):
        vec = {}
        for term, tf in tf_raw[int_id].items():
            # Sublinear TF: diminishing returns for repeated within-doc occurrences
            tf_log = 1.0 + math.log(tf)
            vec[term] = tf_log * idf_smooth[term]
        tfidf_vectors.append(vec)
        doc_norms.append(math.sqrt(sum(v * v for v in vec.values())))

    # STEP 5 — Dirichlet LM: collection unigram distribution p(t|C)
    total_tokens_all = sum(doc_lengths)
    collection_tf = defaultdict(int)
    for int_id in range(N):
        for term, tf in tf_raw[int_id].items():
            collection_tf[term] += tf
    p_collection = {
        term: ctf / total_tokens_all for term, ctf in collection_tf.items()
    }
    mu = 2000

    index = {
        "inverted": dict(inverted),
        "doc_ids": doc_ids,
        "doc_lengths": doc_lengths,
        "N": N,
        "adl": adl,
        "df": dict(df),
        "idf_standard": idf_standard,
        "idf_smooth": idf_smooth,
        "idf_bm25": idf_bm25,
        "idf_bim": idf_bim,
        "tfidf_vectors": tfidf_vectors,
        "doc_norms": doc_norms,
        "p_collection": p_collection,
        "mu": mu,
        "bim_pt": bim_pt,
        "k1": 1.5,
        "b": 0.75,
        "k2": 500,
    }

    with open(out_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    build_index("documents.json")
