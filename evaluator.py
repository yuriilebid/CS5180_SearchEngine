"""
Offline evaluation metrics: AP, precision@k, NDCG@k from ranked doc-id lists.
"""

import json
import math
from collections import Counter, defaultdict

from text_processor import preprocess


def _open_json_first(path):
    for candidate in (path, "dataset/" + path):
        try:
            return open(candidate, "r", encoding="utf-8")
        except OSError:
            continue
    raise FileNotFoundError(path)


def _load_queries_and_qrels(queries_path="queries.json", qrels_path="qrels.json"):
    with _open_json_first(queries_path) as f:
        queries = json.load(f)
    with _open_json_first(qrels_path) as f:
        qrels = json.load(f)
    relevant_map = {}
    for row in qrels:
        qid = row["query_id"]
        relevant_map.setdefault(qid, set()).add(row["doc_id"])
    return queries, relevant_map


def average_precision(ranked_ids, relevant_set):
    hits = 0
    sum_p = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            sum_p += hits / rank
    return sum_p / len(relevant_set) if relevant_set else 0.0


def precision_at_k(ranked_ids, relevant_set, k):
    head = ranked_ids[:k]
    return sum(1 for d in head if d in relevant_set) / k if k else 0.0


def ndcg_at_k(ranked_ids, relevant_set, k):
    # DCG: binary relevance discounted by log2(rank+1) (rank starts at 1)
    dcg = 0.0
    for rank, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    rel_n = len(relevant_set)
    # Ideal DCG@k: min(|rel|, k) relevant items at the top positions
    ones = min(rel_n, k)
    ideal = [1] * ones + [0] * (k - ones)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(score_fn, retrieve_count=25):
    queries, relevant_map = _load_queries_and_qrels()
    results = {}
    for q in queries:
        qid = q["query_id"]
        ranked = score_fn(q["text"], retrieve_count)
        ranked_ids = [doc_id for doc_id, _ in ranked]
        relevant = relevant_map.get(qid, set())
        results[qid] = {
            "ap": average_precision(ranked_ids, relevant),
            "p5": precision_at_k(ranked_ids, relevant, 5),
            "p10": precision_at_k(ranked_ids, relevant, 10),
            "ndcg10": ndcg_at_k(ranked_ids, relevant, 10),
            "query": q["text"],
        }
    return results


def grid_search_bm25(
    inverted,
    doc_ids,
    doc_lengths,
    adl,
    idf_bm25,
    queries,
    relevant_map,
    retrieve_count=25,
):
    """
    Sweep BM25 (k1, b) on frozen postings + idf_bm25; k2 fixed at 500.
    MAP is mean average precision over all judged queries.
    """
    k1_values = [0.9, 1.2, 1.5, 1.8, 2.0]
    b_values = [0.3, 0.5, 0.65, 0.75, 0.9]
    k2 = 500.0

    def ranked_ids_for_query(text, k1, b_):
        tokens = preprocess(text)
        query_tf = Counter(tokens)
        scores = defaultdict(float)
        for term in set(tokens):
            if term not in inverted:
                continue
            qf = query_tf[term]
            query_component = (qf * (k2 + 1.0)) / (qf + k2)
            for int_id, f in inverted[term].items():
                dl = doc_lengths[int_id]
                # BM25 length normalization: b controls doc-length prior vs average adl
                K = k1 * ((1.0 - b_) + b_ * (dl / adl))
                doc_component = (f * (k1 + 1.0)) / (f + K)
                scores[int_id] += idf_bm25[term] * doc_component * query_component
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:retrieve_count]
        return [doc_ids[i] for i, _ in ranked]

    grid_map = {}
    for k1 in k1_values:
        for b_ in b_values:
            aps = []
            for q in queries:
                qid = q["query_id"]
                rel = relevant_map.get(qid, set())
                ranked_ids = ranked_ids_for_query(q["text"], k1, b_)
                aps.append(average_precision(ranked_ids, rel))
            grid_map[(k1, b_)] = sum(aps) / len(aps) if aps else 0.0

    best_pair, best_map = max(grid_map.items(), key=lambda kv: kv[1])

    # Header: rows are BM25 k1; columns are length-normalization hyperparameter b
    # (label not inline in f-string — backslashes inside {...} are a SyntaxError)
    k1_b_corner = "k1\\b"
    print(f"{k1_b_corner:<8} | " + " | ".join(f"{bv:>6.2f}" for bv in b_values))
    width = 8 + 3 + sum(3 + 6 for _ in b_values) + (len(b_values) - 1) * 3
    print("-" * max(width, 60))
    for k1 in k1_values:
        cells = " | ".join(f"{grid_map[(k1, bv)]:6.4f}" for bv in b_values)
        print(f"{k1:<8.2f} | {cells}")
    print("-" * max(width, 60))
    bk1, bb = best_pair
    print(f"Best: k1={bk1:.2f}  b={bb:.2f}  MAP={best_map:.4f}")


def print_results_table(results):
    rows = sorted(results.items(), key=lambda kv: kv[1]["ap"], reverse=True)
    header = f"{'Query ID':<10} | {'AP':>7} | {'P@5':>6} | {'P@10':>6} | {'NDCG@10':>8} | Query (60 chars)"
    print(header)
    print("-" * len(header))
    for qid, m in rows:
        qt = m["query"][:60]
        print(
            f"{qid:<10} | {m['ap']:7.4f} | {m['p5']:6.4f} | {m['p10']:6.4f} | {m['ndcg10']:8.4f} | {qt}"
        )
    print("-" * len(header))
    n = len(results)
    if n == 0:
        print("No queries evaluated.")
        return
    map_ = sum(m["ap"] for m in results.values()) / n
    mean_p5 = sum(m["p5"] for m in results.values()) / n
    mean_p10 = sum(m["p10"] for m in results.values()) / n
    mean_ndcg10 = sum(m["ndcg10"] for m in results.values()) / n
    print(f"MAP = {map_:.4f}  |  mean P@5 = {mean_p5:.4f}  |  mean P@10 = {mean_p10:.4f}  |  mean NDCG@10 = {mean_ndcg10:.4f}")
