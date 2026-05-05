"""
Interactive multi-model search and batch evaluation (--calc-map).
"""

import json
import re
import sys

from bill_score import score_bill_bm25
from evaluator import (
    _load_queries_and_qrels,
    evaluate_model,
    grid_search_bm25,
    print_results_table,
)
from retrieval import (
    adl,
    doc_ids,
    doc_lengths,
    idf_bm25,
    inverted,
    score_bim,
    score_bm25,
    score_hybrid_rrf,
    score_lm_dirichlet,
    score_rocchio_prf,
    score_rm3,
    score_vsm,
)


def _open_json_first(path):
    for candidate in (path, "dataset/" + path):
        try:
            return open(candidate, "r", encoding="utf-8")
        except OSError:
            continue
    raise FileNotFoundError(path)


def _load_doc_text(documents_path="documents.json"):
    with _open_json_first(documents_path) as f:
        documents = json.load(f)
    return {doc["doc_id"]: doc["text"] for doc in documents}


doc_text = _load_doc_text()

_QUERY_ID_COLON = re.compile(r"^(\d+)\s*:\s*(.+)$", re.DOTALL)


def _load_query_id_to_text():
    """query_id -> text from queries.json (empty dict if file missing)."""
    try:
        with _open_json_first("queries.json") as f:
            rows = json.load(f)
    except OSError:
        return {}
    out = {}
    for row in rows:
        qid = str(row.get("query_id", ""))
        if qid:
            out[qid] = row.get("text", "")
    return out


def parse_labeled_query(line, lookup):
    """
    Parse CLI query input: plain text, or labeled forms:
      @query_id     — substitute text from queries.json (lookup)
      id | text     — label + query (pipe)
      id<TAB>text   — label + query (tab)
      123: text     — numeric query_id + colon + text
    Returns (label_or_None, text_for_retrieval). Empty text means skip this round.
    """
    s = line.strip()
    if not s:
        return None, ""
    if s.startswith("@"):
        qid = s[1:].strip()
        if not qid:
            print("(empty id after @)\n")
            return None, ""
        text = lookup.get(qid)
        if text is None:
            print(
                f"(no query for id {qid!r} in queries.json — "
                "use `id | text` or plain text)\n"
            )
            return None, ""
        return qid, text
    if "|" in s:
        left, right = s.split("|", 1)
        left, right = left.strip(), right.strip()
        if left and right:
            return left, right
    if "\t" in s:
        left, right = s.split("\t", 1)
        left, right = left.strip(), right.strip()
        if left and right:
            return left, right
    m = _QUERY_ID_COLON.match(s)
    if m:
        return m.group(1), m.group(2).strip()
    return None, s


def display_results(ranked, query_label=None, query_text=None):
    if query_text:
        qt = query_text.replace("\n", " ")
        if len(qt) > 140:
            qt = qt[:140] + "..."
        if query_label:
            print(f"Query [{query_label}]: {qt}")
        else:
            print(f"Query: {qt}")
        print("────────────────────────────────────────────")
    if not ranked:
        print("(no results)")
        return
    ranked = ranked[:25]
    n_pages = (len(ranked) + 4) // 5
    for page in range(n_pages):
        start = page * 5
        end = min(start + 5, len(ranked))
        for offset, (doc_id, score) in enumerate(ranked[start:end]):
            rank = start + offset + 1
            snippet = doc_text.get(doc_id, "").replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            print(f"Rank {rank} | {doc_id} | score: {score:.4f}")
            print(snippet)
            print("─────────────────────────────────────────────")
        if page < n_pages - 1:
            choice = input("[N]ext / [Q]uit: ").strip().lower()
            if choice == "q":
                break


def interactive():
    lookup = _load_query_id_to_text()
    print(
        "\nOffline-first IR — models use precomputed index.pkl only.\n"
        "Available: BM25, VSM, BIM, Dirichlet LM, Hybrid RRF, Rocchio PRF, RM3, "
        "Bill BM25 (finance-aware index).\n\n"
        "Query input: plain text, or labeled — `id | text`, `id<TAB>text`, "
        "`123: rest of query`, or `@id` to load text from queries.json.\n"
    )
    menu = (
        "[1] BM25\n"
        "[2] VSM (TF-IDF Cosine)\n"
        "[3] BIM\n"
        "[4] Language Model (Dirichlet)\n"
        "[5] Hybrid RRF (BM25 + VSM + RM3)\n"
        "[6] Rocchio PRF\n"
        "[7] RM3 Query Expansion\n"
        "[8] Bill BM25 (finance preprocessor + teammate index)\n"
        "[q] Quit\n"
    )
    dispatch = {
        "1": score_bm25,
        "2": score_vsm,
        "3": score_bim,
        "4": score_lm_dirichlet,
        "5": score_hybrid_rrf,
        "6": score_rocchio_prf,
        "7": score_rm3,
        "8": score_bill_bm25,
    }
    while True:
        print(menu)
        sel = input("Select model: ").strip().lower()
        if sel == "q":
            break
        if sel not in dispatch:
            print("Invalid choice.\n")
            continue
        q_raw = input("Query: ").strip()
        if not q_raw:
            continue
        label, qtext = parse_labeled_query(q_raw, lookup)
        if not qtext:
            continue
        try:
            ranked = dispatch[sel](qtext, 25)
        except FileNotFoundError as err:
            print(f"{err}\n(For Bill BM25, run: python bill_index_builder.py)\n")
            continue
        display_results(ranked, query_label=label, query_text=qtext)
        print()


def _mean_metrics(results):
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        sum(m["ap"] for m in results.values()) / n,
        sum(m["p5"] for m in results.values()) / n,
        sum(m["p10"] for m in results.values()) / n,
        sum(m["ndcg10"] for m in results.values()) / n,
    )


def _print_model_comparison(summary_rows):
    # summary_rows: list of (name, map, p5, p10, ndcg10)
    summary_rows = sorted(summary_rows, key=lambda r: (-r[1], r[0]))
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by MAP)")
    print("=" * 60)
    hdr = f"{'Model':<14} | {'MAP':>7} | {'P@5':>7} | {'P@10':>7} | {'NDCG@10':>8}"
    print(hdr)
    print("-" * len(hdr))
    for i, (name, map_, p5, p10, ndcg) in enumerate(summary_rows):
        suffix = " ← BEST" if i == 0 else ""
        print(f"{name:<14} | {map_:7.4f} | {p5:7.4f} | {p10:7.4f} | {ndcg:8.4f}{suffix}")
    print("-" * len(hdr))


if __name__ == "__main__":
    if "--calc-map" in sys.argv:
        queries, relevant_map = _load_queries_and_qrels()

        print("=" * 60)
        print("BM25 HYPERPARAMETER GRID SEARCH")
        print("=" * 60)
        grid_search_bm25(
            inverted,
            doc_ids,
            doc_lengths,
            adl,
            idf_bm25,
            queries,
            relevant_map,
        )

        models = {
            "BM25": score_bm25,
            "VSM": score_vsm,
            "BIM": score_bim,
            "LM-Dir": score_lm_dirichlet,
            "Hybrid-RRF": score_hybrid_rrf,
            "Rocchio-PRF": score_rocchio_prf,
            "RM3": score_rm3,
            "Bill-BM25": score_bill_bm25,
        }

        summary_rows = []
        for name, fn in models.items():
            print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
            try:
                results = evaluate_model(fn, retrieve_count=25)
            except FileNotFoundError as err:
                if name == "Bill-BM25":
                    print(
                        "Skipped — build Bill's index first: python bill_index_builder.py\n"
                        f"({err})\n"
                    )
                    continue
                raise
            print_results_table(results)
            summary_rows.append((name, *_mean_metrics(results)))

        _print_model_comparison(summary_rows)
    else:
        interactive()
