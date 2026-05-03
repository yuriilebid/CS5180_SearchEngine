"""
bm25.py
-------
BM25 (Okapi BM25) retrieval model.

Academic reference
------------------
Robertson, S.E., Walker, S., Jones, S., Hancock-Beaulieu, M., & Gatford, M.
(1994). Okapi at TREC-3. NIST Special Publication 500-225, pp. 109-126.

Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework:
BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4).

Formula (per query term t, document d)
---------------------------------------
  score(d, Q) = Σ  IDF(t) * [ tf(t,d) * (k1+1) ]
                t      ─────────────────────────────────────────
                        tf(t,d) + k1 * (1 - b + b * dl/avgdl)

Parameters
----------
k1 : float  (default 1.5)
    TF saturation.  Range [1.2, 2.0] in literature.
    Higher → raw term frequency contributes more before plateauing.

b  : float  (default 0.75)
    Length normalization.  Range [0, 1].
    b=0 → no length penalty.
    b=1 → full normalization against avgdl.
    0.75 is the empirically validated default on TREC collections.

Why these defaults for our dataset
------------------------------------
avgdl = 73.4 tokens (short-to-medium documents).
k1=1.5 and b=0.75 are the TREC-validated defaults and a safe starting
point. With short docs, length variance is lower, so b=0.75 is
appropriate (not overly aggressive).
"""

from __future__ import annotations
from .Preprocessor import preprocess  # v2 — finance-aware normalization (Bill)
from .Inverted_indexer import InvertedIndex  # v3 — manifest-based incremental (Bill)


class BM25:
    """
    BM25 ranker that operates on a pre-built InvertedIndex.

    Usage
    -----
    idx = InvertedIndex()
    idx.build(documents)

    ranker = BM25(idx)
    results = ranker.rank(query_text, top_k=25)
    # results → [ (doc_id, score), ... ] sorted descending
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        """
        Parameters
        ----------
        index : InvertedIndex
            Pre-built index (must have called index.build() already).
        k1 : float
            TF saturation parameter.
        b : float
            Length normalization parameter.
        """
        self.index = index
        self.k1    = k1
        self.b     = b

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _score_term(self, term: str, doc_id: str) -> float:
        """
        Compute BM25 contribution of a single (term, doc) pair.

        Steps
        -----
        1. Get tf  — how many times term appears in this doc.
        2. Get idf — how rare the term is across the collection.
        3. Compute length normalization factor.
        4. Apply saturating TF formula.
        5. Return idf * saturated_tf.

        Returns 0.0 if term is not in the index or not in doc.
        """
        postings = self.index.get_postings(term)        # { doc_id → tf }
        tf = postings.get(doc_id, 0)
        if tf == 0:
            return 0.0

        idf  = self.index.get_idf(term)
        dl   = self.index.doc_lengths.get(doc_id, 0)
        avgdl = self.index.avgdl

        # Length normalization factor
        # When dl == avgdl  → norm = 1.0  (no adjustment)
        # When dl >  avgdl  → norm > 1.0  (penalizes longer doc)
        # When dl <  avgdl  → norm < 1.0  (rewards shorter doc)
        norm = 1 - self.b + self.b * (dl / avgdl) if avgdl > 0 else 1.0

        # Saturating TF
        # Numerator:   tf * (k1 + 1)        — grows with tf but is bounded
        # Denominator: tf + k1 * norm       — normalizes by doc length
        # As tf → ∞, the fraction → (k1 + 1), never exceeding it
        saturated_tf = (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

        return idf * saturated_tf

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        """
        BM25 score for a document given preprocessed query tokens.

        score(d, Q) = Σ  BM25_term(t, d)   for t in Q

        Parameters
        ----------
        query_tokens : list[str]
            Already preprocessed (stemmed) query terms.
        doc_id : str
            Document to score.

        Returns
        -------
        float
            BM25 relevance score. Higher = more relevant.
        """
        return sum(self._score_term(term, doc_id) for term in query_tokens)

    # ── Ranking ───────────────────────────────────────────────────────────────

    def rank(self, query_text: str, top_k: int = 25) -> list[tuple[str, float]]:
        """
        Preprocess query, find candidate documents, score and rank them.

        Retrieval strategy
        ------------------
        1. Preprocess query → stemmed token list.
        2. Collect candidate documents — the UNION of postings lists for
           all query terms. Only these docs can score > 0, so we skip the
           remaining (5000 - |candidates|) docs entirely.
        3. Score each candidate with BM25.
        4. Sort descending, return top_k.

        Parameters
        ----------
        query_text : str
            Raw query string (will be preprocessed internally).
        top_k : int
            Number of top results to return (default 25 per spec).

        Returns
        -------
        list[tuple[str, float]]
            [(doc_id, score), ...] sorted by score descending.
        """

        # Step 1 — Preprocess query (same pipeline as documents)
        # Deduplicate tokens: "15 year mortgage vs 30 year paid off in 15"
        # → ['15','year',...] without dedup scores '15' and 'year' twice,
        #   inflating BM25 scores for queries with repeated terms.
        query_tokens = list(set(preprocess(query_text)))

        if not query_tokens:
            return []

        # Step 2 — Candidate generation via posting list union
        # Any document appearing in at least one term's postings is a candidate
        candidates: set[str] = set()
        for term in query_tokens:
            candidates.update(self.index.get_postings(term).keys())

        if not candidates:
            return []

        # Step 3 — Score every candidate
        scored: list[tuple[str, float]] = [
            (doc_id, self.score(query_tokens, doc_id))
            for doc_id in candidates
        ]

        # Step 4 — Sort descending and return top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ── Quick sanity check when run directly ──────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from .Data_loader import load_all

    data_dir      = sys.argv[1] if len(sys.argv) > 1 else "."
    index_path    = os.path.join(data_dir, "index.pkl")
    manifest_path = os.path.join(data_dir, "index_manifest.json")
    doc_path      = os.path.join(data_dir, "documents.json")

    documents, queries, qrels = load_all(data_dir)

    # Build or load index using v3 smart builder
    idx = InvertedIndex()
    if InvertedIndex.exists(index_path):
        print("[BM25] Loading existing index from disk...")
        idx = InvertedIndex.load(index_path)
    else:
        print("[BM25] No index found — building from scratch...")
        idx.build_smart(
            filepath      = doc_path,
            index_path    = index_path,
            manifest_path = manifest_path,
            chunk_size    = 1000,
        )

    # Build ranker
    ranker = BM25(idx, k1=1.5, b=0.75)

    print("=" * 65)
    print("BM25 RANKING DEMO")
    print("=" * 65)

    # Test on 3 sample queries from the dataset
    sample_qids = ["6002", "3724", "6110"]

    for qid in sample_qids:
        query_text = queries[qid]
        results    = ranker.rank(query_text, top_k=5)
        relevant   = qrels.get(qid, set())

        print(f"\nQuery [{qid}]: {query_text}")
        print(f"  Relevant docs: {sorted(relevant)}")
        print(f"  Top 5 results:")
        for rank, (doc_id, score) in enumerate(results, 1):
            hit = "✓ RELEVANT" if doc_id in relevant else ""
            print(f"    {rank}. doc {doc_id:>8}  score={score:.4f}  {hit}")

    # Parameter sensitivity demo
    print("\n" + "=" * 65)
    print("PARAMETER SENSITIVITY (query: '15 year mortgage vs 30 year')")
    print("=" * 65)
    query_text = "15 year mortgage vs 30 year"
    configs = [
        (1.2, 0.75, "standard low-k1"),
        (1.5, 0.75, "our default"),
        (2.0, 0.75, "high-k1 (tf matters more)"),
        (1.5, 0.0,  "b=0 (no length penalty)"),
        (1.5, 1.0,  "b=1 (full length penalty)"),
    ]
    for k1, b, label in configs:
        r = BM25(idx, k1=k1, b=b)
        top = r.rank(query_text, top_k=1)
        doc_id, score = top[0] if top else ("—", 0)
        print(f"  k1={k1}, b={b} ({label:26s}) → top doc={doc_id}, score={score:.4f}")