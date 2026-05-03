"""
Bridge to Bill's BM25 ranker (finance-aware preprocessor + his inverted index).

Uses bill_index.pkl built by bill_index_builder.py — kept separate from the
course-wide index.pkl so Bill's code stays close to the original pipeline.
"""

import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_BILL_INDEX = os.path.join(_PROJECT_ROOT, "bill_index.pkl")

_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is not None:
        return _ranker
    if not os.path.isfile(_BILL_INDEX):
        raise FileNotFoundError(
            f"Missing {_BILL_INDEX}. Run: python bill_index_builder.py"
        )
    from bill_integration.Inverted_indexer import InvertedIndex
    from bill_integration.Retrieval_model_bm25 import BM25

    idx = InvertedIndex.load(_BILL_INDEX)
    _ranker = BM25(idx, k1=1.5, b=0.75)
    return _ranker


def score_bill_bm25(query, top_k):
    """Same contract as retrieval.score_bm25: [(doc_id, score), ...]."""
    return _get_ranker().rank(query, top_k)
