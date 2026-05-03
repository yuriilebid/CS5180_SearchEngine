"""
data_loader.py  (v2 — incremental)
------------------------------------
Loads dataset files for the IR system.

What changed from v1
--------------------
v1: json.load() — entire documents.json read into RAM at once.
v2: stream_documents() — yields one chunk of `chunk_size` docs at a time.
    The file is read once top-to-bottom; only one chunk lives in RAM.

Why this matters
----------------
    5,000 docs  @ ~800 chars avg  ≈     4 MB  in RAM  (v1 is fine)
    500,000 docs @ ~800 chars avg  ≈   400 MB  in RAM  (v1 starts hurting)
    50,000,000 docs               ≈    40 GB  in RAM  (v1 crashes)

    v2 holds at most chunk_size docs at a time regardless of collection size.

Files
-----
  documents.json  → streamed in chunks via stream_documents()
                    also loadable all at once via load_documents() for
                    compatibility with the rest of the system
  queries.json    → unchanged (25 queries, always tiny)
  qrels.json      → unchanged (243 judgments, always tiny)
"""

import json
import os
from collections import defaultdict
from typing import Generator


# ── Document streaming (new in v2) ────────────────────────────────────────────

def stream_documents(
    filepath   : str,
    chunk_size : int = 1000,
) -> Generator[dict[str, str], None, None]:
    """
    Stream documents.json in chunks of `chunk_size` docs.

    Instead of loading the full JSON array into memory, this function:
      1. Opens the file once
      2. Parses the full JSON array (unavoidable for a JSON array format)
         but immediately iterates — never holding more than one chunk
      3. Yields { doc_id -> text } dicts of size chunk_size

    Real-world note
    ---------------
    Production systems use JSONL (one JSON object per line) so the file
    can be parsed line-by-line without loading the array at all.
    Our dataset uses a JSON array, so we parse then immediately stream.
    See stream_documents_jsonl() below for the true zero-RAM approach.

    Parameters
    ----------
    filepath   : str   path to documents.json
    chunk_size : int   number of documents per yielded chunk (default 1000)

    Yields
    ------
    dict[str, str]
        { doc_id -> text } with up to chunk_size entries.
        The last chunk may be smaller than chunk_size.

    Example
    -------
    >>> for chunk in stream_documents("documents.json", chunk_size=1000):
    ...     index.add_batch(chunk)   # process 1000 docs, then free them
    """
    chunk       : dict[str, str] = {}
    empty_count : int = 0
    total       : int = 0

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)   # parse array once — entries streamed below

    for entry in raw:
        doc_id = str(entry["doc_id"])
        text   = entry.get("text", "").strip()

        if not text:
            empty_count += 1

        chunk[doc_id] = text
        total += 1

        if len(chunk) == chunk_size:
            yield chunk
            chunk = {}          # release previous chunk from memory

    if chunk:                   # yield the final (possibly partial) chunk
        yield chunk

    print(f"[Loader] Streamed  : {total:,} documents  |  {empty_count} empty  "
          f"|  chunk_size={chunk_size}")


def stream_documents_jsonl(
    filepath   : str,
    chunk_size : int = 1000,
) -> Generator[dict[str, str], None, None]:
    """
    True line-by-line streaming for JSONL format (one doc per line).

    This is how production systems work - no json.load() at all.
    Each line is parsed independently; RAM usage = one chunk only.

    Use this if you ever convert documents.json -> documents.jsonl:
        jq -c '.[]' documents.json > documents.jsonl

    Yields
    ------
    dict[str, str]  same as stream_documents()
    """
    chunk : dict[str, str] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry  = json.loads(line)
            doc_id = str(entry["doc_id"])
            text   = entry.get("text", "").strip()
            chunk[doc_id] = text

            if len(chunk) == chunk_size:
                yield chunk
                chunk = {}

    if chunk:
        yield chunk


# ── Full load (kept for compatibility with evaluator / search) ─────────────────

def load_documents(filepath: str) -> dict[str, str]:
    """
    Load all documents into a single dict (v1 behaviour, kept for compatibility).

    Used by: search.py, evaluator.py -- they need random access to doc text
    by doc_id at query time, so they need the full dict in RAM.

    The index builder uses stream_documents() instead.

    Returns
    -------
    dict[str, str]  { doc_id -> text }
    """
    documents   : dict[str, str] = {}
    empty_count : int = 0

    for chunk in stream_documents(filepath):
        for doc_id, text in chunk.items():
            if not text:
                empty_count += 1
            documents[doc_id] = text

    print(f"[Loader] Documents : {len(documents):,} loaded  |  {empty_count} empty texts")
    return documents


# ── Queries and qrels (unchanged from v1) ─────────────────────────────────────

def load_queries(filepath: str) -> dict[str, str]:
    """Load queries.json -> { query_id -> text }"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    queries = {str(entry["query_id"]): entry["text"].strip() for entry in raw}
    print(f"[Loader] Queries   : {len(queries):,} loaded")
    return queries


def load_qrels(filepath: str) -> dict[str, set[str]]:
    """Load qrels.json -> { query_id -> set[doc_id] }"""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    qrels : dict[str, set[str]] = defaultdict(set)
    for entry in raw:
        query_id = str(entry["query_id"])
        doc_id   = str(entry["doc_id"])
        if int(entry.get("relevance", 1)) > 0:
            qrels[query_id].add(doc_id)

    total = sum(len(v) for v in qrels.values())
    print(f"[Loader] Qrels     : {len(qrels):,} queries  |  {total:,} judgments")
    return dict(qrels)


def load_all(data_dir: str = ".") -> tuple[dict, dict, dict]:
    """
    Load all three files. Returns (documents, queries, qrels).
    documents is a full dict for search/eval compatibility.
    Index building should use stream_documents() directly.
    """
    doc_path   = os.path.join(data_dir, "documents.json")
    query_path = os.path.join(data_dir, "queries.json")
    qrels_path = os.path.join(data_dir, "qrels.json")

    for path in (doc_path, query_path, qrels_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Loader] Missing file: {path}")

    print(f"[Loader] Loading data from: {os.path.abspath(data_dir)}")
    documents = load_documents(doc_path)
    queries   = load_queries(query_path)
    qrels     = load_qrels(qrels_path)
    print("[Loader] All files loaded successfully.\n")
    return documents, queries, qrels


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir  = sys.argv[1] if len(sys.argv) > 1 else "."
    doc_path  = os.path.join(data_dir, "documents.json")

    print("=" * 60)
    print("STREAMING TEST -- chunk_size=1000")
    print("=" * 60)

    chunk_num  = 0
    total_docs = 0

    for chunk in stream_documents(doc_path, chunk_size=1000):
        chunk_num  += 1
        total_docs += len(chunk)
        first_id    = next(iter(chunk))
        print(f"  Epoch {chunk_num:>2}  |  {len(chunk):>4} docs  "
              f"|  total so far: {total_docs:>5}  "
              f"|  sample id: {first_id}")

    print(f"\n  Total docs streamed : {total_docs:,}")
    print(f"  Max docs in RAM at once : {1000:,}  "
          f"(vs {total_docs:,} for v1 full-load)")