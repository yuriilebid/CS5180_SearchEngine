"""
Build Bill's inverted index + manifest into ./bill_index.pkl (independent of index.pkl).

Usage:
  python bill_index_builder.py
  python bill_index_builder.py --rebuild
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _resolve_documents_path():
    for candidate in ("documents.json", os.path.join("dataset", "documents.json")):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "documents.json not found in cwd or dataset/ — place the course corpus first."
    )


def main():
    from bill_integration.Inverted_indexer import InvertedIndex

    doc_path = _resolve_documents_path()
    index_path = os.path.join(_ROOT, "bill_index.pkl")
    manifest_path = os.path.join(_ROOT, "bill_index_manifest.json")
    force = "--rebuild" in sys.argv

    idx = InvertedIndex()
    idx.build_smart(
        filepath=doc_path,
        index_path=index_path,
        manifest_path=manifest_path,
        chunk_size=1000,
        force_rebuild=force,
    )


if __name__ == "__main__":
    main()
