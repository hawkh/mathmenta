#!/usr/bin/env python
"""
build_index.py
Run this script once to pre-build the FAISS vector index from the knowledge base.
Subsequent app launches will load the cached index instead of rebuilding it.

Usage:
    python build_index.py
    python build_index.py --rebuild   # force rebuild
"""
import sys
from pathlib import Path

# Make sure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    force = "--rebuild" in sys.argv

    print("=" * 60)
    print("Math Mentor – Knowledge Base Index Builder")
    print("=" * 60)

    from rag.vector_store import build_index
    index, chunks, model = build_index(force_rebuild=force)

    print(f"\n✅ Index ready: {index.ntotal} vectors from {len(chunks)} chunks")
    print("You can now run: streamlit run app.py")
