"""
rag/vector_store.py
Build, save, and load the FAISS vector store from the knowledge base documents.
"""
import os
import json
import glob
from pathlib import Path
from typing import List, Dict

import numpy as np

# ── lazy imports (not required at module load time) ──────────────────────────
_faiss = None
_SentenceTransformer = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_st():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer


# ── Constants ────────────────────────────────────────────────────────────────
KB_DIR = Path(__file__).parent / "knowledge_base"
INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(Path(__file__).parent / "faiss_index")))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80      # overlap between chunks


# ── Text chunking ────────────────────────────────────────────────────────────
def _chunk_text(text: str, source: str) -> List[Dict]:
    """Split text into overlapping chunks with metadata."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "source": source,
                "start_char": start,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_knowledge_base() -> List[Dict]:
    """Load all .txt files from the knowledge base directory."""
    all_chunks = []
    for fpath in sorted(KB_DIR.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8")
        chunks = _chunk_text(text, source=fpath.stem)
        all_chunks.extend(chunks)
        print(f"  Loaded {fpath.name}: {len(chunks)} chunks")
    return all_chunks


# ── Build index ──────────────────────────────────────────────────────────────
def build_index(force_rebuild: bool = False) -> tuple:
    """
    Build (or load) FAISS index.
    Returns (index, chunks, model) tuple.
    """
    faiss = _get_faiss()
    ST = _get_st()

    meta_path = INDEX_PATH.with_suffix(".json")

    if not force_rebuild and INDEX_PATH.exists() and meta_path.exists():
        # Load existing index
        print("Loading existing FAISS index …")
        index = faiss.read_index(str(INDEX_PATH))
        with open(meta_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        model = ST(EMBEDDING_MODEL)
        print(f"  Loaded index with {index.ntotal} vectors.")
        return index, chunks, model

    # Build from scratch
    print("Building FAISS index from knowledge base …")
    chunks = load_knowledge_base()
    texts = [c["text"] for c in chunks]

    model = ST(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine sim (normalised)
    index.add(embeddings)

    # Persist
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"  Index built: {index.ntotal} vectors, dim={dim}.")
    return index, chunks, model


# ── Retrieve ─────────────────────────────────────────────────────────────────
def retrieve(query: str, index, chunks: List[Dict], model, top_k: int = 5) -> List[Dict]:
    """Retrieve top-k most relevant chunks for a query."""
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            chunk = dict(chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

    return results


# ── Singleton (cached across Streamlit reruns via @st.cache_resource) ────────
_SINGLETON: tuple | None = None


def get_retriever():
    """Return a ready-to-use (index, chunks, model) tuple, cached globally."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = build_index()
    return _SINGLETON
