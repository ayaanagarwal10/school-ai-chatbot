"""
Loads the pre-built knowledge base and finds relevant chunks via cosine
similarity. No vector database — the corpus is small enough that an
in-memory numpy search is simpler to run and debug.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "knowledge_base.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # must match ingest.py

_model = None
_chunks = None
_embeddings = None


def _load():
    global _model, _chunks, _embeddings
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _chunks is None or _embeddings is None:
        if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError("Knowledge base not found. Run `python ingest.py` first.")
        _chunks = json.loads(CHUNKS_PATH.read_text())
        _embeddings = np.load(EMBEDDINGS_PATH)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    _load()
    query_embedding = _model.encode([query], normalize_embeddings=True)[0]
    scores = _embeddings @ query_embedding  # normalized -> dot product = cosine similarity
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{**_chunks[i], "score": float(scores[i])} for i in top_indices]