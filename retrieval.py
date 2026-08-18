"""
Fast in-memory retrieval for the school chatbot.

Uses a small SentenceTransformer model and pre-built embeddings.
No vector database is needed because the school knowledge base is small.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "knowledge_base.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None
_chunks = None
_embeddings = None


def _load():
    global _model, _chunks, _embeddings

    if _model is None:
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

    if _chunks is None:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Knowledge base not found. Run `python ingest.py` first."
            )

        _chunks = json.loads(
            CHUNKS_PATH.read_text(encoding="utf-8")
        )

    if _embeddings is None:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Embeddings not found. Run `python ingest.py` first."
            )

        _embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r"
        )


def preload():
    """Load the model and knowledge base before the first user request."""
    _load()


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    _load()

    query_embedding = _model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores = _embeddings @ query_embedding

    top_indices = np.argpartition(
        scores,
        -top_k
    )[-top_k:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    return [
        {
            **_chunks[i],
            "score": float(scores[i])
        }
        for i in top_indices
    ]