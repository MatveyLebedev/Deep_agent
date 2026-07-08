"""Embeddings protocol.

Two methods, signature-compatible with LangChain embeddings, but with no
dependency on it: any langchain embedder object satisfies this protocol as-is.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingsLike(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


def model_name_of(embeddings) -> str:
    """Best-effort stable name of the embedding model (used in cache keys and
    the index manifest, so a different model never reuses cached vectors)."""
    for attr in ("model_name", "model", "_model"):
        val = getattr(embeddings, attr, None)
        if isinstance(val, str) and val:
            return val
    return type(embeddings).__name__
