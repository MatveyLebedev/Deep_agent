"""Embeddings seam.

raglib does not ship provider clients. Pass a LangChain embeddings object
straight into ``RagIndex.build(embeddings=...)`` — its ``embed_documents`` /
``embed_query`` methods already satisfy :class:`EmbeddingsLike` (see
``base.py``), so ``langchain_gigachat.GigaChatEmbeddings`` and friends work with
no adapter. ``HashingEmbeddings`` is a deterministic offline stand-in for tests.
"""
from raglib.embeddings.base import EmbeddingsLike, model_name_of
from raglib.embeddings.hashing import HashingEmbeddings

__all__ = ["EmbeddingsLike", "model_name_of", "HashingEmbeddings"]
