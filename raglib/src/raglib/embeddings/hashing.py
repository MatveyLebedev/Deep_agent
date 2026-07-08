"""Deterministic offline embeddings for tests (feature-hashing bag of words).

Not semantically smart, but: no network, stable across runs and platforms,
and cosine similarity is meaningful for token overlap — enough to exercise
the whole FAISS pipeline in CI.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbeddings:
    def __init__(self, dim: int = 256):
        self.dim = int(dim)
        self.model_name = f"hashing-{self.dim}"

    def _vector(self, text: str) -> List[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.sha1(tok.encode("utf-8")).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            v[idx] += sign
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v /= norm
        return v.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)
