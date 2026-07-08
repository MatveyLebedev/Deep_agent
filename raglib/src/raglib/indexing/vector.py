"""FAISS wrapper: exact cosine search over L2-normalized vectors.

`IndexFlatIP` needs no training and no hyperparameters; at target volumes
(1e3–1e5 vectors) a full scan is milliseconds. The index type is recorded in
the manifest as `index_factory`, so growing to HNSW later does not change the
storage format. FAISS ids are implicit ordinals == jsonl line numbers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "faiss-cpu is required for vector search: pip install faiss-cpu "
            "(BM25-only indexes work without it)."
        ) from e


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-9, None)


class FaissFlat:
    """Exact inner-product index over normalized vectors (== cosine)."""

    def __init__(self, index):
        self._faiss = _faiss()
        self.index = index

    @classmethod
    def from_vectors(cls, vectors: np.ndarray) -> "FaissFlat":
        faiss = _faiss()
        vectors = l2_normalize(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @property
    def dim(self) -> int:
        return int(self.index.d)

    def search(self, query_vec: np.ndarray, top_n: int,
               allowed: set[int] | None = None) -> list[tuple[int, float]]:
        """Ranked (row_id, cosine). With `allowed`, scan everything and filter:
        a Flat index is a full scan anyway, so this stays exact and simple."""
        if self.ntotal == 0:
            return []
        qv = l2_normalize(np.asarray(query_vec, dtype=np.float32))
        k = self.ntotal if allowed is not None else min(top_n, self.ntotal)
        scores, ids = self.index.search(qv, k)
        out: list[tuple[int, float]] = []
        for row, score in zip(ids[0], scores[0]):
            if row < 0:
                continue
            if allowed is not None and int(row) not in allowed:
                continue
            out.append((int(row), float(score)))
            if len(out) >= top_n:
                break
        return out

    def save(self, path: Path) -> None:
        self._faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissFlat":
        faiss = _faiss()
        return cls(faiss.read_index(str(path)))


def embed_texts_cached(embeddings, texts: list[str], cache_dir: Path,
                       model_name: str) -> np.ndarray:
    """Embed texts with a per-text sha1 content cache (survives rebuilds within
    the same index dir; keyed by model so switching models never reuses vectors)."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def key_of(text: str) -> str:
        return hashlib.sha1((model_name + "\x00" + text).encode("utf-8")).hexdigest()

    vecs: list[np.ndarray | None] = [None] * len(texts)
    missing: list[int] = []
    for i, t in enumerate(texts):
        p = cache_dir / f"{key_of(t)}.npy"
        if p.exists():
            vecs[i] = np.load(p)
        else:
            missing.append(i)
    if missing:
        got = embeddings.embed_documents([texts[i] for i in missing])
        if len(got) != len(missing):
            raise RuntimeError(
                f"embeddings returned {len(got)} vectors for {len(missing)} texts"
            )
        for i, v in zip(missing, got):
            arr = np.asarray(v, dtype=np.float32)
            np.save(cache_dir / f"{key_of(texts[i])}.npy", arr)
            vecs[i] = arr
    dims = {v.shape[-1] for v in vecs}  # type: ignore[union-attr]
    if len(dims) > 1:
        raise RuntimeError(f"inconsistent embedding dimensions: {sorted(dims)}")
    return np.vstack(vecs)  # type: ignore[arg-type]
