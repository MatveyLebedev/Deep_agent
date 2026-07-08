"""Shared retrieval layer: structure-aware chunking + hybrid BM25/vector search.

One implementation for both execution paths:
  * the deterministic pipeline (extraction.py) builds a Retriever per document
    and pulls per-field context;
  * the agent path's search tools (tools.py) rank the same chunks for subagents.

Vectors are best-effort — if embeddings can't be built (offline / no key /
error) everything transparently degrades to BM25-only, so retrieval always
works in a closed network.
"""
from __future__ import annotations

import os
import re


# ------------------------------------------------------------------ embeddings
def get_embeddings():
    provider = os.environ.get("EMBED_PROVIDER", "openai").lower()
    if provider == "gigachat":
        from gigachat_embeddings import GigaChatEmbeddings
        return GigaChatEmbeddings.from_env()
    # default: any OpenAI-compatible embeddings endpoint (OpenRouter, internal vLLM, ...)
    api_key = os.environ.get("EMBED_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Without credentials the POST would still transmit the document text to
        # the (external by default) endpoint before failing auth — refuse instead.
        # Callers treat this like any embeddings failure and fall back to BM25.
        raise ValueError(
            "EMBED_PROVIDER=openai needs EMBED_API_KEY (or OPENROUTER_API_KEY); "
            "refusing to send document text to an external embeddings endpoint "
            "without explicit credentials."
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.environ.get("EMBED_MODEL", "mistralai/mistral-embed-2312"),
        openai_api_base=os.environ.get("EMBED_API_BASE", "https://openrouter.ai/api/v1"),
        openai_api_key=api_key,
    )


# ------------------------------------------------------------------- chunking
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")                       # ## 9. Совет директоров
_CLAUSE_RE = re.compile(r"^\s*(?:[-*]\s*)?\d+(?:\.\d+)*\.?\s+\S")      # 9 / 9.1 / 9.7.1
_NOISE_RE = re.compile(r"^\s*(?:---\s*Page\s+\d+.*|Total pages:.*|<!--.*-->)\s*$", re.IGNORECASE)


def _size_split(text: str, size: int, overlap: int = 150) -> list[str]:
    size = max(200, size)
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step)] or [text]


def _split_clause_units(lines: list[str]) -> list[str]:
    """Start a new unit at each top-level clause-number line; keep sub-lines with it."""
    units, cur = [], []
    for ln in lines:
        if _CLAUSE_RE.match(ln) and cur:
            units.append("\n".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        units.append("\n".join(cur))
    return units


def _merge_small(units: list[str], target: int) -> list[str]:
    """Merge adjacent clause units up to `target` chars so chunks aren't tiny."""
    merged, buf = [], ""
    for u in units:
        if not buf:
            buf = u
        elif len(buf) + len(u) + 1 <= target:
            buf = buf + "\n" + u
        else:
            merged.append(buf)
            buf = u
    if buf:
        merged.append(buf)
    return merged


def hierarchical_chunks(text: str, max_chars: int = 1800, overlap: int = 150) -> list[str]:
    """Structure-aware chunking for charter markdown: group by markdown headings,
    split each section body at clause-number boundaries, merge tiny clauses, and
    prepend the section heading to every chunk (small-to-large parent context).
    Falls back to fixed-size splitting for unstructured text / oversized clauses."""
    # Drop page-marker/image noise and collapse the long whitespace runs docling
    # emits for table-cell padding (pure token waste for extraction).
    lines = []
    for ln in text.splitlines():
        if _NOISE_RE.match(ln):
            continue
        lines.append(re.sub(r"[ \t]{2,}", " ", ln).rstrip())

    sections: list[tuple[str, list[str]]] = []
    heading, body = "", []
    for ln in lines:
        if _HEADING_RE.match(ln):
            if heading or body:
                sections.append((heading, body))
            heading, body = ln.strip().lstrip("#").strip(), []
        else:
            body.append(ln)
    if heading or body:
        sections.append((heading, body))

    chunks: list[str] = []
    for heading, body in sections:
        for unit in _merge_small(_split_clause_units(body), max_chars - 80):
            unit = unit.strip()
            if not unit:
                continue
            prefix = f"[{heading}]\n" if heading else ""
            if len(prefix) + len(unit) <= max_chars:
                chunks.append(prefix + unit)
            else:
                chunks.extend(prefix + piece for piece in _size_split(unit, max_chars - len(prefix), overlap))

    chunks = [c for c in chunks if c.strip()]
    return chunks or _size_split(text, max_chars, overlap)


# ------------------------------------------------------------------ retrieval
def _build_bm25(chunks: list[str]):
    from rank_bm25 import BM25Okapi
    tokenized = [re.findall(r"\w+", c.lower()) for c in chunks]
    return BM25Okapi(tokenized)


def hybrid_enabled() -> bool:
    return os.getenv("EXTRACTION_HYBRID", "1").strip().lower() in ("1", "true", "yes", "on")


def _rrf(rank_lists: list[tuple[list, float]], k: int = 60) -> dict:
    """Weighted Reciprocal Rank Fusion. rank_lists = [(ranked_indices, weight), ...].
    A doc highly ranked in either list gets a boost; weights tilt the balance."""
    scores: dict[int, float] = {}
    for ranked, weight in rank_lists:
        for rank, idx in enumerate(ranked):
            i = int(idx)
            scores[i] = scores.get(i, 0.0) + weight * (1.0 / (k + rank + 1))
    return scores


class Retriever:
    """Hybrid retriever: BM25 (lexical, precise on clause numbers/terms) fused
    with dense vectors (semantic recall) via RRF. Vectors are best-effort — if
    embeddings can't be built (offline / no key / error) it transparently runs
    BM25-only, so the pipeline always works in a closed network."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.bm25 = _build_bm25(chunks)
        self.vecs = None      # normalized (n, d) matrix, or None
        self.embedder = None
        if hybrid_enabled():
            self._build_vectors()
        else:
            print("[retrieval] BM25 only (hybrid disabled)")

    def _build_vectors(self) -> None:
        try:
            import numpy as np
            emb = get_embeddings()
            mat = np.asarray(emb.embed_documents(self.chunks), dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != len(self.chunks):
                raise ValueError("unexpected embedding shape")
            self.vecs = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            self.embedder = emb
            print(f"[retrieval] hybrid BM25+vector (RRF) over {len(self.chunks)} chunks")
        except Exception as e:
            self.vecs = None
            reason = (str(e).splitlines() or [type(e).__name__])[0][:140]
            print(f"[retrieval] BM25 only — embeddings unavailable ({reason})")

    def _bm25_ranked(self, keywords: str, cand: int) -> list[int]:
        q = re.findall(r"\w+", keywords.lower())
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked if scores[i] > 0][:cand]

    def _vec_ranked(self, keywords: str, cand: int) -> list[int]:
        if self.vecs is None:
            return []
        try:
            import numpy as np
            qv = np.asarray(self.embedder.embed_query(keywords), dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) + 1e-9)
            sims = self.vecs @ qv
            return [int(i) for i in np.argsort(-sims)[:cand]]
        except Exception:
            return []

    def top_indices(self, keywords: str, top_k: int, bm25_weight: float = 0.7) -> list[int]:
        """Fused top-k chunk indices for `keywords`, in relevance order."""
        cand = max(top_k * 3, 15)
        bm_ranked = self._bm25_ranked(keywords, cand)
        vec_ranked = self._vec_ranked(keywords, cand)

        if not vec_ranked:                       # BM25-only path
            chosen = bm_ranked[:top_k]
        else:
            fused = _rrf([(bm_ranked, bm25_weight), (vec_ranked, 1.0 - bm25_weight)])
            chosen = [i for i, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)][:top_k]
            chosen = chosen or bm_ranked[:top_k]
        return chosen

    def context(self, keywords: str, top_k: int, bm25_weight: float = 0.7) -> str:
        """Concatenated top-k chunks, restored to document order."""
        chosen = self.top_indices(keywords, top_k, bm25_weight)
        if not chosen:
            chosen = [0] if self.chunks else []
        chosen = sorted(set(chosen))             # restore document order
        return "\n\n".join(self.chunks[i] for i in chosen)
