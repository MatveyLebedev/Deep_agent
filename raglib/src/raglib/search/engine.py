"""SearchEngine: bm25 / vector (flat|tree) / hybrid (RRF) / grep over one index.

All methods share the same output shape and invariant: matching runs over
internal index units, but results are aggregated to WHOLE clauses (score =
max over the clause's units, deduplicated by clause_id). Every SearchHit
carries the clause number and the full, uncut clause text.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from raglib.indexing.bm25 import BM25Index, make_tokenizer, resolve_normalizer
from raglib.indexing.store import IndexData
from raglib.models import Clause, SearchHit, effective_key, section_path
from raglib.parsing.markdown import is_descendant

RRF_K = 60
_METHODS = ("bm25", "vector", "hybrid")
_STRATEGIES = ("flat", "tree")


class SearchEngine:
    def __init__(self, data: IndexData, embeddings=None,
                 bm25_normalizer: str | None = None):
        self.data = data
        self.embeddings = embeddings
        # (doc_id, section key) -> heading title, and doc_id -> friendly name,
        # for the provenance metadata on every SearchHit. First occurrence wins,
        # mirroring build_sections' by_key.
        self._titles: dict[tuple[str, str], str] = {}
        for s in data.sections:
            self._titles.setdefault((s.doc_id, s.key), s.title)
        self._doc_names = {did: (Path(d.source_path).name or did)
                           for did, d in data.documents.items()}
        # queries must be normalized exactly like the corpus: the mode chosen
        # at build time is recorded in the manifest; load may override it
        # (an "auto" override resolves to the best backend available here)
        self.bm25_normalizer = resolve_normalizer(
            bm25_normalizer
            or (data.manifest.get("bm25") or {}).get("normalizer")
            or "none")
        self._bm25 = BM25Index([ch.text for ch in data.chunks],
                               tokenizer=make_tokenizer(self.bm25_normalizer))
        self._query_vec_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------- helpers
    def _require_vectors(self, what: str) -> None:
        if self.data.chunk_index is None:
            raise RuntimeError(
                f"{what} requires a vector index, but this index was built "
                "BM25-only (embeddings=None). Rebuild with embeddings.")
        if self.embeddings is None:
            model = (self.data.manifest.get("embeddings") or {}).get("model")
            raise RuntimeError(
                f"{what} requires an embeddings client at search time. "
                f"Pass embeddings= to RagIndex.load() (index was built with "
                f"model {model!r}).")

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self._query_vec_cache.get(query)
        if vec is None:
            vec = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
            dim = (self.data.manifest.get("embeddings") or {}).get("dim")
            if dim and vec.shape[-1] != dim:
                raise RuntimeError(
                    f"query embedding dim {vec.shape[-1]} != index dim {dim}: "
                    "a different embedding model was used to build this index.")
            self._query_vec_cache[query] = vec
        return vec

    def _clause_matches(self, clause: Clause, doc: str | None, section: str | None) -> bool:
        if doc is not None and clause.doc_id != doc:
            return False
        if section is not None:
            eff = effective_key(clause)
            if not (is_descendant(eff, section)
                    or is_descendant(clause.section_key, section)):
                return False
        return True

    def _allowed_units(self, doc: str | None, section: str | None,
                       section_scope: list[tuple[str, str]] | None = None
                       ) -> set[int] | None:
        """Set of allowed unit ids, or None when unrestricted."""
        if doc is None and section is None and section_scope is None:
            return None
        allowed: set[int] = set()
        for ch in self.data.chunks:
            clause = self.data.clauses[ch.clause_id]
            if not self._clause_matches(clause, doc, section):
                continue
            if section_scope is not None:
                eff = effective_key(clause)
                if not any(clause.doc_id == sdoc
                           and (is_descendant(eff, skey)
                                or is_descendant(clause.section_key, skey)
                                or (not skey and clause.section_key == skey))
                           for sdoc, skey in section_scope):
                    continue
            allowed.add(ch.unit_id)
        return allowed

    def _tree_scope(self, query: str, top_sections: int = 8) -> list[tuple[str, str]]:
        """Coarse step of coarse-to-fine: top sections by the sections FAISS level."""
        self._require_vectors('strategy="tree"')
        if self.data.section_index is None:
            return []
        qv = self._embed_query(query)
        rows = self.data.section_index.search(qv, top_sections)
        return [(self.data.section_rows[r].doc_id, self.data.section_rows[r].key)
                for r, _score in rows]

    # -------------------------------------------------- unit-level searches
    def _bm25_units(self, query: str, allowed: set[int] | None,
                    top_n: int) -> list[tuple[int, float]]:
        fetch = top_n if allowed is None else len(self.data.chunks)
        ranked = self._bm25.search(query, fetch)
        if allowed is not None:
            ranked = [(u, s) for u, s in ranked if u in allowed]
        return ranked[:top_n]

    def _vector_units(self, query: str, allowed: set[int] | None,
                      top_n: int) -> list[tuple[int, float]]:
        self._require_vectors('method="vector"')
        qv = self._embed_query(query)
        return self.data.chunk_index.search(qv, top_n, allowed=allowed)

    # ------------------------------------------------------- aggregation
    def _aggregate(self, unit_scores: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """Units -> whole clauses: clause score = max over its units; dedupe."""
        best: dict[int, float] = {}
        for unit_id, score in unit_scores:
            cid = self.data.chunks[unit_id].clause_id
            if cid not in best or score > best[cid]:
                best[cid] = score
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    @staticmethod
    def _rrf(rankings: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (cid, _score) in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    def _hit(self, clause_id: int, score: float, method: str) -> SearchHit:
        c = self.data.clauses[clause_id]
        keys = section_path(effective_key(c))
        titles = [self._titles.get((c.doc_id, k), "") for k in keys]
        return SearchHit(clause_number=c.number, text=c.text, score=score,
                         doc_id=c.doc_id, section_path=keys,
                         clause_id=c.clause_id, method=method,
                         doc_name=self._doc_names.get(c.doc_id, c.doc_id),
                         section_titles=titles)

    # ------------------------------------------------------------ public
    def search(self, query: str, *, method: str = "hybrid", top_k: int = 5,
               strategy: str = "flat", doc: str | None = None,
               section: str | None = None) -> list[SearchHit]:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")

        scope = self._tree_scope(query) if strategy == "tree" else None
        allowed = self._allowed_units(doc, section, scope)
        depth = max(top_k * 5, 50)  # over-fetch units before clause aggregation

        if method == "bm25":
            ranked = self._aggregate(self._bm25_units(query, allowed, depth))
        elif method == "vector":
            ranked = self._aggregate(self._vector_units(query, allowed, depth))
        else:  # hybrid: RRF over clause-level rankings
            self._require_vectors('method="hybrid"')
            ranked = self._rrf([
                self._aggregate(self._bm25_units(query, allowed, depth)),
                self._aggregate(self._vector_units(query, allowed, depth)),
            ])
        return [self._hit(cid, score, method) for cid, score in ranked[:top_k]]

    def grep(self, pattern: str, *, top_k: int = 20, doc: str | None = None,
             section: str | None = None, flags: int = re.IGNORECASE) -> list[SearchHit]:
        """Regex over whole clauses (score = number of matches)."""
        rx = re.compile(pattern, flags)
        scored: list[tuple[int, float]] = []
        for c in self.data.clauses:
            if not self._clause_matches(c, doc, section):
                continue
            n = len(rx.findall(c.text))
            if n:
                scored.append((c.clause_id, float(n)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return [self._hit(cid, score, "grep") for cid, score in scored[:top_k]]
