"""RagIndex — the public facade: build / load / search / toc / agentic / delete."""
from __future__ import annotations

from pathlib import Path

from raglib.indexing import builder, store
from raglib.models import SearchHit, SectionRef
from raglib.search.engine import SearchEngine
from raglib.search import toc as toc_tools


class RagIndex:
    """One persistent index over a set of recognized documents.

    All search tools share one output contract: a hit is a WHOLE clause with
    its number (never a cut fragment) — see PLAN.md §1.
    """

    def __init__(self, data: store.IndexData, embeddings=None,
                 bm25_normalizer: str | None = None):
        self._data = data
        self._engine = SearchEngine(data, embeddings=embeddings,
                                    bm25_normalizer=bm25_normalizer)
        self._closed = False

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def build(cls, inputs, index_dir, *, recognizer=None, embeddings=None,
              chunk_size: int = 1500, chunk_overlap: int = 150,
              section_embedding: str = "mean",
              bm25_normalizer: str = "auto",
              ocr_number_repair: bool = True) -> "RagIndex":
        """Build and persist an index, then load it back (so the load path is
        exercised on every build). ocr_number_repair=True tolerates OCR errors in
        clause numbers (О→0, l→1, comma-for-dot); set False for strict matching."""
        root = builder.build_index(
            inputs, index_dir, recognizer=recognizer, embeddings=embeddings,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            section_embedding=section_embedding, bm25_normalizer=bm25_normalizer,
            ocr_number_repair=ocr_number_repair)
        return cls.load(root, embeddings=embeddings)

    @classmethod
    def load(cls, index_dir, *, embeddings=None,
             bm25_normalizer: str | None = None) -> "RagIndex":
        """bm25_normalizer overrides the mode stored in the manifest
        (BM25 is rebuilt in memory on load, so no re-embedding is needed)."""
        return cls(store.load_index(Path(index_dir)), embeddings=embeddings,
                   bm25_normalizer=bm25_normalizer)

    @staticmethod
    def delete_index(index_dir) -> bool:
        """Delete an index folder (validated as a raglib index first).
        Idempotent: returns False if it is already gone."""
        return store.delete_index(index_dir)

    def delete(self) -> bool:
        """Delete this loaded index from disk and close the object."""
        self._check_open()
        root = self._data.root
        result = store.delete_index(root) if root is not None else False
        self._closed = True
        return result

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("This RagIndex was deleted/closed.")

    # ------------------------------------------------------------ properties
    @property
    def root(self) -> Path | None:
        return self._data.root

    @property
    def manifest(self) -> dict:
        return self._data.manifest

    @property
    def documents(self) -> list[str]:
        return list(self._data.documents)

    # ------------------------------------------------------------ search
    def search(self, query: str, *, method: str = "hybrid", top_k: int = 5,
               strategy: str = "flat", doc: str | None = None,
               section: str | None = None) -> list[SearchHit]:
        self._check_open()
        return self._engine.search(query, method=method, top_k=top_k,
                                   strategy=strategy, doc=doc, section=section)

    def grep(self, pattern: str, *, top_k: int = 20, doc: str | None = None,
             section: str | None = None) -> list[SearchHit]:
        self._check_open()
        return self._engine.grep(pattern, top_k=top_k, doc=doc, section=section)

    # ------------------------------------------------------------ toc tools
    def toc(self, doc: str | None = None, *, preview: bool = False,
            preview_chars: int = 80, clauses: bool = False) -> str:
        """Outline with section keys and enriched titles; preview=True adds the
        first content sentence of every section; clauses=True lists the clause
        numbers belonging to each section."""
        self._check_open()
        return toc_tools.toc_outline(self._data, doc=doc, preview=preview,
                                     preview_chars=preview_chars, clauses=clauses)

    def toc_entries(self, doc: str | None = None):
        """Structured outline (list[TocEntry]) for programmatic navigation."""
        self._check_open()
        return toc_tools.toc_entries(self._data, doc=doc)

    def find_section(self, query: str, *, semantic: bool = False,
                     top_k: int = 3) -> list[SectionRef]:
        self._check_open()
        return toc_tools.find_section(self._data, query, semantic=semantic,
                                      engine=self._engine, top_k=top_k)

    def read_section(self, doc_id: str, section: str) -> str:
        self._check_open()
        return toc_tools.read_section(self._data, doc_id, section)

    # ------------------------------------------------------------ agentic
    def agentic_search(self, prompt: str, *, llm=None, top_k: int = 8,
                       max_iters: int = 3, max_llm_calls: int = 8):
        """Prompt in -> code-driven search loop with LLM reflection (PLAN.md §6).
        llm=None degrades to plain hybrid/bm25; returns AgenticResult."""
        self._check_open()
        from raglib.agent.searcher import AgenticSearcher
        return AgenticSearcher(self._engine, llm, top_k=top_k,
                               max_iters=max_iters,
                               max_llm_calls=max_llm_calls).run(prompt)
