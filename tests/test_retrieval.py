"""Phase-4: shared retrieval module (RRF fusion, Retriever) and search tools."""
import pytest

import retrieval
import tools
from retrieval import Retriever, _rrf

CHARTER_MD = """## Статья 11. ОБЩЕЕ СОБРАНИЕ АКЦИОНЕРОВ

11.1. Компетенция Общего собрания акционеров определяется законом

## Статья 12. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ

12.1.4. одобрение крупных сделок стоимостью от 25 до 50 процентов
балансовой стоимости активов Компании

## Статья 13. ИСПОЛНИТЕЛЬНЫЕ ОРГАНЫ

13.1. Генеральный директор осуществляет руководство текущей деятельностью
"""


class TestRRF:
    def test_agreement_wins(self):
        # doc 0 is ranked #1 by both lists; doc 1 only by one
        scores = _rrf([([0, 1], 0.5), ([0, 2], 0.5)])
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_weights_tilt_the_balance(self):
        heavy_first = _rrf([([0], 0.9), ([1], 0.1)])
        assert heavy_first[0] > heavy_first[1]
        light_first = _rrf([([0], 0.1), ([1], 0.9)])
        assert light_first[1] > light_first[0]


class TestRetrieverBM25Only:
    @pytest.fixture(autouse=True)
    def _no_hybrid(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_HYBRID", "0")

    def test_finds_relevant_chunk(self):
        r = Retriever(retrieval.hierarchical_chunks(CHARTER_MD))
        ctx = r.context("крупная сделка процент балансовой стоимости активов", top_k=1)
        assert "12.1.4" in ctx
        assert "Генеральный директор" not in ctx

    def test_context_restores_document_order(self):
        # chunk 2 is MORE relevant (ranked first), yet context must join
        # selected chunks in document order: chunk 1 before chunk 2.
        chunks = [
            "фоновый шум без совпадений",
            "сделка упоминается здесь один раз",
            "директор директор директор — самый релевантный кусок",
        ]
        r = Retriever(chunks)
        assert r.top_indices("директор сделка", top_k=2)[0] == 2
        ctx = r.context("директор сделка", top_k=2)
        assert ctx.index("сделка упоминается") < ctx.index("директор директор")

    def test_no_match_falls_back_to_first_chunk(self):
        r = Retriever(["только текст без совпадений"])
        assert r.context("нерелевантный запрос", top_k=3) == "только текст без совпадений"


class _FakeEmbeddings:
    """Deterministic embedder: vector = term counts over a tiny vocabulary."""
    VOCAB = ["сделка", "директор", "собрание"]

    def _vec(self, text):
        low = text.lower()
        return [float(low.count(w)) + 1e-6 for w in self.VOCAB]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class TestRetrieverHybrid:
    def test_vector_signal_fused_with_bm25(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_HYBRID", "1")
        monkeypatch.setattr(retrieval, "get_embeddings", lambda: _FakeEmbeddings())
        chunks = [
            "12.1.4. одобрение крупных сделок сделка сделка",
            "13.1. Генеральный директор директор",
            "11.1. Общее собрание собрание",
        ]
        r = Retriever(chunks)
        assert r.vecs is not None
        top = r.top_indices("сделка", top_k=1)
        assert top == [0]

    def test_broken_embeddings_degrade_to_bm25(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_HYBRID", "1")
        def _boom():
            raise RuntimeError("no network")
        monkeypatch.setattr(retrieval, "get_embeddings", _boom)
        r = Retriever(["12.1.4. крупная сделка"])
        assert r.vecs is None
        assert "12.1.4" in r.context("крупная сделка", top_k=1)


class TestChunksForAndCacheKey:
    def test_hierarchical_by_default(self):
        chunks = tools._chunks_for(CHARTER_MD)
        assert any(c.startswith("[Статья 12") for c in chunks)

    def test_explicit_pattern_uses_fixed_splitter(self):
        chunks = tools._chunks_for("a## b## c", split_pattern="##")
        assert chunks == ["a", "b", "c"]

    def test_cache_key_depends_on_chunking_mode(self, monkeypatch):
        monkeypatch.setenv("EXTRACTION_CHUNKING", "hierarchical")
        k_hier = tools._embed_cache_key("текст", "")
        monkeypatch.setenv("EXTRACTION_CHUNKING", "fixed")
        k_fixed = tools._embed_cache_key("текст", "")
        assert k_hier != k_fixed


class TestSearchHybridTool:
    @pytest.fixture
    def scratch_doc(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORK_ROOT", str(tmp_path))
        monkeypatch.setenv("EXTRACTION_HYBRID", "0")  # no network in tests
        tools._retriever_cache.clear()
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "doc.md").write_text(CHARTER_MD, encoding="utf-8")
        return "/scratch/doc.md"

    def test_returns_ranked_chunks_with_provenance(self, scratch_doc):
        out = tools.search_hybrid.func([scratch_doc], "крупные сделки процент активов", top_k=2)
        assert "--- /scratch/doc.md (hybrid #1) ---" in out
        assert "12.1.4" in out

    def test_retriever_cached_between_calls(self, scratch_doc):
        tools.search_hybrid.func([scratch_doc], "крупные сделки", top_k=1)
        tools.search_hybrid.func([scratch_doc], "генеральный директор", top_k=1)
        assert len(tools._retriever_cache) == 1

    def test_no_content(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORK_ROOT", str(tmp_path))
        (tmp_path / "scratch").mkdir()
        assert "No content" in tools.search_hybrid.func(["/scratch/missing.md"], "q")
