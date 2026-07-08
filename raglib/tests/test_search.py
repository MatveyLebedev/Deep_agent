import re

import pytest

from raglib import RagIndex
from raglib.embeddings import HashingEmbeddings


def _assert_output_invariants(index: RagIndex, hits):
    """PLAN.md §1: every hit is a WHOLE clause with its number — an exact,
    uncut substring of the recognized markdown; no clause appears twice."""
    docs = index._data.documents
    seen = set()
    for h in hits:
        assert isinstance(h.clause_number, str)
        assert h.text == docs[h.doc_id].md_text[
            index._data.clauses[h.clause_id].span[0]:
            index._data.clauses[h.clause_id].span[1]]
        assert h.text in docs[h.doc_id].md_text
        assert h.clause_id not in seen
        seen.add(h.clause_id)
        if h.clause_number:
            assert h.section_path[-1] == h.clause_number
        # provenance metadata invariants (doc name + titled path chapter->paragraph)
        assert h.doc_name                                    # friendly name present
        assert len(h.section_titles) == len(h.section_path)  # titles align with path
        assert h.breadcrumb == " › ".join(h.path)            # readable path consistent


def test_bm25_returns_whole_numbered_clause(bm25_index):
    hits = bm25_index.search("нотариального удостоверения протокола", method="bm25")
    assert hits[0].clause_number == "14.1"
    assert hits[0].text.startswith("14.1.")
    _assert_output_invariants(bm25_index, hits)


def test_invariants_hold_for_every_method(vec_index):
    query = "балансовой стоимости активов"
    for method in ("bm25", "vector", "hybrid"):
        hits = vec_index.search(query, method=method, top_k=8)
        assert hits, method
        _assert_output_invariants(vec_index, hits)
    _assert_output_invariants(vec_index, vec_index.grep(r"процентов"))


def test_oversized_clause_returned_once_and_uncut(vec_index):
    # 'утверждение независимого аудитора' lives in the tail of clause 12.1.4 —
    # beyond the first 1500-char window. The unit matches, the WHOLE clause returns.
    for method in ("bm25", "vector"):
        hits = vec_index.search("утверждение независимого аудитора",
                                method=method, top_k=5)
        numbers = [h.clause_number for h in hits]
        assert numbers.count("12.1.4") == 1, method
        hit = next(h for h in hits if h.clause_number == "12.1.4")
        assert len(hit.text) > 1500
        assert "н) утверждение независимого аудитора" in hit.text


def test_hits_carry_document_name_and_titled_path(bm25_index):
    # provenance metadata: friendly document name + hierarchical path (titles +
    # numbers) from the top chapter down to the leaf paragraph.
    hits = bm25_index.search("независимого аудитора", method="bm25", top_k=8)
    hit = next(h for h in hits if h.clause_number == "12.1.4")

    # document name is the source filename, not just the sanitized doc_id
    assert hit.doc_id == "charter"
    assert hit.doc_name == "charter.md"

    # section_titles align 1:1 with the numeric section_path (chapter -> paragraph)
    assert hit.section_path == ["12", "12.1", "12.1.4"]
    assert len(hit.section_titles) == len(hit.section_path)
    assert hit.section_titles[0] == "Статья 12. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ"
    assert hit.section_titles[1] == "12.1. Компетенция Наблюдательного совета"
    assert hit.section_titles[2] == ""            # leaf clause has no heading of its own

    # readable path: the title where present, the bare number for the leaf
    assert hit.path == ["Статья 12. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ",
                        "12.1. Компетенция Наблюдательного совета", "12.1.4"]
    assert hit.breadcrumb == ("Статья 12. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ"
                              " › 12.1. Компетенция Наблюдательного совета › 12.1.4")


def test_hybrid_rrf(vec_index):
    hits = vec_index.search("крупной сделки балансовой стоимости", method="hybrid",
                            top_k=5)
    numbers = {h.clause_number for h in hits}
    assert numbers & {"13.1", "13.2"}
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_tree_strategy_matches_flat_on_small_corpus(vec_index):
    query = "балансовой стоимости активов"
    flat = vec_index.search(query, method="vector", strategy="flat", top_k=5)
    tree = vec_index.search(query, method="vector", strategy="tree", top_k=5)
    assert tree
    _assert_output_invariants(vec_index, tree)
    assert flat[0].clause_id in {h.clause_id for h in tree}


def test_filters(vec_index):
    hits = vec_index.search("балансовой стоимости активов", method="bm25",
                            section="13")
    assert hits and all(h.section_path[0] == "13" for h in hits)

    hits = vec_index.search("доступ", method="bm25", doc="policy")
    assert hits and all(h.doc_id == "policy" for h in hits)
    assert all(h.clause_number == "" for h in hits)  # unnumbered document


def test_grep(vec_index):
    rx = r"\d+\s*(?:и более\s*)?процентов"
    hits = vec_index.grep(rx)
    assert "13.1" in [h.clause_number for h in hits]
    assert all(re.search(rx, h.text, re.IGNORECASE) for h in hits)

    scoped = vec_index.grep(r"процентов", section="13")
    assert scoped and all(h.section_path[0] == "13" for h in scoped)


def test_method_and_strategy_validation(bm25_index):
    with pytest.raises(ValueError, match="method"):
        bm25_index.search("тест", method="fuzzy")
    with pytest.raises(ValueError, match="strategy"):
        bm25_index.search("тест", strategy="magic")


def test_toc(vec_index):
    outline = vec_index.toc()
    assert "=== charter ===" in outline
    assert "[12.1]" in outline and "[13]" in outline
    policy_only = vec_index.toc(doc="policy")
    assert "Назначение" in policy_only and "charter" not in policy_only
    with pytest.raises(KeyError):
        vec_index.toc(doc="missing")


def test_find_section_by_key_and_title(vec_index):
    assert vec_index.find_section("Статья 12")[0].key == "12"
    assert vec_index.find_section("12.1")[0].key == "12.1"
    refs = vec_index.find_section("наблюдательный совет")
    assert refs and refs[0].key in ("12", "12.1")


def test_find_section_semantic(vec_index, bm25_index):
    refs = vec_index.find_section("согласие на совершение крупной сделки",
                                  semantic=True)
    assert refs
    scores = [r.score for r in refs]
    assert scores == sorted(scores, reverse=True)
    with pytest.raises(RuntimeError, match="BM25-only"):
        bm25_index.find_section("крупные сделки", semantic=True)


def test_read_section_whole_with_descendants(vec_index):
    text = vec_index.read_section("charter", "12")
    assert "12.1.4." in text and "12.2." in text          # descendants included
    assert "Статья 13" not in text                        # neighbour excluded
    assert "13.3." in vec_index.read_section("charter", "Статья 13")
    with pytest.raises(KeyError, match="Available keys"):
        vec_index.read_section("charter", "99")
    with pytest.raises(KeyError):
        vec_index.read_section("missing", "1")


def test_vector_search_is_deterministic(tmp_path, fixture_files, embeddings):
    other = RagIndex.build(fixture_files, tmp_path / "idx2", embeddings=embeddings)
    q = "утверждение независимого аудитора"
    a = [h.clause_number for h in other.search(q, method="vector", top_k=3)]
    b = [h.clause_number for h in
         RagIndex.load(other.root, embeddings=HashingEmbeddings()).search(
             q, method="vector", top_k=3)]
    assert a == b
