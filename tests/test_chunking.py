"""hierarchical_chunks — structure-aware chunking of docling charter markdown."""
from extraction import _size_split, hierarchical_chunks

CHARTER_MD = """Total pages: 35

--- Page 1/35 ---
<!-- image -->

## Статья 12. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ

12.1. Компетенция Наблюдательного совета:

12.1.1. одобрение крупных сделок, предметом которых является имущество,
стоимость которого составляет от 25 до 50 процентов балансовой стоимости активов

12.1.2. образование исполнительных органов

--- Page 2/35 ---

## Статья 13. ИСПОЛНИТЕЛЬНЫЕ ОРГАНЫ

13.1. Генеральный директор осуществляет руководство текущей деятельностью
"""


class TestHierarchicalChunks:
    def test_heading_prefix_on_every_chunk(self):
        chunks = hierarchical_chunks(CHARTER_MD)
        assert chunks
        for c in chunks:
            assert c.startswith("["), f"chunk without heading prefix: {c[:60]!r}"

    def test_page_markers_and_noise_removed(self):
        joined = "\n".join(hierarchical_chunks(CHARTER_MD))
        assert "--- Page" not in joined
        assert "Total pages" not in joined
        assert "<!-- image -->" not in joined

    def test_clause_text_preserved_under_its_section(self):
        chunks = hierarchical_chunks(CHARTER_MD)
        major = [c for c in chunks if "12.1.1" in c]
        assert major, "clause 12.1.1 lost during chunking"
        assert all(c.startswith("[Статья 12") for c in major)

    def test_sections_not_mixed(self):
        chunks = hierarchical_chunks(CHARTER_MD)
        for c in chunks:
            if "Генеральный директор" in c:
                assert c.startswith("[Статья 13")

    def test_oversized_clause_split_keeps_prefix(self):
        text = "## Статья 5. ДЛИННАЯ\n\n5.1. " + "слово " * 800
        chunks = hierarchical_chunks(text, max_chars=600)
        assert len(chunks) > 1
        assert all(c.startswith("[Статья 5") for c in chunks)
        assert all(len(c) <= 600 + 80 for c in chunks)

    def test_unstructured_text_falls_back(self):
        text = "просто текст без заголовков " * 200
        chunks = hierarchical_chunks(text, max_chars=500)
        assert chunks
        assert all(chunks)

    def test_whitespace_runs_collapsed(self):
        text = "## Статья 1. ТЕСТ\n\n1.1. колонка    таблицы      с   пробелами"
        joined = "\n".join(hierarchical_chunks(text))
        assert "колонка таблицы с пробелами" in joined


class TestSizeSplit:
    def test_step_and_overlap(self):
        text = "a" * 1000
        chunks = _size_split(text, size=400, overlap=150)
        assert chunks[0] == "a" * 400
        assert len(chunks) == 4  # starts at 0, 250, 500, 750

    def test_minimum_size_enforced(self):
        chunks = _size_split("abc", size=10)
        assert chunks == ["abc"]
