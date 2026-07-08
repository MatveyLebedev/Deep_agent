"""Section outline / scoped reading over docling markdown (tools.py).

read_pdf flattens a charter to flat `##` headings, so section scope comes from
the clause number (12 ⊃ 12.1 ⊃ 12.1.4), not the heading level — these tests
pin that contract.
"""
import pytest

from tools import _is_descendant, _section_key, list_sections, read_section

DOC_MD = """## Статья 11. ОБЩЕЕ СОБРАНИЕ АКЦИОНЕРОВ

## 11.1 . Компетенция Общего собрания

text A

## 11.1.1. вопросы, решения по которым принимаются большинством

text B

## Статья 12 . НАБЛЮДАТЕЛЬНЫЙ СОВЕТ

text C
"""


@pytest.fixture
def scratch_doc(tmp_path, monkeypatch):
    monkeypatch.setenv("WORK_ROOT", str(tmp_path))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "doc.md").write_text(DOC_MD, encoding="utf-8")
    return "/scratch/doc.md"


class TestSectionKey:
    @pytest.mark.parametrize("title,key", [
        ("Статья 12 . НАБЛЮДАТЕЛЬНЫЙ СОВЕТ", "12"),
        ("Статья 7. ПРАВА И ОБЯЗАННОСТИ", "7"),
        ("12.1.4. Решение по которым", "12.1.4"),
        ("11.1 . Компетенция Общего собрания", "11.1"),
        ("6.3. Увеличение уставного капитала", "6.3"),
        ("СОДЕРЖАНИЕ", ""),
        ("Глава 3 О ЧЁМ-ТО", "3"),
    ])
    def test_key_extraction(self, title, key):
        assert _section_key(title) == key


class TestIsDescendant:
    def test_self(self):
        assert _is_descendant("12", "12")

    def test_child_and_grandchild(self):
        assert _is_descendant("12.1", "12")
        assert _is_descendant("12.1.4", "12")

    def test_prefix_of_digits_is_not_descendant(self):
        assert not _is_descendant("121", "12")

    def test_sibling(self):
        assert not _is_descendant("13", "12")

    def test_empty_keys(self):
        assert not _is_descendant("", "12")
        assert not _is_descendant("12", "")


class TestListSections:
    def test_outline_shows_keys_and_titles(self, scratch_doc):
        out = list_sections.func(scratch_doc)
        assert "[11]" in out and "[11.1]" in out and "[11.1.1]" in out and "[12]" in out
        assert "НАБЛЮДАТЕЛЬНЫЙ СОВЕТ" in out

    def test_no_headings_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORK_ROOT", str(tmp_path))
        (tmp_path / "scratch").mkdir()
        (tmp_path / "scratch" / "flat.md").write_text("plain text", encoding="utf-8")
        assert "No headings" in list_sections.func("/scratch/flat.md")


class TestReadSection:
    def test_parent_includes_all_descendants(self, scratch_doc):
        body = read_section.func(scratch_doc, "11")
        assert "text A" in body and "text B" in body
        assert "text C" not in body

    def test_exact_subclause_only(self, scratch_doc):
        body = read_section.func(scratch_doc, "11.1.1")
        assert "text B" in body
        assert "text A" not in body and "text C" not in body

    def test_lookup_by_statya_prefix(self, scratch_doc):
        body = read_section.func(scratch_doc, "Статья 12")
        assert "text C" in body

    def test_lookup_by_title_word(self, scratch_doc):
        body = read_section.func(scratch_doc, "Наблюдательный")
        assert "text C" in body

    def test_not_found_lists_available_keys(self, scratch_doc):
        out = read_section.func(scratch_doc, "99")
        assert "not found" in out
        assert "11.1.1" in out

    def test_truncation_marker(self, scratch_doc):
        out = read_section.func(scratch_doc, "11", max_chars=40)
        assert "section truncated" in out

    def test_pdf_path_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WORK_ROOT", str(tmp_path))
        inp = tmp_path / "input"
        inp.mkdir()
        (inp / "doc.pdf").write_bytes(b"%PDF-1.4")
        out = read_section.func("/input/doc.pdf", "1")
        assert "read_pdf" in out

    def test_path_outside_virtual_roots_rejected(self, scratch_doc):
        out = read_section.func("/etc/passwd", "1")
        assert "Error" in out
