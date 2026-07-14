import pytest

from raglib import RagIndex

# docling-style output: page markers, image placeholders, a bare-number
# heading ('## 7.') whose real title lives in the body, and a table
SCANNED = """Total pages: 2

--- Page 1/2 ---
<!-- image -->

## УСТАВ АКЦИОНЕРНОГО ОБЩЕСТВА «ВЕКТОР»

## 7.

Ревизионная комиссия Общества избирается годовым Общим собранием акционеров сроком до следующего годового Общего собрания акционеров.

7.1. Ревизионная комиссия осуществляет контроль за финансово-хозяйственной деятельностью Общества.

--- Page 2/2 ---

| Показатель | Значение |
| --- | --- |
| Обыкновенных акций | 100 |

## 8. ПРОТОКОЛ ОБЩЕГО СОБРАНИЯ

8.1. Протокол Общего собрания акционеров подтверждается лицом, осуществляющим ведение реестра акционеров Общества и выполняющим функции счётной комиссии.
"""


@pytest.fixture(scope="module")
def scanned_index(tmp_path_factory) -> RagIndex:
    d = tmp_path_factory.mktemp("scanned")
    (d / "vector_charter.md").write_text(SCANNED, encoding="utf-8")
    return RagIndex.build([d / "vector_charter.md"], d / "idx", embeddings=None)


def test_bare_number_title_promoted_to_content(scanned_index):
    entries = {e.key: e for e in scanned_index.toc_entries() if e.key}
    # '## 7.' carries no words -> the first content sentence becomes the title
    assert entries["7"].title.startswith("Ревизионная комиссия Общества избирается")
    # a normal heading keeps its own title, preview comes from the body
    assert entries["8"].title == "8. ПРОТОКОЛ ОБЩЕГО СОБРАНИЯ"
    assert entries["8"].preview.startswith("8.1. Протокол Общего собрания")


def test_preview_skips_recognition_noise(scanned_index):
    for e in scanned_index.toc_entries():
        low = e.preview.lower()
        assert "page" not in low and "<!--" not in low
        assert not e.preview.startswith("|")
        assert "total pages" not in low


def test_outline_preview_flag(scanned_index):
    plain = scanned_index.toc()
    assert "[7] Ревизионная комиссия Общества избирается" in plain
    assert "—" not in plain.replace("«", "")  # no previews unless asked

    rich = scanned_index.toc(preview=True)
    assert "[8] 8. ПРОТОКОЛ ОБЩЕГО СОБРАНИЯ — 8.1. Протокол Общего собрания" in rich
    # enriched title == preview for the bare heading: no duplicated '— …' tail
    line7 = next(ln for ln in rich.splitlines() if ln.strip().startswith("[7]"))
    assert "—" not in line7


def test_find_section_matches_enriched_title_and_preview(scanned_index):
    # bare '## 7.' is findable by the words of its promoted title
    refs = scanned_index.find_section("ревизионная комиссия")
    assert refs and refs[0].key == "7"
    # and a section is findable by words that occur only in its body preview
    refs = scanned_index.find_section("ведение реестра акционеров")
    assert refs and refs[0].key == "8" and refs[0].score == pytest.approx(0.3)


def test_navigation_driven_search(vec_index):
    """The full navigation flow answers a question without knowing keys upfront:
    outline -> pick section -> read it whole -> ranked search scoped to it."""
    # 1) programmatic outline of one document
    entries = vec_index.toc_entries(doc="charter")
    target = next(e for e in entries if "ПРОТОКОЛ" in e.title.upper())
    assert target.key == "14"

    # 2) whole section, descendants included
    section_text = vec_index.read_section("charter", target.key)
    assert "нотариального удостоверения" in section_text

    # 3) ranked search narrowed to the found section
    hits = vec_index.search("нотариальное удостоверение протокола",
                            method="bm25", doc="charter", section=target.key)
    assert hits and hits[0].clause_number == "14.1"
    assert hits[0].text.startswith("14.1.")  # whole clause, number preserved

    # the same flow across documents: find by title words, then read
    ref = vec_index.find_section("порядок предоставления доступа")[0]
    assert ref.doc_id == "policy"
    assert "заявки" in vec_index.read_section(ref.doc_id, ref.title)


def test_unnumbered_sections_are_navigable(vec_index):
    """A section with no number is still listed with an addressable key, found,
    and read — both by its synthetic key and by its title."""
    entries = vec_index.toc_entries(doc="policy")
    отзыв = next(e for e in entries if e.title == "Отзыв доступа")
    assert отзыв.key and отзыв.key not in ("",)          # has an addressable key

    # the outline tags it like a numbered section, so the key is copy-pasteable
    assert f"[{отзыв.key}] Отзыв доступа" in vec_index.toc(doc="policy")

    # find_section returns that key; read_section accepts it AND the title
    ref = vec_index.find_section("отзыв доступа")[0]
    assert ref.key == отзыв.key
    by_key = vec_index.read_section("policy", ref.key)
    by_title = vec_index.read_section("policy", "Отзыв доступа")
    assert by_key == by_title
    assert "увольнения" in by_key


def test_toc_entries_unknown_doc_raises(vec_index):
    with pytest.raises(KeyError, match="missing"):
        vec_index.toc_entries(doc="missing")


def test_toc_clause_numbers(vec_index, scanned_index):
    entries = {e.key: e for e in vec_index.toc_entries(doc="charter") if e.key}
    assert entries["13"].clause_numbers == ["13", "13.1", "13.2", "13.3"]
    # «12.2» physically sits inside section 12.1's scope, but BY NUMBERING it
    # belongs to [12] — that's where the outline shows it
    assert "12.2" in entries["12"].clause_numbers
    assert "12.2" not in entries["12.1"].clause_numbers
    assert entries["12.1"].clause_numbers == ["12.1", "12.1.1", "12.1.2",
                                              "12.1.3", "12.1.4"]

    outline = scanned_index.toc(clauses=True)
    assert "пункты: 7, 7.1" in outline
    assert "пункты: 8, 8.1" in outline
    # default stays compact — no clause listings unless asked
    assert "пункты:" not in scanned_index.toc(preview=True)
