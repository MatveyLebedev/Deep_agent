from raglib.models import Document
from raglib.parsing.clauses import segment_clauses
from raglib.parsing.markdown import build_sections, section_key_of
from raglib.parsing.ocr import repair_number


def make_doc(text: str, doc_id: str = "doc") -> Document:
    return Document(doc_id=doc_id, source_path="x", md_text=text,
                    sections=build_sections(doc_id, text))


def test_repair_number_fixes_typical_ocr_errors():
    assert repair_number("12.1") == "12.1"          # clean → no-op
    assert repair_number("12,1") == "12.1"          # comma-for-dot
    assert repair_number("12 . 1") == "12.1"        # spaces inside
    assert repair_number("1О.2") == "10.2"          # Cyrillic О → 0
    assert repair_number("1O.2") == "10.2"          # Latin O → 0
    assert repair_number("l2.3") == "12.3"          # l → 1
    assert repair_number("5.З") == "5.3"            # Cyrillic З → 3
    assert repair_number("1б.4") == "16.4"          # б → 6
    assert repair_number("1О , 2") == "10.2"        # mixed glyph + comma + spaces


def test_repair_number_guards():
    assert repair_number("Зона") == ""              # no real digit anchor
    assert repair_number("II") == ""                # Roman numeral, no digit
    assert repair_number("") == ""
    assert repair_number("05.01") == ""             # leading zero → date/amount
    assert repair_number("О5.О1.2О24") == ""        # OCR date still rejected
    assert repair_number("12.05.2024") == ""        # leading-zero middle part


def test_section_key_of_tolerates_ocr():
    assert section_key_of("Статья 1О. НАБЛЮДАТЕЛЬНЫЙ СОВЕТ") == "10"
    assert section_key_of("1О.2. Компетенция") == "10.2"
    assert section_key_of("12,3 Название") == "12.3"
    # look-alike-leading words and Roman numerals must NOT yield a key
    assert section_key_of("Общие положения") == ""
    assert section_key_of("Зона ответственности") == ""
    assert section_key_of("Раздел II") == ""


def test_segment_detects_ocr_clause_numbers():
    text = ("## Статья 1О. ОСНОВНЫЕ ПОЛОЖЕНИЯ\n\n"   # heading OCR: 1О → 10
            "1О.1. Кириллическая О вместо нуля.\n\n"   # 10.1
            "10,2. Точка распозналась как запятая.\n\n"  # 10.2 (comma)
            "1O.3. Латинская O вместо нуля.\n")          # 10.3
    on = {c.number for c in segment_clauses(make_doc(text), ocr_repair=True)}
    assert {"10", "10.1", "10.2", "10.3"} <= on

    # with the fallback off, the OCR'd paragraph numbers are NOT split out
    # (the heading key is still repaired — headings are high-confidence)
    off = {c.number for c in segment_clauses(make_doc(text), ocr_repair=False)}
    assert "10.1" not in off and "10.2" not in off and "10.3" not in off
    assert "10" in off


def test_ocr_fallback_does_not_invent_clauses():
    # a word and a date must never open a clause even with the fallback on
    text = ("Зона ответственности сторон определяется договором.\n\n"
            "12.05.2024. Договор заключён между сторонами.\n")
    clauses = segment_clauses(make_doc(text), ocr_repair=True)
    assert [c.number for c in clauses] == [""]       # one unnumbered block, no clause
