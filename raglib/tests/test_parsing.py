from raglib.models import Document
from raglib.parsing.clauses import (
    match_clause_paragraph,
    segment_clauses,
    window_spans,
)
from raglib.parsing.markdown import (
    build_sections,
    is_descendant,
    is_numbered_key,
    is_toc_title,
    section_key_of,
)


def make_doc(text: str, doc_id: str = "doc") -> Document:
    return Document(doc_id=doc_id, source_path="x", md_text=text,
                    sections=build_sections(doc_id, text))


def test_section_key_of():
    assert section_key_of("Статья 12 . НАБЛЮДАТЕЛЬНЫЙ СОВЕТ") == "12"
    assert section_key_of("12.1.4. Решение принимается") == "12.1.4"
    assert section_key_of("11.1 . Компетенция") == "11.1"
    assert section_key_of("СОДЕРЖАНИЕ") == ""
    assert section_key_of("Глава 3. Управление") == "3"


def test_is_descendant():
    assert is_descendant("12", "12")
    assert is_descendant("12.1.4", "12")
    assert not is_descendant("121", "12")
    assert not is_descendant("11", "1")   # dotted prefix, not string prefix
    assert not is_descendant("12", "")


def test_section_tree_on_charter(charter_text):
    sections = build_sections("charter", charter_text)
    keys = [s.key for s in sections if s.key]
    assert {"1", "11", "12", "12.1", "13", "14"} <= set(keys)

    by_key = {s.key: s for s in sections if s.key}
    assert by_key["12.1"].parent_key == "12"
    assert "12.1" in by_key["12"].children
    # numbered scope: Статья 12 ends where Статья 13 starts
    assert by_key["12"].line_end == by_key["13"].line_start
    # unnumbered heading (СОДЕРЖАНИЕ) closes at the very next heading and gets a
    # stable synthetic key (addressable like a numbered section)
    toc = next(s for s in sections if "СОДЕРЖАНИЕ" in s.title)
    assert not is_numbered_key(toc.key) and toc.key and toc.line_end > toc.line_start


def test_clause_segmentation_on_charter(charter_text):
    doc = make_doc(charter_text, "charter")
    clauses = segment_clauses(doc)
    numbers = {c.number for c in clauses}
    assert {"1.1", "1.2", "11.1", "11.2", "12.1.1", "12.1.4", "12.2",
            "13.1", "13.2", "13.3", "14.1", "14.2"} <= numbers
    # dates like 18.02.2025 never become clause numbers
    assert not any("2025" in n for n in numbers)

    # the output invariant holds by construction: text is an exact source slice
    for c in clauses:
        assert c.text == charter_text[c.span[0]:c.span[1]]
        assert c.text.strip() == c.text  # spans are trimmed

    # nested а)/б) enumerations stay inside their parent clause
    c1142 = next(c for c in clauses if c.number == "12.1.4")
    assert "а) определение приоритетных" in c1142.text
    assert "н) утверждение независимого аудитора" in c1142.text
    assert len(c1142.text) > 1500  # oversized clause stays whole

    # the numbered paragraph 12.2 sits inside section 12.1's markdown scope
    assert next(c for c in clauses if c.number == "12.2").section_key == "12.1"


def test_clause_para_regex_rejects_noise():
    assert match_clause_paragraph("12.1. Компетенция совета")
    assert match_clause_paragraph("12.1.4) Решение принимается")
    # recognition emits ООО clauses as list items, with or without punctuation
    assert match_clause_paragraph("- 1.1 Общество учреждено и действует")
    assert match_clause_paragraph("- 12.1.4. Решение принимается")
    assert not match_clause_paragraph("1. обычный пункт списка")   # single number
    assert not match_clause_paragraph("- 1. обычный пункт списка")
    assert not match_clause_paragraph("01.02.2025. Дата встречи")  # leading zero
    assert not match_clause_paragraph("- 01.02.2025 дата в списке")
    assert not match_clause_paragraph("1.1 га земли под застройку")  # bare, no bullet
    assert not match_clause_paragraph("| 12.1 | таблица |")


def test_clause_para_regex_recognition_artifacts():
    from raglib.parsing.clauses import clause_number_of

    # spaces INSIDE the number (real docling output)
    m = match_clause_paragraph("- 7 . 2. Акционеры, не полностью оплатившие акции")
    assert m and clause_number_of(m) == "7.2"
    m = match_clause_paragraph("6. 5.1) утверждение решения о выпуске акций")
    assert m and clause_number_of(m) == "6.5.1"
    # no space between the punctuation and the text
    m = match_clause_paragraph("- 7.3.1.оплачивать акции в сроки и порядке")
    assert m and clause_number_of(m) == "7.3.1"
    # clause rendered as a table row
    m = match_clause_paragraph("| 8.1.24  Распределение чистой прибыли | голосов |")
    assert m and clause_number_of(m) == "8.1.24"
    # deep numbering, five parts
    m = match_clause_paragraph("11.4.4.5.1. К материалам относятся")
    assert m and clause_number_of(m) == "11.4.4.5.1"

    # guards: dotted amounts and spaced dates never open a clause
    assert not match_clause_paragraph("3.682.482.815 рублей уставного капитала")
    assert not match_clause_paragraph("- 25 . 05 . 2025 протокол заседания")
    assert not match_clause_paragraph("| 3.682.482 | 815 |")
    assert not match_clause_paragraph("| Статья 1. ОБЩИЕ ПОЛОЖЕНИЯ ..... 4 |")


def test_bullet_clauses_segmentation():
    text = ("## 1. Общие положения\n\n"
            "- 1.1 Общество учреждено и действует в соответствии с законом.\n"
            "- 1.2 Общество имеет полное фирменное наименование.\n\n"
            "Прочий текст без номера остаётся в пункте 1.2.\n")
    doc = make_doc(text)
    clauses = segment_clauses(doc)
    numbers = [c.number for c in clauses]
    assert numbers == ["1", "1.1", "1.2"]
    assert "Прочий текст" in clauses[-1].text  # unnumbered tail stays in 1.2


def test_inline_table_clauses_are_split_out():
    """Recognition flattens tables into single lines: sub-clauses sit mid-line
    ('…Устав  8.1.1  Внесение…'). They must become separate whole clauses."""
    text = ("## 8. Общее собрание участников\n\n"
            "8.1. К компетенции Общего собрания относятся:\n\n"
            "| Вопрос  8.1.1  Внесение изменений в устав Общества  Большинство "
            "в 2/3 голосов  8.1.2  Реорганизация Общества  Единогласно   |\n"
            "|---|\n"
            "| продолжение таблицы  8.1.3  Ликвидация Общества  Единогласно |\n")
    doc = make_doc(text)
    clauses = segment_clauses(doc)
    numbers = [c.number for c in clauses]
    assert numbers == ["8", "8.1", "8.1.1", "8.1.2", "8.1.3"]
    for c in clauses:  # the output invariant survives mid-line splitting
        assert c.text == text[c.span[0]:c.span[1]]
    assert next(c for c in clauses if c.number == "8.1.2").text.startswith("8.1.2")


def test_inline_split_guards():
    # unnumbered digest (here an appendix) with table markup: no split
    digest = ("## ПРИЛОЖЕНИЕ\n\n"
              "| Статья 6. УСТАВНЫЙ КАПИТАЛ  6  6.1. Размер уставного "
              "капитала.  6  6.2. Объявленные акции  7 |\n")
    clauses = segment_clauses(make_doc(digest))
    assert [c.number for c in clauses] == [""]

    # unrelated numbering inside a numbered clause's table: no split either
    other = ("## 3. Права\n\n"
             "3.1. Права участника:\n\n"
             "| ставка  12.5  процентов годовых  повышенная  14.1  процентов |\n")
    clauses = segment_clauses(make_doc(other))
    assert [c.number for c in clauses] == ["3", "3.1"]


def test_context_aware_numbering():
    text = ("## Статья 12. СОВЕТ\n\n"
            "12.1.1. К компетенции относятся:\n"
            "- 1) первый вопрос;\n"
            "- 20.1) особый вопрос из перечня совета;\n"   # элемент перечня
            "- 2) второй вопрос;\n\n"
            "## Статья 21. ДОКУМЕНТЫ\n\n"
            "- 21.1. Компания обязана обеспечить доступ к документам.\n"
            "- 9. 21.2. По требованию акционера предоставляются копии.\n"  # OCR-склейка
            "21.3. По требованию акционера, владеющего 25 процентами акций.\n")
    numbers = [c.number for c in segment_clauses(make_doc(text))]
    assert "20.1" not in numbers        # чужой элемент перечня — не пункт документа
    assert "9.21.2" not in numbers      # склейка «9. 21.2.» распозналась как 21.2
    assert [n for n in numbers if n.startswith("21")] == ["21", "21.1", "21.2", "21.3"]


def test_unnumbered_document_yields_whole_blocks():
    text = "Просто заметка.\n\nБез единого заголовка и нумерации."
    clauses = segment_clauses(make_doc(text))
    assert len(clauses) == 1
    assert clauses[0].number == "" and clauses[0].text == text


def test_unnumbered_sections_get_synthetic_keys():
    """Titled sections without a number (a preamble, «Назначение», «Отзыв…»)
    become first-class: each gets a distinct synthetic key, and a clause under
    one keeps number="" but records that key as section_key for provenance."""
    text = ("# Регламент доступа\n\n"
            "Действует с даты утверждения.\n\n"
            "## Назначение\n\n"
            "Определяет порядок предоставления доступа.\n\n"
            "## Отзыв доступа\n\n"
            "Доступ отзывается в день увольнения.\n")
    doc = make_doc(text)
    unnum = [s for s in doc.sections if not is_numbered_key(section_key_of(s.title))]
    keys = [s.key for s in unnum]
    assert all(k and not is_numbered_key(k) for k in keys)  # synthetic
    assert len(keys) == len(set(keys)) == 3                 # distinct per section

    clauses = segment_clauses(doc)
    отзыв = next(c for c in clauses if "отзывается" in c.text)
    assert отзыв.number == ""                               # no real number
    assert отзыв.section_key and not is_numbered_key(отзыв.section_key)


def test_table_of_contents_is_not_a_clause():
    """A table of contents matches almost any query (it lists every article
    title), so it must not be a retrieval unit — but it stays in the section
    tree for navigation. A genuinely unnumbered preamble is still a clause."""
    text = ("# УСТАВ ООО «Ромашка»\n\n"
            "## СОДЕРЖАНИЕ\n\n"
            "- Статья 1. Общие положения\n"
            "- Статья 13. Крупные сделки\n\n"
            "## Статья 13. КРУПНЫЕ СДЕЛКИ\n\n"
            "13.1. Крупной сделкой признаётся сделка или несколько сделок.\n")
    doc = make_doc(text)
    clauses = segment_clauses(doc)
    numbers = [c.number for c in clauses]
    assert not any("СОДЕРЖАНИЕ" in c.text for c in clauses)  # TOC digest dropped
    assert "13" in numbers and "13.1" in numbers             # real clauses kept
    assert any(c.number == "" and "УСТАВ" in c.text for c in clauses)  # preamble kept
    # the СОДЕРЖАНИЕ section itself survives for toc()/find_section navigation
    assert any(is_toc_title(s.title) for s in doc.sections)


def test_clauses_reconstruct_section(charter_text):
    doc = make_doc(charter_text, "charter")
    clauses = [c for c in segment_clauses(doc)
               if c.number and (c.number == "13" or c.number.startswith("13."))]
    assert [c.number for c in clauses] == ["13", "13.1", "13.2", "13.3"]
    sec = next(s for s in doc.sections if s.key == "13")
    lines = charter_text.split("\n")
    section_body = "\n".join(lines[sec.line_start:sec.line_end])
    for c in clauses:
        assert c.text in section_body


def test_window_spans():
    text = "x" * 100
    spans = window_spans(text, size=30, overlap=10)
    assert spans[0] == (0, 30)
    assert spans[-1][1] == 100                      # full coverage
    assert all(b - a <= 30 for a, b in spans)
    assert window_spans("short", 30, 10) == [(0, 5)]  # short clause -> one unit
