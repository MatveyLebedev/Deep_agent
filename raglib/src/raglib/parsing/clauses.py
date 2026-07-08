"""Clause segmentation: the document -> whole clauses (units of output),
plus windowing of oversized clauses into index units.

A clause boundary is:
  * every markdown heading (the heading + its intro text = one clause, numbered
    with the section key), or
  * a NUMBERED PARAGRAPH — a body line starting with a dotted multi-part number
    ('12.1.', '12.1.4)', …). In recognized charters most clauses are paragraphs,
    not headings.

Single-number lines ('1.', '2)') do NOT open a clause: those are ordinary
markdown list items / nested enumerations and stay inside the parent clause.
Leading-zero parts ('01.02.2025') are rejected to avoid matching dates.

Invariant: clause.text == md_text[span[0]:span[1]] — an exact substring of the
recognized markdown, so output can be verified and processed programmatically.
"""
from __future__ import annotations

import re

from raglib.models import Clause, Document, Section
from raglib.parsing.markdown import is_descendant

# Dotted number with 2..5 parts followed by '.' or ')'. Recognition artifacts
# tolerated (all seen in real docling output):
#   * optional list bullet / bold marks before the number  ("- 7.2. текст")
#   * spaces INSIDE the number                             ("7 . 2.", "6. 5.1)")
#   * no space between the punctuation and the text        ("7.3.1.оплачивать")
# Guards: no leading zeros in any part (rejects dates 01.02.2025 / 25 . 05 . 2025);
# without a space after the punctuation the next char must be a LETTER, so
# dotted amounts like "3.682.482.815" never open a clause.
CLAUSE_PARA_RE = re.compile(
    r"^\s{0,3}(?:[-*]\s+)?(?:\*{1,2}\s*)?"
    r"(?!0\d)(\d+(?:\s*\.\s*(?!0\d)\d+){1,4})"
    r"\s*[.)](?:\*{1,2})?(?:\s+\S|(?=[^\W\d_]))"
)
# recognition often emits ООО-style clauses as list items WITHOUT punctuation
# after the number ("- 1.1 Общество ..."): allowed only when bulleted, so bare
# "1.1 га" measurements in running text never open a clause
_BULLET_CLAUSE_RE = re.compile(
    r"^\s{0,3}[-*]\s+(?!0\d)(\d+(?:\.(?!0\d)\d+){1,4})\s+\S"
)
# clause rendered as a table row ("| 8.1.24  Распределение прибыли … |"):
# the number opens the first cell and the cell must contain a letter, so
# numeric data cells ("| 3.682.482 |") never open a clause
_TABLE_ROW_CLAUSE_RE = re.compile(
    r"^\s*\|\s*(?:\*{1,2}\s*)?(?!0\d)(\d+(?:\.(?!0\d)\d+){1,4})"
    r"(?:\s*[.)])?\s+[^|]*[^\W\d_]"
)


def match_clause_paragraph(line: str) -> re.Match | None:
    """Match a body line that OPENS a new clause; group(1) is the number
    (may contain spaces around dots — normalize with clause_number_of)."""
    return (CLAUSE_PARA_RE.match(line) or _BULLET_CLAUSE_RE.match(line)
            or _TABLE_ROW_CLAUSE_RE.match(line))


def clause_number_of(match: re.Match) -> str:
    """Normalized clause number from a match ('7 . 2' -> '7.2')."""
    return re.sub(r"\s+", "", match.group(1))


# Recognition sometimes flattens a whole table into a few giant single-line
# cells, so sub-clauses sit MID-LINE: "…Устав  8.1.1  Внесение изменений…".
# Signature: the number is framed by 2+ spaces (table-cell padding) and
# followed by a letter.
_INLINE_TABLE_CLAUSE_RE = re.compile(
    r"\s{2,}(?!0\d)(\d+(?:\.(?!0\d)\d+){1,4})(?:\s*[.)])?\s{2,}(?=[^\W\d_])"
)


def _family_key(number: str) -> str:
    """The numbering scope inline sub-clauses must belong to: the clause's own
    number or its parent ('8.1.24' -> siblings under '8.1' are fine too)."""
    return number.rsplit(".", 1)[0] if "." in number else number


def _consistent_with_section(num: str, section_key: str) -> bool:
    """Does the number belong to the section's numbering family?
    Section '12.1' accepts 12.1.x (descendants) and 12.x (siblings)."""
    if not section_key:
        return True  # no numbering context to check against
    return (is_descendant(num, section_key)
            or is_descendant(num, _family_key(section_key)))


def _split_inline_table_clauses(clause: Clause) -> list[Clause]:
    """Split a table-bearing clause at mid-line sub-clause numbers.

    Precision guards: only clauses that contain table markup ('|'), only
    numbers that are descendants of the clause itself or of its parent
    (so СОДЕРЖАНИЕ-style digests never split), only table-cell spacing."""
    if not clause.number or "|" not in clause.text:
        return [clause]
    own, family = clause.number, _family_key(clause.number)
    cuts: list[tuple[int, str]] = []  # (offset in clause.text, number)
    for m in _INLINE_TABLE_CLAUSE_RE.finditer(clause.text):
        num = re.sub(r"\s+", "", m.group(1))
        if num != own and (is_descendant(num, own) or is_descendant(num, family)):
            cuts.append((m.start(1), num))
    if not cuts:
        return [clause]

    pieces: list[Clause] = []
    bounds = [(0, clause.number)] + cuts + [(len(clause.text), "")]
    base = clause.span[0]
    for (start, number), (end, _next) in zip(bounds, bounds[1:]):
        raw = clause.text[start:end]
        text = raw.strip()
        if not text:
            continue
        a = base + start + (len(raw) - len(raw.lstrip()))
        pieces.append(Clause(clause_id=-1, doc_id=clause.doc_id, number=number,
                             section_key=clause.section_key,
                             span=(a, a + len(text)), text=text))
    return pieces or [clause]


def _line_offsets(md_text: str) -> list[int]:
    """Char offset of the start of each line (split by '\\n')."""
    offsets = [0]
    for i, ch in enumerate(md_text):
        if ch == "\n":
            offsets.append(i + 1)
    return offsets


def _innermost_section(sections: list[Section], line: int) -> Section | None:
    best: Section | None = None
    for s in sections:
        if s.line_start <= line < s.line_end:
            if best is None or s.line_start >= best.line_start:
                best = s
    return best


def segment_clauses(doc: Document) -> list[Clause]:
    """Split a document into whole clauses. clause_id is left as -1;
    the index builder assigns global ids."""
    md = doc.md_text
    lines = md.split("\n")
    offsets = _line_offsets(md)
    heading_lines = {s.line_start for s in doc.sections}

    # boundaries: (line_index, number, section_key)
    boundaries: list[tuple[int, str, str]] = []
    for s in doc.sections:
        boundaries.append((s.line_start, s.key, s.key))
    for i, ln in enumerate(lines):
        if i in heading_lines:
            continue
        m = match_clause_paragraph(ln)
        if not m:
            continue
        sec = _innermost_section(doc.sections, i)
        sk = sec.key if sec else ""
        num = clause_number_of(m)
        # OCR merge ("- 9. 21.2. текст"): a spaced number inconsistent with the
        # section resolves to its section-consistent suffix ("21.2")
        if re.search(r"\s", m.group(1)) and not _consistent_with_section(num, sk):
            parts = num.split(".")
            for j in range(1, len(parts) - 1):
                cand = ".".join(parts[j:])
                if _consistent_with_section(cand, sk):
                    num = cand
                    break
        # ")"-terminated numbers are enumeration items ("- 20.1) вопрос") —
        # they open a clause only when they belong to the section's numbering
        after = m.group(0)[m.end(1):].lstrip()
        if after.startswith(")") and not _consistent_with_section(num, sk):
            continue
        boundaries.append((i, num, sk))
    boundaries.sort(key=lambda b: b[0])
    if not boundaries or boundaries[0][0] > 0:
        boundaries.insert(0, (0, "", ""))  # preamble before any boundary

    clauses: list[Clause] = []
    for bi, (start_line, number, section_key) in enumerate(boundaries):
        end_line = boundaries[bi + 1][0] if bi + 1 < len(boundaries) else len(lines)
        a = offsets[start_line]
        b = offsets[end_line] if end_line < len(offsets) else len(md)
        raw = md[a:b]
        stripped_left = len(raw) - len(raw.lstrip())
        stripped_right = len(raw) - len(raw.rstrip())
        a2, b2 = a + stripped_left, b - stripped_right
        if a2 >= b2:
            continue  # empty block (e.g. consecutive headings)
        text = md[a2:b2]
        clauses.append(Clause(
            clause_id=-1, doc_id=doc.doc_id, number=number or section_key,
            section_key=section_key, span=(a2, b2), text=text,
        ))

    # second pass: recognition may flatten tables into single lines, hiding
    # sub-clauses mid-line — split those out (guarded, see the helper)
    out: list[Clause] = []
    for c in clauses:
        out.extend(_split_inline_table_clauses(c))
    return out


def window_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Sliding-window spans over an oversized clause. A short clause yields
    exactly one span covering the whole text (1:1 clause->unit)."""
    n = len(text)
    if n <= size:
        return [(0, n)]
    step = max(1, size - overlap)
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        j = min(i + size, n)
        spans.append((i, j))
        if j >= n:
            break
        i += step
    return spans
