"""Markdown section tree, keyed by clause numbering.

Recognition output flattens heading levels, so section scope is derived from
the CLAUSE NUMBER in the heading title (12 ⊃ 12.1 ⊃ 12.1.4), not from the
number of '#'. Ported from the battle-tested Deep agent tools.py logic.
"""
from __future__ import annotations

import re
from typing import Iterator

from raglib.models import Document, Section
from raglib.parsing.ocr import OCR_PART, repair_number

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(\S.*)$")
# number parts are OCR-tolerant (О→0, l→1, …) but each requires a real digit
# (OCR_PART), so a look-alike-leading word ('Общие', 'Зона') or a Roman numeral
# ('II') never yields a bogus key; repair_number canonicalizes the match.
_WORD_KEY_RE = re.compile(rf"\s*(?:статья|глава|раздел)\s+({OCR_PART})", re.IGNORECASE)
_NUM_KEY_RE = re.compile(rf"\s*({OCR_PART}(?:\s*[.,]\s*{OCR_PART})*)")
# Heading whose body is a table of contents (a digest of other sections), not
# content. Whole-title match so "Статья 5. Содержание деятельности" is unaffected.
_TOC_TITLE_RE = re.compile(
    r"^(?:содержание|оглавление|contents|table\s+of\s+contents)\b[\s:.]*$",
    re.IGNORECASE)


def is_toc_title(title: str) -> bool:
    """True if a heading titles a table of contents (СОДЕРЖАНИЕ / ОГЛАВЛЕНИЕ /
    CONTENTS). Such a section is navigation, not searchable content — it stays in
    the section tree but is not emitted as a clause (see segment_clauses)."""
    return bool(_TOC_TITLE_RE.match(title.strip()))


# Unnumbered sections (СОДЕРЖАНИЕ, «Назначение», a titled preamble, …) get a
# stable per-document key with this prefix, so they are addressable in
# read_section / find_section / filters and carry provenance on hits, exactly
# like numbered ones. It never collides with a real numbering key —
# section_key_of only ever returns dotted digits or "".
SYNTHETIC_KEY_PREFIX = "§"


def is_numbered_key(key: str) -> bool:
    """True for a real numbering key ('12.1'); False for '' and for the
    synthetic key of an unnumbered section ('§3')."""
    return bool(key) and not key.startswith(SYNTHETIC_KEY_PREFIX)


def section_key_of(title: str) -> str:
    """Normalized section key from a heading title.
    'Статья 12 . НАБЛЮДАТЕЛЬНЫЙ СОВЕТ' -> '12'; '12.1.4. Решение…' -> '12.1.4';
    'Статья 1О' -> '10' (OCR); 'СОДЕРЖАНИЕ' -> '' (no number)."""
    m = _WORD_KEY_RE.match(title)
    if m:
        return repair_number(m.group(1))
    m = _NUM_KEY_RE.match(title)
    if m:
        return repair_number(m.group(1))
    return ""


def is_descendant(key: str, parent: str) -> bool:
    """True if `key` is `parent` itself or a dotted sub-key of it."""
    return bool(parent) and bool(key) and (key == parent or key.startswith(parent + "."))


def iter_headings(lines: list[str]) -> Iterator[tuple[int, int, str, str]]:
    """Yield (line_index, level, title, key) for each markdown heading line."""
    for i, ln in enumerate(lines):
        m = HEADING_RE.match(ln)
        if m:
            title = m.group(2).strip()
            yield i, len(m.group(1)), title, section_key_of(title)


def build_sections(doc_id: str, md_text: str) -> list[Section]:
    """Build the flat section list with numbering-based scopes and parent links."""
    lines = md_text.split("\n")
    heads = list(iter_headings(lines))
    sections: list[Section] = []
    for idx, (li, lvl, title, key) in enumerate(heads):
        end = len(lines)
        for lj, _lvl2, _t2, kj in heads[idx + 1:]:
            if key:
                # numbered section: only a later NON-descendant numbered heading closes it
                if kj and not is_descendant(kj, key):
                    end = lj
                    break
            else:  # unnumbered section: closed by the very next heading
                end = lj
                break
        sections.append(Section(doc_id=doc_id, key=key, title=title, level=lvl,
                                line_start=li, line_end=end))

    by_key: dict[str, Section] = {}
    for s in sections:
        if s.key and s.key not in by_key:
            by_key[s.key] = s
    for s in sections:
        if s.key and "." in s.key:
            parts = s.key.split(".")
            for cut in range(len(parts) - 1, 0, -1):
                cand = ".".join(parts[:cut])
                if cand in by_key:
                    s.parent_key = cand
                    by_key[cand].children.append(s.key)
                    break

    # give every remaining unnumbered section a stable synthetic key, so it is
    # addressable and provides provenance just like a numbered one. Done after
    # numbering-based parent linking (synthetic keys carry no hierarchy).
    unnumbered = 0
    for s in sections:
        if not s.key:
            unnumbered += 1
            s.key = f"{SYNTHETIC_KEY_PREFIX}{unnumbered}"
    return sections


def find_section_in_doc(doc: Document, query: str) -> Section | None:
    """Locate a section by key ('12.1'), 'Статья 12', or a title substring."""
    q = query.strip()
    q_key = section_key_of(q) or q
    for s in doc.sections:
        if s.key and s.key == q_key:
            return s
    ql = q.lower()
    for s in doc.sections:
        if ql in s.title.lower():
            return s
    return None


def section_text(doc: Document, section: Section) -> str:
    """Full text of a section INCLUDING nested sub-sections (no truncation)."""
    lines = doc.md_text.split("\n")
    return "\n".join(lines[section.line_start:section.line_end]).strip()
