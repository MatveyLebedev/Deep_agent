"""Core data model.

Two-tier retrieval model (see PLAN.md §5):

* ``Clause`` — the unit of OUTPUT. A whole numbered clause of the document
  («пункт»), never split. ``text`` is always the exact substring
  ``md_text[span[0]:span[1]]`` of the recognized markdown, so search results
  can be processed programmatically and verified byte-for-byte.
* ``Chunk`` — the unit of INDEXING. A window inside an oversized clause
  (short clause -> exactly one chunk). Chunks never leave the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Section:
    """A node of the document's section tree, keyed by clause numbering
    (12 ⊃ 12.1 ⊃ 12.1.4), not by markdown heading level."""
    doc_id: str
    key: str                    # dotted number ("12.1"); "" for unnumbered headings
    title: str
    level: int                  # markdown heading level (1-6)
    line_start: int             # heading line index in the doc's markdown
    line_end: int               # exclusive; covers all descendant sections
    parent_key: str | None = None
    children: list[str] = field(default_factory=list)
    vec_row: int | None = None  # row in sections.faiss, if the section has a vector


@dataclass
class Document:
    doc_id: str
    source_path: str
    md_text: str                # recognized markdown, stored verbatim in the index
    sections: list[Section] = field(default_factory=list)


@dataclass
class Clause:
    """Unit of output: a whole clause with its number. Atomic — never split."""
    clause_id: int              # global ordinal == line number in clauses.jsonl
    doc_id: str
    number: str                 # most specific clause number ("12.1.4"); may be ""
    section_key: str            # key of the innermost enclosing section ("" if none)
    span: tuple[int, int]       # char span in Document.md_text; text == md_text[a:b]
    text: str


@dataclass
class Chunk:
    """Unit of indexing: a window inside a clause. Internal only."""
    unit_id: int                # global ordinal == line number in chunks.jsonl == FAISS id
    clause_id: int
    doc_id: str
    section_key: str
    span: tuple[int, int]       # char span RELATIVE to the clause text
    text: str


@dataclass
class SearchHit:
    """One search result: a whole clause + its number (output invariant)."""
    clause_number: str
    text: str                   # full clause text, exact substring of the source md
    score: float
    doc_id: str
    section_path: list[str]     # numeric ancestor chain, e.g. ["12", "12.1", "12.1.4"]
    clause_id: int
    method: str                 # bm25 | vector | hybrid | grep | agentic
    verdict: str | None = None  # agentic search: relevant | partial
    # --- provenance metadata ---
    doc_name: str = ""          # friendly document name (source filename; "" -> use doc_id)
    section_titles: list[str] = field(default_factory=list)
    # heading title of each section_path level, aligned 1:1 ("" where a level has
    # no heading of its own, e.g. the leaf clause). Titles already carry the number.

    @property
    def path(self) -> list[str]:
        """Readable hierarchical path, chapter -> paragraph: the section title at
        each level (titles include their own number) or the bare number as a
        fallback, e.g. ["Статья 12. Наблюдательный совет", "12.1. Компетенция…",
        "12.1.4"]."""
        out: list[str] = []
        for i, num in enumerate(self.section_path):
            title = self.section_titles[i] if i < len(self.section_titles) else ""
            out.append(title or num)
        return out

    @property
    def breadcrumb(self) -> str:
        """`path` joined for display: 'Статья 12. … › 12.1. … › 12.1.4'."""
        return " › ".join(self.path)

    @property
    def locator(self) -> str:
        """A stable, always-non-empty reference for the hit — for programmatic
        keying and citation. The clause number when it has one; otherwise the
        section path (an unnumbered clause under a titled section); and, failing
        even that, a document-relative id. So a numberless clause is never
        anonymous."""
        return self.clause_number or self.breadcrumb or f"{self.doc_name or self.doc_id}#{self.clause_id}"


@dataclass
class SectionRef:
    """A section reference returned by find_section()."""
    doc_id: str
    key: str
    title: str
    score: float = 0.0


@dataclass
class TocEntry:
    """One outline row for programmatic navigation. `title` is enriched:
    bare-number headings ('12.1.3 .') are replaced by the content preview."""
    doc_id: str
    key: str
    title: str
    preview: str     # first meaningful sentence of the section body ("" if none)
    level: int
    clause_numbers: list[str] = field(default_factory=list)
    # numbers of clauses belonging to this section BY NUMBERING (a clause is
    # attributed to the section with the longest ancestor key, so «12.2» shows
    # under [12] even if recognition placed it inside section 12.1's scope)


def effective_key(clause: Clause) -> str:
    """The key that positions a clause in the hierarchy."""
    return clause.number or clause.section_key


def section_path(key: str) -> list[str]:
    """Ancestor chain for a dotted key: '12.1.4' -> ['12', '12.1', '12.1.4']."""
    if not key:
        return []
    parts = key.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]
