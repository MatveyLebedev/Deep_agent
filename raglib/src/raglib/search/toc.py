"""Table-of-contents tools: enriched outline, find a section (by key, title,
content preview or meaning), read a whole section including sub-sections.

Recognition output often carries junk headings ('12.1.3 .', bare numbers) and
noise lines (page markers, image placeholders, tables). The outline therefore
enriches every section with a PREVIEW — the first meaningful sentence of the
section body. Poor titles are replaced by the preview, and find_section()
matches the enriched title AND the preview, so navigation stays usable on
real scanned documents.
"""
from __future__ import annotations

import re

from raglib.indexing.store import IndexData
from raglib.models import Document, Section, SectionRef, TocEntry
from raglib.parsing.markdown import (
    HEADING_RE,
    find_section_in_doc,
    is_descendant,
    section_key_of,
    section_text,
)

_NOISE_RE = re.compile(
    r"^\s*(?:---\s*page\s+\d+\s*/\s*\d+\s*---"   # docling page markers
    r"|<!--.*?-->"                               # <!-- image --> placeholders
    r"|-{3,}"                                    # horizontal rules
    r"|\|.*"                                     # table rows
    r"|total\s+pages:.*)\s*$",
    re.IGNORECASE)
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*]\s+")
_LETTERS_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _shorten(text: str, max_chars: int) -> str:
    text = _clean(text)
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[:cut if cut > max_chars // 2 else max_chars] + "…"


def section_preview(document: Document, section: Section,
                    max_chars: int = 80) -> str:
    """First meaningful content line of the section's own body. Page markers,
    image placeholders, tables and nested headings are skipped; list bullets
    are stripped. Returns "" when the section has no own text."""
    lines = document.md_text.split("\n")
    for ln in lines[section.line_start + 1:section.line_end]:
        if not ln.strip() or _NOISE_RE.match(ln) or HEADING_RE.match(ln):
            continue
        return _shorten(_BULLET_PREFIX_RE.sub("", ln), max_chars)
    return ""


def _is_poor_title(title: str) -> bool:
    """True for titles carrying no words beyond the number ('12.1.3 .', '7.')."""
    return sum(len(m) for m in _LETTERS_RE.findall(title)) < 3


def _enriched(document: Document, section: Section,
              preview_chars: int) -> tuple[str, str]:
    """(title, preview): bare-number titles are promoted to the preview."""
    preview = section_preview(document, section, preview_chars)
    title = _clean(section.title)
    if _is_poor_title(title) and preview:
        title = preview
    return title, preview


def _numeric_sort_key(number: str):
    return [int(p) if p.isdigit() else 0 for p in number.split(".")]


def _clauses_by_section(data: IndexData) -> dict[tuple[str, str], list[str]]:
    """(doc_id, section_key) -> sorted clause numbers. A clause is attributed
    to the section with the LONGEST key that is its numbering ancestor."""
    keys_by_doc: dict[str, list[str]] = {}
    for doc_id, document in data.documents.items():
        keys_by_doc[doc_id] = [s.key for s in document.sections if s.key]
    grouped: dict[tuple[str, str], list[str]] = {}
    for c in data.clauses:
        if not c.number:
            continue
        ancestors = [k for k in keys_by_doc.get(c.doc_id, ())
                     if is_descendant(c.number, k)]
        home = max(ancestors, key=len) if ancestors else c.section_key
        grouped.setdefault((c.doc_id, home), []).append(c.number)
    for numbers in grouped.values():
        numbers.sort(key=_numeric_sort_key)
    return grouped


def toc_entries(data: IndexData, doc: str | None = None,
                preview_chars: int = 80) -> list[TocEntry]:
    """Structured outline for programmatic navigation."""
    if doc is not None and doc not in data.documents:
        raise KeyError(f"Document not found: {doc!r} "
                       f"(have: {sorted(data.documents)})")
    grouped = _clauses_by_section(data)
    entries: list[TocEntry] = []
    for doc_id, document in data.documents.items():
        if doc is not None and doc_id != doc:
            continue
        for s in document.sections:
            title, preview = _enriched(document, s, preview_chars)
            entries.append(TocEntry(
                doc_id=doc_id, key=s.key, title=title, preview=preview,
                level=s.level,
                clause_numbers=grouped.get((doc_id, s.key), []) if s.key else []))
    return entries


def toc_outline(data: IndexData, doc: str | None = None, *,
                preview: bool = False, preview_chars: int = 80,
                clauses: bool = False, max_clause_numbers: int = 14) -> str:
    """Human-readable outline; each line shows the key usable in read_section.
    preview=True adds the first content sentence of every section;
    clauses=True adds the clause numbers belonging to each section — makes
    segmentation completeness visible at a glance."""
    entries = toc_entries(data, doc=doc, preview_chars=preview_chars)
    out: list[str] = []
    for doc_id in ([doc] if doc is not None else list(data.documents)):
        out.append(f"=== {doc_id} ===")
        doc_entries = [e for e in entries if e.doc_id == doc_id]
        if not doc_entries:
            out.append("  (no headings)")
            continue
        for e in doc_entries:
            indent = "  " * max(0, e.level - 1)
            tag = f"[{e.key}] " if e.key else ""
            line = f"{indent}{tag}{e.title}"
            if preview and e.preview and e.preview != e.title:
                line += f" — {e.preview}"
            out.append(line)
            if clauses and e.clause_numbers:
                nums = e.clause_numbers
                shown = ", ".join(nums[:max_clause_numbers])
                if len(nums) > max_clause_numbers:
                    shown += f", … (всего {len(nums)})"
                out.append(f"{indent}      пункты: {shown}")
    return "\n".join(out)


def find_section(data: IndexData, query: str, *, semantic: bool = False,
                 engine=None, top_k: int = 3,
                 preview_chars: int = 300) -> list[SectionRef]:
    """By key/'Статья N'/title substring/preview substring; with semantic=True —
    by meaning via the sections FAISS level (needs a vector index + embeddings).
    preview_chars: how much of the section's first content line participates in
    substring matching (wider than the display preview in toc())."""
    if semantic:
        if engine is None:
            raise ValueError("semantic find_section needs the search engine")
        engine._require_vectors("find_section(semantic=True)")
        if data.section_index is None:
            return []
        qv = engine._embed_query(query)
        rows = data.section_index.search(qv, top_k)
        refs = []
        for r, score in rows:
            s = data.section_rows[r]
            title, _ = _enriched(data.documents[s.doc_id], s, 80)
            refs.append(SectionRef(doc_id=s.doc_id, key=s.key,
                                   title=title, score=score))
        return refs

    q = query.strip()
    q_key = section_key_of(q) or q
    ql = q.lower()
    scored: dict[tuple[str, str, str], SectionRef] = {}

    def add(e: TocEntry, score: float) -> None:
        ident = (e.doc_id, e.key, e.title)
        if ident not in scored or scored[ident].score < score:
            scored[ident] = SectionRef(e.doc_id, e.key, e.title, score)

    for e in toc_entries(data, preview_chars=preview_chars):
        if e.key and e.key == q_key:
            add(e, 1.0)                       # exact key / «Статья N»
        elif ql and ql in e.title.lower():
            add(e, 0.5)                       # enriched-title substring
        elif ql and e.preview and ql in e.preview.lower():
            add(e, 0.3)                       # content-preview substring
    return sorted(scored.values(), key=lambda r: -r.score)[:top_k]


def read_section(data: IndexData, doc_id: str, section: str) -> str:
    """Whole section text, including nested sub-sections. Never truncated."""
    document = data.documents.get(doc_id)
    if document is None:
        raise KeyError(f"Document not found: {doc_id!r} "
                       f"(have: {sorted(data.documents)})")
    sec = find_section_in_doc(document, section)
    if sec is None:
        keys = ", ".join(s.key for s in document.sections if s.key)
        raise KeyError(f"Section {section!r} not found in {doc_id!r}. "
                       f"Available keys: {keys or '(none)'}")
    return section_text(document, sec)
