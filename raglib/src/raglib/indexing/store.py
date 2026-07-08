"""On-disk index format: one folder, no services.

  manifest.json   format marker, version, documents, chunking params, embeddings meta
  docs/<id>.md    recognized markdown, byte-exact (clause spans point into it)
  sections.jsonl  section-tree nodes; vec_row -> row in sections.faiss
  clauses.jsonl   clauses (units of OUTPUT); line number == clause_id
  chunks.jsonl    index units (windows); line number == FAISS row in chunks.faiss
  chunks.faiss    unit vectors (absent in BM25-only mode)
  sections.faiss  section vectors (absent in BM25-only mode)
  embed_cache/    sha1 content cache of raw embedding vectors
  toc.json        outline per document (informational)

Clause/chunk texts are NOT duplicated in the jsonl files: they are exact spans
into docs/<id>.md, which enforces the output invariant by construction.
The manifest is written LAST and doubles as the "index is complete" marker.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from raglib.models import Chunk, Clause, Document, Section

FORMAT_NAME = "raglib-index"
FORMAT_VERSION = 1
MANIFEST = "manifest.json"


class NotARaglibIndexError(ValueError):
    """The directory is not a raglib index; refusing to touch it."""


@dataclass
class IndexData:
    root: Path | None
    manifest: dict
    documents: dict[str, Document]
    sections: list[Section]
    clauses: list[Clause]
    chunks: list[Chunk]
    chunk_index: object | None = None      # FaissFlat
    section_index: object | None = None    # FaissFlat
    section_rows: list[Section] = field(default_factory=list)  # faiss row -> Section

    @property
    def has_vectors(self) -> bool:
        return self.chunk_index is not None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_index_dir(root: Path) -> bool:
    mf = Path(root) / MANIFEST
    if not mf.is_file():
        return False
    try:
        return json.loads(mf.read_text(encoding="utf-8")).get("format") == FORMAT_NAME
    except (ValueError, OSError):
        return False


def save_index(root: Path, *, manifest: dict, documents: dict[str, Document],
               sections: list[Section], clauses: list[Clause], chunks: list[Chunk],
               chunk_index=None, section_index=None) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    docs_dir = root / "docs"
    docs_dir.mkdir(exist_ok=True)
    for d in documents.values():
        (docs_dir / f"{d.doc_id}.md").write_bytes(d.md_text.encode("utf-8"))

    _write_jsonl(root / "sections.jsonl", [
        {"doc_id": s.doc_id, "key": s.key, "title": s.title, "level": s.level,
         "line_start": s.line_start, "line_end": s.line_end,
         "parent_key": s.parent_key, "children": s.children, "vec_row": s.vec_row}
        for s in sections
    ])
    _write_jsonl(root / "clauses.jsonl", [
        {"clause_id": c.clause_id, "doc_id": c.doc_id, "number": c.number,
         "section_key": c.section_key, "start": c.span[0], "end": c.span[1]}
        for c in clauses
    ])
    _write_jsonl(root / "chunks.jsonl", [
        {"unit_id": ch.unit_id, "clause_id": ch.clause_id,
         "start": ch.span[0], "end": ch.span[1]}
        for ch in chunks
    ])

    toc = {}
    for s in sections:
        toc.setdefault(s.doc_id, []).append(
            {"key": s.key, "title": s.title, "level": s.level})
    (root / "toc.json").write_text(
        json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")

    if chunk_index is not None:
        chunk_index.save(root / "chunks.faiss")
    if section_index is not None:
        section_index.save(root / "sections.faiss")

    # manifest last: its presence marks a complete, valid index
    (root / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(root: Path) -> IndexData:
    root = Path(root)
    mf_path = root / MANIFEST
    if not mf_path.is_file():
        raise NotARaglibIndexError(f"No {MANIFEST} in {root} — not a raglib index.")
    manifest = json.loads(mf_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_NAME:
        raise NotARaglibIndexError(f"{root} is not a raglib index (format mismatch).")
    if int(manifest.get("version", -1)) > FORMAT_VERSION:
        raise NotARaglibIndexError(
            f"Index version {manifest.get('version')} is newer than supported "
            f"({FORMAT_VERSION}); upgrade raglib.")

    documents: dict[str, Document] = {}
    for meta in manifest.get("documents", []):
        doc_id = meta["doc_id"]
        md_text = (root / "docs" / f"{doc_id}.md").read_bytes().decode("utf-8")
        documents[doc_id] = Document(doc_id=doc_id,
                                     source_path=meta.get("source_path", ""),
                                     md_text=md_text)

    sections: list[Section] = []
    for row in _read_jsonl(root / "sections.jsonl"):
        s = Section(doc_id=row["doc_id"], key=row["key"], title=row["title"],
                    level=row["level"], line_start=row["line_start"],
                    line_end=row["line_end"], parent_key=row.get("parent_key"),
                    children=row.get("children", []), vec_row=row.get("vec_row"))
        sections.append(s)
        if s.doc_id in documents:
            documents[s.doc_id].sections.append(s)

    clauses: list[Clause] = []
    for row in _read_jsonl(root / "clauses.jsonl"):
        doc = documents[row["doc_id"]]
        a, b = int(row["start"]), int(row["end"])
        clauses.append(Clause(clause_id=int(row["clause_id"]), doc_id=doc.doc_id,
                              number=row["number"], section_key=row["section_key"],
                              span=(a, b), text=doc.md_text[a:b]))

    chunks: list[Chunk] = []
    for row in _read_jsonl(root / "chunks.jsonl"):
        cl = clauses[int(row["clause_id"])]
        a, b = int(row["start"]), int(row["end"])
        chunks.append(Chunk(unit_id=int(row["unit_id"]), clause_id=cl.clause_id,
                            doc_id=cl.doc_id, section_key=cl.section_key,
                            span=(a, b), text=cl.text[a:b]))

    chunk_index = section_index = None
    if (root / "chunks.faiss").exists():
        from raglib.indexing.vector import FaissFlat
        chunk_index = FaissFlat.load(root / "chunks.faiss")
        if (root / "sections.faiss").exists():
            section_index = FaissFlat.load(root / "sections.faiss")

    section_rows: list[Section] = sorted(
        (s for s in sections if s.vec_row is not None),
        key=lambda s: s.vec_row)  # type: ignore[arg-type,return-value]

    return IndexData(root=root, manifest=manifest, documents=documents,
                     sections=sections, clauses=clauses, chunks=chunks,
                     chunk_index=chunk_index, section_index=section_index,
                     section_rows=section_rows)


def delete_index(root) -> bool:
    """Delete an index folder. Validates it IS a raglib index before rmtree —
    never deletes an arbitrary directory. Returns False if already gone
    (idempotent), True if deleted; raises NotARaglibIndexError otherwise."""
    root = Path(root)
    if not root.exists():
        return False
    if not root.is_dir():
        raise NotARaglibIndexError(f"{root} is not a directory.")
    if not is_index_dir(root):
        raise NotARaglibIndexError(
            f"{root} does not look like a raglib index (no valid {MANIFEST}); "
            "refusing to delete.")
    shutil.rmtree(root)
    return True
