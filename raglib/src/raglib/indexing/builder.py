"""Index builder: files -> recognizer -> section tree -> clauses -> units ->
BM25 (implicit, rebuilt on load) + FAISS (chunks + sections) -> on-disk folder.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

from raglib.embeddings.base import model_name_of
from raglib.indexing import store
from raglib.models import Chunk, Clause, Document, Section
from raglib.parsing.clauses import segment_clauses, window_spans
from raglib.parsing.markdown import build_sections
from raglib.recognition.base import TextRecognizer, expand_inputs
from raglib.recognition.mock import MockRecognizer


def _doc_id_for(path: Path, taken: set[str]) -> str:
    base = re.sub(r"[^\w\-.]+", "_", path.stem, flags=re.UNICODE).strip("_") or "doc"
    doc_id, n = base, 1
    while doc_id in taken:
        n += 1
        doc_id = f"{base}-{n}"
    taken.add(doc_id)
    return doc_id


def _section_char_span(doc: Document, section: Section) -> tuple[int, int]:
    """Char span of a section's line range in the document markdown."""
    offsets = [0]
    for i, ch in enumerate(doc.md_text):
        if ch == "\n":
            offsets.append(i + 1)
    n = len(doc.md_text)
    a = offsets[section.line_start] if section.line_start < len(offsets) else n
    b = offsets[section.line_end] if section.line_end < len(offsets) else n
    return a, b


def build_index(
    inputs,
    index_dir,
    *,
    recognizer: TextRecognizer | None = None,
    embeddings=None,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    section_embedding: str = "mean",       # "mean" | "title_lead"
    section_lead_chars: int = 400,
    bm25_normalizer: str = "auto",         # "auto" | "none" | "stem" | "lemma"
) -> Path:
    recognizer = recognizer or MockRecognizer()
    files = expand_inputs(inputs)
    if not files:
        raise ValueError("No input files found.")
    if section_embedding not in ("mean", "title_lead"):
        raise ValueError("section_embedding must be 'mean' or 'title_lead'")
    from raglib.indexing.bm25 import BUILD_NORMALIZERS, resolve_normalizer
    if bm25_normalizer not in BUILD_NORMALIZERS:
        raise ValueError(f"bm25_normalizer must be one of {BUILD_NORMALIZERS}, "
                         f"got {bm25_normalizer!r}")
    # resolve "auto" to the best available backend and record the concrete
    # choice, so search-time tokenization always matches the corpus
    bm25_normalizer = resolve_normalizer(bm25_normalizer)

    root = Path(index_dir)
    if root.exists() and any(root.iterdir()):
        if store.is_index_dir(root):
            store.delete_index(root)  # rebuild over a previous raglib index
        else:
            raise store.NotARaglibIndexError(
                f"{root} exists, is not empty and is not a raglib index; "
                "refusing to overwrite.")

    documents: dict[str, Document] = {}
    sections: list[Section] = []
    clauses: list[Clause] = []
    chunks: list[Chunk] = []
    taken: set[str] = set()

    for f in files:
        md_text = recognizer.recognize(f)
        doc_id = _doc_id_for(f, taken)
        doc = Document(doc_id=doc_id, source_path=str(f), md_text=md_text,
                       sections=build_sections(doc_id, md_text))
        documents[doc_id] = doc
        sections.extend(doc.sections)
        for cl in segment_clauses(doc):
            cl.clause_id = len(clauses)
            clauses.append(cl)
            for (a, b) in window_spans(cl.text, chunk_size, chunk_overlap):
                chunks.append(Chunk(unit_id=len(chunks), clause_id=cl.clause_id,
                                    doc_id=doc_id, section_key=cl.section_key,
                                    span=(a, b), text=cl.text[a:b]))

    chunk_index = section_index = None
    emb_meta: dict = {"model": None, "dim": None, "index_factory": "Flat",
                      "section_embedding": section_embedding}
    if embeddings is not None and chunks:
        import numpy as np

        from raglib.indexing.vector import FaissFlat, embed_texts_cached, l2_normalize

        model_name = model_name_of(embeddings)
        cache_dir = root / "embed_cache"
        unit_vecs = l2_normalize(embed_texts_cached(
            embeddings, [ch.text for ch in chunks], cache_dir, model_name))
        chunk_index = FaissFlat.from_vectors(unit_vecs)

        # section vectors, bottom-up. Containment is by char spans (exact for
        # numbered and unnumbered sections alike).
        sec_vecs: list = []
        if section_embedding == "title_lead":
            texts, holders = [], []
            for s in sections:
                doc = documents[s.doc_id]
                a, b = _section_char_span(doc, s)
                lead = doc.md_text[a:b][:section_lead_chars]
                if lead.strip():
                    holders.append(s)
                    texts.append(f"{s.title}\n{lead}")
            if texts:
                tl_vecs = l2_normalize(embed_texts_cached(
                    embeddings, texts, cache_dir, model_name))
                for s, v in zip(holders, tl_vecs):
                    s.vec_row = len(sec_vecs)
                    sec_vecs.append(v)
        else:  # mean-pooling of contained unit vectors: zero extra embed calls
            for s in sections:
                doc = documents[s.doc_id]
                a, b = _section_char_span(doc, s)
                rows = [ch.unit_id for ch in chunks
                        if ch.doc_id == s.doc_id
                        and clauses[ch.clause_id].span[0] >= a
                        and clauses[ch.clause_id].span[1] <= b]
                if not rows:
                    continue
                v = l2_normalize(unit_vecs[rows].mean(axis=0))[0]
                s.vec_row = len(sec_vecs)
                sec_vecs.append(v)
        if sec_vecs:
            section_index = FaissFlat.from_vectors(np.vstack(sec_vecs))

        emb_meta.update(model=model_name, dim=int(unit_vecs.shape[1]))

    manifest = {
        "format": store.FORMAT_NAME,
        "version": store.FORMAT_VERSION,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "documents": [{"doc_id": d.doc_id, "source_path": d.source_path}
                      for d in documents.values()],
        "chunking": {"size": chunk_size, "overlap": chunk_overlap},
        "bm25": {"normalizer": bm25_normalizer},
        "embeddings": emb_meta,
        "counts": {"documents": len(documents), "sections": len(sections),
                   "clauses": len(clauses), "chunks": len(chunks)},
    }
    store.save_index(root, manifest=manifest, documents=documents,
                     sections=sections, clauses=clauses, chunks=chunks,
                     chunk_index=chunk_index, section_index=section_index)
    return root
