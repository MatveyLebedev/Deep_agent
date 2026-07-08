"""raglib — RAG library: files -> (mocked) text recognition -> hierarchical
on-disk index -> multi-tool search over whole numbered clauses."""
from raglib.api import RagIndex
from raglib.indexing.store import NotARaglibIndexError
from raglib.models import (
    Chunk,
    Clause,
    Document,
    SearchHit,
    Section,
    SectionRef,
    TocEntry,
)

__version__ = "0.1.0"

__all__ = [
    "RagIndex",
    "SearchHit",
    "SectionRef",
    "TocEntry",
    "Document",
    "Section",
    "Clause",
    "Chunk",
    "NotARaglibIndexError",
]
