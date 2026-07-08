import json

import pytest

from raglib import NotARaglibIndexError, RagIndex
from raglib.embeddings import HashingEmbeddings
from tests.conftest import FIXTURES


def test_roundtrip_counts_and_files(vec_index):
    root = vec_index.root
    for name in ("manifest.json", "clauses.jsonl", "chunks.jsonl",
                 "sections.jsonl", "toc.json", "chunks.faiss", "sections.faiss"):
        assert (root / name).exists(), name
    counts = vec_index.manifest["counts"]
    data = vec_index._data
    assert counts["clauses"] == len(data.clauses)
    assert counts["chunks"] == len(data.chunks)
    assert counts["chunks"] > counts["clauses"]  # the long clause split into units
    # loaded clause texts are exact slices of the stored markdown
    for c in data.clauses:
        assert c.text == data.documents[c.doc_id].md_text[c.span[0]:c.span[1]]


def test_bm25_only_index_has_no_faiss(bm25_index):
    assert not (bm25_index.root / "chunks.faiss").exists()
    assert bm25_index.manifest["embeddings"]["model"] is None
    with pytest.raises(RuntimeError, match="BM25-only"):
        bm25_index.search("тест", method="vector")
    with pytest.raises(RuntimeError, match="BM25-only"):
        bm25_index.search("тест", method="hybrid")


def test_load_requires_matching_embeddings(vec_index):
    # load without an embeddings client: bm25 works, vector explains what to do
    plain = RagIndex.load(vec_index.root)
    assert plain.search("нотариального удостоверения", method="bm25")
    with pytest.raises(RuntimeError, match="embeddings client"):
        plain.search("тест", method="vector")
    # wrong-dimension client is caught on the first query
    wrong = RagIndex.load(vec_index.root, embeddings=HashingEmbeddings(dim=64))
    with pytest.raises(RuntimeError, match="dim"):
        wrong.search("тест", method="vector")


def test_rebuild_over_existing_index(tmp_path):
    root = tmp_path / "idx"
    RagIndex.build([FIXTURES / "notes.md"], root, embeddings=None)
    index = RagIndex.build([FIXTURES / "policy.md"], root, embeddings=None)
    assert index.documents == ["policy"]  # fully rebuilt, not merged


def test_build_refuses_foreign_directory(tmp_path):
    target = tmp_path / "precious"
    target.mkdir()
    (target / "data.txt").write_text("not an index", encoding="utf-8")
    with pytest.raises(NotARaglibIndexError):
        RagIndex.build([FIXTURES / "notes.md"], target, embeddings=None)
    assert (target / "data.txt").exists()  # untouched


def test_delete_index_validates_and_is_idempotent(tmp_path):
    root = tmp_path / "idx"
    RagIndex.build([FIXTURES / "notes.md"], root, embeddings=None)
    assert RagIndex.delete_index(root) is True
    assert not root.exists()
    assert RagIndex.delete_index(root) is False  # already gone: no error

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "file.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(NotARaglibIndexError, match="refusing"):
        RagIndex.delete_index(foreign)
    assert (foreign / "file.txt").exists()


def test_instance_delete_closes_object(tmp_path):
    index = RagIndex.build([FIXTURES / "notes.md"], tmp_path / "idx", embeddings=None)
    assert index.delete() is True
    with pytest.raises(RuntimeError, match="deleted"):
        index.search("тест", method="bm25")


def test_load_rejects_non_index_and_newer_versions(tmp_path):
    with pytest.raises(NotARaglibIndexError):
        RagIndex.load(tmp_path)  # no manifest

    root = tmp_path / "idx"
    RagIndex.build([FIXTURES / "notes.md"], root, embeddings=None)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = 99
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(NotARaglibIndexError, match="newer"):
        RagIndex.load(root)
