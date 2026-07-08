import json

import pytest

from raglib import RagIndex
from raglib.indexing import bm25 as bm25mod
from raglib.indexing.bm25 import make_tokenizer, resolve_normalizer
from tests.conftest import FIXTURES


def test_invalid_normalizer_rejected(tmp_path):
    with pytest.raises(ValueError, match="normalizer"):
        make_tokenizer("porter")
    with pytest.raises(ValueError, match="bm25_normalizer"):
        RagIndex.build([FIXTURES / "notes.md"], tmp_path / "idx",
                       embeddings=None, bm25_normalizer="porter")


def test_resolve_normalizer_prefers_available(monkeypatch):
    # concrete names pass through untouched
    for name in ("none", "stem", "lemma"):
        assert resolve_normalizer(name) == name

    avail = {"stem": True, "lemma": True}
    monkeypatch.setattr(bm25mod, "_backend_available", lambda n: avail.get(n, True))
    assert resolve_normalizer("auto") == "stem"          # best available
    avail["stem"] = False
    assert resolve_normalizer("auto") == "lemma"         # contour: no snowball
    avail["lemma"] = False
    assert resolve_normalizer("auto") == "none"          # bare core


def test_auto_records_concrete_choice_in_manifest(tmp_path, monkeypatch):
    # simulate a contour without snowballstemmer: auto must resolve to lemma
    pytest.importorskip("pymorphy3")
    monkeypatch.setattr(bm25mod, "_backend_available",
                        lambda n: False if n == "stem" else True)
    index = RagIndex.build([FIXTURES / "charter.md"], tmp_path / "idx",
                           embeddings=None, bm25_normalizer="auto")
    manifest = json.loads((index.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bm25"]["normalizer"] == "lemma"     # concrete, not "auto"
    # end-to-end: a form-mismatch query only works if lemma is really applied
    # ("нотариальным удостоверением протоколов" vs "нотариального удостоверения
    # протокола" in clause 14.1)
    assert index.search(FORM_MISMATCH_QUERY,
                        method="bm25")[0].clause_number == "14.1"


def test_stem_collapses_word_forms():
    pytest.importorskip("snowballstemmer")
    tok = make_tokenizer("stem")
    assert tok("заинтересованностью") == tok("заинтересованность")
    assert tok("крупной сделки") == tok("крупная сделка")
    assert tok("протоколом") == tok("протокол")


def test_lemma_collapses_word_forms():
    pytest.importorskip("pymorphy3")
    tok = make_tokenizer("lemma")
    assert tok("сделки сделок сделкам") == ["сделка"] * 3
    assert tok("наблюдательным советом") == ["наблюдательный", "совет"]
    assert tok("удостоверения") == tok("удостоверение")


# query and document use DIFFERENT word forms; the discriminative stems
# (нотариальн-, удостоверен-) occur only in clause 14.1 of the fixture
FORM_MISMATCH_QUERY = "нотариальным удостоверением протоколов"


@pytest.mark.parametrize("normalizer", ["stem", "lemma"])
def test_normalizer_bridges_form_mismatch(tmp_path, normalizer):
    """Exact-form BM25 finds nothing for a paraphrased query; normalized BM25
    ranks the right clause first."""
    pytest.importorskip("snowballstemmer" if normalizer == "stem" else "pymorphy3")
    index = RagIndex.build([FIXTURES / "charter.md"], tmp_path / "idx",
                           embeddings=None, bm25_normalizer=normalizer)
    hits = index.search(FORM_MISMATCH_QUERY, method="bm25", top_k=3)
    assert hits and hits[0].clause_number == "14.1"

    # manifest roundtrip: plain load keeps the build-time normalizer
    reloaded = RagIndex.load(tmp_path / "idx")
    assert reloaded._engine.bm25_normalizer == normalizer
    assert reloaded.search(FORM_MISMATCH_QUERY,
                           method="bm25", top_k=3)[0].clause_number == "14.1"


def test_load_override_changes_normalizer(bm25_index):
    pytest.importorskip("pymorphy3")
    # exact word forms: the paraphrased query matches nothing at all
    assert bm25_index._engine.bm25_normalizer == "none"
    assert bm25_index.search(FORM_MISMATCH_QUERY, method="bm25", top_k=3) == []

    override = RagIndex.load(bm25_index.root, bm25_normalizer="lemma")
    assert override._engine.bm25_normalizer == "lemma"
    hits = override.search(FORM_MISMATCH_QUERY, method="bm25", top_k=3)
    assert hits and hits[0].clause_number == "14.1"
