import pytest

from raglib import RagIndex
from raglib.recognition import (
    FixtureRecognizer,
    MockRecognizer,
    RecognitionError,
    expand_inputs,
)
from tests.conftest import FIXTURES


def test_mock_recognizer_passes_markdown_through():
    md = MockRecognizer().recognize(FIXTURES / "charter.md")
    assert md == (FIXTURES / "charter.md").read_bytes().decode("utf-8")


def test_mock_recognizer_refuses_binary(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(RecognitionError, match="target system"):
        MockRecognizer().recognize(pdf)


def test_fixture_recognizer_contract(tmp_path):
    """Any recognizer that emits markdown is indistinguishable from native .md:
    a fake 'pdf' mapped to prepared markdown indexes and searches normally."""
    pdf = tmp_path / "charter_scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    rec = FixtureRecognizer({pdf: FIXTURES / "charter.md"})

    index = RagIndex.build([pdf], tmp_path / "idx", recognizer=rec, embeddings=None)
    assert index.documents == ["charter_scan"]
    hits = index.search("нотариального удостоверения протокола", method="bm25")
    assert hits and hits[0].clause_number == "14.1"


def test_expand_inputs(tmp_path):
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("h", encoding="utf-8")
    files = expand_inputs(tmp_path)
    assert [f.name for f in files] == ["a.md", "b.md"]  # sorted, hidden skipped
    # file + dir + dedupe
    files = expand_inputs([tmp_path / "a.md", tmp_path])
    assert [f.name for f in files] == ["a.md", "b.md"]
    with pytest.raises(FileNotFoundError):
        expand_inputs(tmp_path / "missing.md")
