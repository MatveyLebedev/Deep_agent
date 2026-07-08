from pathlib import Path

import pytest

from raglib import RagIndex
from raglib.embeddings import HashingEmbeddings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_files() -> list[Path]:
    return [FIXTURES / "charter.md", FIXTURES / "policy.md", FIXTURES / "notes.md"]


@pytest.fixture(scope="session")
def charter_text() -> str:
    return (FIXTURES / "charter.md").read_bytes().decode("utf-8")


@pytest.fixture(scope="session")
def bm25_index(tmp_path_factory, fixture_files) -> RagIndex:
    # exact-form BM25 (normalizer="none"): the baseline that the RU
    # normalization tests compare against and override
    root = tmp_path_factory.mktemp("idx_bm25") / "index"
    return RagIndex.build(fixture_files, root, embeddings=None,
                          bm25_normalizer="none")


@pytest.fixture(scope="session")
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings(dim=256)


@pytest.fixture(scope="session")
def vec_index(tmp_path_factory, fixture_files, embeddings) -> RagIndex:
    root = tmp_path_factory.mktemp("idx_vec") / "index"
    return RagIndex.build(fixture_files, root, embeddings=embeddings)
