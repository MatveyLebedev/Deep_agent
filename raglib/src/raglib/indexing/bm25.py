"""BM25 lexical index over index units. Rebuilt from chunks.jsonl on load
(fast at target volumes) — no pickle on disk.

Russian morphology support: the tokenizer has three concrete modes, recorded
in the index manifest so queries are normalized exactly like the corpus was.

  none   \\w+ lowercase (exact word forms)          — no extra deps
  stem   Snowball russian stemmer                    (extra: raglib[stem])
  lemma  pymorphy3 normal forms, cached              (extra: raglib[ru])

The build-time default is ``auto``: pick the best AVAILABLE backend
(stem > lemma > none) and store the concrete choice in the manifest. So an
index built in an environment without snowballstemmer transparently uses
lemma (or none), and search-time tokenization always matches the corpus.
"""
from __future__ import annotations

import importlib.util
import re
from functools import lru_cache
from typing import Callable, List, Sequence

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

NORMALIZERS = ("none", "stem", "lemma")
BUILD_NORMALIZERS = NORMALIZERS + ("auto",)

Tokenizer = Callable[[str], List[str]]


def _base_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _backend_available(normalizer: str) -> bool:
    """Whether the backend a normalizer needs is importable (no import cost)."""
    if normalizer == "stem":
        return importlib.util.find_spec("snowballstemmer") is not None
    if normalizer == "lemma":
        return importlib.util.find_spec("pymorphy3") is not None
    return True  # "none" needs nothing


def resolve_normalizer(normalizer: str) -> str:
    """Resolve ``auto`` to the best available concrete normalizer
    (stem > lemma > none). Concrete names pass through unchanged."""
    if normalizer != "auto":
        return normalizer
    for candidate in ("stem", "lemma", "none"):
        if _backend_available(candidate):
            return candidate
    return "none"


def make_tokenizer(normalizer: str = "none") -> Tokenizer:
    """Build a tokenizer for the given normalization mode."""
    if normalizer == "none":
        return _base_tokens

    if normalizer == "stem":
        try:
            import snowballstemmer
        except ImportError as e:
            raise RuntimeError(
                "bm25_normalizer='stem' requires snowballstemmer: "
                "pip install raglib[stem]  (or build with bm25_normalizer='auto' "
                "to fall back to lemma/none automatically)") from e
        stemmer = snowballstemmer.stemmer("russian")

        def tokenize_stem(text: str) -> List[str]:
            return stemmer.stemWords(_base_tokens(text))

        return tokenize_stem

    if normalizer == "lemma":
        try:
            import pymorphy3
        except ImportError as e:
            raise RuntimeError(
                "bm25_normalizer='lemma' requires pymorphy3: "
                "pip install raglib[ru]") from e
        morph = pymorphy3.MorphAnalyzer()

        @lru_cache(maxsize=200_000)
        def normal_form(word: str) -> str:
            return morph.parse(word)[0].normal_form

        def tokenize_lemma(text: str) -> List[str]:
            return [normal_form(w) for w in _base_tokens(text)]

        return tokenize_lemma

    raise ValueError(f"bm25 normalizer must be one of {NORMALIZERS}, "
                     f"got {normalizer!r}")


class BM25Index:
    def __init__(self, texts: Sequence[str], tokenizer: Tokenizer | None = None):
        from rank_bm25 import BM25Okapi

        self._tokenize = tokenizer or _base_tokens
        self._n = len(texts)
        self._bm25 = (BM25Okapi([self._tokenize(t) for t in texts])
                      if self._n else None)

    def search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        """Ranked (unit_id, score) with score > 0."""
        if self._bm25 is None:
            return []
        q = self._tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(self._n), key=lambda i: scores[i], reverse=True)
        out: list[tuple[int, float]] = []
        for i in order[:top_n]:
            if scores[i] <= 0:
                break
            out.append((i, float(scores[i])))
        return out
