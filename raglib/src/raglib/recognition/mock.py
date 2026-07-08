"""Mock recognizers.

``MockRecognizer`` mirrors the production contract (file in -> markdown out)
but only passes through files that are already text (.md/.txt) — exactly what
the target system emits. Anything else raises with a clear message.

``FixtureRecognizer`` maps arbitrary input files to prepared markdown — for
integration tests of the "pdf goes in, markdown comes out" flow.
"""
from __future__ import annotations

from pathlib import Path

from raglib.recognition.base import RecognitionError

_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}


def _read_text(path: Path) -> str:
    # bytes round-trip: no newline translation, offsets in the index stay exact
    return path.read_bytes().decode("utf-8", errors="replace")


class MockRecognizer:
    """Pass .md/.txt through unchanged; refuse binary formats."""

    def recognize(self, path: Path) -> str:
        path = Path(path)
        if path.suffix.lower() in _TEXT_SUFFIXES:
            return _read_text(path)
        raise RecognitionError(
            f"MockRecognizer cannot recognize {path.name!r}: in production this "
            "is done by the target system (which emits .md). Feed .md/.txt here, "
            "or provide a real TextRecognizer adapter / FixtureRecognizer."
        )


class FixtureRecognizer:
    """Map input files to prepared markdown (path -> md path or literal text)."""

    def __init__(self, mapping: dict):
        self._mapping = {str(Path(k)): v for k, v in mapping.items()}

    def recognize(self, path: Path) -> str:
        key = str(Path(path))
        if key not in self._mapping:
            raise RecognitionError(f"No fixture registered for {key!r}")
        value = self._mapping[key]
        vp = Path(value)
        if vp.suffix.lower() in _TEXT_SUFFIXES and vp.is_file():
            return _read_text(vp)
        return str(value)
