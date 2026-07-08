"""Text-recognition seam.

In production, recognition is performed by the target corporate system, which
emits markdown. The library only defines the contract: *file in -> markdown
out*. Plug the real system in by implementing ``TextRecognizer``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


class RecognitionError(RuntimeError):
    """Raised when a recognizer cannot handle the given file."""


@runtime_checkable
class TextRecognizer(Protocol):
    def recognize(self, path: Path) -> str:
        """Return the recognized markdown text for one input file."""
        ...


def expand_inputs(inputs) -> list[Path]:
    """Expand file / directory / list-of-paths into an ordered list of files."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(f for f in sorted(p.iterdir())
                         if f.is_file() and not f.name.startswith("."))
        else:
            raise FileNotFoundError(f"Input path not found: {p}")
    # de-duplicate, keep order
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            out.append(f)
    return out
