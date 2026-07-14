"""OCR-error tolerance for clause numbers.

Recognition of scanned charters confuses digits with look-alike letters and
dots with commas, so a пункт «10.2» comes out as «1О.2», «l0.2» or «10,2» and the
strict segmenter would miss it. These helpers list the TYPICAL confusions and
canonicalize a number token to real digits, used by clause segmentation
(``parsing/clauses.py``) and heading-key extraction (``parsing/markdown.py``).

Safety: a token is repaired only if it carries at least one REAL digit as an
anchor — so a word like «Зона» or a Roman numeral «II» never becomes «30»/«11» —
and the result is rejected if it looks like a date/amount (a part with a leading
zero). ``ocr_number_repair=False`` on ``RagIndex.build`` turns the fallback off.
"""
from __future__ import annotations

import re

# Typical digit ← look-alike glyph confusions in Russian OCR (Latin + Cyrillic).
# Conservative on purpose: only high-frequency, low-ambiguity glyphs.
#   0 ← O o (Latin)  О о (Cyrillic)
#   1 ← l I (Latin)  |  (pipe)  і (Cyrillic i)
#   3 ← З з (Cyrillic Ze)
#   5 ← S s (Latin)
#   6 ← б   (Cyrillic be)
OCR_DIGIT_MAP = {
    "O": "0", "o": "0", "О": "0", "о": "0",
    "l": "1", "I": "1", "|": "1", "і": "1",
    "З": "3", "з": "3",
    "S": "5", "s": "5",
    "б": "6",
}

# Glyphs safe to admit into a DETECTION character class. Excludes '|' because it
# is markdown table syntax (still repaired by repair_number if it reaches a token).
_DETECT_GLYPHS = "OoОоlIіЗзSsб"
# Ready for use inside a regex character class: [<OCR_DIGIT>]
OCR_DIGIT = "0-9" + _DETECT_GLYPHS
# A numbering PART tolerant of look-alike glyphs but REQUIRING at least one real
# digit — so a word ('Общие', 'Зона') or a Roman numeral is never taken as a
# number part (which would let '1. Общие' become '1.06'). Use in a regex as-is.
OCR_PART = rf"[0-9{_DETECT_GLYPHS}]*[0-9][0-9{_DETECT_GLYPHS}]*"

_CANONICAL_RE = re.compile(r"\d+(?:\.\d+)*")


def repair_number(raw: str) -> str:
    """Canonicalize an OCR'd clause-number token to real digits.

    Maps look-alike glyphs to digits, comma→dot, drops spaces:
    ``'1О , 2'`` → ``'10.2'``, ``'7 . 2'`` → ``'7.2'``. Returns ``''`` when the
    token has no real digit to anchor on (``'Зона'``, ``'II'``), doesn't form a
    dotted number, or has a leading-zero part (a date/amount like ``'05.01'``).
    """
    if not raw or not any(ch.isdigit() for ch in raw):
        return ""                       # need a real digit as anchor
    out: list[str] = []
    for ch in raw:
        if ch.isspace():
            continue
        out.append("." if ch == "," else OCR_DIGIT_MAP.get(ch, ch))
    s = re.sub(r"\.{2,}", ".", "".join(out)).strip(".")
    if not s or not _CANONICAL_RE.fullmatch(s):
        return ""
    if any(len(p) > 1 and p[0] == "0" for p in s.split(".")):
        return ""                       # leading zero -> date / amount, not a clause
    return s
