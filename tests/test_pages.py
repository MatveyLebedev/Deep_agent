"""Page-spec parsing for read_pdf (tools.py)."""
import pytest

from tools import _parse_pages


@pytest.mark.parametrize("spec,total,expected", [
    ("1,3,5-8", 10, [1, 3, 5, 6, 7, 8]),
    ("5", 10, [5]),
    ("3-7", 10, [3, 4, 5, 6, 7]),
    ("0-3", 10, [1, 2, 3]),          # clamped to page 1
    ("8-99", 10, [8, 9, 10]),        # clamped to total
    ("5", 3, []),                    # out of range
    ("abc", 10, []),                 # garbage
    ("2,abc,4", 10, [2, 4]),         # garbage part skipped
    ("3-1", 10, []),                 # inverted range
    ("2, 4 , 6", 10, [2, 4, 6]),     # spaces tolerated
])
def test_parse_pages(spec, total, expected):
    assert _parse_pages(spec, total) == expected
