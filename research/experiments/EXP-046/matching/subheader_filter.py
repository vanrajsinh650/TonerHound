"""EXP-046 Stage 10: Universal Sub-Header Filter.

Detects and classifies intermediate sub-headers, sector titles, and category banners
using structural pattern checks (percentage annotations, title casing, lack of currency values).
"""

from __future__ import annotations

import re
from typing import Sequence


_PCT_PATTERN = re.compile(r"(\d+(\.\d+)?%|—\s*\d+(\.\d+)?%)")
_CURRENCY_OR_NUM_PATTERN = re.compile(r"(\$\s*[\d,]+|[\d,]+\.\d{2}|(?<!\w)\d{3,}(?!\w))")


class SubHeaderFilter:
    """Classifies rows as sub-headers to prevent row-index accumulation drift."""

    def is_subheader(self, cells_text: Sequence[str]) -> bool:
        """Evaluate 3-condition rule: if 2 of 3 match, row is a sub-header."""
        non_empty = [c.strip() for c in cells_text if c and c.strip()]
        if not non_empty:
            return False

        full_text = " ".join(non_empty)

        # Condition 1: Lacks numeric/currency cells
        has_num = bool(_CURRENCY_OR_NUM_PATTERN.search(full_text))
        cond1 = not has_num

        # Condition 2: Contains percentage with category / short title
        has_pct = bool(_PCT_PATTERN.search(full_text))
        cond2 = has_pct

        # Condition 3: All-caps or title case with minimal punctuation
        cleaned = re.sub(r"[\s\d%—\-\(\)\.,:]+", "", full_text)
        cond3 = cleaned.isupper() or full_text.istitle()

        match_count = sum([cond1, cond2, cond3])
        return match_count >= 2


def filter_subheaders_from_rows(rows: list[list[str]]) -> list[tuple[int, list[str]]]:
    """Return filtered data rows with original 0-indexed positions."""
    f = SubHeaderFilter()
    filtered = []
    for idx, r in enumerate(rows):
        if not f.is_subheader(r):
            filtered.append((idx, r))
    return filtered
