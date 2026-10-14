"""EXP-046 Stage 16: Numeric Canonical Matcher V2.

Canonicalizes numeric tokens across diverse formatting permutations
($1,250.00, 1250, 1250.0, 1,250, (1,250.00)) with floating point tolerance <= 0.01.
"""

from __future__ import annotations

import re
from typing import Any

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage


def parse_canonical_float(text: str) -> float | None:
    """Extract canonical floating-point numerical value from token text."""
    s = text.strip()
    is_negative = ("(" in s and ")" in s) or s.startswith("-")
    cleaned = re.sub(r"[^\d.]", "", s)
    if not cleaned or cleaned.count(".") > 1:
        return None
    try:
        val = float(cleaned)
        return -val if is_negative else val
    except ValueError:
        return None


class NumericCanonicalMatcherV2:
    """Matches numerical values by canonicalized numerical equivalence."""

    def __init__(self, tolerance: float = 0.01) -> None:
        self.tolerance = tolerance

    def match_numeric_target(
        self, target_value: Any, page_obj: DocumentPage
    ) -> tuple[BBox | None, str | None]:
        """Search page for numerically equivalent tokens within tolerance."""
        target_f = parse_canonical_float(str(target_value))
        if target_f is None:
            return None, None

        best_diff = float("inf")
        best_tok = None

        for t in page_obj.tokens:
            tok_f = parse_canonical_float(t.text)
            if tok_f is not None:
                diff = abs(target_f - tok_f)
                if diff <= self.tolerance and diff < best_diff:
                    best_diff = diff
                    best_tok = t

        if best_tok:
            b = best_tok.bbox
            return BBox(b.x, b.y, b.width, b.height, page_obj.page_number), best_tok.text

        return None, None
