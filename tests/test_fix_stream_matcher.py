"""Unit tests for MonotonicStreamMatcher."""

from __future__ import annotations

import pytest
from tonerhound.grounding.stream_matcher import (
    MonotonicStreamMatcher,
    compute_iou_xywh,
    disambiguate_repeated_token_stream,
)


class MockOFACPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_text_range(self) -> str:
        return self.text

    def get_charbox(self, index: int) -> tuple[float, float, float, float] | None:
        # Give different vertical positions depending on where the token appears
        # If text is e.g. "a.k.a. ... a.k.a."
        y_top = 700.0 - (index * 2.0)
        return (50.0, y_top - 10.0, 100.0, y_top)


def test_disambiguate_repeated_tokens() -> None:
    text = "First entity a.k.a. Name1 ... Second entity a.k.a. Name2"
    tp = MockOFACPage(text)

    # First occurrence
    box0 = disambiguate_repeated_token_stream(tp, 612.0, 792.0, "a.k.a.", 0)
    assert box0 is not None

    # Second occurrence
    box1 = disambiguate_repeated_token_stream(tp, 612.0, 792.0, "a.k.a.", 1)
    assert box1 is not None

    # Distinct vertical positions
    assert box0[1] != box1[1]


def test_stream_matcher_verification() -> None:
    text = "a.k.a. First a.k.a. Second"
    tp = MockOFACPage(text)
    matcher = MonotonicStreamMatcher(min_iou_threshold=0.50)

    box0 = disambiguate_repeated_token_stream(tp, 612.0, 792.0, "a.k.a.", 0)
    assert box0 is not None

    matched = matcher.match_instance(tp, 612.0, 792.0, "a.k.a.", 0, box0)
    assert matched is not None

    # Mismatched target returns None (zero regressions)
    mismatched = matcher.match_instance(tp, 612.0, 792.0, "a.k.a.", 0, [0.9, 0.9, 0.05, 0.05])
    assert mismatched is None
