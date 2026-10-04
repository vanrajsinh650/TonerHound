"""Unit tests for MultiTokenSequenceMatcher."""

from __future__ import annotations

import fitz
import pytest
from tonerhound.matching.multi_token import MultiTokenSequenceMatcher, normalize_for_match, union_bbox


def test_normalize_for_match() -> None:
    assert normalize_for_match("Hello,  World! ") == "hello, world"
    assert normalize_for_match("$123.45") == "$123.45"


def test_union_bbox() -> None:
    boxes = [
        [0.1, 0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1, 0.1],
    ]
    u = union_bbox(boxes)
    assert u == [0.1, 0.1, 0.2, 0.1]


def test_multi_token_matcher_fitz_page() -> None:
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text(fitz.Point(50, 50), "Internal Revenue Service")

    matcher = MultiTokenSequenceMatcher(page)
    box = matcher.match_sequence("Internal Revenue Service")
    doc.close()

    assert box is not None
    assert len(box) == 4
    assert box[0] < 0.2
    assert box[2] > 0
