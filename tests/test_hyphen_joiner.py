"""Unit tests for join_hyphenated_pairs."""

from __future__ import annotations

import fitz
import pytest
from tonerhound.geometry.hyphen_joiner import join_hyphenated_pairs


def test_join_hyphenated_pairs_fitz() -> None:
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text(fitz.Point(50, 50), "Inter-")
    page.insert_text(fitz.Point(50, 65), "national")

    res = join_hyphenated_pairs(page, "International")
    doc.close()

    assert res is not None
    assert len(res) == 4
    assert res[2] > 0
    assert res[3] > 0
