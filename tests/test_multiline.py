"""Unit tests for multiline assembler."""

from __future__ import annotations

import fitz
import pytest
from tonerhound.geometry.multiline import assemble_multiline_bbox, normalize_for_match


def test_normalize_for_match() -> None:
    assert normalize_for_match("  Address Line 1  ") == "address line 1"


def test_assemble_multiline_bbox_fitz() -> None:
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text(fitz.Point(50, 50), "123 Main Street")
    page.insert_text(fitz.Point(50, 65), "Suite 400")

    box = assemble_multiline_bbox(page, "123 Main Street Suite 400")
    doc.close()

    assert box is not None
    assert len(box) == 4
    assert box[2] > 0
    assert box[3] > 0
