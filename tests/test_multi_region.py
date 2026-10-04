"""Unit tests for MultiRegionAssembler."""

from __future__ import annotations

import fitz
import pytest
from tonerhound.geometry.multi_region import MultiRegionAssembler, union_bbox


def test_union_bbox() -> None:
    assert union_bbox([]) is None
    boxes = [[0.1, 0.2, 0.3, 0.4]]
    assert union_bbox(boxes) == [0.1, 0.2, 0.3, 0.4]


def test_multi_region_assembler_clusters() -> None:
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text(fitz.Point(50, 50), "Section A")
    page.insert_text(fitz.Point(50, 60), "Section A continuation")

    assembler = MultiRegionAssembler(page)
    regions = assembler.assemble_regions("Section A")
    doc.close()

    assert len(regions) >= 1
    assert len(regions[0]) == 4
