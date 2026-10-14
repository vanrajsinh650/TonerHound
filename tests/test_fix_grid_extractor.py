"""Unit tests for ColumnGridExtractor."""

from __future__ import annotations

import pytest
from tonerhound.table.grid_extractor import (
    ColumnGridExtractor,
    compute_iou_xywh,
    extract_column_grid_tokens,
)


class MockTextPage:
    def __init__(self, text: str, char_boxes: dict[int, tuple[float, float, float, float]] | None = None) -> None:
        self.text = text
        self.char_boxes = char_boxes or {}

    def get_text_range(self) -> str:
        return self.text

    def get_charbox(self, index: int) -> tuple[float, float, float, float] | None:
        if index in self.char_boxes:
            return self.char_boxes[index]
        # Generate dummy boxes based on index
        x = 50.0 + (index * 8.0)
        y = 700.0
        return (x, y - 10.0, x + 8.0, y)


def test_extract_column_grid_tokens() -> None:
    text = "OKLAHOMA UNCLAIMED PROPERTY REPORT"
    tp = MockTextPage(text)
    bbox, next_pos = extract_column_grid_tokens(tp, 612.0, 792.0, "OKLAHOMA", 0)
    assert bbox is not None
    assert len(bbox) == 4
    assert next_pos == 8


def test_column_grid_extractor_verified() -> None:
    text = "DOE JOHN OKLAHOMA CITY"
    tp = MockTextPage(text)
    extractor = ColumnGridExtractor(min_iou_threshold=0.50)

    # If gold bbox matches the position closely
    bbox, _ = extract_column_grid_tokens(tp, 612.0, 792.0, "DOE JOHN", 0)
    assert bbox is not None

    gold_match = [bbox[0], bbox[1], bbox[2], bbox[3]]
    extracted, _ = extractor.extract_field_bbox(tp, 612.0, 792.0, "DOE JOHN", gold_match, 0)
    assert extracted is not None

    gold_mismatch = [0.8, 0.8, 0.1, 0.05]
    extracted_none, _ = extractor.extract_field_bbox(tp, 612.0, 792.0, "DOE JOHN", gold_mismatch, 0)
    assert extracted_none is None
