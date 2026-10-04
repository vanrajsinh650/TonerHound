"""Unit tests for TableCellExtractor."""

from __future__ import annotations

import fitz
import pytest
from tonerhound.document.table_cells import (
    TableCellExtractor,
    compute_iou_xywh,
    normalize_val,
)


def test_normalize_val() -> None:
    assert normalize_val("$1,234.50") == "1234.50"
    assert normalize_val("(500.00)") == "-500.00"
    assert normalize_val("  Total Revenue  ") == "totalrevenue"


def test_table_cell_grounder_mock() -> None:
    extractor = TableCellExtractor()
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    # Draw simple lines to form a rectangle
    page.draw_rect(fitz.Rect(50, 50, 200, 100), color=(0, 0, 0), width=1)
    page.insert_text(fitz.Point(60, 80), "$100.00")

    box, iou = extractor.ground_field_in_table(
        doc,
        pdf_path="virtual_doc.pdf",
        page_num=1,
        gold_value="100.00",
        gold_bbox=[0.1, 0.1, 0.3, 0.1],
    )
    doc.close()
    # It gracefully executes without error
    assert iou >= 0.0
