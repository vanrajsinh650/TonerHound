"""Explicit tests for OCR repair mechanisms and error categories.

Covers:
1. OCR spelling errors
2. Merged tokens
3. Split tokens
4. Missing punctuation
5. Numeric confusion
6. 0/O confusion
7. 1/I confusion
8. Decimal confusion
9. Currency corruption
10. Checkbox fields
11. Checkmarks
12. Rotated text / bounding boxes
"""

from __future__ import annotations

import pytest
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, ExtractionInput, VisualLine
from tonerhound.document.index import DocumentIndex
from tonerhound.normalization.normalizers import (
    detect_checkbox_state,
    is_number_equal,
    parse_numeric_value,
    repair_numeric_ocr,
    repair_ocr_text,
)
from tonerhound.resolution.resolver import EvidenceResolver


def test_ocr_0_vs_o_repair() -> None:
    """Test deterministic repair of 0 vs O/o in numeric tokens."""
    assert repair_numeric_ocr("1O5.5O") == "105.50"
    assert repair_numeric_ocr("5O836") == "50836"
    assert parse_numeric_value("1O5.5O") == 105.50
    assert parse_numeric_value("5O836") == 50836.0


def test_ocr_1_vs_l_i_repair() -> None:
    """Test deterministic repair of 1 vs l/I in numeric tokens."""
    assert repair_numeric_ocr("l033982") == "1033982"
    assert repair_numeric_ocr("I450.00") == "1450.00"
    assert parse_numeric_value("l033982") == 1033982.0
    assert parse_numeric_value("I450.00") == 1450.00


def test_ocr_decimal_confusion() -> None:
    """Test handling of decimal confusion with commas, spaces, or currency symbols."""
    assert parse_numeric_value("3,070.00") == 3070.0
    assert parse_numeric_value("$53,180.00") == 53180.0
    assert parse_numeric_value("700.88") == 700.88
    assert is_number_equal("3070", "3,070.00")


def test_ocr_currency_corruption() -> None:
    """Test handling of currency symbols and parenthesized negatives."""
    assert parse_numeric_value("($1,234.56)") == -1234.56
    assert parse_numeric_value("€99.90") == 99.90
    assert parse_numeric_value("£500") == 500.0


def test_ocr_noise_stripping() -> None:
    """Test stripping of underline prefixes/suffixes, pipes, and form artifacts."""
    assert repair_ocr_text("__Milestone") == "Milestone"
    assert repair_ocr_text("Bianco | |") == "Bianco"
    assert repair_ocr_text("Name:_____") == "Name:"


def test_ocr_spelling_and_fuzzy_resolution() -> None:
    """Test resolution of OCR spelling variations via fuzzy matching."""
    tok = DocumentToken(
        text="Dapartment",
        bbox=BBox(x=0.1, y=0.1, width=0.1, height=0.02, page=1),
        page=1,
        char_index_in_page=0,
    )
    line = VisualLine(tokens=[tok], page=1, line_index=0, bbox=tok.bbox)
    page = DocumentPage(page_number=1, width=600, height=800, tokens=[tok], lines=[line])
    idx = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(idx)

    res = resolver.resolve(ExtractionInput(field="agency", value="Department"))
    assert res.is_grounded
    assert res.page == 1
    assert res.bbox is not None and res.bbox.iou(tok.bbox) >= 0.99



def test_ocr_merged_and_split_tokens() -> None:
    """Test matching when OCR split or merged adjacent tokens."""
    t1 = DocumentToken(text="Mi", bbox=BBox(0.1, 0.1, 0.04, 0.02, 1), page=1, char_index_in_page=0)
    t2 = DocumentToken(text="lestone", bbox=BBox(0.14, 0.1, 0.08, 0.02, 1), page=1, char_index_in_page=3)
    line = VisualLine(tokens=[t1, t2], page=1, line_index=0, bbox=BBox(0.1, 0.1, 0.12, 0.02, 1))
    page = DocumentPage(page_number=1, width=600, height=800, tokens=[t1, t2], lines=[line])
    idx = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(idx)

    res = resolver.resolve(ExtractionInput(field="operator", value="Milestone"))
    assert res.is_grounded
    assert res.bbox is not None
    assert abs(res.bbox.width - 0.12) < 1e-4


def test_ocr_missing_punctuation() -> None:
    """Test matching when OCR omitted punctuation in names or addresses."""
    t1 = DocumentToken(text="ALGEBROS", bbox=BBox(0.1, 0.1, 0.05, 0.02, 1), page=1, char_index_in_page=0)
    t2 = DocumentToken(text="LLC", bbox=BBox(0.16, 0.1, 0.03, 0.02, 1), page=1, char_index_in_page=9)
    t3 = DocumentToken(text="THE", bbox=BBox(0.20, 0.1, 0.03, 0.02, 1), page=1, char_index_in_page=13)
    line = VisualLine(tokens=[t1, t2, t3], page=1, line_index=0, bbox=BBox(0.1, 0.1, 0.13, 0.02, 1))
    page = DocumentPage(page_number=1, width=600, height=800, tokens=[t1, t2, t3], lines=[line])
    idx = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(idx)

    # Extraction has comma "ALGEBROS LLC, THE" while OCR text lacks comma
    res = resolver.resolve(ExtractionInput(field="vendor", value="ALGEBROS LLC, THE"))
    assert res.is_grounded
    assert res.page == 1


def test_ocr_checkbox_detection() -> None:
    """Test deterministic detection of checkbox states without hallucinations."""
    assert detect_checkbox_state("[X]") is True
    assert detect_checkbox_state("[x]") is True
    assert detect_checkbox_state("☒") is True
    assert detect_checkbox_state("✔") is True
    assert detect_checkbox_state("[ ]") is False
    assert detect_checkbox_state("☐") is False
    assert detect_checkbox_state("Random Text") is None


def test_rotated_and_normalized_bbox() -> None:
    """Test BBox invariants under non-standard geometries."""
    box = BBox.from_xyxy(x0=0.8, y0=0.5, x1=0.2, y1=0.1, page=1)
    assert box.x == pytest.approx(0.2)
    assert box.y == pytest.approx(0.1)
    assert box.width == pytest.approx(0.6)
    assert box.height == pytest.approx(0.4)
