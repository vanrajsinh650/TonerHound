"""Tests for TonerHound DocumentIndex."""

from pathlib import Path

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


def test_synthetic_document_indexing():
    # Construct a 1-page document with 2 lines:
    # Line 0: "Invoice Number: INV-2024-001"
    # Line 1: "Total Amount: $1,450.00"
    t1 = DocumentToken("Invoice", BBox(0.1, 0.1, 0.08, 0.02, page=1), page=1, char_index_in_page=0)
    t2 = DocumentToken("Number:", BBox(0.19, 0.1, 0.08, 0.02, page=1), page=1, char_index_in_page=8)
    t3 = DocumentToken("INV-2024-001", BBox(0.3, 0.1, 0.15, 0.02, page=1), page=1, char_index_in_page=16)

    t4 = DocumentToken("Total", BBox(0.1, 0.15, 0.06, 0.02, page=1), page=1, char_index_in_page=29)
    t5 = DocumentToken("Amount:", BBox(0.17, 0.15, 0.08, 0.02, page=1), page=1, char_index_in_page=35)
    t6 = DocumentToken("$1,450.00", BBox(0.3, 0.15, 0.1, 0.02, page=1), page=1, char_index_in_page=43)

    line1 = VisualLine([t1, t2, t3], page=1, line_index=0, bbox=BBox(0.1, 0.1, 0.35, 0.02, page=1))
    line2 = VisualLine([t4, t5, t6], page=1, line_index=1, bbox=BBox(0.1, 0.15, 0.3, 0.02, page=1))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=[t1, t2, t3, t4, t5, t6], lines=[line1, line2])
    index = DocumentIndex.from_pages([page])

    assert index.total_pages == 1
    matches = index.search_exact("INV-2024-001")
    assert len(matches) == 1
    box, matched_text = matches[0]
    assert "INV-2024-001" in matched_text
    assert abs(box.x - 0.3) < 1e-5
    assert abs(box.y - 0.1) < 1e-5


def test_real_pdf_indexing():
    pdf_path = Path("research/reference/groundmark/tests/data/two_pages.pdf")
    if not pdf_path.exists():
        return

    index = DocumentIndex.from_pdf(pdf_path)
    assert index.total_pages == 2
    page1 = index.get_page(1)
    assert page1 is not None
    assert len(page1.tokens) > 0
    assert len(page1.lines) > 0
    # Search for known text on page 1
    matches = index.search_exact("Page", page=1)
    assert len(matches) >= 1
