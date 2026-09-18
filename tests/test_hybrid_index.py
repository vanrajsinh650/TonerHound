"""Tests for TonerHound Hybrid DocumentIndex (LiteParse + OCR fallback).

Verifies:
1. Interface conformance with DocumentIndex (usable by ExtractBenchAdapter with zero code changes).
2. Native LiteParse path selection on digital PDFs with layout blocks (tables, paragraphs, headings).
3. Seamless OCR fallback on corrupted / scanned PDFs via PDFium + Tesseract + repair_ocr_text.
4. Gate control flags: force_ocr, force_liteparse, enable_ocr, min_digital_tokens.
5. Layout block hints exposure to ExtractBenchAdapter and spatial block queries.
6. Multi-page / mixed document routing.
7. Synthetic page fixture construction via from_pages.
8. Disk caching persistence and reloading.
9. Resilient fallback on LiteParse runtime exceptions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tonerhound import (
    DocumentIndex,
    DocumentPage,
    DocumentToken,
    HybridDocumentIndex,
    HybridIndex,
    LayoutBlockHint,
    VisualLine,
)
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.geometry.coordinates import BBox

DIGITAL_PDF_PATH = Path("research/data/full/short/00581-2011-p0050.pdf")
CLEAN_DIGITAL_PDF = Path("research/data/full/short/07021-2016-p0029.pdf")
CORRUPTED_PDF_PATH = Path("research/data/full/short/caterpillar_spec_sheet_312c_excavator_corrupted.pdf")
VEHICLE_DIGITAL_PDF = Path("research/data/full/medium/ccc_online_0003_geico_ford_crown_victoria.pdf")


# ===========================================================================
# 1. Interface Conformance
# ===========================================================================

def test_hybrid_index_interface_contract() -> None:
    """Verify that HybridDocumentIndex strictly inherits and implements DocumentIndex."""
    assert issubclass(HybridDocumentIndex, DocumentIndex)
    assert HybridIndex is HybridDocumentIndex

    # Construct minimal synthetic index
    t = DocumentToken(text="Test", bbox=BBox(0.1, 0.1, 0.1, 0.02, page=1), page=1, char_index_in_page=0)
    line = VisualLine(tokens=[t], page=1, line_index=0, bbox=t.bbox)
    page = DocumentPage(page_number=1, width=612.0, height=792.0, tokens=[t], lines=[line])

    idx = HybridDocumentIndex.from_pages([page])
    assert isinstance(idx, DocumentIndex)
    assert idx.total_pages == 1
    assert idx.get_page(1) is page
    assert len(idx.pages) == 1

    # Inverted search indexes must be functional
    matches = idx.search_exact("Test")
    assert len(matches) == 1
    assert matches[0][1] == "Test"

    # Layout block helpers must exist
    assert hasattr(idx, "layout_blocks")
    assert hasattr(idx, "layout_blocks_by_page")
    assert hasattr(idx, "page_modes")
    assert hasattr(idx, "get_layout_blocks")
    assert hasattr(idx, "get_table_blocks")
    assert hasattr(idx, "get_paragraph_blocks")
    assert hasattr(idx, "get_heading_blocks")
    assert hasattr(idx, "find_blocks_at")


# ===========================================================================
# 2. Digital PDF: Native LiteParse Routing & Layout Blocks
# ===========================================================================

def test_digital_pdf_uses_liteparse_path() -> None:
    """Digital PDF must route to LiteParse, extract tokens, lines, and layout blocks."""
    assert CLEAN_DIGITAL_PDF.exists(), f"Missing fixture {CLEAN_DIGITAL_PDF}"

    idx = HybridDocumentIndex.from_pdf(CLEAN_DIGITAL_PDF, use_cache=False)
    assert idx.total_pages == 1
    assert idx.page_modes[1] == "liteparse"

    page = idx.get_page(1)
    assert page is not None
    assert len(page.tokens) > 0
    assert len(page.lines) > 0

    # Verify all bounding boxes are normalized in [0, 1]
    for token in page.tokens:
        b = token.bbox
        assert 0.0 <= b.x <= 1.0
        assert 0.0 <= b.y <= 1.0
        assert 0.0 <= b.width <= 1.0
        assert 0.0 <= b.height <= 1.0
        assert b.page == 1

    # Verify line clustering and line_index on tokens
    for line in page.lines:
        assert len(line.tokens) > 0
        for tok in line.tokens:
            assert tok.line_index == line.line_index

    # Verify layout blocks were extracted
    assert len(idx.layout_blocks) > 0
    assert len(idx.layout_blocks_by_page[1]) == len(idx.layout_blocks)
    assert len(page.blocks) == len(idx.layout_blocks)

    # Check paragraph and heading blocks
    paragraphs = idx.get_paragraph_blocks(page_num=1)
    assert len(paragraphs) > 0
    for p in paragraphs:
        assert p.is_paragraph
        assert p.page == 1
        assert 0.0 <= p.bbox.x <= 1.0
        assert 0.0 <= p.bbox.y <= 1.0

    headings = idx.get_heading_blocks(page_num=1)
    for h in headings:
        assert h.is_heading


def test_digital_pdf_table_layout_blocks() -> None:
    """Verify table layout blocks with cell rows and bboxes on digital PDF."""
    assert VEHICLE_DIGITAL_PDF.exists(), f"Missing fixture {VEHICLE_DIGITAL_PDF}"

    idx = HybridDocumentIndex.from_pdf(VEHICLE_DIGITAL_PDF, max_pages=2, use_cache=False)
    assert idx.total_pages == 2
    assert idx.page_modes[1] == "liteparse"
    assert idx.page_modes[2] == "liteparse"

    tables = idx.get_table_blocks(page_num=1)
    assert len(tables) >= 1

    table0 = tables[0]
    assert table0.is_table
    assert table0.rows is not None
    assert len(table0.rows) >= 3
    assert table0.cell_bboxes is not None
    assert len(table0.cell_bboxes) == len(table0.rows)

    # Check cell contents and normalized bboxes
    first_row_texts = table0.rows[0]
    assert any("loss vehicle" in txt.lower() or "ford" in txt.lower() for txt in first_row_texts)
    for row_boxes in table0.cell_bboxes:
        for cbox in row_boxes:
            assert 0.0 <= cbox.x <= 1.0
            assert 0.0 <= cbox.y <= 1.0
            assert cbox.page == 1


# ===========================================================================
# 3. Corrupted PDF: Seamless Fallback to TonerHound OCR
# ===========================================================================

def test_corrupted_pdf_seamless_ocr_fallback() -> None:
    """Corrupted / scanned PDF with negligible digital text must fall back to TonerHound OCR."""
    assert CORRUPTED_PDF_PATH.exists(), f"Missing fixture {CORRUPTED_PDF_PATH}"

    idx = HybridDocumentIndex.from_pdf(CORRUPTED_PDF_PATH, use_cache=False)
    assert idx.total_pages == 1
    assert idx.page_modes[1] == "ocr"

    page = idx.get_page(1)
    assert page is not None
    # OCR recovers abundant tokens from the raster scan
    assert len(page.tokens) > 200

    # Tokens must have valid normalized geometries
    for tok in page.tokens:
        assert 0.0 <= tok.bbox.x <= 1.0
        assert 0.0 <= tok.bbox.y <= 1.0
        assert 0.0 <= tok.bbox.width <= 1.0
        assert 0.0 <= tok.bbox.height <= 1.0

    # Text search on OCR-recovered content
    matches = idx.search_exact("Cylinder", page=1)
    assert len(matches) >= 1
    box, txt = matches[0]
    assert "Cylinder" in txt
    assert box.page == 1


# ===========================================================================
# 4. Gate Control Flags (force_ocr, force_liteparse, enable_ocr)
# ===========================================================================

def test_force_ocr_flag_on_digital_pdf() -> None:
    """force_ocr=True must route a digital PDF to TonerHound OCR."""
    idx = HybridDocumentIndex.from_pdf(DIGITAL_PDF_PATH, force_ocr=True, use_cache=False)
    assert idx.page_modes[1] == "ocr"
    page = idx.get_page(1)
    assert page is not None
    assert len(page.tokens) > 0


def test_force_liteparse_flag_on_digital_pdf() -> None:
    """force_liteparse=True must preserve LiteParse routing."""
    idx = HybridDocumentIndex.from_pdf(DIGITAL_PDF_PATH, force_liteparse=True, use_cache=False)
    assert idx.page_modes[1] == "liteparse"
    page = idx.get_page(1)
    assert page is not None
    assert len(page.tokens) == 433


def test_enable_ocr_false_on_corrupted_pdf() -> None:
    """enable_ocr=False prevents OCR execution even when LiteParse extracts 0 tokens."""
    idx = HybridDocumentIndex.from_pdf(CORRUPTED_PDF_PATH, enable_ocr=False, use_cache=False)
    # Without OCR enabled, page cannot fall back to OCR
    assert idx.page_modes[1] == "liteparse"
    page = idx.get_page(1)
    assert page is not None
    assert len(page.tokens) == 0


# ===========================================================================
# 5. ExtractBenchAdapter Integration (Zero Code Changes)
# ===========================================================================

def test_extractbench_adapter_grounding_on_hybrid_digital() -> None:
    """ExtractBenchAdapter consumes HybridDocumentIndex with zero code changes on digital PDF."""
    idx = HybridDocumentIndex.from_pdf(DIGITAL_PDF_PATH, use_cache=False)
    adapter = ExtractBenchAdapter(idx)

    extracted_data = {
        "form_number": "651108",
        "tax_year": 2008,
        "company_name": "TELCO EXPERTS LLC",
        "omb_number": "1545-0099",
    }

    grounded = adapter.ground_extracted_data(extracted_data, example_id="test_exp006_k1")
    assert grounded["task_type"] == "extract"
    assert grounded["example_id"] == "test_exp006_k1"

    citations = {c["field_path"]: c for c in grounded["field_citations"]}
    assert "form_number" in citations
    assert "tax_year" in citations
    assert "company_name" in citations
    assert "omb_number" in citations

    for field_name, cit in citations.items():
        assert cit["page"] == 1
        assert cit["confidence"] >= 0.80
        bbox = cit["bbox"]
        assert len(bbox) == 4
        assert all(0.0 <= v <= 1.0 for v in bbox)


def test_extractbench_adapter_grounding_on_hybrid_corrupted() -> None:
    """ExtractBenchAdapter consumes HybridDocumentIndex on corrupted PDF via OCR fallback."""
    idx = HybridDocumentIndex.from_pdf(CORRUPTED_PDF_PATH, use_cache=False)
    adapter = ExtractBenchAdapter(idx)

    extracted_data = {
        "model": "312C",
    }

    grounded = adapter.ground_extracted_data(extracted_data, example_id="test_exp006_cat")
    citations = {c["field_path"]: c for c in grounded["field_citations"]}
    assert "model" in citations
    assert citations["model"]["page"] == 1
    assert citations["model"]["confidence"] >= 0.80


# ===========================================================================
# 6. Layout Block Exposure to ExtractBenchAdapter & Spatial Queries
# ===========================================================================

def test_layout_block_hints_exposed_to_adapter() -> None:
    """ExtractBenchAdapter exposes layout blocks directly from underlying hybrid index."""
    idx = HybridDocumentIndex.from_pdf(VEHICLE_DIGITAL_PDF, max_pages=1, use_cache=False)
    adapter = ExtractBenchAdapter(idx)

    # Exposed properties on adapter
    assert len(adapter.layout_blocks) > 0
    tables = adapter.get_table_blocks(page_num=1)
    paragraphs = adapter.get_paragraph_blocks(page_num=1)

    assert len(tables) >= 1
    assert len(paragraphs) >= 1
    assert all(t.is_table for t in tables)
    assert all(p.is_paragraph for p in paragraphs)


def test_spatial_block_lookup() -> None:
    """find_blocks_at returns blocks intersecting a target query bounding box."""
    idx = HybridDocumentIndex.from_pdf(VEHICLE_DIGITAL_PDF, max_pages=1, use_cache=False)
    tables = idx.get_table_blocks(page_num=1)
    assert len(tables) >= 1

    t0_bbox = tables[0].bbox
    # Exact bbox query finds the table
    found = idx.find_blocks_at(page_num=1, bbox=t0_bbox)
    assert any(b.id == tables[0].id for b in found)

    # Unrelated region (bottom right corner) does not match top table
    empty_query = BBox(x=0.90, y=0.90, width=0.08, height=0.08, page=1)
    found_empty = idx.find_blocks_at(page_num=1, bbox=empty_query)
    assert not any(b.id == tables[0].id for b in found_empty)


# ===========================================================================
# 7. Synthetic Document Construction via from_pages
# ===========================================================================

def test_from_pages_synthetic_construction() -> None:
    """Verify synthetic DocumentPages and LayoutBlockHints via from_pages."""
    t1 = DocumentToken("Total", BBox(0.1, 0.2, 0.08, 0.02, 1), 1, 0, line_index=0, block_index=1)
    t2 = DocumentToken("$500.00", BBox(0.2, 0.2, 0.10, 0.02, 1), 1, 6, line_index=0, block_index=1)
    line = VisualLine(tokens=[t1, t2], page=1, line_index=0, bbox=BBox(0.1, 0.2, 0.2, 0.02, 1))

    table_hint = LayoutBlockHint(
        kind="table",
        bbox=BBox(0.05, 0.15, 0.40, 0.20, 1),
        page=1,
        rows=[["Total", "$500.00"]],
    )

    page = DocumentPage(page_number=1, width=612.0, height=792.0, tokens=[t1, t2], lines=[line], blocks=[table_hint])
    idx = HybridDocumentIndex.from_pages([page])

    assert idx.total_pages == 1
    assert len(idx.layout_blocks) == 1
    assert len(idx.get_table_blocks()) == 1
    assert idx.get_table_blocks()[0].rows == [["Total", "$500.00"]]

    # Numeric inverted index works
    cands = idx._numeric_index.get(500.0, [])
    assert len(cands) == 1
    assert cands[0][0].text == "$500.00"


# ===========================================================================
# 8. Caching Persistence and Reloading
# ===========================================================================

def test_hybrid_index_caching_persistence(tmp_path: Path) -> None:
    """Verify disk caching creates pickle file and subsequent load uses cache."""
    cache_dir = tmp_path / "hybrid_cache"

    # Initial load: writes to disk cache
    idx1 = HybridDocumentIndex.from_pdf(DIGITAL_PDF_PATH, use_cache=True, cache_dir=cache_dir)
    cache_files = list(cache_dir.glob("*.pkl"))
    assert len(cache_files) == 1

    # Second load: loads from disk cache
    idx2 = HybridDocumentIndex.from_pdf(DIGITAL_PDF_PATH, use_cache=True, cache_dir=cache_dir)
    assert idx2.total_pages == idx1.total_pages
    assert idx2.page_modes == idx1.page_modes
    assert len(idx2.pages[0].tokens) == len(idx1.pages[0].tokens)
    assert len(idx2.layout_blocks) == len(idx1.layout_blocks)


# ===========================================================================
# 9. Resilient Fallback on LiteParse Exception
# ===========================================================================

def test_resilient_fallback_on_liteparse_exception() -> None:
    """If LiteParse raises a runtime exception, hybrid index seamlessly falls back to PDFium or OCR."""
    with patch("tonerhound.document.hybrid_index.LiteParse") as mock_lp:
        instance = MagicMock()
        instance.parse.side_effect = RuntimeError("Simulated LiteParse native crash")
        mock_lp.return_value = instance

        # Digital PDF with native text falls back to fast PDFium extraction
        idx = HybridDocumentIndex.from_pdf(DIGITAL_PDF_PATH, use_cache=False)
        assert idx.total_pages == 1
        assert idx.page_modes[1] == "pdfium"
        page = idx.get_page(1)
        assert page is not None
        assert len(page.tokens) > 0

        # Scanned PDF without native text falls back to OCR
        idx_corrupt = HybridDocumentIndex.from_pdf(CORRUPTED_PDF_PATH, use_cache=False)
        assert idx_corrupt.total_pages == 1
        assert idx_corrupt.page_modes[1] == "ocr"
        page_corrupt = idx_corrupt.get_page(1)
        assert page_corrupt is not None
        assert len(page_corrupt.tokens) > 0


# ===========================================================================
# 10. DocumentIndex Factory Integration
# ===========================================================================

def test_document_index_factory_hybrid_backend() -> None:
    """Verify DocumentIndex.from_pdf(..., backend='hybrid') instantiates HybridDocumentIndex."""
    idx = DocumentIndex.from_pdf(CLEAN_DIGITAL_PDF, backend="hybrid", use_cache=False)
    assert isinstance(idx, HybridDocumentIndex)
    assert idx.page_modes[1] == "liteparse"
    assert idx.total_pages == 1
