"""Unit tests for TonerHound LiteParse DocumentIndex integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tonerhound import DocumentIndex, LiteParseDocumentIndex
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.liteparse_index import (
    _extract_tokens_from_liteparse_page,
    _normalize_bbox_coords,
)
from tonerhound.matching.matcher import EvidenceMatcher

REAL_PDF_PATH = Path("research/data/full/short/00581-2011-p0050.pdf")
MULTI_PAGE_PDF_PATH = Path("research/reference/groundmark/tests/data/two_pages.pdf")


def test_initializes_from_real_pdf():
    """Verify LiteParseDocumentIndex initializes properly from a real PDF."""
    assert REAL_PDF_PATH.exists(), f"Sample PDF missing at {REAL_PDF_PATH}"

    # 1. Initialize via LiteParseDocumentIndex.from_pdf
    index_direct = LiteParseDocumentIndex.from_pdf(REAL_PDF_PATH, use_cache=False)
    assert isinstance(index_direct, LiteParseDocumentIndex)
    assert isinstance(index_direct, DocumentIndex)
    assert index_direct.total_pages == 1

    page = index_direct.get_page(1)
    assert page is not None
    assert page.page_number == 1
    assert page.width > 0
    assert page.height > 0
    assert len(page.tokens) > 0
    assert len(page.lines) > 0

    # Verify statistics: 433 tokens clustered into 51 lines
    num_tokens = len(page.tokens)
    num_lines = len(page.lines)
    assert num_tokens == 433, f"Expected 433 tokens, got {num_tokens}"
    assert num_lines == 51, f"Expected 51 lines, got {num_lines}"

    # 2. Seamless integration via DocumentIndex.from_pdf(..., backend="liteparse")
    index_factory = DocumentIndex.from_pdf(str(REAL_PDF_PATH), backend="liteparse", use_cache=False)
    assert isinstance(index_factory, LiteParseDocumentIndex)
    assert index_factory.total_pages == 1
    assert len(index_factory.get_page(1).tokens) == num_tokens

    # 3. Initialize from raw bytes
    with open(REAL_PDF_PATH, "rb") as f:
        pdf_bytes = f.read()
    index_bytes = LiteParseDocumentIndex.from_pdf(pdf_bytes, use_cache=False)
    assert index_bytes.total_pages == 1
    assert len(index_bytes.get_page(1).tokens) == num_tokens


def test_bounding_boxes_strictly_in_0_1_range():
    """Verify that all extracted bounding boxes are strictly normalized in [0.0, 1.0]."""
    index = LiteParseDocumentIndex.from_pdf(REAL_PDF_PATH, use_cache=False)

    for page in index.pages:
        # Check every token
        for token in page.tokens:
            b = token.bbox
            assert 0.0 <= b.x <= 1.0, f"Token bbox.x out of bounds: {b.x} (text: {token.text})"
            assert 0.0 <= b.y <= 1.0, f"Token bbox.y out of bounds: {b.y} (text: {token.text})"
            assert 0.0 <= b.width <= 1.0, f"Token bbox.width out of bounds: {b.width}"
            assert 0.0 <= b.height <= 1.0, f"Token bbox.height out of bounds: {b.height}"
            assert 0.0 <= b.x0 <= 1.0, f"Token bbox.x0 out of bounds: {b.x0}"
            assert 0.0 <= b.x1 <= 1.0, f"Token bbox.x1 out of bounds: {b.x1}"
            assert 0.0 <= b.y0 <= 1.0, f"Token bbox.y0 out of bounds: {b.y0}"
            assert 0.0 <= b.y1 <= 1.0, f"Token bbox.y1 out of bounds: {b.y1}"
            assert b.x1 >= b.x0
            assert b.y1 >= b.y0
            assert b.page == page.page_number

        # Check every visual line
        for line in page.lines:
            lb = line.bbox
            assert 0.0 <= lb.x0 <= 1.0, f"Line bbox.x0 out of bounds: {lb.x0} (line: {line.text})"
            assert 0.0 <= lb.x1 <= 1.0, f"Line bbox.x1 out of bounds: {lb.x1} (line: {line.text})"
            assert 0.0 <= lb.y0 <= 1.0, f"Line bbox.y0 out of bounds: {lb.y0} (line: {line.text})"
            assert 0.0 <= lb.y1 <= 1.0, f"Line bbox.y1 out of bounds: {lb.y1} (line: {line.text})"
            assert lb.page == page.page_number


def test_normalize_bbox_coords_edge_cases():
    """Verify edge case clamping in coordinate normalization."""
    # Negative point coordinates clamp to 0.0
    b_neg = _normalize_bbox_coords(x=-50.0, y=-20.0, width=10.0, height=20.0, page_width=612.0, page_height=792.0, page_num=1)
    assert b_neg.x == 0.0
    assert b_neg.y == 0.0
    assert 0.0 <= b_neg.x1 <= 1.0
    assert 0.0 <= b_neg.y1 <= 1.0

    # Excessively large point coordinates clamp to 1.0
    b_huge = _normalize_bbox_coords(x=700.0, y=900.0, width=200.0, height=100.0, page_width=612.0, page_height=792.0, page_num=1)
    assert b_huge.x == 1.0
    assert b_huge.y == 1.0
    assert b_huge.width == 0.0
    assert b_huge.height == 0.0

    # Zero or negative page dimensions fallback cleanly without ZeroDivisionError
    b_zero_dim = _normalize_bbox_coords(x=100.0, y=100.0, width=50.0, height=20.0, page_width=0.0, page_height=-10.0, page_num=1)
    assert 0.0 <= b_zero_dim.x <= 1.0
    assert 0.0 <= b_zero_dim.y <= 1.0
    assert 0.0 <= b_zero_dim.x1 <= 1.0
    assert 0.0 <= b_zero_dim.y1 <= 1.0


def test_inverted_search_indexes_match_pdfium():
    """Verify that inverted search indexes work identically between LiteParse and PDFium."""
    lp_index = LiteParseDocumentIndex.from_pdf(REAL_PDF_PATH, use_cache=False)
    pdfium_index = DocumentIndex.from_pdf(REAL_PDF_PATH, backend="pdfium", use_cache=False)

    # 1. Exact string search across document
    test_queries = [
        "651108",
        "Schedule K-1",
        "2008",
        "1545-0099",
        "TELCO",
        "Internal Revenue Service",
    ]

    for q in test_queries:
        lp_matches = lp_index.search_exact(q)
        pdfium_matches = pdfium_index.search_exact(q)

        assert len(lp_matches) >= 1, f"LiteParse index failed to find query '{q}'"
        assert len(pdfium_matches) >= 1, f"PDFium index failed to find query '{q}'"

        # Compare matching bounding boxes for the top match
        lp_box, lp_text = lp_matches[0]
        pdf_box, pdf_text = pdfium_matches[0]
        assert q in lp_text
        assert q in pdf_text
        # Both bboxes should be on page 1 and close in space
        assert lp_box.page == pdf_box.page == 1
        assert abs(lp_box.x - pdf_box.x) < 0.05, f"Horizontal position discrepancy for '{q}'"
        assert abs(lp_box.y - pdf_box.y) < 0.05, f"Vertical position discrepancy for '{q}'"

    # 2. Numeric inverted index
    assert len(lp_index._numeric_index) > 0
    assert len(pdfium_index._numeric_index) > 0

    lp_matcher = EvidenceMatcher(lp_index)
    pdf_matcher = EvidenceMatcher(pdfium_index)

    # Search for numeric key 651108.0
    lp_cands = lp_matcher.find_normalized_numeric_candidates(651108.0)
    pdf_cands = pdf_matcher.find_normalized_numeric_candidates(651108.0)
    assert len(lp_cands) >= 1
    assert len(pdf_cands) >= 1
    assert lp_cands[0].page == 1
    assert pdf_cands[0].page == 1
    assert "651108" in lp_cands[0].matched_text
    assert "651108" in pdf_cands[0].matched_text

    # Search for numeric year 2008.0
    lp_year = lp_matcher.find_normalized_numeric_candidates(2008.0)
    pdf_year = pdf_matcher.find_normalized_numeric_candidates(2008.0)
    assert len(lp_year) >= 1
    assert len(pdf_year) >= 1

    # 3. Token inverted index
    for token_key in ["651108", "schedule", "2008", "telco"]:
        assert token_key in lp_index._token_index, f"Key '{token_key}' missing in LiteParse _token_index"
        assert token_key in pdfium_index._token_index, f"Key '{token_key}' missing in PDFium _token_index"


def test_extractbench_adapter_grounding_on_liteparse_index():
    """Verify that ExtractBenchAdapter runs grounding directly on LiteParseDocumentIndex."""
    index = LiteParseDocumentIndex.from_pdf(REAL_PDF_PATH, use_cache=False)
    adapter = ExtractBenchAdapter(index)

    extracted_data = {
        "form_number": "651108",
        "tax_year": 2008,
        "company_name": "TELCO EXPERTS LLC",
        "omb_number": "1545-0099",
    }

    grounded = adapter.ground_extracted_data(extracted_data, example_id="test_exp006_k1")

    assert grounded["task_type"] == "extract"
    assert grounded["example_id"] == "test_exp006_k1"
    citations = grounded["field_citations"]
    assert len(citations) == 4, f"Expected 4 citations, got {len(citations)}"

    citations_by_path = {c["field_path"]: c for c in citations}
    for field_path in ["form_number", "tax_year", "company_name", "omb_number"]:
        assert field_path in citations_by_path
        cit = citations_by_path[field_path]
        assert cit["page"] == 1
        assert cit["confidence"] >= 0.8
        bbox = cit["bbox"]
        assert len(bbox) == 4
        assert all(0.0 <= coord <= 1.0 for coord in bbox)


def test_caching_mechanism(tmp_path: Path):
    """Verify that LiteParseDocumentIndex disk caching works atomically."""
    cache_dir = tmp_path / "cache"

    # First load: creates cache
    idx1 = LiteParseDocumentIndex.from_pdf(REAL_PDF_PATH, use_cache=True, cache_dir=cache_dir)
    cache_files = list(cache_dir.glob("*_liteparse_*.pkl"))
    assert len(cache_files) == 1, "Cache file was not created"

    # Second load: loads from cache
    idx2 = LiteParseDocumentIndex.from_pdf(REAL_PDF_PATH, use_cache=True, cache_dir=cache_dir)
    assert idx2.total_pages == idx1.total_pages
    assert len(idx2.pages[0].tokens) == len(idx1.pages[0].tokens)

    # Search on loaded cached index works
    matches = idx2.search_exact("651108")
    assert len(matches) == 1
    assert matches[0][1] == "651108"


def test_fallback_to_item_text_when_no_words():
    """Verify that when word boxes are absent, extraction falls back to item.text."""
    # 1. Through LiteParse with emit_word_boxes=False
    idx_no_words = LiteParseDocumentIndex.from_pdf(
        REAL_PDF_PATH,
        emit_word_boxes=False,
        use_cache=False,
    )
    assert idx_no_words.total_pages == 1
    page = idx_no_words.get_page(1)
    assert len(page.tokens) > 0
    # Every token has valid [0, 1] bbox
    for t in page.tokens:
        assert 0.0 <= t.bbox.x <= 1.0
        assert 0.0 <= t.bbox.y <= 1.0

    # 2. Direct unit test on _extract_tokens_from_liteparse_page with simulated items
    item_without_words = MagicMock()
    item_without_words.words = None
    item_without_words.text = "Fallback Text Item"
    item_without_words.x = 50.0
    item_without_words.y = 100.0
    item_without_words.width = 150.0
    item_without_words.height = 20.0

    mock_page = MagicMock()
    mock_page.text_items = [item_without_words]

    tokens = _extract_tokens_from_liteparse_page(
        page=mock_page,
        page_num=1,
        page_width=600.0,
        page_height=800.0,
        emit_word_boxes=True,
    )

    assert len(tokens) == 1
    assert tokens[0].text == "Fallback Text Item"
    assert abs(tokens[0].bbox.x - 50.0 / 600.0) < 1e-5
    assert abs(tokens[0].bbox.y - 100.0 / 800.0) < 1e-5


def test_multi_page_pdf_support():
    """Verify multi-page PDF extraction and indexing with LiteParse."""
    if not MULTI_PAGE_PDF_PATH.exists():
        pytest.skip("Multi-page test PDF not found")

    idx = LiteParseDocumentIndex.from_pdf(MULTI_PAGE_PDF_PATH, use_cache=False)
    assert idx.total_pages == 2

    p1 = idx.get_page(1)
    p2 = idx.get_page(2)
    assert p1 is not None and p2 is not None
    assert len(p1.tokens) > 0
    assert len(p2.tokens) > 0

    # Text on page 1
    m1 = idx.search_exact("first page")
    assert len(m1) >= 1
    assert m1[0][0].page == 1

    # Text on page 2
    m2 = idx.search_exact("second page")
    assert len(m2) >= 1
    assert m2[0][0].page == 2
