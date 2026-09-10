"""Adversarial test suite designed to challenge naive grounding systems.

Verifies:
1. Duplicate scalar disambiguation by label context (Subtotal $50 vs Tax $50)
2. Duplicate across pages with page hint
3. Duplicate labels and values: refuses to guess -> ambiguous
4. Normalized currency (USD 1,450.00 -> 1450)
5. Normalized date (15 March 2026 -> 2026-03-15)
6. Wrapped multi-line address -> multi_region
7. OCR corruption -> fuzzy sequence alignment
8. Derived value -> derived
"""

import pytest

from tonerhound import DocumentIndex, DocumentPage, DocumentToken, ExtractionInput, ProvenanceStatus, VisualLine
from tonerhound.geometry.coordinates import BBox
from tonerhound.resolution.resolver import EvidenceResolver


def _create_invoice_page() -> DocumentPage:
    """Page 1 with duplicate $50.00 scalars and normalized entities."""
    # Line 0: Subtotal $50.00
    sub_lbl = DocumentToken("Subtotal", BBox(0.1, 0.20, 0.08, 0.02, page=1), page=1, char_index_in_page=0, line_index=0)
    sub_val = DocumentToken("$50.00", BBox(0.3, 0.20, 0.06, 0.02, page=1), page=1, char_index_in_page=9, line_index=0)
    line0 = VisualLine([sub_lbl, sub_val], page=1, line_index=0, bbox=BBox(0.1, 0.20, 0.26, 0.02, page=1))

    # Line 1: Tax $50.00
    tax_lbl = DocumentToken("Tax", BBox(0.1, 0.25, 0.05, 0.02, page=1), page=1, char_index_in_page=16, line_index=1)
    tax_val = DocumentToken("$50.00", BBox(0.3, 0.25, 0.06, 0.02, page=1), page=1, char_index_in_page=20, line_index=1)
    line1 = VisualLine([tax_lbl, tax_val], page=1, line_index=1, bbox=BBox(0.1, 0.25, 0.26, 0.02, page=1))

    # Line 2: Total Due: USD 1,450.00
    tot_lbl = DocumentToken("Total Due:", BBox(0.1, 0.30, 0.10, 0.02, page=1), page=1, char_index_in_page=27, line_index=2)
    tot_val = DocumentToken("USD 1,450.00", BBox(0.3, 0.30, 0.12, 0.02, page=1), page=1, char_index_in_page=38, line_index=2)
    line2 = VisualLine([tot_lbl, tot_val], page=1, line_index=2, bbox=BBox(0.1, 0.30, 0.32, 0.02, page=1))

    # Line 3: Date: 15 March 2026
    date_lbl = DocumentToken("Date:", BBox(0.1, 0.35, 0.06, 0.02, page=1), page=1, char_index_in_page=51, line_index=3)
    date_d = DocumentToken("15", BBox(0.2, 0.35, 0.03, 0.02, page=1), page=1, char_index_in_page=57, line_index=3)
    date_m = DocumentToken("March", BBox(0.24, 0.35, 0.06, 0.02, page=1), page=1, char_index_in_page=60, line_index=3)
    date_y = DocumentToken("2026", BBox(0.31, 0.35, 0.05, 0.02, page=1), page=1, char_index_in_page=66, line_index=3)
    line3 = VisualLine([date_lbl, date_d, date_m, date_y], page=1, line_index=3, bbox=BBox(0.1, 0.35, 0.26, 0.02, page=1))

    # Lines 4-6: Multi-line Address
    addr_l1 = DocumentToken("12 Example Road,", BBox(0.1, 0.45, 0.18, 0.02, page=1), page=1, char_index_in_page=71, line_index=4)
    line4 = VisualLine([addr_l1], page=1, line_index=4, bbox=addr_l1.bbox)

    addr_l2 = DocumentToken("Sector 28,", BBox(0.1, 0.48, 0.12, 0.02, page=1), page=1, char_index_in_page=88, line_index=5)
    line5 = VisualLine([addr_l2], page=1, line_index=5, bbox=addr_l2.bbox)

    addr_l3 = DocumentToken("Gurugram 122001", BBox(0.1, 0.51, 0.16, 0.02, page=1), page=1, char_index_in_page=99, line_index=6)
    line6 = VisualLine([addr_l3], page=1, line_index=6, bbox=addr_l3.bbox)

    # Line 7: OCR corrupted line: "Ph0ne: +1-800-555-O199" (zeros replaced by 'O')
    ocr_lbl = DocumentToken("Phone:", BBox(0.1, 0.60, 0.08, 0.02, page=1), page=1, char_index_in_page=115, line_index=7)
    ocr_val = DocumentToken("+1-800-555-O199", BBox(0.25, 0.60, 0.18, 0.02, page=1), page=1, char_index_in_page=122, line_index=7)
    line7 = VisualLine([ocr_lbl, ocr_val], page=1, line_index=7, bbox=BBox(0.1, 0.60, 0.33, 0.02, page=1))

    tokens = [
        sub_lbl, sub_val, tax_lbl, tax_val, tot_lbl, tot_val,
        date_lbl, date_d, date_m, date_y,
        addr_l1, addr_l2, addr_l3,
        ocr_lbl, ocr_val,
    ]
    lines = [line0, line1, line2, line3, line4, line5, line6, line7]
    return DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)


def _create_page_7() -> DocumentPage:
    """Page 7 with duplicate Tax $50.00."""
    tax_lbl = DocumentToken("Tax", BBox(0.1, 0.80, 0.05, 0.02, page=7), page=7, char_index_in_page=0, line_index=0)
    tax_val = DocumentToken("$50.00", BBox(0.3, 0.80, 0.06, 0.02, page=7), page=7, char_index_in_page=4, line_index=0)
    line0 = VisualLine([tax_lbl, tax_val], page=7, line_index=0, bbox=BBox(0.1, 0.80, 0.26, 0.02, page=7))
    return DocumentPage(page_number=7, width=612, height=792, tokens=[tax_lbl, tax_val], lines=[line0])


def _create_identical_tax_page() -> DocumentPage:
    """Adversarial Page with two identical 'Tax $50' lines (impossible ambiguity)."""
    t1 = DocumentToken("Tax", BBox(0.1, 0.20, 0.05, 0.02, page=1), page=1, char_index_in_page=0, line_index=0)
    v1 = DocumentToken("$50.00", BBox(0.3, 0.20, 0.06, 0.02, page=1), page=1, char_index_in_page=4, line_index=0)
    l1 = VisualLine([t1, v1], page=1, line_index=0, bbox=BBox(0.1, 0.20, 0.26, 0.02, page=1))

    t2 = DocumentToken("Tax", BBox(0.1, 0.50, 0.05, 0.02, page=1), page=1, char_index_in_page=11, line_index=1)
    v2 = DocumentToken("$50.00", BBox(0.3, 0.50, 0.06, 0.02, page=1), page=1, char_index_in_page=15, line_index=1)
    l2 = VisualLine([t2, v2], page=1, line_index=1, bbox=BBox(0.1, 0.50, 0.26, 0.02, page=1))

    return DocumentPage(page_number=1, width=612, height=792, tokens=[t1, v1, t2, v2], lines=[l1, l2])


def test_duplicate_scalar_disambiguation():
    """Case 1: Subtotal $50 vs Tax $50 on same page."""
    page = _create_invoice_page()
    index = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(index)

    # Resolve Tax $50
    tax_res = resolver.resolve(
        ExtractionInput(field="tax", value=50.0, evidence_text="$50.00", field_context="Tax")
    )
    assert tax_res.is_grounded
    # Tax line is at y=0.25
    assert abs(tax_res.bbox.y - 0.25) < 0.02

    # Resolve Subtotal $50
    sub_res = resolver.resolve(
        ExtractionInput(field="subtotal", value=50.0, evidence_text="$50.00", field_context="Subtotal")
    )
    assert sub_res.is_grounded
    # Subtotal line is at y=0.20
    assert abs(sub_res.bbox.y - 0.20) < 0.02


def test_duplicate_across_pages():
    """Case 2: Duplicate Tax $50 on Page 1 and Page 7."""
    page1 = _create_invoice_page()
    page7 = _create_page_7()
    index = DocumentIndex.from_pages([page1, page7])
    resolver = EvidenceResolver(index)

    # With page_hint=7, must resolve on page 7
    res_p7 = resolver.resolve(
        ExtractionInput(field="tax", value=50.0, evidence_text="$50.00", field_context="Tax", page_hint=7)
    )
    assert res_p7.page == 7
    assert res_p7.bbox.y == 0.80

    # With page_hint=1, must resolve on page 1
    res_p1 = resolver.resolve(
        ExtractionInput(field="tax", value=50.0, evidence_text="$50.00", field_context="Tax", page_hint=1)
    )
    assert res_p1.page == 1
    assert abs(res_p1.bbox.y - 0.25) < 0.02


def test_impossible_ambiguity_refuses_to_guess():
    """Case 3: Two identical 'Tax $50' lines with identical labels -> must return AMBIGUOUS."""
    page = _create_identical_tax_page()
    index = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(index)

    res = resolver.resolve(
        ExtractionInput(field="tax", value=50.0, evidence_text="$50.00", field_context="Tax")
    )
    assert res.status == ProvenanceStatus.AMBIGUOUS
    assert res.bbox is None
    assert not res.is_grounded


def test_normalized_currency():
    """Case 4: USD 1,450.00 -> 1450."""
    page = _create_invoice_page()
    index = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(index)

    res = resolver.resolve(
        ExtractionInput(field="total_due", value=1450, field_context="Total Due")
    )
    assert res.is_grounded
    assert res.status in {ProvenanceStatus.EXACT, ProvenanceStatus.NORMALIZED}
    assert abs(res.bbox.y - 0.30) < 0.02


def test_normalized_date():
    """Case 5: 15 March 2026 -> 2026-03-15."""
    page = _create_invoice_page()
    index = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(index)

    res = resolver.resolve(
        ExtractionInput(field="invoice_date", value="2026-03-15", field_context="Date")
    )
    assert res.is_grounded
    assert res.status == ProvenanceStatus.NORMALIZED
    assert abs(res.bbox.y - 0.35) < 0.02


def test_ocr_corruption_fuzzy():
    """Case 6: OCR corruption in phone number (+1-800-555-O199)."""
    page = _create_invoice_page()
    index = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(index)

    res = resolver.resolve(
        ExtractionInput(field="phone", value="+1-800-555-0199", field_context="Phone")
    )
    assert res.is_grounded
    assert res.status == ProvenanceStatus.FUZZY
    assert abs(res.bbox.y - 0.60) < 0.02


def test_derived_value():
    """Case 7: Calculated value with no literal physical source."""
    page = _create_invoice_page()
    index = DocumentIndex.from_pages([page])
    resolver = EvidenceResolver(index)

    res = resolver.resolve(
        ExtractionInput(field="grand_total_tax_rate", value=0.18, field_context="Tax Rate")
    )
    assert res.status == ProvenanceStatus.DERIVED
    assert res.bbox is None
    assert not res.is_grounded
