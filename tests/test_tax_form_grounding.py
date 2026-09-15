"""Tests for Form 1040 tax-form structural grounding engine."""

from __future__ import annotations

from tonerhound import DocumentIndex, DocumentPage, DocumentToken, VisualLine
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.geometry.coordinates import BBox
from tonerhound.resolution.resolver import EvidenceResolver
from tonerhound.tax.grounder import (
    TaxFormGrounder,
    is_form_1040_tax_return,
)


def _build_tax_form_1040_fixture() -> DocumentIndex:
    """Build a synthetic 2-page IRS Form 1040 package."""
    # Page 1: Form 1040 Header, Names, Filing Status, Dependents, Lines 1a & 1z
    p1_tokens: list[DocumentToken] = []
    p1_lines: list[VisualLine] = []

    # Title line: "Form 1040 U.S. Individual Income Tax Return 2023"
    t_title1 = DocumentToken("Form", BBox(0.05, 0.03, 0.05, 0.02, 1), 1, 0, 0)
    t_title2 = DocumentToken("1040", BBox(0.11, 0.03, 0.06, 0.02, 1), 1, 5, 0)
    t_title3 = DocumentToken("Individual", BBox(0.20, 0.03, 0.10, 0.02, 1), 1, 10, 0)
    p1_tokens.extend([t_title1, t_title2, t_title3])
    p1_lines.append(VisualLine([t_title1, t_title2, t_title3], 1, 0, BBox(0.05, 0.03, 0.25, 0.02, 1)))

    # Row 1: Taxpayer Name: "John" "Doe" (y=0.09)
    t_tp_fn = DocumentToken("John", BBox(0.06, 0.09, 0.06, 0.02, 1), 1, 25, 1)
    t_tp_ln = DocumentToken("Doe", BBox(0.40, 0.09, 0.05, 0.02, 1), 1, 30, 1)
    p1_tokens.extend([t_tp_fn, t_tp_ln])
    p1_lines.append(VisualLine([t_tp_fn, t_tp_ln], 1, 1, BBox(0.06, 0.09, 0.39, 0.02, 1)))

    # Row 2: Spouse Name: "Jane" "Doe" (y=0.13)
    t_sp_fn = DocumentToken("Jane", BBox(0.06, 0.13, 0.06, 0.02, 1), 1, 35, 2)
    t_sp_ln = DocumentToken("Doe", BBox(0.40, 0.13, 0.05, 0.02, 1), 1, 40, 2)
    p1_tokens.extend([t_sp_fn, t_sp_ln])
    p1_lines.append(VisualLine([t_sp_fn, t_sp_ln], 1, 2, BBox(0.06, 0.13, 0.39, 0.02, 1)))

    # Filing status line (y=0.245)
    t_fs_lbl = DocumentToken("Filing", BBox(0.05, 0.245, 0.05, 0.02, 1), 1, 45, 3)
    t_fs_box = DocumentToken("[X]", BBox(0.145, 0.245, 0.015, 0.015, 1), 1, 52, 3)
    t_fs_val = DocumentToken("Married filing jointly", BBox(0.17, 0.245, 0.20, 0.02, 1), 1, 56, 3)
    p1_tokens.extend([t_fs_lbl, t_fs_box, t_fs_val])
    p1_lines.append(VisualLine([t_fs_lbl, t_fs_box, t_fs_val], 1, 3, BBox(0.05, 0.245, 0.32, 0.02, 1)))

    # Dependents section header (y=0.42)
    t_dep_hdr = DocumentToken("Dependents (see instructions):", BBox(0.05, 0.42, 0.30, 0.02, 1), 1, 80, 4)
    t_dep_ctc = DocumentToken("Child tax credit", BBox(0.75, 0.42, 0.10, 0.02, 1), 1, 115, 4)
    p1_tokens.extend([t_dep_hdr, t_dep_ctc])
    p1_lines.append(VisualLine([t_dep_hdr, t_dep_ctc], 1, 4, BBox(0.05, 0.42, 0.80, 0.02, 1)))

    # Dependent row 0 (y=0.435)
    t_dep0_name = DocumentToken("Alice Doe", BBox(0.05, 0.435, 0.15, 0.02, 1), 1, 135, 5)
    t_dep0_box = DocumentToken("[X]", BBox(0.775, 0.435, 0.015, 0.015, 1), 1, 152, 5)
    p1_tokens.extend([t_dep0_name, t_dep0_box])
    p1_lines.append(VisualLine([t_dep0_name, t_dep0_box], 1, 5, BBox(0.05, 0.435, 0.74, 0.02, 1)))

    # Line 1a: Total W-2 Wages = $125,500 (y=0.50, x=0.85)
    t_l1a_lbl = DocumentToken("1a Total amount from Form(s) W-2, box 1", BBox(0.05, 0.50, 0.45, 0.02, 1), 1, 160, 6)
    t_l1a_val = DocumentToken("125,500", BBox(0.85, 0.50, 0.08, 0.02, 1), 1, 205, 6)
    p1_tokens.extend([t_l1a_lbl, t_l1a_val])
    p1_lines.append(VisualLine([t_l1a_lbl, t_l1a_val], 1, 6, BBox(0.05, 0.50, 0.88, 0.02, 1)))

    # Line 1z: Add lines 1a through 1h = $125,500 (y=0.64, x=0.85)
    t_l1z_lbl = DocumentToken("1z Add lines 1a through 1h", BBox(0.05, 0.64, 0.35, 0.02, 1), 1, 215, 7)
    t_l1z_val = DocumentToken("125,500", BBox(0.85, 0.64, 0.08, 0.02, 1), 1, 255, 7)
    p1_tokens.extend([t_l1z_lbl, t_l1z_val])
    p1_lines.append(VisualLine([t_l1z_lbl, t_l1z_val], 1, 7, BBox(0.05, 0.64, 0.88, 0.02, 1)))

    page1 = DocumentPage(page_number=1, width=612, height=792, tokens=p1_tokens, lines=p1_lines)

    # Page 2: Lines 16, 24, 25a, 33 (Total payments)
    p2_tokens: list[DocumentToken] = []
    p2_lines: list[VisualLine] = []

    # Form 1040 Page 2 Header: "Form 1040 (2023) Page 2"
    t_p2_hdr = DocumentToken("Form 1040 (2023) Page 2", BBox(0.05, 0.02, 0.30, 0.02, 2), 2, 0, 0)
    p2_tokens.append(t_p2_hdr)
    p2_lines.append(VisualLine([t_p2_hdr], 2, 0, BBox(0.05, 0.02, 0.30, 0.02, 2)))

    # Line 16: Tax = $29,881 (y=0.05)
    t_l16_lbl = DocumentToken("16 Tax (see instructions)", BBox(0.05, 0.05, 0.30, 0.02, 2), 2, 30, 1)
    t_l16_val = DocumentToken("29,881", BBox(0.85, 0.05, 0.08, 0.02, 2), 2, 65, 1)
    p2_tokens.extend([t_l16_lbl, t_l16_val])
    p2_lines.append(VisualLine([t_l16_lbl, t_l16_val], 2, 1, BBox(0.05, 0.05, 0.88, 0.02, 2)))

    # Line 24: Total tax = $29,881 (y=0.17)
    t_l24_lbl = DocumentToken("24 Total tax. Add lines 22 and 23", BBox(0.05, 0.17, 0.40, 0.02, 2), 2, 75, 2)
    t_l24_val = DocumentToken("29,881", BBox(0.85, 0.17, 0.08, 0.02, 2), 2, 120, 2)
    p2_tokens.extend([t_l24_lbl, t_l24_val])
    p2_lines.append(VisualLine([t_l24_lbl, t_l24_val], 2, 2, BBox(0.05, 0.17, 0.88, 0.02, 2)))

    # Line 25a: Federal income tax withheld from Form(s) W-2 = $15,292 (y=0.20, x=0.70)
    t_l25a_lbl = DocumentToken("25a Federal income tax withheld from Form(s) W-2", BBox(0.05, 0.20, 0.55, 0.02, 2), 2, 130, 3)
    t_l25a_val = DocumentToken("15,292", BBox(0.68, 0.20, 0.08, 0.02, 2), 2, 190, 3)
    p2_tokens.extend([t_l25a_lbl, t_l25a_val])
    p2_lines.append(VisualLine([t_l25a_lbl, t_l25a_val], 2, 3, BBox(0.05, 0.20, 0.76, 0.02, 2)))

    # Line 33: Total payments = $15,292 (y=0.37, x=0.85)
    t_l33_lbl = DocumentToken("33 Total payments. Add lines 25d, 26, and 32", BBox(0.05, 0.37, 0.50, 0.02, 2), 2, 200, 4)
    t_l33_val = DocumentToken("15,292", BBox(0.85, 0.37, 0.08, 0.02, 2), 2, 255, 4)
    p2_tokens.extend([t_l33_lbl, t_l33_val])
    p2_lines.append(VisualLine([t_l33_lbl, t_l33_val], 2, 4, BBox(0.05, 0.37, 0.88, 0.02, 2)))

    # Presidential Campaign Fund Checkbox (y=0.22, x=0.81)
    t_pres_lbl = DocumentToken("Presidential Election Campaign fund", BBox(0.05, 0.22, 0.40, 0.02, 1), 1, 300, 8)
    t_pres_cb = DocumentToken("[]", BBox(0.81, 0.22, 0.015, 0.015, 1), 1, 345, 8)
    page1.tokens.extend([t_pres_lbl, t_pres_cb])
    page1.lines.append(VisualLine([t_pres_lbl, t_pres_cb], 1, 8, BBox(0.05, 0.22, 0.78, 0.02, 1)))

    page2 = DocumentPage(page_number=2, width=612, height=792, tokens=p2_tokens, lines=p2_lines)

    return DocumentIndex.from_pages([page1, page2])


def test_is_form_1040_signature_detection() -> None:
    """Check that is_form_1040_tax_return correctly recognizes 1040 schemas and rejects others."""
    # Positive case: Form 1040 extraction leaves
    leaves_1040 = [
        ("taxpayer_first_name_mi", "John", None, "taxpayer first name", "root"),
        ("line_1a_total_w2_wages", 125500, None, "line 1a wages", "root"),
        ("line_24_total_tax", 29881, None, "line 24 tax", "root"),
    ]
    assert is_form_1040_tax_return(leaves_1040) is True

    # Negative case: Form 13F / N-PORT / generic invoice
    leaves_invoice = [
        ("vendor_name", "ACME Corp", None, "vendor name", "root"),
        ("invoice_number", "INV-123", None, "invoice number", "root"),
        ("total_amount", 500.0, None, "total amount", "root"),
    ]
    assert is_form_1040_tax_return(leaves_invoice) is False


def test_taxpayer_vs_spouse_name_disambiguation() -> None:
    """Test that John and Jane are correctly disambiguated by row level y-coordinate."""
    idx = _build_tax_form_1040_fixture()
    resolver = EvidenceResolver(idx)
    grounder = TaxFormGrounder(idx, resolver)

    leaves = [
        ("taxpayer_first_name_mi", "John", None, "taxpayer first name", "root"),
        ("taxpayer_last_name", "Doe", None, "taxpayer last name", "root"),
        ("spouse_first_name_mi", "Jane", None, "spouse first name", "root"),
        ("spouse_last_name", "Doe", None, "spouse last name", "root"),
        ("line_1a_total_w2_wages", 125500, None, "w2 wages", "root"),
    ]
    res = grounder.ground(leaves, {})
    citations = {c["field_path"]: c for c in res["field_citations"]}

    assert "taxpayer_first_name_mi" in citations
    assert citations["taxpayer_first_name_mi"]["page"] == 1
    assert abs(citations["taxpayer_first_name_mi"]["bbox"][1] - 0.09) < 0.02
    assert citations["taxpayer_first_name_mi"]["reference_text"] == "John"

    assert "spouse_first_name_mi" in citations
    assert citations["spouse_first_name_mi"]["page"] == 1
    assert abs(citations["spouse_first_name_mi"]["bbox"][1] - 0.13) < 0.02
    assert citations["spouse_first_name_mi"]["reference_text"] == "Jane"


def test_duplicate_amount_line_disambiguation() -> None:
    """Test that duplicate amounts (125,500 on 1a vs 1z; 29,881 on 16 vs 24; 15,292 on 25a vs 33)

    are accurately anchored to their respective line positions rather than defaulting to the first match.
    """
    idx = _build_tax_form_1040_fixture()
    resolver = EvidenceResolver(idx)
    grounder = TaxFormGrounder(idx, resolver)

    leaves = [
        ("taxpayer_first_name_mi", "John", None, "taxpayer first name", "root"),
        ("line_1a_total_w2_wages", 125500, None, "w2 wages", "root"),
        ("line_1z_total_wages", 125500, None, "total wages", "root"),
        ("line_16_tax", 29881, None, "line 16 tax", "root"),
        ("line_24_total_tax", 29881, None, "line 24 tax", "root"),
        ("line_25a_federal_income_tax_withheld_w2", 15292, None, "line 25a withholding", "root"),
        ("line_33_total_payments", 15292, None, "line 33 total payments", "root"),
    ]

    res = grounder.ground(leaves, {})
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # Page 1: 1a (y~0.50) vs 1z (y~0.64)
    assert "line_1a_total_w2_wages" in citations
    assert citations["line_1a_total_w2_wages"]["page"] == 1
    assert abs(citations["line_1a_total_w2_wages"]["bbox"][1] - 0.50) < 0.03

    assert "line_1z_total_wages" in citations
    assert citations["line_1z_total_wages"]["page"] == 1
    assert abs(citations["line_1z_total_wages"]["bbox"][1] - 0.64) < 0.03

    # Page 2: 16 (y~0.05) vs 24 (y~0.17)
    assert "line_16_tax" in citations
    assert citations["line_16_tax"]["page"] == 2
    assert abs(citations["line_16_tax"]["bbox"][1] - 0.05) < 0.03

    assert "line_24_total_tax" in citations
    assert citations["line_24_total_tax"]["page"] == 2
    assert abs(citations["line_24_total_tax"]["bbox"][1] - 0.17) < 0.03

    # Page 2: 25a (y~0.20, x~0.68) vs 33 (y~0.37, x~0.85)
    assert "line_25a_federal_income_tax_withheld_w2" in citations
    assert citations["line_25a_federal_income_tax_withheld_w2"]["page"] == 2
    assert abs(citations["line_25a_federal_income_tax_withheld_w2"]["bbox"][1] - 0.20) < 0.03

    assert "line_33_total_payments" in citations
    assert citations["line_33_total_payments"]["page"] == 2
    assert abs(citations["line_33_total_payments"]["bbox"][1] - 0.37) < 0.03


def test_checkbox_spatial_grounding() -> None:
    """Test presidential campaign fund and dependents checkboxes."""
    idx = _build_tax_form_1040_fixture()
    resolver = EvidenceResolver(idx)
    grounder = TaxFormGrounder(idx, resolver)

    leaves = [
        ("taxpayer_first_name_mi", "John", None, "name", "root"),
        ("line_1a_total_w2_wages", 125500, None, "w2", "root"),
        ("line_24_total_tax", 29881, None, "tax", "root"),
        ("presidential_campaign_you_box", False, None, "presidential fund you", "root"),
        ("dependents[0].child_tax_credit_box", True, None, "dependent child credit", "dependents[0]"),
    ]

    res = grounder.ground(leaves, {})
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # Presidential checkbox: x~0.81, y~0.22, page 1
    assert "presidential_campaign_you_box" in citations
    cb_pres = citations["presidential_campaign_you_box"]
    assert cb_pres["page"] == 1
    assert abs(cb_pres["bbox"][0] + cb_pres["bbox"][2] / 2.0 - 0.81) < 0.03
    assert abs(cb_pres["bbox"][1] + cb_pres["bbox"][3] / 2.0 - 0.22) < 0.03

    # Dependent 0 child tax credit box: x~0.778, y~dep_header_y + 0.0145
    assert "dependents[0].child_tax_credit_box" in citations
    cb_dep = citations["dependents[0].child_tax_credit_box"]
    assert cb_dep["page"] == 1
    assert abs(cb_dep["bbox"][0] + cb_dep["bbox"][2] / 2.0 - 0.778) < 0.03


def test_adapter_end_to_end_dispatch() -> None:
    """Test end-to-end integration of Form 1040 via ExtractBenchAdapter."""
    idx = _build_tax_form_1040_fixture()
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "taxpayer_first_name_mi": "John",
        "taxpayer_last_name": "Doe",
        "spouse_first_name_mi": "Jane",
        "spouse_last_name": "Doe",
        "filing_status": "Married filing jointly",
        "line_1a_total_w2_wages": 125500,
        "line_1z_total_wages": 125500,
        "line_16_tax": 29881,
        "line_24_total_tax": 29881,
        "line_25a_federal_income_tax_withheld_w2": 15292,
        "line_33_total_payments": 15292,
    }

    res = adapter.ground_extracted_data(payload)
    assert res["task_type"] == "extract"
    citations = {c["field_path"]: c for c in res["field_citations"]}

    assert len(citations) == 11
    # Check page distribution
    assert citations["line_1a_total_w2_wages"]["page"] == 1
    assert citations["line_1z_total_wages"]["page"] == 1
    assert citations["line_16_tax"]["page"] == 2
    assert citations["line_24_total_tax"]["page"] == 2
    assert citations["line_25a_federal_income_tax_withheld_w2"]["page"] == 2
    assert citations["line_33_total_payments"]["page"] == 2
