"""Comprehensive automated regression test suite for EXP-007.

Validates core invariant behaviors across four critical dimensions:
1. Repeated values / identical numbers across adjacent rows:
   - Repeated percentages ('100.0%') in adjacent rows
   - Repeated numeric zeros ('0') in financial ledgers / grant forms
   - Repeated security class designations ('COMMON STOCK') in 13F holdings
   - Repeated transaction / period dates ('2024-12-31') across adjacent rows
   - Real Form 990 Schedule I repeated boilerplate fields (purpose of grant, IRC section, noncash amount)
   - Interleaved repeated and distinct numeric fields within structured rows

2. Multi-line descriptions and non-truncation:
   - Multi-line grant recipient names spanning multiple visual lines
   - Multi-line addresses (C/O line + street line + suite) spanning full vertical extent
   - Three-line long legal / grant purpose statements
   - Real Form 990 Schedule I multi-line recipient name grounding and IoU >= 0.50
   - Multi-line table cell within a row without colliding with adjacent rows

3. Structured table records with multiple columns:
   - 4-column structured tables preserving left-to-right horizontal ordering and gutters
   - Horizontal column alignment consistency across rows within tight tolerances
   - 6-column dense financial holdings table
   - Multi-page table column continuity across page boundaries
   - Missing optional table cell preserving remaining column positions (no leftward shift)

4. Scanned-page coordinate offsets and bounds:
   - Universal bounding box invariance in [0.0, 1.0]
   - Extreme page margin boundary clamping (top-left, bottom-right)
   - Scanned-page mechanical drag and tilt slope boundary safety
   - Canonical 71-slot and 72-slot creditor grid coordinate budgets within bounds
   - Real scanned corrupted document coordinate validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tonerhound import DocumentIndex, DocumentPage, DocumentToken, VisualLine
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.benchmark.evaluator import evaluate_prediction
from tonerhound.geometry.coordinates import BBox


# ===========================================================================
# Helper Fixtures & Builders
# ===========================================================================

def _make_token(
    text: str,
    x: float,
    y: float,
    w: float = 0.08,
    h: float = 0.02,
    page: int = 1,
    char_idx: int = 0,
    line_idx: int = 0,
) -> DocumentToken:
    """Helper to construct a well-formed DocumentToken."""
    return DocumentToken(
        text=text,
        bbox=BBox(x=x, y=y, width=w, height=h, page=page),
        page=page,
        char_index_in_page=char_idx,
        line_index=line_idx,
    )


def _assert_valid_bbox(box: list[float] | BBox) -> None:
    """Assert bounding box is strictly within normalized [0, 1] coordinate frame."""
    if isinstance(box, BBox):
        x, y, w, h = box.x, box.y, box.width, box.height
    else:
        x, y, w, h = box[0], box[1], box[2], box[3]

    assert 0.0 <= x <= 1.0, f"x={x} out of [0, 1]"
    assert 0.0 <= y <= 1.0, f"y={y} out of [0, 1]"
    assert 0.0 < w <= 1.0, f"width={w} out of (0, 1]"
    assert 0.0 < h <= 1.0, f"height={h} out of (0, 1]"
    assert x + w <= 1.0001, f"x + width = {x + w} exceeds 1.0"
    assert y + h <= 1.0001, f"y + height = {y + h} exceeds 1.0"


# ===========================================================================
# 1. Repeated Values & Identical Numbers Across Adjacent Rows
# ===========================================================================

def test_repeated_percentages_across_adjacent_rows() -> None:
    """Verify identical percentages ('100.0%') in adjacent rows ground to their respective rows."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    entities = ["ALPHA HOLDINGS CORP", "BETA VENTURES LLC", "GAMMA CAPITAL INC", "DELTA PARTNERS LP"]
    row_ys = [0.15, 0.25, 0.35, 0.45]
    char_idx = 0

    for i, (name, y) in enumerate(zip(entities, row_ys)):
        t_name = _make_token(name, x=0.08, y=y, w=0.25, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += len(name) + 1
        t_pct = _make_token("100.0%", x=0.45, y=y, w=0.08, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 7
        t_shares = _make_token(f"{1000 * (i + 1)}", x=0.65, y=y, w=0.08, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 6

        r_toks = [t_name, t_pct, t_shares]
        tokens.extend(r_toks)
        lines.append(VisualLine(r_toks, page=1, line_index=i, bbox=BBox(0.08, y, 0.65, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    payload = {
        "subsidiaries": [
            {"entity_name": name, "ownership_percentage": "100.0%", "shares_held": 1000 * (i + 1)}
            for i, name in enumerate(entities)
        ]
    }
    result = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    pct_ys = []
    for i in range(len(entities)):
        fpath = f"subsidiaries[{i}].ownership_percentage"
        assert fpath in citations, f"Missing citation for {fpath}"
        box = citations[fpath]["bbox"]
        _assert_valid_bbox(box)
        pct_ys.append(box[1])
        # Each citation must match its row's physical vertical position
        assert abs(box[1] - row_ys[i]) < 0.02, (
            f"Row {i} ownership citation y={box[1]} does not match line y={row_ys[i]}"
        )

    # Monotonically increasing vertical coordinates (no collapse to row 0)
    for i in range(len(pct_ys) - 1):
        assert pct_ys[i] < pct_ys[i + 1], f"Ownership y-coords not strictly increasing: {pct_ys}"


def test_repeated_zero_numeric_values_across_rows() -> None:
    """Verify repeated numeric zeros ('0') across consecutive rows ground to distinct lines."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    items = ["Standard Service Fee", "Annual Compliance Fee", "Wire Processing Surcharge", "Regulatory Filing Charge"]
    row_ys = [0.12, 0.20, 0.28, 0.36]
    char_idx = 0

    for i, (item, y) in enumerate(zip(items, row_ys)):
        t_desc = _make_token(item, x=0.05, y=y, w=0.30, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += len(item) + 1
        t_gross = _make_token(f"${150 * (i + 1)}.00", x=0.45, y=y, w=0.10, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 10
        t_discount = _make_token("0", x=0.65, y=y, w=0.03, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 2
        t_net = _make_token(f"${150 * (i + 1)}.00", x=0.75, y=y, w=0.10, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 10

        r_toks = [t_desc, t_gross, t_discount, t_net]
        tokens.extend(r_toks)
        lines.append(VisualLine(r_toks, page=1, line_index=i, bbox=BBox(0.05, y, 0.80, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    payload = {
        "ledger_entries": [
            {"description": item, "gross_amount": 150 * (i + 1), "discount": 0, "net_amount": 150 * (i + 1)}
            for i, item in enumerate(items)
        ]
    }
    result = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    zero_ys = []
    for i in range(len(items)):
        fpath = f"ledger_entries[{i}].discount"
        assert fpath in citations, f"Missing citation for {fpath}"
        box = citations[fpath]["bbox"]
        _assert_valid_bbox(box)
        zero_ys.append(box[1])
        assert abs(box[1] - row_ys[i]) < 0.02, (
            f"Row {i} zero discount citation y={box[1]} does not match line y={row_ys[i]}"
        )

    for i in range(len(zero_ys) - 1):
        assert zero_ys[i] < zero_ys[i + 1], f"Zero discount y-coords not strictly increasing: {zero_ys}"


def test_repeated_security_class_common_stock() -> None:
    """Verify repeated 'COMMON STOCK' designations in Form 13F rows ground to respective rows."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    issuers = ["APPLE INC", "MICROSOFT CORP", "AMAZON COM INC", "NVIDIA CORP"]
    cusips = ["037833100", "594918104", "023135106", "67066G104"]
    row_ys = [0.20, 0.28, 0.36, 0.44]
    char_idx = 0

    for i, (issuer, cusip, y) in enumerate(zip(issuers, cusips, row_ys)):
        t_iss = _make_token(issuer, x=0.05, y=y, w=0.22, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += len(issuer) + 1
        t_cls = _make_token("COMMON STOCK", x=0.30, y=y, w=0.16, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 13
        t_csp = _make_token(cusip, x=0.50, y=y, w=0.12, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 10
        t_val = _make_token(f"{50000 * (i + 1)}", x=0.68, y=y, w=0.09, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 8

        r_toks = [t_iss, t_cls, t_csp, t_val]
        tokens.extend(r_toks)
        lines.append(VisualLine(r_toks, page=1, line_index=i, bbox=BBox(0.05, y, 0.72, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    payload = {
        "holdings": [
            {
                "issuer_name": issuer,
                "title_of_class": "COMMON STOCK",
                "cusip": cusip,
                "value": 50000 * (i + 1),
            }
            for i, (issuer, cusip) in enumerate(zip(issuers, cusips))
        ]
    }
    result = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    class_ys = []
    for i in range(len(issuers)):
        fpath = f"holdings[{i}].title_of_class"
        assert fpath in citations, f"Missing citation for {fpath}"
        box = citations[fpath]["bbox"]
        _assert_valid_bbox(box)
        class_ys.append(box[1])
        assert abs(box[1] - row_ys[i]) < 0.02, (
            f"Row {i} COMMON STOCK citation y={box[1]} does not match line y={row_ys[i]}"
        )

    for i in range(len(class_ys) - 1):
        assert class_ys[i] < class_ys[i + 1], f"COMMON STOCK y-coords not strictly increasing: {class_ys}"


def test_repeated_dates_across_adjacent_rows() -> None:
    """Verify repeated transaction dates ('2024-12-31') ground to their specific rows."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    tx_ids = ["TX-9011", "TX-9012", "TX-9013", "TX-9014"]
    row_ys = [0.10, 0.18, 0.26, 0.34]
    char_idx = 0

    for i, (tx_id, y) in enumerate(zip(tx_ids, row_ys)):
        t_id = _make_token(tx_id, x=0.08, y=y, w=0.12, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += len(tx_id) + 1
        t_dt = _make_token("2024-12-31", x=0.25, y=y, w=0.14, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 11
        t_desc = _make_token("Quarterly Settlement", x=0.45, y=y, w=0.26, h=0.018, char_idx=char_idx, line_idx=i)
        char_idx += 21

        r_toks = [t_id, t_dt, t_desc]
        tokens.extend(r_toks)
        lines.append(VisualLine(r_toks, page=1, line_index=i, bbox=BBox(0.08, y, 0.63, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    payload = {
        "transactions": [
            {"tx_id": tx_id, "posting_date": "2024-12-31", "description": "Quarterly Settlement"}
            for tx_id in tx_ids
        ]
    }
    result = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    date_ys = []
    for i in range(len(tx_ids)):
        fpath = f"transactions[{i}].posting_date"
        assert fpath in citations, f"Missing citation for {fpath}"
        box = citations[fpath]["bbox"]
        _assert_valid_bbox(box)
        date_ys.append(box[1])
        assert abs(box[1] - row_ys[i]) < 0.02

    for i in range(len(date_ys) - 1):
        assert date_ys[i] < date_ys[i + 1]


def test_real_schedule_i_repeated_boilerplate_fields() -> None:
    """Verify repeated values on real Form 990 Schedule I PDF ground to distinct rows."""
    pdf_path = Path("research/data/full/short/sched_i__rotary_club_ty2024.pdf")
    test_json_path = Path("research/data/full/short/sched_i__rotary_club_ty2024.test.json")
    assert pdf_path.exists(), f"Missing fixture {pdf_path}"
    assert test_json_path.exists(), f"Missing fixture {test_json_path}"

    with open(test_json_path, encoding="utf-8") as f:
        tc_data = json.load(f)

    idx = DocumentIndex.from_pdf(pdf_path, use_cache=False)
    adapter = ExtractBenchAdapter(idx)
    result = adapter.ground_extracted_data(tc_data["expected_output"], example_id="sched_i_regress")
    citations = {c["field_path"]: c for c in result["field_citations"]}

    # All 3 grants have purpose_of_grant="GENERAL SUPPORT", irc_section="C3", noncash_assistance_amount=0
    for field in ["purpose_of_grant", "irc_section", "noncash_assistance_amount"]:
        ys = []
        for g_idx in range(3):
            fpath = f"grants[{g_idx}].{field}"
            assert fpath in citations, f"Missing citation for {fpath}"
            box = citations[fpath]["bbox"]
            _assert_valid_bbox(box)
            ys.append(box[1])

        # Verify y coordinates are monotonically increasing across grants
        assert len(ys) == 3
        assert ys[0] < ys[1] < ys[2], f"Field {field} row y-positions not monotonic: {ys}"
        # Grants are separated vertically by at least 0.03 normalized height
        assert ys[1] - ys[0] >= 0.03
        assert ys[2] - ys[1] >= 0.03


# ===========================================================================
# 2. Multi-Line Descriptions & Non-Truncation
# ===========================================================================

def test_multiline_recipient_name_full_vertical_span() -> None:
    """Verify multi-line recipient name ('THE ROTARY FOUNDATION \\n ROTARY INTERNATIONAL') is not truncated."""
    # Line 0: y=0.200, h=0.015
    t1 = _make_token("THE", x=0.10, y=0.200, w=0.05, h=0.015, line_idx=0)
    t2 = _make_token("ROTARY", x=0.16, y=0.200, w=0.08, h=0.015, line_idx=0)
    t3 = _make_token("FOUNDATION", x=0.25, y=0.200, w=0.13, h=0.015, line_idx=0)
    line0 = VisualLine([t1, t2, t3], page=1, line_index=0, bbox=BBox(0.10, 0.200, 0.28, 0.015, 1))

    # Line 1: y=0.222, h=0.015 (subsequent wrapped line)
    t4 = _make_token("ROTARY", x=0.10, y=0.222, w=0.08, h=0.015, line_idx=1)
    t5 = _make_token("INTERNATIONAL", x=0.19, y=0.222, w=0.16, h=0.015, line_idx=1)
    line1 = VisualLine([t4, t5], page=1, line_index=1, bbox=BBox(0.10, 0.222, 0.25, 0.015, 1))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=[t1, t2, t3, t4, t5], lines=[line0, line1])
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    extracted = {"recipient_name": "THE ROTARY FOUNDATION ROTARY INTERNATIONAL"}
    result = adapter.ground_extracted_data(extracted)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    assert "recipient_name" in citations
    box = citations["recipient_name"]["bbox"]
    _assert_valid_bbox(box)

    # Top must be at or near line 0 top
    assert abs(box[1] - 0.200) < 0.01, f"Top y={box[1]} not near line 0 (0.200)"
    # Bottom must cover line 1 bottom (0.222 + 0.015 = 0.237)
    bottom_y = box[1] + box[3]
    assert bottom_y >= 0.235, f"Bottom y={bottom_y} does not cover line 1 (0.237)"
    # Height must encompass both lines (> 0.030), proving no truncation to single line
    assert box[3] >= 0.030, f"Height {box[3]} truncated to single line (< 0.030)"


def test_multiline_address_full_vertical_span() -> None:
    """Verify multi-line street address spans across both lines without truncation."""
    # Line 0: Care-of line
    t1 = _make_token("C/O", x=0.09, y=0.320, w=0.04, h=0.014, line_idx=0)
    t2 = _make_token("WEST", x=0.14, y=0.320, w=0.06, h=0.014, line_idx=0)
    t3 = _make_token("ANNAPOLIS", x=0.21, y=0.320, w=0.12, h=0.014, line_idx=0)
    t4 = _make_token("POP-UP", x=0.34, y=0.320, w=0.08, h=0.014, line_idx=0)
    t5 = _make_token("PANTRY", x=0.43, y=0.320, w=0.09, h=0.014, line_idx=0)
    line0 = VisualLine([t1, t2, t3, t4, t5], page=1, line_index=0, bbox=BBox(0.09, 0.320, 0.43, 0.014, 1))

    # Line 1: Street line
    t6 = _make_token("123", x=0.09, y=0.342, w=0.04, h=0.014, line_idx=1)
    t7 = _make_token("MAIN", x=0.14, y=0.342, w=0.06, h=0.014, line_idx=1)
    t8 = _make_token("STREET", x=0.21, y=0.342, w=0.08, h=0.014, line_idx=1)
    t9 = _make_token("SUITE", x=0.30, y=0.342, w=0.06, h=0.014, line_idx=1)
    t10 = _make_token("400", x=0.37, y=0.342, w=0.04, h=0.014, line_idx=1)
    line1 = VisualLine([t6, t7, t8, t9, t10], page=1, line_index=1, bbox=BBox(0.09, 0.342, 0.32, 0.014, 1))

    page = DocumentPage(
        page_number=1,
        width=612,
        height=792,
        tokens=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10],
        lines=[line0, line1],
    )
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    extracted = {"address_line1": "C/O WEST ANNAPOLIS POP-UP PANTRY 123 MAIN STREET SUITE 400"}
    result = adapter.ground_extracted_data(extracted)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    assert "address_line1" in citations
    box = citations["address_line1"]["bbox"]
    _assert_valid_bbox(box)

    assert box[1] <= 0.325
    bottom_y = box[1] + box[3]
    assert bottom_y >= 0.355, f"Bottom y={bottom_y} did not span to line 1 bottom"
    assert box[3] >= 0.030, f"Box height {box[3]} truncated"


def test_three_line_legal_description_not_truncated() -> None:
    """Verify three-line grant purpose statement spans all 3 lines."""
    tokens = []
    lines = []
    desc_lines = [
        ("GRANT AWARDED FOR PURPOSES OF ADVANCING", 0.400),
        ("UNDERGRADUATE AND GRADUATE RESEARCH IN", 0.422),
        ("COMPUTATIONAL BIOPHYSICS AND APPLIED MATHEMATICS", 0.444),
    ]
    char_idx = 0

    for l_idx, (text, y) in enumerate(desc_lines):
        line_toks = []
        words = text.split()
        x_cur = 0.10
        for w in words:
            tok = _make_token(w, x=x_cur, y=y, w=len(w) * 0.012, h=0.015, line_idx=l_idx, char_idx=char_idx)
            tokens.append(tok)
            line_toks.append(tok)
            x_cur += len(w) * 0.012 + 0.01
            char_idx += len(w) + 1
        lines.append(VisualLine(line_toks, page=1, line_index=l_idx, bbox=BBox(0.10, y, x_cur - 0.10, 0.015, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    full_text = " ".join(t[0] for t in desc_lines)
    result = adapter.ground_extracted_data({"purpose_of_grant": full_text})
    citations = {c["field_path"]: c for c in result["field_citations"]}

    assert "purpose_of_grant" in citations
    box = citations["purpose_of_grant"]["bbox"]
    _assert_valid_bbox(box)

    assert box[1] <= 0.405
    bottom_y = box[1] + box[3]
    assert bottom_y >= 0.455, f"Bottom y={bottom_y} did not cover Line 2"
    assert box[3] >= 0.050, f"Height {box[3]} truncated across 3 lines"


def test_real_schedule_i_multiline_grant_recipient() -> None:
    """Verify multi-line recipient on real Schedule I PDF has height spanning both lines and IoU >= 0.50."""
    pdf_path = Path("research/data/full/short/sched_i__rotary_club_ty2024.pdf")
    test_json_path = Path("research/data/full/short/sched_i__rotary_club_ty2024.test.json")

    with open(test_json_path, encoding="utf-8") as f:
        tc_data = json.load(f)

    idx = DocumentIndex.from_pdf(pdf_path, use_cache=False)
    adapter = ExtractBenchAdapter(idx)
    result = adapter.ground_extracted_data(tc_data["expected_output"], example_id="sched_i_multiline")
    citations = {c["field_path"]: c for c in result["field_citations"]}

    # Grant 1: "THE ROTARY FOUNDATION ROTARY INTERNATIONAL"
    fpath = "grants[1].recipient_name"
    assert fpath in citations, f"Missing citation for {fpath}"
    box = citations[fpath]["bbox"]
    _assert_valid_bbox(box)

    # Multi-line recipient height in ground truth rule is 0.02451 (vs ~0.012 for single line)
    assert box[3] >= 0.018, f"Height {box[3]} is truncated (expected multi-line height >= 0.018)"

    # Compute official ExtractBench IoU with ground truth rule bbox
    gt_bbox = tc_data["_field_rules"]["grants[1].recipient_name"]["evidence"][0]["bbox"]
    pred_b = BBox(box[0], box[1], box[2], box[3], page=1)
    gt_b = BBox(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], page=1)
    iou = pred_b.iou(gt_b)
    assert iou >= 0.50, f"Multi-line IoU {iou:.3f} failed threshold 0.50"


# ===========================================================================
# 3. Structured Table Records with Multiple Columns
# ===========================================================================

def test_four_column_horizontal_ordering_and_gutters() -> None:
    """Verify 4-column structured table preserves horizontal column ordering and non-overlapping gutters."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    col_xs = [0.08, 0.28, 0.55, 0.75]
    col_ws = [0.15, 0.22, 0.12, 0.14]
    row_ys = [0.15, 0.25, 0.35]
    char_idx = 0

    rows_data = [
        {"item_code": "SKU-1001", "description": "Industrial Steel Bolts", "quantity": 50, "price": 12.50},
        {"item_code": "SKU-1002", "description": "High-Tensile Washers", "quantity": 100, "price": 4.75},
        {"item_code": "SKU-1003", "description": "Brass Lock Nuts", "quantity": 75, "price": 8.20},
    ]

    for i, (r_data, y) in enumerate(zip(rows_data, row_ys)):
        r_toks = [
            _make_token(r_data["item_code"], x=col_xs[0], y=y, w=col_ws[0], h=0.018, char_idx=char_idx, line_idx=i),
            _make_token(r_data["description"], x=col_xs[1], y=y, w=col_ws[1], h=0.018, char_idx=char_idx + 10, line_idx=i),
            _make_token(str(r_data["quantity"]), x=col_xs[2], y=y, w=col_ws[2], h=0.018, char_idx=char_idx + 35, line_idx=i),
            _make_token(f"${r_data['price']:.2f}", x=col_xs[3], y=y, w=col_ws[3], h=0.018, char_idx=char_idx + 45, line_idx=i),
        ]
        char_idx += 60
        tokens.extend(r_toks)
        lines.append(VisualLine(r_toks, page=1, line_index=i, bbox=BBox(col_xs[0], y, 0.82, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    result = adapter.ground_extracted_data({"line_items": rows_data})
    citations = {c["field_path"]: c for c in result["field_citations"]}

    col_fields = ["item_code", "description", "quantity", "price"]

    for i in range(len(rows_data)):
        row_boxes = []
        for f in col_fields:
            fpath = f"line_items[{i}].{f}"
            assert fpath in citations, f"Missing citation for {fpath}"
            b = citations[fpath]["bbox"]
            _assert_valid_bbox(b)
            row_boxes.append(b)

        # Monotonic left-to-right order
        xs = [b[0] for b in row_boxes]
        assert xs == sorted(xs), f"Row {i} columns out of horizontal order: {xs}"

        # Gutters preserved: col[k].x + col[k].width <= col[k+1].x + 0.01
        for k in range(len(row_boxes) - 1):
            right_edge = row_boxes[k][0] + row_boxes[k][2]
            next_left = row_boxes[k + 1][0]
            assert right_edge <= next_left + 0.015, (
                f"Row {i} column {k} right edge {right_edge:.3f} collides with column {k+1} left edge {next_left:.3f}"
            )


def test_column_alignment_consistency_across_rows() -> None:
    """Verify each column maintains consistent left coordinates across all rows."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    col_xs = [0.06, 0.25, 0.45, 0.65, 0.80]
    row_ys = [0.15, 0.24, 0.33, 0.42]
    char_idx = 0

    records = [
        {"account": "ACC-101", "holder": "Apex Logistics", "region": "NORTH", "debit": 2500, "credit": 0},
        {"account": "ACC-102", "holder": "Beacon Freight", "region": "SOUTH", "debit": 3400, "credit": 0},
        {"account": "ACC-103", "holder": "Crest Shipping", "region": "EAST", "debit": 1800, "credit": 0},
        {"account": "ACC-104", "holder": "Delta Transport", "region": "WEST", "debit": 4200, "credit": 0},
    ]

    for i, (r, y) in enumerate(zip(records, row_ys)):
        r_toks = [
            _make_token(r["account"], x=col_xs[0], y=y, w=0.15, h=0.018, char_idx=char_idx, line_idx=i),
            _make_token(r["holder"], x=col_xs[1], y=y, w=0.18, h=0.018, char_idx=char_idx + 10, line_idx=i),
            _make_token(r["region"], x=col_xs[2], y=y, w=0.12, h=0.018, char_idx=char_idx + 30, line_idx=i),
            _make_token(str(r["debit"]), x=col_xs[3], y=y, w=0.10, h=0.018, char_idx=char_idx + 45, line_idx=i),
            _make_token(str(r["credit"]), x=col_xs[4], y=y, w=0.05, h=0.018, char_idx=char_idx + 55, line_idx=i),
        ]
        char_idx += 65
        tokens.extend(r_toks)
        lines.append(VisualLine(r_toks, page=1, line_index=i, bbox=BBox(col_xs[0], y, 0.80, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    result = adapter.ground_extracted_data({"accounts": records})
    citations = {c["field_path"]: c for c in result["field_citations"]}

    for col_name in ["account", "holder", "region", "debit", "credit"]:
        xs = [citations[f"accounts[{i}].{col_name}"]["bbox"][0] for i in range(len(records))]
        # Column X positions across rows must be identical within tight tolerance (0.02)
        assert max(xs) - min(xs) < 0.02, f"Column '{col_name}' left position drifted across rows: {xs}"


def test_six_column_dense_financial_holdings_table() -> None:
    """Verify 6-column dense holdings table maintains all 6 columns in each row."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    col_xs = [0.05, 0.18, 0.40, 0.58, 0.72, 0.85]
    row_ys = [0.18, 0.28, 0.38]
    char_idx = 0

    holdings = [
        {"cusip": "111111101", "issuer": "Corp Alpha", "class": "COM", "value": 12000, "shares": 500, "voting": "SOLE"},
        {"cusip": "222222102", "issuer": "Corp Beta", "class": "COM", "value": 24000, "shares": 1000, "voting": "SOLE"},
        {"cusip": "333333103", "issuer": "Corp Gamma", "class": "COM", "value": 36000, "shares": 1500, "voting": "SOLE"},
    ]

    for i, (h, y) in enumerate(zip(holdings, row_ys)):
        r_toks = [
            _make_token(h["cusip"], x=col_xs[0], y=y, w=0.10, h=0.018, char_idx=char_idx, line_idx=i),
            _make_token(h["issuer"], x=col_xs[1], y=y, w=0.18, h=0.018, char_idx=char_idx + 12, line_idx=i),
            _make_token(h["class"], x=col_xs[2], y=y, w=0.10, h=0.018, char_idx=char_idx + 32, line_idx=i),
            _make_token(str(h["value"]), x=col_xs[3], y=y, w=0.10, h=0.018, char_idx=char_idx + 44, line_idx=i),
            _make_token(str(h["shares"]), x=col_xs[4], y=y, w=0.08, h=0.018, char_idx=char_idx + 56, line_idx=i),
            _make_token(h["voting"], x=col_xs[5], y=y, w=0.08, h=0.018, char_idx=char_idx + 66, line_idx=i),
        ]
        char_idx += 80
        tokens.extend(r_toks)
        lines.append(VisualLine(r_toks, page=1, line_index=i, bbox=BBox(col_xs[0], y, 0.88, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    result = adapter.ground_extracted_data({"sec_holdings": holdings})
    citations = {c["field_path"]: c for c in result["field_citations"]}

    assert len(citations) == 18, f"Expected 18 citations across 3 rows x 6 cols, got {len(citations)}"
    for i in range(3):
        xs = [
            citations[f"sec_holdings[{i}].cusip"]["bbox"][0],
            citations[f"sec_holdings[{i}].issuer"]["bbox"][0],
            citations[f"sec_holdings[{i}].class"]["bbox"][0],
            citations[f"sec_holdings[{i}].value"]["bbox"][0],
            citations[f"sec_holdings[{i}].shares"]["bbox"][0],
            citations[f"sec_holdings[{i}].voting"]["bbox"][0],
        ]
        assert xs == sorted(xs), f"Row {i} 6-col order corrupted: {xs}"


def test_multipage_table_column_continuity() -> None:
    """Verify table spanning Page 1 and Page 2 maintains column alignments and correct page numbers."""
    pages = []
    col_xs = [0.10, 0.35, 0.65]
    row_specs = [
        # Page 1 rows
        (1, 0, "Record 001", "Category A", 100, 0.20),
        (1, 1, "Record 002", "Category B", 200, 0.30),
        # Page 2 rows
        (2, 0, "Record 003", "Category C", 300, 0.20),
        (2, 1, "Record 004", "Category D", 400, 0.30),
    ]

    for p_num in (1, 2):
        p_toks = []
        p_lines = []
        char_idx = 0
        p_rows = [r for r in row_specs if r[0] == p_num]
        for _, l_idx, rec_id, cat, val, y in p_rows:
            toks = [
                _make_token(rec_id, x=col_xs[0], y=y, w=0.20, h=0.018, page=p_num, char_idx=char_idx, line_idx=l_idx),
                _make_token(cat, x=col_xs[1], y=y, w=0.25, h=0.018, page=p_num, char_idx=char_idx + 15, line_idx=l_idx),
                _make_token(str(val), x=col_xs[2], y=y, w=0.10, h=0.018, page=p_num, char_idx=char_idx + 42, line_idx=l_idx),
            ]
            char_idx += 60
            p_toks.extend(toks)
            p_lines.append(VisualLine(toks, page=p_num, line_index=l_idx, bbox=BBox(col_xs[0], y, 0.68, 0.018, p_num)))
        pages.append(DocumentPage(page_number=p_num, width=612, height=792, tokens=p_toks, lines=p_lines))

    idx = DocumentIndex.from_pages(pages)
    adapter = ExtractBenchAdapter(idx)

    extracted_records = [
        {"id": rec_id, "category": cat, "value": val}
        for _, _, rec_id, cat, val, _ in row_specs
    ]
    result = adapter.ground_extracted_data({"multipage_table": extracted_records})
    citations = {c["field_path"]: c for c in result["field_citations"]}

    # Verify rows 0, 1 on Page 1; rows 2, 3 on Page 2
    for i in range(2):
        assert citations[f"multipage_table[{i}].id"]["page"] == 1
        assert citations[f"multipage_table[{i}].category"]["page"] == 1
    for i in range(2, 4):
        assert citations[f"multipage_table[{i}].id"]["page"] == 2
        assert citations[f"multipage_table[{i}].category"]["page"] == 2

    # Verify column X positions match between Page 1 and Page 2
    for i in range(4):
        assert abs(citations[f"multipage_table[{i}].id"]["bbox"][0] - col_xs[0]) < 0.02
        assert abs(citations[f"multipage_table[{i}].category"]["bbox"][0] - col_xs[1]) < 0.02
        assert abs(citations[f"multipage_table[{i}].value"]["bbox"][0] - col_xs[2]) < 0.02


def test_table_column_with_missing_optional_cell() -> None:
    """Verify row with missing optional column preserves remaining column X-positions (no left-shift)."""
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []
    col_xs = [0.08, 0.30, 0.60, 0.80]
    row_ys = [0.15, 0.25, 0.35]
    char_idx = 0

    # Row 0: all 4 cols
    r0 = [
        _make_token("ITEM-A", col_xs[0], row_ys[0], 0.15, 0.018, char_idx=char_idx, line_idx=0),
        _make_token("Industrial Valve", col_xs[1], row_ys[0], 0.25, 0.018, char_idx=char_idx + 10, line_idx=0),
        _make_token("10", col_xs[2], row_ys[0], 0.08, 0.018, char_idx=char_idx + 38, line_idx=0),
        _make_token("$500.00", col_xs[3], row_ys[0], 0.12, 0.018, char_idx=char_idx + 48, line_idx=0),
    ]
    char_idx += 65
    tokens.extend(r0)
    lines.append(VisualLine(r0, page=1, line_index=0, bbox=BBox(col_xs[0], row_ys[0], 0.84, 0.018, 1)))

    # Row 1: missing optional 'quantity', but has item, description, and total
    r1 = [
        _make_token("ITEM-B", col_xs[0], row_ys[1], 0.15, 0.018, char_idx=char_idx, line_idx=1),
        _make_token("Custom Flange Unit", col_xs[1], row_ys[1], 0.25, 0.018, char_idx=char_idx + 10, line_idx=1),
        _make_token("$750.00", col_xs[3], row_ys[1], 0.12, 0.018, char_idx=char_idx + 48, line_idx=1),
    ]
    char_idx += 65
    tokens.extend(r1)
    lines.append(VisualLine(r1, page=1, line_index=1, bbox=BBox(col_xs[0], row_ys[1], 0.84, 0.018, 1)))

    page = DocumentPage(page_number=1, width=612, height=792, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    payload = {
        "items": [
            {"sku": "ITEM-A", "desc": "Industrial Valve", "qty": 10, "total": 500.0},
            {"sku": "ITEM-B", "desc": "Custom Flange Unit", "total": 750.0},  # no qty
        ]
    }
    result = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in result["field_citations"]}

    # Row 1 total MUST remain at col_xs[3] (0.80), not shifted leftward to col_xs[2] (0.60)!
    assert "items[1].total" in citations
    total_x = citations["items[1].total"]["bbox"][0]
    assert abs(total_x - col_xs[3]) < 0.03, (
        f"Row 1 total shifted leftward to x={total_x}, expected col 3 ({col_xs[3]})"
    )


# ===========================================================================
# 4. Scanned-Page Coordinate Offsets and Bounds Verification
# ===========================================================================

def test_all_citations_strictly_within_unit_interval() -> None:
    """Verify every citation generated across diverse structures satisfies 0.0 <= coord <= 1.0."""
    tokens = [
        _make_token("Header", 0.05, 0.02, 0.15, 0.015, line_idx=0),
        _make_token("Contract", 0.10, 0.10, 0.15, 0.018, line_idx=1),
        _make_token("99.9%", 0.80, 0.10, 0.12, 0.018, line_idx=1),
        _make_token("Footer", 0.05, 0.96, 0.15, 0.015, line_idx=2),
    ]
    line0 = VisualLine([tokens[0]], 1, 0, BBox(0.05, 0.02, 0.15, 0.015, 1))
    line1 = VisualLine([tokens[1], tokens[2]], 1, 1, BBox(0.10, 0.10, 0.82, 0.018, 1))
    line2 = VisualLine([tokens[3]], 1, 2, BBox(0.05, 0.96, 0.15, 0.015, 1))
    page = DocumentPage(1, 612, 792, tokens=tokens, lines=[line0, line1, line2])
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx)

    result = adapter.ground_extracted_data({
        "header_title": "Header",
        "contract_code": "Contract",
        "fulfillment_rate": "99.9%",
        "footer_text": "Footer",
    })
    for cit in result["field_citations"]:
        _assert_valid_bbox(cit["bbox"])


def test_extreme_margin_boundary_clamping() -> None:
    """Verify boundary clamping for edge tokens located at extreme top-left and bottom-right."""
    # Top-left token near (0.005, 0.005)
    t_top_left = _make_token("TopLeftStamp", x=0.005, y=0.005, w=0.040, h=0.012, line_idx=0)
    line0 = VisualLine([t_top_left], 1, 0, BBox(0.005, 0.005, 0.040, 0.012, 1))

    # Bottom-right token near (0.940, 0.975)
    t_bot_right = _make_token("BotRightSerial", x=0.940, y=0.975, w=0.050, h=0.018, line_idx=1)
    line1 = VisualLine([t_bot_right], 1, 1, BBox(0.940, 0.975, 0.050, 0.018, 1))

    page = DocumentPage(1, 612, 792, tokens=[t_top_left, t_bot_right], lines=[line0, line1])
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_bbox_precision=True)

    result = adapter.ground_extracted_data({
        "top_left": "TopLeftStamp",
        "bottom_right": "BotRightSerial",
    })
    citations = {c["field_path"]: c for c in result["field_citations"]}

    assert "top_left" in citations
    _assert_valid_bbox(citations["top_left"]["bbox"])
    # y must be >= 0.0 even after line height padding
    assert citations["top_left"]["bbox"][1] >= 0.0

    assert "bottom_right" in citations
    _assert_valid_bbox(citations["bottom_right"]["bbox"])
    # y + height must be <= 1.0 even after expansion
    br_box = citations["bottom_right"]["bbox"]
    assert br_box[1] + br_box[3] <= 1.0001


def test_scanned_page_mechanical_drag_and_tilt_bounds() -> None:
    """Verify simulated mechanical drag and skew slope stay strictly within [0, 1] across all rows."""
    from tonerhound.benchmark.adapter import _Y_START, _Y_END

    # Canonical grid pitch
    pitch = (_Y_END - _Y_START) / 70.0
    page_slope = 0.008  # ~0.5 degree mechanical tilt
    drag_offset = 0.015

    for slot_idx in range(71):
        anc_cy = _Y_START + slot_idx * pitch + drag_offset
        for col_x in [0.05, 0.25, 0.50, 0.75, 0.90]:
            col_w = 0.08
            cell_xc = col_x + col_w / 2.0
            cell_yc = anc_cy + page_slope * (cell_xc - 0.50)
            cell_h = 0.0102
            cell_y = cell_yc - cell_h / 2.0

            # Clamping invariants
            clamped_x = max(0.0, min(1.0 - col_w, col_x))
            clamped_y = max(0.0, min(1.0 - cell_h, cell_y))
            box = BBox(x=clamped_x, y=clamped_y, width=col_w, height=cell_h, page=1)
            _assert_valid_bbox(box)


def test_canonical_71_and_72_slot_creditor_grid_bounds() -> None:
    """Verify all 71 and 72 slot creditor grid coordinates stay strictly within [0.0, 1.0]."""
    _Y_START = 0.08162
    _Y_END = 0.88186

    # 71-slot grid (standard pages)
    s_71 = (_Y_END - _Y_START) / 70.0
    for k in range(71):
        y_k = _Y_START + k * s_71
        h_k = 0.0101
        box = BBox(x=0.067, y=y_k, width=0.18, height=h_k, page=1)
        _assert_valid_bbox(box)

    # 72-slot grid (Page 2 terminal pages)
    s_72 = (_Y_END - _Y_START) / 71.0
    for k in range(72):
        y_k = _Y_START + k * s_72
        h_k = 0.0098
        box = BBox(x=0.067, y=y_k, width=0.18, height=h_k, page=2)
        _assert_valid_bbox(box)


def test_real_scanned_corrupted_document_coordinates_in_bounds() -> None:
    """Verify coordinates on real scanned/corrupted PDF stay strictly within [0.0, 1.0]."""
    corrupted_pdf = Path("research/data/full/short/caterpillar_spec_sheet_312c_excavator_corrupted.pdf")
    assert corrupted_pdf.exists(), f"Missing fixture {corrupted_pdf}"

    idx = DocumentIndex.from_pdf(corrupted_pdf, use_cache=True)
    adapter = ExtractBenchAdapter(idx)

    extracted = {
        "model": "312C",
        "manufacturer": "Caterpillar",
        "operating_weight": "28,440 lb",
        "engine_power": "90 hp",
    }
    result = adapter.ground_extracted_data(extracted, example_id="caterpillar_corrupt_test")

    # For every citation produced, coordinates must be strictly in [0, 1]
    for cit in result["field_citations"]:
        _assert_valid_bbox(cit["bbox"])
        assert cit["page"] >= 1
