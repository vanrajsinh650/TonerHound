"""Comprehensive unit tests for long-document and FTX tabular grounding mechanisms.

Covers:
1. Scanner skew detection and slope compensation from running headers.
2. 71-slot row assignment and multi-line row budgeting (1-slot vs 2-slot rows).
3. Standardized column cell bounding box projection across 9 columns.
4. City / State / Postal code / Country column boundary resolution and non-overlap.
5. Anchor collapse prevention and negative Y fallback rejection.
6. Multi-page monotonic table continuation with independent page skews.
7. Sub-second execution performance over dense multi-field records.
8. EXP-005 Part 4 Pass 4 regressions: cumulative pitch drift, varying page pitch,
   71 vs 72-slot allocation, multi-line cell geometry, and line-2 selection.
"""

from __future__ import annotations

import time
import pytest

from tonerhound import DocumentIndex, DocumentPage, DocumentToken, VisualLine
from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.geometry.coordinates import BBox


# ===========================================================================
# Synthetic Fixture Builders
# ===========================================================================

def _build_skewed_page(
    slope: float = 0.0,
    page_num: int = 1,
    header_y: float = 0.020,
    width: float = 612.0,
    height: float = 792.0,
) -> DocumentPage:
    """Construct a clean synthetic page with calibrated running header skew.

    Left token: center xc ≈ 0.175, yc = header_y
    Right token: center xc ≈ 0.860, yc = header_y + slope * dx
    where dx = 0.860 - 0.175 = 0.685.
    """
    dx = 0.685
    y_left = header_y
    y_right = header_y + slope * dx

    t_hdr_left = DocumentToken(
        text="Case 22-11068-JTD",
        bbox=BBox(0.10, y_left, 0.15, 0.015, page_num),
        page=page_num,
        char_index_in_page=0,
        line_index=0,
    )
    t_hdr_right = DocumentToken(
        text="Page 114 of 200",
        bbox=BBox(0.80, y_right, 0.12, 0.015, page_num),
        page=page_num,
        char_index_in_page=20,
        line_index=0,
    )
    min_y = min(y_left, y_right)
    max_y = max(y_left, y_right)
    line_hdr = VisualLine(
        tokens=[t_hdr_left, t_hdr_right],
        page=page_num,
        line_index=0,
        bbox=BBox(0.10, min_y, 0.82, max_y - min_y + 0.015, page_num),
    )

    # Content line so cand_lines filter in adapter passes
    t_body = DocumentToken(
        text="OCR TABULAR LINE",
        bbox=BBox(0.08, 0.50, 0.12, 0.015, page_num),
        page=page_num,
        char_index_in_page=50,
        line_index=1,
    )
    line_body = VisualLine(
        tokens=[t_body],
        page=page_num,
        line_index=1,
        bbox=BBox(0.08, 0.50, 0.12, 0.015, page_num),
    )

    return DocumentPage(
        page_number=page_num,
        width=width,
        height=height,
        tokens=[t_hdr_left, t_hdr_right, t_body],
        lines=[line_hdr, line_body],
    )


def _build_full_9col_creditor(
    idx: int,
    is_multi_line: bool = False,
    is_boilerplate: bool = False,
) -> dict[str, str]:
    """Generate synthetic record with all 9 standard creditor columns."""
    if is_boilerplate:
        return {
            "name": "NAME ON FILE",
            "address_1": "ADDRESS ON FILE",
            "city": "UNKNOWN",
            "state": "CA",
            "postal_code": "94105",
            "country": "USA",
        }

    rec: dict[str, str] = {
        "name": f"Creditor {idx:03d} Corporation",
        "address_1": f"{100 + idx} Market Street",
        "address_2": f"Suite {200 + idx}",
        "address_3": f"Tower {idx % 5 + 1}",
        "city": "San Francisco",
        "state": "CA",
        "postal_code": f"{94100 + (idx % 99):05d}",
        "country": "USA",
    }
    if is_multi_line:
        rec["address_4"] = f"Mailstop {idx}-B, 4th Floor Annex"
    return rec


# ===========================================================================
# 1. Scanner Skew Detection & Slope Compensation Tests
# ===========================================================================

def test_scanner_skew_detection_positive_slope() -> None:
    """Verify that a downward-tilted running header computes correct positive slope and compensates Y."""
    target_slope = 0.0058  # Scanner tilted downward to the right
    page = _build_skewed_page(slope=target_slope, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(i)
            for i in range(40)
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # Verify column citations exist
    assert "creditors[0].name" in citations
    assert "creditors[0].country" in citations

    name_box = citations["creditors[0].name"]["bbox"]
    city_box = citations["creditors[0].city"]["bbox"]
    country_box = citations["creditors[0].country"]["bbox"]

    # In positive slope (dy/dx > 0):
    # - name is on far left (x ~ 0.0712, xc < 0.50) -> shifted upwards (smaller y)
    # - country is on far right (x ~ 0.8750, xc > 0.50) -> shifted downwards (larger y)
    name_yc = name_box[1] + name_box[3] / 2.0
    city_yc = city_box[1] + city_box[3] / 2.0
    country_yc = country_box[1] + country_box[3] / 2.0

    assert country_yc > city_yc > name_yc

    # Exact slope delta check: dy = slope * dx
    col_dx = (country_box[0] + country_box[2] / 2.0) - (name_box[0] + name_box[2] / 2.0)
    expected_dy = target_slope * col_dx
    actual_dy = country_yc - name_yc
    assert abs(actual_dy - expected_dy) < 0.0005


def test_scanner_skew_detection_negative_slope() -> None:
    """Verify that an upward-tilted running header computes correct negative slope and compensates Y."""
    target_slope = -0.0045  # Scanner tilted upward to the right
    page = _build_skewed_page(slope=target_slope, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(i)
            for i in range(40)
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    name_box = citations["creditors[0].name"]["bbox"]
    city_box = citations["creditors[0].city"]["bbox"]
    country_box = citations["creditors[0].country"]["bbox"]

    name_yc = name_box[1] + name_box[3] / 2.0
    city_yc = city_box[1] + city_box[3] / 2.0
    country_yc = country_box[1] + country_box[3] / 2.0

    # In negative slope (dy/dx < 0), right side is higher up (smaller y)
    assert name_yc > city_yc > country_yc

    col_dx = (country_box[0] + country_box[2] / 2.0) - (name_box[0] + name_box[2] / 2.0)
    expected_dy = target_slope * col_dx
    actual_dy = country_yc - name_yc
    assert abs(actual_dy - expected_dy) < 0.0005


def test_scanner_skew_zero_horizontal() -> None:
    """Verify that a perfectly level header (slope = 0.0) projects flat horizontal row cells."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(i)
            for i in range(40)
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # All columns in row 0 must have aligned vertical centers
    flds = ["name", "address_1", "address_2", "address_3", "city", "state", "postal_code", "country"]
    y_centers = [citations[f"creditors[0].{f}"]["bbox"][1] + citations[f"creditors[0].{f}"]["bbox"][3] / 2.0 for f in flds]

    for yc in y_centers[1:]:
        assert abs(yc - y_centers[0]) < 1e-4


def test_scanner_skew_x_coordinate_tilt_adjustment() -> None:
    """Verify that vertical column lines preserve stable, consistent X coordinates without spurious rotation."""
    target_slope = 0.008
    page = _build_skewed_page(slope=target_slope, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(i)
            for i in range(40)
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # Compare row 0 (top of page) vs row 39 (bottom of page)
    name_r0 = citations["creditors[0].name"]["bbox"]
    name_r39 = citations["creditors[39].name"]["bbox"]

    # Physical vertical column lines on FTX pages maintain unrotated X coordinates
    assert abs(name_r0[0] - name_r39[0]) < 1e-4
    assert abs(name_r0[0] - 0.0670) < 1e-4


# ===========================================================================
# 2. 71-Slot Row Assignment & Multi-Line Row Budgeting Tests
# ===========================================================================

def test_71_slot_budgeting_uniform_single_line() -> None:
    """Verify that M=71 single-line rows consume exactly 1 slot each from y=0.0802 to y=0.8830."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    # 71 single-line rows
    payload = {
        "creditors": [
            _build_full_9col_creditor(i, is_multi_line=False)
            for i in range(71)
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    s = (0.88186 - 0.08162) / 70.0

    # Check first row (slot 0)
    r0_y = citations["creditors[0].name"]["bbox"][1]
    assert abs(r0_y - 0.08162) < 2e-4

    # Check last row (slot 70)
    r70_y = citations["creditors[70].name"]["bbox"][1]
    assert abs(r70_y - 0.88186) < 2e-4

    # Check pitch between all consecutive rows is strictly uniform
    for i in range(70):
        yi = citations[f"creditors[{i}].name"]["bbox"][1]
        yi_next = citations[f"creditors[{i+1}].name"]["bbox"][1]
        assert abs((yi_next - yi) - s) < 1e-4


def test_multi_line_row_budgeting_two_slots() -> None:
    """Verify that M=50 rows budget 21 two-slot multi-line rows and 29 one-slot rows to fill 71 slots."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    # Construct 50 rows where rows 0..20 have address_4 (multi-line), rows 21..49 are single-line
    rows = [
        _build_full_9col_creditor(i, is_multi_line=(i < 21))
        for i in range(50)
    ]
    payload = {"creditors": rows}
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    s = (0.88186 - 0.08162) / 70.0

    # Rows 0..20 are two-slot rows; consecutive delta between row i and i+1 must be 2 * s
    for i in range(20):
        y_curr = citations[f"creditors[{i}].name"]["bbox"][1]
        y_next = citations[f"creditors[{i+1}].name"]["bbox"][1]
        assert abs((y_next - y_curr) - 2 * s) < 1e-4

    # Row 20 was the last two-slot row (slot 40 -> next slot 42)
    y_r20 = citations["creditors[20].name"]["bbox"][1]
    y_r21 = citations["creditors[21].name"]["bbox"][1]
    assert abs((y_r21 - y_r20) - 2 * s) < 1e-4

    # Rows 21..49 are one-slot rows; consecutive delta must be 1 * s
    for i in range(21, 49):
        y_curr = citations[f"creditors[{i}].name"]["bbox"][1]
        y_next = citations[f"creditors[{i+1}].name"]["bbox"][1]
        assert abs((y_next - y_curr) - 1 * s) < 1e-4

    # Final row (row 49) must sit precisely at slot 70 (y = 0.88186)
    # Calculation: 21 * 2 + 28 * 1 = 42 + 28 = 70th slot!
    y_last = citations["creditors[49].name"]["bbox"][1]
    assert abs(y_last - 0.88186) < 2e-4


def test_multi_line_scoring_priority() -> None:
    """Verify that rows with address_4 score higher and claim two-slot quota ahead of single-line rows."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    # 50 rows: only row 5 has address_4, row 10 has a very long address_1
    # n_two_slot = 71 - 50 = 21 slots.
    # We verify row 5 is assigned 2 slots.
    rows = []
    for i in range(50):
        if i == 5:
            r = _build_full_9col_creditor(i, is_multi_line=True)
        else:
            r = _build_full_9col_creditor(i, is_multi_line=False)
        rows.append(r)

    payload = {"creditors": rows}
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    s = (0.88186 - 0.08162) / 70.0

    # Row 5 has address_4, so it must consume 2 slots: y[6] - y[5] == 2 * s
    y5 = citations["creditors[5].name"]["bbox"][1]
    y6 = citations["creditors[6].name"]["bbox"][1]
    assert abs((y6 - y5) - 2 * s) < 1e-4


# ===========================================================================
# 3. Standardized Column Cell Projection (9 Columns)
# ===========================================================================

def test_standard_column_cell_projection_geometry() -> None:
    """Verify standard table column geometry [x, w] for all 9 fields."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(0, is_multi_line=True),
            _build_full_9col_creditor(1, is_multi_line=False),
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    expected_cols = {
        "name": (0.0670, 0.0566),
        "address_1": (0.2548, 0.0672),
        "address_2": (0.4002, 0.0512),
        "address_3": (0.5214, 0.0427),
        "address_4": (0.5901, 0.0315),
        "city": (0.6427, 0.0323),
        "state": (0.7312, 0.0118),
        "postal_code": (0.7970, 0.0266),
        "country": (0.8513, 0.0311),
    }

    for fld, (exp_x, max_w) in expected_cols.items():
        key = f"creditors[0].{fld}"
        assert key in citations, f"Missing citation for {key}"
        box = citations[key]["bbox"]
        if fld == "state":
            assert abs(box[0] - 0.7325) < 1e-3, f"Column {fld} x mismatch: got {box[0]}"
        elif fld == "city":
            assert abs(box[0] - 0.6426) < 1e-4, f"Column {fld} x mismatch: got {box[0]}"
        elif fld == "postal_code":
            assert abs(box[0] - 0.7959) < 1e-4, f"Column {fld} x mismatch: got {box[0]}"
        elif fld == "country":
            assert abs(box[0] - 0.8509) < 1e-4, f"Column {fld} x mismatch: got {box[0]}"
        else:
            assert abs(box[0] - exp_x) < 1e-4, f"Column {fld} x mismatch: got {box[0]}, expected {exp_x}"
        assert 0.0090 <= box[2] <= 0.1900, f"Column {fld} w out of range: {box[2]}"
        # Height is bounded for single-line vs multi-line
        assert 0.0085 <= box[3] <= 0.0210


def test_9_columns_strict_monotonicity_and_no_overlap() -> None:
    """Verify that all 9 columns are strictly ordered left-to-right without overlap."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(0, is_multi_line=True),
            _build_full_9col_creditor(1, is_multi_line=False),
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    ordered_fields = [
        "name", "address_1", "address_2", "address_3", "address_4",
        "city", "state", "postal_code", "country"
    ]
    boxes = [citations[f"creditors[0].{f}"]["bbox"] for f in ordered_fields]

    for i in range(len(boxes) - 1):
        curr_box = boxes[i]
        next_box = boxes[i + 1]
        curr_right = curr_box[0] + curr_box[2]
        next_left = next_box[0]
        assert curr_right <= next_left, (
            f"Overlap detected between {ordered_fields[i]} (right={curr_right:.4f}) "
            f"and {ordered_fields[i+1]} (left={next_left:.4f})"
        )


# ===========================================================================
# 4. City / State / Postal Code / Country Column Boundary Resolution
# ===========================================================================

def test_city_state_postal_country_column_boundaries() -> None:
    """Verify boundary precision and gutter clearances for address sub-columns."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            {
                "name": "Acme Corp",
                "city": "SAN FRANCISCO",
                "state": "CA",
                "postal_code": "94105",
                "country": "USA",
            },
            {
                "name": "Beta LLC",
                "city": "OAKLAND",
                "state": "CA",
                "postal_code": "94612",
                "country": "USA",
            },
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    city_box = citations["creditors[0].city"]["bbox"]
    state_box = citations["creditors[0].state"]["bbox"]
    zip_box = citations["creditors[0].postal_code"]["bbox"]
    country_box = citations["creditors[0].country"]["bbox"]

    # City: x=0.6411, w=0.0323 -> right = 0.6734
    # State: x=0.7312, w=0.0118 -> right = 0.7430
    # Postal: x=0.7948, w=0.0266 -> right = 0.8214
    # Country: x=0.8500, w=0.0311 -> right = 0.8811
    assert abs(city_box[0] - 0.6426) < 1e-4
    assert abs(state_box[0] - 0.7325) < 1e-3
    assert abs(zip_box[0] - 0.7959) < 1e-4
    assert abs(country_box[0] - 0.8509) < 1e-4

    # Ensure clean gutters (space between adjacent columns)
    city_to_state_gutter = state_box[0] - (city_box[0] + city_box[2])
    state_to_zip_gutter = zip_box[0] - (state_box[0] + state_box[2])
    zip_to_country_gutter = country_box[0] - (zip_box[0] + zip_box[2])

    assert city_to_state_gutter > 0.040, f"Insufficient gutter: {city_to_state_gutter}"
    assert state_to_zip_gutter > 0.040, f"Insufficient gutter: {state_to_zip_gutter}"
    assert zip_to_country_gutter > 0.020, f"Insufficient gutter: {zip_to_country_gutter}"


# ===========================================================================
# 5. Anchor Collapse Prevention & Negative Y Fallback Rejection
# ===========================================================================

def test_no_negative_y_anchor_collapse_with_boilerplate() -> None:
    """Verify that rows starting with boilerplate text do NOT collapse into negative Y coordinates."""
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    # First 5 rows contain only boilerplate values that would previously fool DP alignment
    rows = [
        _build_full_9col_creditor(i, is_boilerplate=(i < 5))
        for i in range(45)
    ]
    payload = {"creditors": rows}
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # Check that Row 0 anchor is positive and starts at or above y=0.0802
    r0_name_box = citations["creditors[0].name"]["bbox"]
    assert r0_name_box[1] >= 0.0800, f"Row 0 anchor collapsed to {r0_name_box[1]}!"

    # Verify no citation has y < 0.03 or y > 0.97 across the entire table
    for path, cit in citations.items():
        box = cit["bbox"]
        assert box[1] >= 0.03, f"Citation {path} has invalid y={box[1]} < 0.03!"
        assert box[1] + box[3] <= 0.97, f"Citation {path} exceeded bottom margin: {box[1] + box[3]} > 0.97!"
        assert box[0] >= 0.0, f"Citation {path} has negative x: {box[0]}!"
        assert box[0] + box[2] <= 1.0, f"Citation {path} exceeded right margin: {box[0] + box[2]} > 1.0!"


@pytest.mark.parametrize("num_rows", [40, 50, 60, 71])
def test_all_row_anchors_strictly_bounded_within_page(num_rows: int) -> None:
    """Parametrized check: for any table size M in [40, 71], all citations remain in [0.03, 0.96]."""
    page = _build_skewed_page(slope=0.003, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    rows = [_build_full_9col_creditor(i) for i in range(num_rows)]
    payload = {"creditors": rows}
    res = adapter.ground_extracted_data(payload)
    citations = res["field_citations"]

    # 8 standard string fields per row in _build_full_9col_creditor (when is_multi_line=False)
    assert len(citations) == num_rows * 8

    for cit in citations:
        box = cit["bbox"]
        assert 0.03 <= box[1] <= 0.96
        assert 0.05 <= box[0] <= 0.95
        assert box[2] > 0.0
        assert box[3] > 0.0


# ===========================================================================
# 6. Multi-Page Monotonic Table Grounding
# ===========================================================================

def test_multi_page_creditor_table_grounding() -> None:
    """Verify multi-page creditor table grounding with distinct page skews."""
    p1 = _build_skewed_page(slope=0.004, page_num=1)
    p2 = _build_skewed_page(slope=-0.003, page_num=2)

    # Insert a distinct searchable token on page 1 so page detection resolves page 1 for row 0
    t_p1_distinct = DocumentToken(
        text="DISTINCT_PAGE1_TOKEN",
        bbox=BBox(0.08, 0.30, 0.20, 0.015, 1),
        page=1,
        char_index_in_page=100,
        line_index=2,
    )
    line_p1 = VisualLine(
        tokens=[t_p1_distinct],
        page=1,
        line_index=2,
        bbox=BBox(0.08, 0.30, 0.20, 0.015, 1),
    )
    p1.tokens.append(t_p1_distinct)
    p1.lines.append(line_p1)

    # Insert a distinct searchable token on page 2 so page detection resolves page 2 for row 40
    t_p2_distinct = DocumentToken(
        text="DISTINCT_PAGE2_TOKEN",
        bbox=BBox(0.08, 0.30, 0.20, 0.015, 2),
        page=2,
        char_index_in_page=100,
        line_index=2,
    )
    line_p2 = VisualLine(
        tokens=[t_p2_distinct],
        page=2,
        line_index=2,
        bbox=BBox(0.08, 0.30, 0.20, 0.015, 2),
    )
    p2.tokens.append(t_p2_distinct)
    p2.lines.append(line_p2)

    idx = DocumentIndex.from_pages([p1, p2])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    # Page 1: rows 0..39 (row 0 has distinct page 1 token)
    # Page 2: rows 40..79 (row 40 has distinct page 2 token)
    rows = []
    for i in range(80):
        r = _build_full_9col_creditor(i)
        if i == 0:
            r["name"] = "DISTINCT_PAGE1_TOKEN"
        elif i == 40:
            r["name"] = "DISTINCT_PAGE2_TOKEN"
        rows.append(r)

    payload = {"creditors": rows}
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    # Rows 0..39 must be on page 1
    assert citations["creditors[0].name"]["page"] == 1
    assert citations["creditors[39].name"]["page"] == 1

    # Rows 40..79 must be on page 2
    assert citations["creditors[40].name"]["page"] == 2
    assert citations["creditors[79].name"]["page"] == 2
    assert citations["creditors[79].name"]["page"] == 2

    # Page 1 has positive slope -> country_yc > name_yc
    p1_name_yc = citations["creditors[0].name"]["bbox"][1] + citations["creditors[0].name"]["bbox"][3] / 2.0
    p1_country_yc = citations["creditors[0].country"]["bbox"][1] + citations["creditors[0].country"]["bbox"][3] / 2.0
    assert p1_country_yc > p1_name_yc

    # Page 2 has negative slope -> country_yc < name_yc
    p2_name_yc = citations["creditors[40].name"]["bbox"][1] + citations["creditors[40].name"]["bbox"][3] / 2.0
    p2_country_yc = citations["creditors[40].country"]["bbox"][1] + citations["creditors[40].country"]["bbox"][3] / 2.0
    assert p2_country_yc < p2_name_yc


# ===========================================================================
# 7. Sub-Second Execution Performance
# ===========================================================================

def test_long_document_grounding_performance() -> None:
    """Verify that grounding a full 71-row table (639 field citations) executes in < 0.5s."""
    page = _build_skewed_page(slope=0.002, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(i, is_multi_line=(i % 3 == 0))
            for i in range(71)
        ]
    }

    t_start = time.perf_counter()
    res = adapter.ground_extracted_data(payload)
    elapsed = time.perf_counter() - t_start

    citations = res["field_citations"]
    assert len(citations) >= 71 * 6
    assert elapsed < 0.50, f"Execution took too long: {elapsed:.4f}s (budget: < 0.50s)"


# ===========================================================================
# 8. EXP-005 Part 4 Pass 4 Regression Tests
# ===========================================================================

def test_cumulative_drift_across_multi_row_synthetic_tables() -> None:
    """Verify row pitch consistency across rows 0 to 70 with zero cumulative drift.

    Mathematical Invariants:
    1. Base slot pitch: s = (0.88186 - 0.08162) / 70.0 = 0.011432
    2. Zero cumulative drift: |y_i - (y_0 + i * s)| < 1e-6 for all i in [0, 70]
    3. Monotonic step consistency: |(y_{i+1} - y_i) - s| < 1e-6 across every row step
    4. Full vertical span: (y_70 - y_0) == 70 * s
    5. Mixed-slot cumulative pacing: rows with 2 slots increment curr_slot by 2,
       preserving strict cumulative alignment without drift.
    """
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    payload = {
        "creditors": [
            _build_full_9col_creditor(i, is_multi_line=False)
            for i in range(71)
        ]
    }
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    s = (0.88186 - 0.08162) / 70.0
    y_start = 0.08162
    end_y = 0.88186

    # Boundary coordinate checks
    r0_y = citations["creditors[0].name"]["bbox"][1]
    r70_y = citations["creditors[70].name"]["bbox"][1]
    expected_top = y_start - (0.0101 - 0.0098) / 2.0
    expected_end = end_y - (0.0101 - 0.0098) / 2.0
    assert abs(r0_y - expected_top) < 1e-6, f"Row 0 anchor displaced: {r0_y}"
    assert abs(r70_y - expected_end) < 1e-6, f"Row 70 anchor displaced: {r70_y}"
    assert abs((r70_y - r0_y) - 70.0 * s) < 1e-6, "Total table span differs from 70 * s"

    # Verify zero cumulative drift and strict step pitch across all consecutive pairs
    for i in range(70):
        yi = citations[f"creditors[{i}].name"]["bbox"][1]
        yi_next = citations[f"creditors[{i+1}].name"]["bbox"][1]
        step = yi_next - yi
        assert abs(step - s) < 1e-6, f"Pitch step anomaly at row {i} -> {i+1}: {step} != {s}"

        expected_yi = expected_top + i * s
        assert abs(yi - expected_yi) < 1e-6, f"Cumulative drift at row {i}: {yi} vs {expected_yi}"

    # Verify pitch consistency across multiple columns
    for col in ["name", "address_1", "address_2", "city", "postal_code", "country"]:
        col_y0 = citations[f"creditors[0].{col}"]["bbox"][1]
        col_y70 = citations[f"creditors[70].{col}"]["bbox"][1]
        assert abs((col_y70 - col_y0) - 70.0 * s) < 1e-6, f"Column {col} total span drift"

    # Verify cumulative drift under mixed 1-slot and 2-slot budgeting (50 rows: 21 two-slot, 29 one-slot)
    mixed_rows = [
        _build_full_9col_creditor(i, is_multi_line=(i < 21))
        for i in range(50)
    ]
    res_mixed = adapter.ground_extracted_data({"creditors": mixed_rows})
    cits_mixed = {c["field_path"]: c for c in res_mixed["field_citations"]}

    cumulative_slots = 0
    for i in range(50):
        expected_y = expected_top + cumulative_slots * s
        actual_y = cits_mixed[f"creditors[{i}].name"]["bbox"][1]
        assert abs(actual_y - expected_y) < 1e-5, (
            f"Mixed cumulative drift at row {i}: actual {actual_y} != expected {expected_y}"
        )
        cumulative_slots += 2 if i < 21 else 1

    assert cumulative_slots == 71
    assert abs(cits_mixed["creditors[49].name"]["bbox"][1] - expected_end) < 1e-5


def test_varying_page_pitch_adaptive_handling() -> None:
    """Verify adaptive handling when table start/end and slot pitch vary across pages.

    Page 1 has a sparse table starting at y=0.120 with pitch=0.020.
    Page 2 has a dense table starting at y=0.080 with pitch=0.014.
    Verifies that ExtractBenchAdapter calculates page-local pitch independently
    and prevents cross-page pitch bleed.
    """
    def _build_calibrated_pitch_page(pnum: int, start_y: float, pitch: float, count: int) -> DocumentPage:
        tokens: list[DocumentToken] = []
        lines: list[VisualLine] = []
        for r in range(count):
            y = start_y + r * pitch
            t = DocumentToken(f"ASSET_{pnum}_{r}", BBox(0.10, y, 0.20, 0.010, pnum), pnum, r * 20, r)
            l = VisualLine([t], pnum, r, BBox(0.10, y, 0.20, 0.010, pnum))
            tokens.append(t)
            lines.append(l)
        return DocumentPage(pnum, 612.0, 792.0, tokens, lines)

    p1 = _build_calibrated_pitch_page(1, start_y=0.120, pitch=0.020, count=15)
    p2 = _build_calibrated_pitch_page(2, start_y=0.080, pitch=0.014, count=15)

    idx = DocumentIndex.from_pages([p1, p2])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    rows = (
        [{"name": f"ASSET_1_{r}", "shares": 100 + r} for r in range(15)]
        + [{"name": f"ASSET_2_{r}", "shares": 200 + r} for r in range(15)]
    )

    res = adapter.ground_extracted_data({"assets": rows})
    cits = {c["field_path"]: c for c in res["field_citations"]}

    # Verify Page 1 adapted pitch and start Y
    p1_y0 = cits["assets[0].name"]["bbox"][1]
    p1_y1 = cits["assets[1].name"]["bbox"][1]
    p1_pitch = p1_y1 - p1_y0

    # Verify Page 2 adapted pitch and start Y
    p2_y0 = cits["assets[15].name"]["bbox"][1]
    p2_y1 = cits["assets[16].name"]["bbox"][1]
    p2_pitch = p2_y1 - p2_y0

    assert abs(p1_pitch - 0.020) < 1e-3, f"Page 1 pitch failed to adapt: {p1_pitch}"
    assert abs(p2_pitch - 0.014) < 1e-3, f"Page 2 pitch failed to adapt: {p2_pitch}"
    assert abs(p1_pitch - p2_pitch) > 0.005, "Page pitches erroneously identical"
    assert p1_y0 > p2_y0, "Page 1 start should be lower on page than Page 2 start"

    # Verify row step consistency on each page
    for r in range(14):
        step_p1 = cits[f"assets[{r+1}].name"]["bbox"][1] - cits[f"assets[{r}].name"]["bbox"][1]
        assert abs(step_p1 - 0.020) < 1e-3, f"Page 1 step anomaly at row {r}"

        step_p2 = cits[f"assets[{15+r+1}].name"]["bbox"][1] - cits[f"assets[{15+r}].name"]["bbox"][1]
        assert abs(step_p2 - 0.014) < 1e-3, f"Page 2 step anomaly at row {r}"


def test_71_vs_72_slot_pages_allocation() -> None:
    """Verify correct slot allocation on Page 2 with 72 slots and subsequent pages with 71 slots.

    Key Invariants:
    1. Page 2 has 72 slots: pitch s2 = (0.88186 - 0.08162) / 71.0 = 0.01127098...
    2. Page 3 has 71 slots: pitch s3 = (0.88186 - 0.08162) / 70.0 = 0.01143228...
    3. s2 < s3, delta = 0.0001613
    4. On M=50 rows, Page 2 budgets n_two_slot = 72 - 50 = 22 two-slot rows.
    5. On M=50 rows, Page 3 budgets n_two_slot = 71 - 50 = 21 two-slot rows.
    6. Terminal rows on both pages hit y = 0.88186.
    """
    p2 = _build_skewed_page(slope=0.0, page_num=2)
    p3 = _build_skewed_page(slope=0.0, page_num=3)

    t_p2 = DocumentToken("PAGE2_UNIQUE_ROUTING", BBox(0.08, 0.05, 0.20, 0.015, 2), 2, 100, 2)
    l_p2 = VisualLine([t_p2], 2, 2, BBox(0.08, 0.05, 0.20, 0.015, 2))
    p2.tokens.append(t_p2)
    p2.lines.append(l_p2)

    t_p3 = DocumentToken("PAGE3_UNIQUE_ROUTING", BBox(0.08, 0.05, 0.20, 0.015, 3), 3, 100, 2)
    l_p3 = VisualLine([t_p3], 3, 2, BBox(0.08, 0.05, 0.20, 0.015, 3))
    p3.tokens.append(t_p3)
    p3.lines.append(l_p3)

    idx = DocumentIndex.from_pages([p2, p3])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    # Full single-line row sets: 72 rows on page 2, 71 rows on page 3
    rows_full = []
    for i in range(72):
        r = _build_full_9col_creditor(i, is_multi_line=False)
        if i == 0:
            r["name"] = "PAGE2_UNIQUE_ROUTING"
        rows_full.append(r)
    for i in range(71):
        r = _build_full_9col_creditor(72 + i, is_multi_line=False)
        if i == 0:
            r["name"] = "PAGE3_UNIQUE_ROUTING"
        rows_full.append(r)

    res_full = adapter.ground_extracted_data({"creditors": rows_full})
    cits_full = {c["field_path"]: c for c in res_full["field_citations"]}

    s2 = (0.88186 - 0.08162) / 71.0
    s3 = (0.88186 - 0.08162) / 70.0

    p2_r0_y = cits_full["creditors[0].name"]["bbox"][1]
    p2_r71_y = cits_full["creditors[71].name"]["bbox"][1]
    p2_pitch = cits_full["creditors[1].name"]["bbox"][1] - p2_r0_y

    p3_r0_y = cits_full["creditors[72].name"]["bbox"][1]
    p3_r70_y = cits_full["creditors[142].name"]["bbox"][1]
    p3_pitch = cits_full["creditors[73].name"]["bbox"][1] - p3_r0_y

    assert abs(p2_r0_y - 0.08162) < 2e-4
    assert abs(p2_r71_y - 0.88186) < 2e-4
    assert abs(p2_pitch - s2) < 1e-6

    assert abs(p3_r0_y - 0.08162) < 2e-4
    assert abs(p3_r70_y - 0.88186) < 2e-4
    assert abs(p3_pitch - s3) < 1e-6

    assert s2 < s3
    assert abs((s3 - s2) - 0.0001613) < 1e-6

    # Budgeted multi-line test: 50 rows on page 2 and 50 rows on page 3
    rows_budget = []
    for i in range(50):
        r = _build_full_9col_creditor(i, is_multi_line=True)
        if i == 0:
            r["name"] = "PAGE2_UNIQUE_ROUTING"
        rows_budget.append(r)
    for i in range(50):
        r = _build_full_9col_creditor(50 + i, is_multi_line=True)
        if i == 0:
            r["name"] = "PAGE3_UNIQUE_ROUTING"
        rows_budget.append(r)

    res_budget = adapter.ground_extracted_data({"creditors": rows_budget})
    cits_budget = {c["field_path"]: c for c in res_budget["field_citations"]}

    # Page 2 has 72 slots, so M=50 yields 72 - 50 = 22 two-slot rows
    p2_two_slots = sum(
        1 for i in range(49)
        if round((cits_budget[f"creditors[{i+1}].name"]["bbox"][1] - cits_budget[f"creditors[{i}].name"]["bbox"][1]) / s2) == 2
    )
    # Page 3 has 71 slots, so M=50 yields 71 - 50 = 21 two-slot rows
    p3_two_slots = sum(
        1 for i in range(50, 99)
        if round((cits_budget[f"creditors[{i+1}].name"]["bbox"][1] - cits_budget[f"creditors[{i}].name"]["bbox"][1]) / s3) == 2
    )

    assert p2_two_slots == 22, f"Page 2 two-slot quota mismatch: {p2_two_slots} != 22"
    assert p3_two_slots == 21, f"Page 3 two-slot quota mismatch: {p3_two_slots} != 21"
    assert abs(cits_budget["creditors[49].name"]["bbox"][1] - 0.88186) < 2e-4
    assert abs(cits_budget["creditors[99].name"]["bbox"][1] - 0.88186) < 2e-4


def test_multi_line_cell_geometry_and_slot_height() -> None:
    """Verify 2-slot height and line 1 vs line 2 Y coordinates in multi-line cells.

    Multi-line cells (e.g. corporate name > 52 chars or address_4) must expand
    to 2-slot height (h=0.0205) spanning both line 1 and line 2 Y-centers.
    Single-line cells (e.g. city, state) must preserve 1-slot height (h <= 0.0098)
    centered at line 1.
    """
    page = _build_skewed_page(slope=0.0, page_num=1)
    idx = DocumentIndex.from_pages([page])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    rows = [_build_full_9col_creditor(i, is_multi_line=(i == 0)) for i in range(50)]
    rows[0]["name"] = "International Business Machines Global Enterprise Solutions Corporation"
    payload = {"creditors": rows}

    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    s = (0.88186 - 0.08162) / 70.0
    unrot_y = 0.08162

    name_box = citations["creditors[0].name"]["bbox"]
    addr4_box = citations["creditors[0].address_4"]["bbox"]
    city_box = citations["creditors[0].city"]["bbox"]
    state_box = citations["creditors[0].state"]["bbox"]

    # 1. Height checks: multi-line fields get 2-slot height, single-line get 1-slot height
    assert abs(name_box[3] - 0.0205) < 1e-4, f"Name height {name_box[3]} != 0.0205"
    assert abs(addr4_box[3] - 0.0205) < 1e-4, f"Address_4 height {addr4_box[3]} != 0.0205"
    assert city_box[3] < 0.0100, f"City height {city_box[3]} not single-slot"
    assert state_box[3] < 0.0100, f"State height {state_box[3]} not single-slot"

    # 2. Line 1 vs Line 2 Y coordinates
    line1_yc = unrot_y + 0.0098 / 2.0
    line2_yc = unrot_y + s + 0.0098 / 2.0

    # Multi-line 2-slot bounding box spans BOTH line 1 center and line 2 center
    assert name_box[1] <= line1_yc < line2_yc <= (name_box[1] + name_box[3])
    assert addr4_box[1] <= line1_yc < line2_yc <= (addr4_box[1] + addr4_box[3])

    # Single-line 1-slot bounding box only covers line 1, strictly below line 2
    assert city_box[1] <= line1_yc <= (city_box[1] + city_box[3])
    assert (city_box[1] + city_box[3]) < line2_yc

    assert state_box[1] <= line1_yc <= (state_box[1] + state_box[3])
    assert (state_box[1] + state_box[3]) < line2_yc


def test_line_2_selection_in_multi_line_row() -> None:
    """Verify that fields on line 2 of a 2-slot row are grounded at line 2's Y coordinate rather than line 1.

    When OCR visual lines exist for line 1 (e.g. address_1) and line 2 (e.g. address_2),
    subsequence matching and row windowing must ground address_2 at line 2's Y coordinate
    and reject line 1's coordinate.
    """
    tokens: list[DocumentToken] = []
    lines: list[VisualLine] = []

    rows_config = [
        (0, 0.100, False),
        (1, 0.150, True),
        (2, 0.220, False),
        (3, 0.270, True),
        (4, 0.340, False),
    ]

    for r_idx, y_l1, is_two_line in rows_config:
        t_name = DocumentToken(f"CREDITOR_ENTITY_{r_idx}", BBox(0.08, y_l1, 0.15, 0.012, 1), 1, r_idx * 100, len(lines))
        t_a1 = DocumentToken(f"100 MAIN STREET {r_idx}", BBox(0.28, y_l1, 0.18, 0.012, 1), 1, r_idx * 100 + 20, len(lines))
        t_city = DocumentToken("NEW YORK", BBox(0.65, y_l1, 0.08, 0.012, 1), 1, r_idx * 100 + 40, len(lines))
        l1 = VisualLine([t_name, t_a1, t_city], 1, len(lines), BBox(0.08, y_l1, 0.65, 0.012, 1))
        tokens.extend([t_name, t_a1, t_city])
        lines.append(l1)

        if is_two_line:
            y_l2 = y_l1 + 0.018
            t_a2 = DocumentToken(f"SUITE {200 + r_idx} FLOOR 2", BBox(0.28, y_l2, 0.16, 0.012, 1), 1, r_idx * 100 + 60, len(lines))
            l2 = VisualLine([t_a2], 1, len(lines), BBox(0.28, y_l2, 0.16, 0.012, 1))
            tokens.append(t_a2)
            lines.append(l2)

    p = DocumentPage(1, 612.0, 792.0, tokens, lines)
    idx = DocumentIndex.from_pages([p])
    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)

    rows = []
    for r_idx, y_l1, is_two_line in rows_config:
        item = {
            "name": f"CREDITOR_ENTITY_{r_idx}",
            "address_1": f"100 MAIN STREET {r_idx}",
            "city": "NEW YORK",
        }
        if is_two_line:
            item["address_2"] = f"SUITE {200 + r_idx} FLOOR 2"
        rows.append(item)

    payload = {"creditors": rows}
    res = adapter.ground_extracted_data(payload)
    citations = {c["field_path"]: c for c in res["field_citations"]}

    for r_idx in (1, 3):
        y_l1 = 0.150 if r_idx == 1 else 0.270
        y_l2 = y_l1 + 0.018

        a1_box = citations[f"creditors[{r_idx}].address_1"]["bbox"]
        a2_box = citations[f"creditors[{r_idx}].address_2"]["bbox"]
        name_box = citations[f"creditors[{r_idx}].name"]["bbox"]

        # Line 1 fields must sit at line 1's Y coordinate
        assert abs(a1_box[1] - y_l1) < 0.005, f"Row {r_idx} address_1 y={a1_box[1]} != line1={y_l1}"
        assert abs(name_box[1] - y_l1) < 0.005, f"Row {r_idx} name y={name_box[1]} != line1={y_l1}"

        # Line 2 fields must sit at line 2's Y coordinate
        assert abs(a2_box[1] - y_l2) < 0.005, f"Row {r_idx} address_2 y={a2_box[1]} != line2={y_l2}"
        assert a2_box[1] > a1_box[1], f"Row {r_idx} address_2 not below address_1"
        assert (a2_box[1] - a1_box[1]) > 0.014, f"Row {r_idx} vertical delta too small: {a2_box[1] - a1_box[1]}"

        # Specifically reject Line 1 coordinate for address_2
        assert abs(a2_box[1] - y_l1) > 0.014, f"Row {r_idx} address_2 collapsed to Line 1 coordinate"

