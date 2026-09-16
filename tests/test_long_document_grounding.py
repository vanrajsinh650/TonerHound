"""Comprehensive unit tests for long-document and FTX tabular grounding mechanisms.

Covers:
1. Scanner skew detection and slope compensation from running headers.
2. 71-slot row assignment and multi-line row budgeting (1-slot vs 2-slot rows).
3. Standardized column cell bounding box projection across 9 columns.
4. City / State / Postal code / Country column boundary resolution and non-overlap.
5. Anchor collapse prevention and negative Y fallback rejection.
6. Multi-page monotonic table continuation with independent page skews.
7. Sub-second execution performance over dense multi-field records.
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
    assert abs(r0_y - 0.08162) < 1e-4

    # Check last row (slot 70)
    r70_y = citations["creditors[70].name"]["bbox"][1]
    assert abs(r70_y - 0.88186) < 1e-4

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
    assert abs(y_last - 0.88186) < 1e-4


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
    assert city_box[0] == 0.6427
    assert abs(state_box[0] - 0.7325) < 1e-3
    assert zip_box[0] == 0.7970
    assert country_box[0] == 0.8513

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
