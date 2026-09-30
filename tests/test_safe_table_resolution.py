"""Regression tests for EXP-028D: Safe Table Resolution Integration.

Verifies the four diagnosed defect fixes and structural scoring invariants:
1. repeated CUSIP/code on same page (monotonic assignment, no anchor collapse)
2. wrong hardcoded column boundary override (dynamic header projection)
3. substring bbox requiring character slicing (safe character span enhancement)
4. repeated identical numeric values across rows (monotonic vertical disambiguation)
5. multi-record monotonic ordering (monotonic y progression across records)
"""

from __future__ import annotations

import pytest

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine
from tonerhound.resolution.joint_record_resolver import (
    JointRecordResolver,
    RecordFieldLeaf,
    ResolverMode,
    StructuralRecord,
)


# ==============================================================================
# 1. Repeated CUSIP/code on same page
# ==============================================================================

def test_repeated_cusip_code_monotonic_binding() -> None:
    """Verify that multiple rows sharing identical CUSIP bind monotonically to successive lines without collapsing."""
    # Build 3 lines on page 1 with identical CUSIP "037833100" at y=0.10, y=0.15, y=0.20
    # Line 0
    t0_name = DocumentToken("APPLE", BBox(0.05, 0.10, 0.10, 0.02, 1), page=1, char_index_in_page=0)
    t0_cusip = DocumentToken("037833100", BBox(0.30, 0.10, 0.12, 0.02, 1), page=1, char_index_in_page=6)
    t0_shrs = DocumentToken("100", BBox(0.50, 0.10, 0.05, 0.02, 1), page=1, char_index_in_page=16)
    l0 = VisualLine([t0_name, t0_cusip, t0_shrs], page=1, line_index=0, bbox=BBox(0.05, 0.10, 0.50, 0.02, 1))

    # Line 1 (different holding, same CUSIP or repeated class)
    t1_name = DocumentToken("APPLE", BBox(0.05, 0.15, 0.10, 0.02, 1), page=1, char_index_in_page=20)
    t1_cusip = DocumentToken("037833100", BBox(0.30, 0.15, 0.12, 0.02, 1), page=1, char_index_in_page=26)
    t1_shrs = DocumentToken("200", BBox(0.50, 0.15, 0.05, 0.02, 1), page=1, char_index_in_page=36)
    l1 = VisualLine([t1_name, t1_cusip, t1_shrs], page=1, line_index=1, bbox=BBox(0.05, 0.15, 0.50, 0.02, 1))

    # Line 2
    t2_name = DocumentToken("APPLE", BBox(0.05, 0.20, 0.10, 0.02, 1), page=1, char_index_in_page=40)
    t2_cusip = DocumentToken("037833100", BBox(0.30, 0.20, 0.12, 0.02, 1), page=1, char_index_in_page=46)
    t2_shrs = DocumentToken("300", BBox(0.50, 0.20, 0.05, 0.02, 1), page=1, char_index_in_page=56)
    l2 = VisualLine([t2_name, t2_cusip, t2_shrs], page=1, line_index=2, bbox=BBox(0.05, 0.20, 0.50, 0.02, 1))

    tokens = [t0_name, t0_cusip, t0_shrs, t1_name, t1_cusip, t1_shrs, t2_name, t2_cusip, t2_shrs]
    page = DocumentPage(page_number=1, width=600, height=800, tokens=tokens, lines=[l0, l1, l2])
    idx = DocumentIndex.from_pages([page])

    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)
    payload = {
        "holdings": [
            {"issuer": "APPLE", "cusip": "037833100", "shares": 100},
            {"issuer": "APPLE", "cusip": "037833100", "shares": 200},
            {"issuer": "APPLE", "cusip": "037833100", "shares": 300},
        ]
    }

    result = adapter.ground_extracted_data(payload)
    cits = {c["field_path"]: c for c in result["field_citations"]}

    # Row 0 cusip must be near y=0.10
    assert "holdings[0].cusip" in cits
    assert abs(cits["holdings[0].cusip"]["bbox"][1] - 0.10) < 0.02

    # Row 1 cusip must be near y=0.15, NOT y=0.10!
    assert "holdings[1].cusip" in cits
    assert abs(cits["holdings[1].cusip"]["bbox"][1] - 0.15) < 0.02

    # Row 2 cusip must be near y=0.20, NOT y=0.10!
    assert "holdings[2].cusip" in cits
    assert abs(cits["holdings[2].cusip"]["bbox"][1] - 0.20) < 0.02

    # Verify monotonic progression: y0 < y1 < y2
    y0 = cits["holdings[0].cusip"]["bbox"][1]
    y1 = cits["holdings[1].cusip"]["bbox"][1]
    y2 = cits["holdings[2].cusip"]["bbox"][1]
    assert y0 < y1 < y2


# ==============================================================================
# 2. Dynamic column boundary override vs static constants
# ==============================================================================

def test_dynamic_column_boundary_override() -> None:
    """Verify that columns placed outside hardcoded constants (e.g. at x=0.94) are recognized via dynamic header tokens."""
    # Header line at y=0.05 defines column "NONE" at x=0.92..0.96
    h_name = DocumentToken("NAME", BBox(0.05, 0.05, 0.10, 0.02, 1), page=1, char_index_in_page=0)
    h_cusip = DocumentToken("CUSIP", BBox(0.30, 0.05, 0.10, 0.02, 1), page=1, char_index_in_page=5)
    h_sole = DocumentToken("SOLE", BBox(0.70, 0.05, 0.08, 0.02, 1), page=1, char_index_in_page=11)
    h_shared = DocumentToken("SHARED", BBox(0.80, 0.05, 0.08, 0.02, 1), page=1, char_index_in_page=16)
    h_none = DocumentToken("NONE", BBox(0.92, 0.05, 0.05, 0.02, 1), page=1, char_index_in_page=23)
    header_line = VisualLine([h_name, h_cusip, h_sole, h_shared, h_none], page=1, line_index=0, bbox=BBox(0.05, 0.05, 0.92, 0.02, 1))

    # Data row 0 at y=0.15: voting authority none is "5000" located at x=0.94 (beyond old 0.92 limit)
    d_name = DocumentToken("MICROSOFT", BBox(0.05, 0.15, 0.15, 0.02, 1), page=1, char_index_in_page=28)
    d_cusip = DocumentToken("594918104", BBox(0.30, 0.15, 0.12, 0.02, 1), page=1, char_index_in_page=38)
    d_sole = DocumentToken("1000", BBox(0.72, 0.15, 0.06, 0.02, 1), page=1, char_index_in_page=48)
    d_shared = DocumentToken("0", BBox(0.82, 0.15, 0.02, 0.02, 1), page=1, char_index_in_page=53)
    d_none = DocumentToken("5000", BBox(0.94, 0.15, 0.04, 0.02, 1), page=1, char_index_in_page=55)
    data_line = VisualLine([d_name, d_cusip, d_sole, d_shared, d_none], page=1, line_index=1, bbox=BBox(0.05, 0.15, 0.93, 0.02, 1))

    tokens = [h_name, h_cusip, h_sole, h_shared, h_none, d_name, d_cusip, d_sole, d_shared, d_none]
    page = DocumentPage(page_number=1, width=600, height=800, tokens=tokens, lines=[header_line, data_line])
    idx = DocumentIndex.from_pages([page])

    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)
    payload = {
        "holdings": [
            {
                "issuer": "MICROSOFT",
                "cusip": "594918104",
                "voting_authority": {"sole": 1000, "shared": 0, "none": 5000},
            }
        ]
    }

    result = adapter.ground_extracted_data(payload)
    cits = {c["field_path"]: c for c in result["field_citations"]}

    assert "holdings[0].voting_authority.none" in cits
    # The bbox x must be at x=0.94 (or within the none column), not rejected
    none_bbox = cits["holdings[0].voting_authority.none"]["bbox"]
    assert abs(none_bbox[0] - 0.94) < 0.02
    assert abs(none_bbox[1] - 0.15) < 0.02


# ==============================================================================
# 3. Substring bbox requiring character slicing
# ==============================================================================

def test_substring_bbox_character_slicing() -> None:
    """Verify that geometry enhancements trim composite token bboxes down to exact character spans."""
    # Token text: "$1,234.56" spanning x=0.40 to 0.48 (width=0.08)
    # Target value: "1,234.56" (excluding the leading $)
    t_num = DocumentToken("$1,234.56", BBox(0.40, 0.20, 0.08, 0.02, 1), page=1, char_index_in_page=0)
    t_desc = DocumentToken("EQUIPMENT", BBox(0.10, 0.20, 0.15, 0.02, 1), page=1, char_index_in_page=10)
    line0 = VisualLine([t_desc, t_num], page=1, line_index=0, bbox=BBox(0.10, 0.20, 0.38, 0.02, 1))

    page = DocumentPage(page_number=1, width=600, height=800, tokens=[t_desc, t_num], lines=[line0])
    idx = DocumentIndex.from_pages([page])

    adapter = ExtractBenchAdapter(
        idx,
        enable_structural_disambiguation=True,
        enable_character_span=True,
    )
    payload = {
        "items": [
            {"description": "EQUIPMENT", "amount": "1,234.56"}
        ]
    }

    result = adapter.ground_extracted_data(payload)
    cits = {c["field_path"]: c for c in result["field_citations"]}

    assert "items[0].amount" in cits
    amt_bbox = cits["items[0].amount"]["bbox"]
    # Box should be trimmed: x must start after 0.40 (because $ is stripped) and width < 0.08
    assert amt_bbox[0] > 0.40
    assert amt_bbox[2] < 0.08


# ==============================================================================
# 4. Repeated identical numeric values across rows
# ==============================================================================

def test_repeated_numeric_values_across_rows() -> None:
    """Verify that identical numeric values (e.g. 0) in consecutive rows resolve to their respective rows."""
    r0_name = DocumentToken("ITEM A", BBox(0.10, 0.12, 0.10, 0.02, 1), page=1, char_index_in_page=0)
    r0_qty = DocumentToken("0", BBox(0.50, 0.12, 0.02, 0.02, 1), page=1, char_index_in_page=7)
    l0 = VisualLine([r0_name, r0_qty], page=1, line_index=0, bbox=BBox(0.10, 0.12, 0.42, 0.02, 1))

    r1_name = DocumentToken("ITEM B", BBox(0.10, 0.18, 0.10, 0.02, 1), page=1, char_index_in_page=9)
    r1_qty = DocumentToken("0", BBox(0.50, 0.18, 0.02, 0.02, 1), page=1, char_index_in_page=16)
    l1 = VisualLine([r1_name, r1_qty], page=1, line_index=1, bbox=BBox(0.10, 0.18, 0.42, 0.02, 1))

    page = DocumentPage(page_number=1, width=600, height=800, tokens=[r0_name, r0_qty, r1_name, r1_qty], lines=[l0, l1])
    idx = DocumentIndex.from_pages([page])

    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)
    payload = {
        "items": [
            {"name": "ITEM A", "qty": 0},
            {"name": "ITEM B", "qty": 0},
        ]
    }

    result = adapter.ground_extracted_data(payload)
    cits = {c["field_path"]: c for c in result["field_citations"]}

    assert "items[0].qty" in cits
    assert abs(cits["items[0].qty"]["bbox"][1] - 0.12) < 0.02

    assert "items[1].qty" in cits
    assert abs(cits["items[1].qty"]["bbox"][1] - 0.18) < 0.02


# ==============================================================================
# 5. Multi-record monotonic ordering
# ==============================================================================

def test_multi_record_monotonic_ordering() -> None:
    """Verify that records in a 5-row table have strictly monotonic vertical progression."""
    tokens = []
    lines = []
    y_coords = [0.10, 0.15, 0.20, 0.25, 0.30]
    char_idx = 0

    for i, y in enumerate(y_coords):
        t_desc = DocumentToken(f"SERVICE {i}", BBox(0.10, y, 0.15, 0.02, 1), page=1, char_index_in_page=char_idx)
        char_idx += 10
        t_amt = DocumentToken(f"{100 * (i + 1)}", BBox(0.40, y, 0.08, 0.02, 1), page=1, char_index_in_page=char_idx)
        char_idx += 10
        ln = VisualLine([t_desc, t_amt], page=1, line_index=i, bbox=BBox(0.10, y, 0.38, 0.02, 1))
        tokens.extend([t_desc, t_amt])
        lines.append(ln)

    page = DocumentPage(page_number=1, width=600, height=800, tokens=tokens, lines=lines)
    idx = DocumentIndex.from_pages([page])

    adapter = ExtractBenchAdapter(idx, enable_structural_disambiguation=True)
    payload = {
        "services": [
            {"desc": f"SERVICE {i}", "cost": 100 * (i + 1)}
            for i in range(5)
        ]
    }

    result = adapter.ground_extracted_data(payload)
    cits = {c["field_path"]: c for c in result["field_citations"]}

    y_positions = [cits[f"services[{i}].desc"]["bbox"][1] for i in range(5)]
    # Must be strictly increasing
    for k in range(len(y_positions) - 1):
        assert y_positions[k] < y_positions[k + 1], f"Monotonic ordering violated at row {k}: {y_positions}"
