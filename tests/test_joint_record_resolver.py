"""Comprehensive unit tests for JointRecordResolver (EXP-028B2).

Tests all Phase 3 requirements:
- repeated numeric values
- repeated text values
- repeated codes
- same column, different rows
- same row, different columns
- multi-field records
- tables with missing headers
- tables spanning pages
- tables with wrapped cells
- irregular tables
- ambiguity / abstention gating
- ablation modes A through F
"""

import pytest
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.resolution.joint_record_resolver import (
    FieldResolutionResult,
    JointRecordResolver,
    RecordFieldLeaf,
    ResolverMode,
    StructuralRecord,
)


@pytest.fixture
def resolver() -> JointRecordResolver:
    return JointRecordResolver()


def _make_candidate(text: str, bbox: BBox, page: int, sim: float = 1.0) -> MatchCandidate:
    return MatchCandidate(
        matched_text=text,
        bbox=bbox,
        page=page,
        tokens=(),
        match_type="exact",
        raw_similarity=sim,
    )


# ==============================================================================
# 1. Repeated Numeric Values & Same Column, Different Rows
# ==============================================================================

def test_repeated_numeric_values_different_rows(resolver: JointRecordResolver):
    """Test disambiguation of identical number '0' across two different rows in same column."""
    rec_row1 = StructuralRecord(
        table_name="holdings",
        row_index=10,
        record_path="holdings[10]",
        page=5,
        row_y_center=0.300,
        row_height=0.012,
        is_anchored=True,
    )
    rec_row2 = StructuralRecord(
        table_name="holdings",
        row_index=11,
        record_path="holdings[11]",
        page=5,
        row_y_center=0.320,
        row_height=0.012,
        is_anchored=True,
    )

    # Two identical '0' tokens at same X column (0.80) on different rows
    cand_row1 = _make_candidate("0", BBox(x=0.80, y=0.295, width=0.01, height=0.010, page=5), page=5)
    cand_row2 = _make_candidate("0", BBox(x=0.80, y=0.315, width=0.01, height=0.010, page=5), page=5)

    # When scoring for row 1: cand_row1 must win decisively
    ranked_1 = resolver.rank_record_candidates([cand_row1, cand_row2], "shared", rec_row1, is_numeric=True)
    assert ranked_1[0][0] == cand_row1
    assert ranked_1[0][1] > ranked_1[1][1]

    # When scoring for row 2: cand_row2 must win decisively
    ranked_2 = resolver.rank_record_candidates([cand_row1, cand_row2], "shared", rec_row2, is_numeric=True)
    assert ranked_2[0][0] == cand_row2
    assert ranked_2[0][1] > ranked_2[1][1]


# ==============================================================================
# 2. Repeated Values Across Columns (Same Row, Different Columns)
# ==============================================================================

def test_repeated_values_different_columns(resolver: JointRecordResolver):
    """Test disambiguation of identical value '0' across two different columns in the same row."""
    rec = StructuralRecord(
        table_name="holdings",
        row_index=5,
        record_path="holdings[5]",
        page=3,
        row_y_center=0.450,
        row_height=0.012,
        is_anchored=True,
    )

    col_corridors = {
        "shared": (0.75, 0.82),
        "none": (0.85, 0.92),
    }

    cand_shared = _make_candidate("0", BBox(x=0.79, y=0.445, width=0.01, height=0.010, page=3), page=3)
    cand_none = _make_candidate("0", BBox(x=0.89, y=0.445, width=0.01, height=0.010, page=3), page=3)

    # For 'shared': cand_shared must win
    ranked_shared = resolver.rank_record_candidates(
        [cand_shared, cand_none],
        "shared",
        rec,
        column_corridor=col_corridors["shared"],
        all_column_corridors=col_corridors,
        is_numeric=True,
    )
    assert ranked_shared[0][0] == cand_shared

    # For 'none': cand_none must win
    ranked_none = resolver.rank_record_candidates(
        [cand_shared, cand_none],
        "none",
        rec,
        column_corridor=col_corridors["none"],
        all_column_corridors=col_corridors,
        is_numeric=True,
    )
    assert ranked_none[0][0] == cand_none


# ==============================================================================
# 3. Repeated Text & Boilerplate Disambiguation
# ==============================================================================

def test_repeated_text_boilerplate(resolver: JointRecordResolver):
    """Test disambiguation of boilerplate string 'SOLE' across multiple columns and rows."""
    rec = StructuralRecord(
        table_name="holdings",
        row_index=3,
        record_path="holdings[3]",
        page=2,
        row_y_center=0.250,
        row_height=0.010,
        is_anchored=True,
    )
    col_corridors = {
        "investment_discretion": (0.55, 0.65),
        "voting_authority": (0.70, 0.80),
    }

    cand_inv = _make_candidate("SOLE", BBox(x=0.58, y=0.245, width=0.04, height=0.010, page=2), page=2)
    cand_vot = _make_candidate("SOLE", BBox(x=0.73, y=0.245, width=0.04, height=0.010, page=2), page=2)
    cand_other_row = _make_candidate("SOLE", BBox(x=0.58, y=0.300, width=0.04, height=0.010, page=2), page=2)

    # Discretion must pick cand_inv
    res_inv = resolver.rank_record_candidates(
        [cand_other_row, cand_vot, cand_inv],
        "investment_discretion",
        rec,
        column_corridor=col_corridors["investment_discretion"],
        all_column_corridors=col_corridors,
    )
    assert res_inv[0][0] == cand_inv

    # Voting must pick cand_vot
    res_vot = resolver.rank_record_candidates(
        [cand_other_row, cand_vot, cand_inv],
        "voting_authority",
        rec,
        column_corridor=col_corridors["voting_authority"],
        all_column_corridors=col_corridors,
    )
    assert res_vot[0][0] == cand_vot


# ==============================================================================
# 4. Repeated Codes (e.g. CUSIP / Ticker / State Codes)
# ==============================================================================

def test_repeated_codes_as_anchors(resolver: JointRecordResolver):
    """Test that a distinctive code like CUSIP '037833100' is preferred as anchor over generic strings."""
    fields = [
        RecordFieldLeaf(path="holdings[0].title", field_name="title", value="COM"),
        RecordFieldLeaf(path="holdings[0].shares", field_name="shares", value=1500),
        RecordFieldLeaf(path="holdings[0].cusip", field_name="cusip", value="037833100"),
    ]
    cands = {
        "title": [_make_candidate("COM", BBox(x=0.20, y=0.15, width=0.03, height=0.01, page=1), 1)],
        "shares": [_make_candidate("1500", BBox(x=0.50, y=0.15, width=0.03, height=0.01, page=1), 1)],
        "cusip": [_make_candidate("037833100", BBox(x=0.30, y=0.15, width=0.06, height=0.01, page=1), 1)],
    }
    anc_fld, anc_cand = resolver.identify_record_anchor(fields, cands)
    assert anc_fld == "cusip"
    assert anc_cand.matched_text == "037833100"


# ==============================================================================
# 5. Tables Spanning Pages & Wrong Page Penalty
# ==============================================================================

def test_wrong_page_penalty(resolver: JointRecordResolver):
    """Test candidate on wrong page receives a severe penalty even if coordinates match."""
    rec = StructuralRecord(
        table_name="holdings",
        row_index=1,
        record_path="holdings[1]",
        page=72,
        row_y_center=0.793,
        row_height=0.010,
        is_anchored=True,
    )

    cand_p72 = _make_candidate("SOLE", BBox(x=0.63, y=0.790, width=0.03, height=0.010, page=72), page=72)
    cand_p73 = _make_candidate("SOLE", BBox(x=0.63, y=0.790, width=0.03, height=0.010, page=73), page=73)

    ranked = resolver.rank_record_candidates([cand_p73, cand_p72], "investment_discretion", rec)
    assert ranked[0][0] == cand_p72
    assert ranked[0][1] > 20.0
    assert ranked[1][1] < 0.0


# ==============================================================================
# 6. Sibling Co-Linearity Bonus & Multi-Field Records
# ==============================================================================

def test_sibling_colinearity_bonus(resolver: JointRecordResolver):
    """Test that existing resolved fields in the same row boost co-linear candidates."""
    rec = StructuralRecord(
        table_name="holdings",
        row_index=1,
        record_path="holdings[1]",
        page=10,
        row_y_center=0.500,
        row_height=0.012,
        is_anchored=True,
        resolved_fields={
            "name_of_issuer": BBox(x=0.05, y=0.495, width=0.15, height=0.010, page=10),
            "cusip": BBox(x=0.30, y=0.495, width=0.05, height=0.010, page=10),
        },
    )

    cand_colinear = _make_candidate("1,000", BBox(x=0.50, y=0.495, width=0.04, height=0.010, page=10), page=10, sim=0.85)
    cand_other_row = _make_candidate("1,000", BBox(x=0.50, y=0.540, width=0.04, height=0.010, page=10), page=10, sim=0.95)

    ranked = resolver.rank_record_candidates([cand_other_row, cand_colinear], "shares", rec, is_numeric=True)
    assert ranked[0][0] == cand_colinear


# ==============================================================================
# 7. Tables with Wrapped Cells (Multi-line row expansion)
# ==============================================================================

def test_wrapped_cells_row_expansion(resolver: JointRecordResolver):
    """Test that wrapped cells spanning 2 lines are tolerated within expanded row height."""
    rec = StructuralRecord(
        table_name="creditors",
        row_index=4,
        record_path="creditors[4]",
        page=1,
        row_y_center=0.200,
        row_height=0.024,  # Multi-line row height
        is_anchored=True,
    )
    # Line 1 of wrapped cell (y ~ 0.192), Line 2 of wrapped cell (y ~ 0.208)
    cand_line1 = _make_candidate("123 MAIN STREET", BBox(x=0.25, y=0.192, width=0.12, height=0.010, page=1), 1)
    cand_line2 = _make_candidate("SUITE 400", BBox(x=0.25, y=0.208, width=0.08, height=0.010, page=1), 1)
    cand_far = _make_candidate("456 OTHER ROAD", BBox(x=0.25, y=0.250, width=0.12, height=0.010, page=1), 1)

    r1 = resolver.rank_record_candidates([cand_far, cand_line1], "address_1", rec)
    r2 = resolver.rank_record_candidates([cand_far, cand_line2], "address_2", rec)

    assert r1[0][0] == cand_line1
    assert r2[0][0] == cand_line2
    assert r1[0][1] > 10.0
    assert r2[0][1] > 10.0


# ==============================================================================
# 8. Tables with Missing Headers (Unanchored Corridors)
# ==============================================================================

def test_tables_missing_headers_joint_resolution(resolver: JointRecordResolver):
    """Test joint resolution when explicit column corridors are missing but row anchor is known."""
    rec = StructuralRecord(
        table_name="items",
        row_index=2,
        record_path="items[2]",
        page=1,
        row_y_center=0.400,
        row_height=0.012,
        is_anchored=True,
    )
    fields = [
        RecordFieldLeaf(path="items[2].desc", field_name="desc", value="WIDGET A"),
        RecordFieldLeaf(path="items[2].qty", field_name="qty", value=5),
        RecordFieldLeaf(path="items[2].price", field_name="price", value=19.99),
    ]
    cands = {
        "desc": [
            _make_candidate("WIDGET A", BBox(x=0.10, y=0.395, width=0.10, height=0.010, page=1), 1),
            _make_candidate("WIDGET A", BBox(x=0.10, y=0.550, width=0.10, height=0.010, page=1), 1),
        ],
        "qty": [
            _make_candidate("5", BBox(x=0.30, y=0.395, width=0.02, height=0.010, page=1), 1),
            _make_candidate("5", BBox(x=0.30, y=0.550, width=0.02, height=0.010, page=1), 1),
        ],
        "price": [
            _make_candidate("19.99", BBox(x=0.45, y=0.395, width=0.05, height=0.010, page=1), 1),
            _make_candidate("19.99", BBox(x=0.45, y=0.550, width=0.05, height=0.010, page=1), 1),
        ],
    }
    # No column corridors provided
    results = resolver.resolve_record(rec, fields, cands, column_corridors=None)

    assert results["desc"].candidate.bbox.y == pytest.approx(0.395)
    assert results["qty"].candidate.bbox.y == pytest.approx(0.395)
    assert results["price"].candidate.bbox.y == pytest.approx(0.395)


# ==============================================================================
# 9. Irregular Tables with Missing / Sparse Values
# ==============================================================================

def test_irregular_tables_sparse_fields(resolver: JointRecordResolver):
    """Test resolution when some cells are empty or missing candidates."""
    rec = StructuralRecord(
        table_name="grants",
        row_index=0,
        record_path="grants[0]",
        page=4,
        row_y_center=0.600,
        row_height=0.015,
        is_anchored=True,
    )
    fields = [
        RecordFieldLeaf(path="grants[0].recipient", field_name="recipient", value="UNIVERSITY OF X"),
        RecordFieldLeaf(path="grants[0].purpose", field_name="purpose", value=None),  # Missing
        RecordFieldLeaf(path="grants[0].amount", field_name="amount", value=50000),
    ]
    cands = {
        "recipient": [_make_candidate("UNIVERSITY OF X", BBox(x=0.05, y=0.595, width=0.20, height=0.010, page=4), 4)],
        "purpose": [],
        "amount": [_make_candidate("50,000", BBox(x=0.70, y=0.595, width=0.06, height=0.010, page=4), 4)],
    }
    results = resolver.resolve_record(rec, fields, cands)
    assert results["recipient"].candidate is not None
    assert results["purpose"].candidate is None
    assert results["amount"].candidate is not None


# ==============================================================================
# 10. Ambiguity / Abstention Gating (Mode F)
# ==============================================================================

def test_ambiguity_abstention_gating():
    """Test that Mode F abstains when two candidates cannot be distinguished structurally."""
    resolver_gated = JointRecordResolver(enable_ambiguity_gating=True, mode=ResolverMode.AMBIGUITY_GATED)

    rec = StructuralRecord(
        table_name="misc",
        row_index=0,
        record_path="misc[0]",
        page=1,
        # No row center, no column info: totally ambiguous
        row_y_center=None,
        is_anchored=False,
    )
    fields = [RecordFieldLeaf(path="misc[0].val", field_name="val", value="ABC")]
    # Two identical candidates on same page with identical similarity and no row/column constraints
    c1 = _make_candidate("ABC", BBox(x=0.10, y=0.20, width=0.05, height=0.01, page=1), 1)
    c2 = _make_candidate("ABC", BBox(x=0.10, y=0.80, width=0.05, height=0.01, page=1), 1)
    cands = {"val": [c1, c2]}

    res = resolver_gated.resolve_record(rec, fields, cands)
    assert res["val"].is_ambiguous is True
    assert res["val"].is_abstained is True
    assert res["val"].candidate is None


# ==============================================================================
# 11. Controlled Ablation Modes A through F
# ==============================================================================

def test_ablation_modes_a_through_f():
    """Verify that all Phase 4 ablation modes (A, B, C, D, E, F) execute and produce distinct behaviors."""
    rec = StructuralRecord(
        table_name="test_table",
        row_index=1,
        record_path="test_table[1]",
        page=2,
        row_y_center=0.500,
        row_height=0.010,
        is_anchored=True,
    )
    cols = {"amt": (0.60, 0.70)}

    # C_good: matches page, row, and column
    c_good = _make_candidate("100", BBox(x=0.65, y=0.495, width=0.03, height=0.01, page=2), 2, sim=0.80)
    # C_high_sim_bad_geom: higher text sim, but wrong row and wrong column
    c_high_sim = _make_candidate("100", BBox(x=0.10, y=0.800, width=0.03, height=0.01, page=2), 2, sim=1.00)

    candidates = [c_high_sim, c_good]

    # Mode A (Baseline): pure text sim -> c_high_sim wins
    res_a = JointRecordResolver().rank_record_candidates(candidates, "amt", rec, column_corridor=cols["amt"], mode=ResolverMode.BASELINE)
    assert res_a[0][0] == c_high_sim

    # Mode B (Row-Anchor Only): row score boosts c_good over c_high_sim
    res_b = JointRecordResolver().rank_record_candidates(candidates, "amt", rec, column_corridor=cols["amt"], mode=ResolverMode.ROW_ANCHOR_ONLY)
    assert res_b[0][0] == c_good

    # Mode C (Column Only): column score boosts c_good over c_high_sim
    res_c = JointRecordResolver().rank_record_candidates(candidates, "amt", rec, column_corridor=cols["amt"], mode=ResolverMode.COLUMN_ONLY)
    assert res_c[0][0] == c_good

    # Mode D (Row + Column): combines both
    res_d = JointRecordResolver().rank_record_candidates(candidates, "amt", rec, column_corridor=cols["amt"], mode=ResolverMode.ROW_AND_COLUMN)
    assert res_d[0][0] == c_good
    assert res_d[0][1] > res_b[0][1]

    # Mode E (Joint Record): includes sibling context
    rec.resolved_fields["title"] = BBox(x=0.20, y=0.495, width=0.10, height=0.01, page=2)
    res_e = JointRecordResolver().rank_record_candidates(candidates, "amt", rec, column_corridor=cols["amt"], mode=ResolverMode.JOINT_RECORD)
    assert res_e[0][0] == c_good
    assert res_e[0][1] > res_d[0][1]  # Higher score due to sibling co-linearity

    # Mode F (Ambiguity Gated): verify execution
    res_f = JointRecordResolver().rank_record_candidates(candidates, "amt", rec, column_corridor=cols["amt"], mode=ResolverMode.AMBIGUITY_GATED)
    assert res_f[0][0] == c_good
