"""Unit tests for EXP-017R Structure-Aware Table/Array Classifier."""

from __future__ import annotations

import pytest

from tonerhound.geometry.coordinates import BBox
from tonerhound.geometry.same_line_recovery import extend_same_line_tokens
from tonerhound.geometry.structure_classifier import (
    StructureProfile,
    TableStructureType,
    classify_table_structure,
)
from tonerhound.models.types import DocumentToken


class TestStructureClassifier:
    """Test suite for TableStructureType and classify_table_structure."""

    def test_insufficient_records_returns_unknown(self) -> None:
        """Arrays with fewer than 2 records lack structural evidence and fall back to UNKNOWN."""
        prof = classify_table_structure(
            table_name="single_item_table",
            records_count=1,
            col_samples={"amount": (0.50, 0.10)},
        )
        assert prof.structure_type == TableStructureType.UNKNOWN
        assert prof.reason == "insufficient_records_for_inference"
        assert prof.num_records == 1

    def test_single_field_sequence_returns_linear_record(self) -> None:
        """1D arrays where each record has only 1 field cannot contain co-linear columns."""
        row_leaves = {
            0: [("box_14[0].amount", 100, 1, "", "box_14[0]")],
            1: [("box_14[1].amount", 200, 1, "", "box_14[1]")],
            2: [("box_14[2].amount", 300, 1, "", "box_14[2]")],
        }
        prof = classify_table_structure(
            table_name="box_14",
            records_count=3,
            row_leaves_map=row_leaves,
        )
        assert prof.structure_type == TableStructureType.LINEAR_RECORD
        assert prof.reason == "single_field_1d_sequence"
        assert prof.num_columns == 1

    def test_verified_dense_grid_returns_true_2d_table(self) -> None:
        """Tables with verified row pitch regularity from Pass 1 classify as TRUE_2D_TABLE."""
        prof = classify_table_structure(
            table_name="holdings",
            records_count=50,
            col_samples={"name": (0.05, 0.20), "cusip": (0.30, 0.10)},
            is_dense_grid=True,
        )
        assert prof.structure_type == TableStructureType.TRUE_2D_TABLE
        assert prof.is_dense_grid is True
        assert prof.reason == "verified_dense_tabular_grid"

    def test_multi_column_rails_returns_true_2d_table(self) -> None:
        """Tables with 4 or more distinct column rails classify as TRUE_2D_TABLE."""
        col_samples = {
            "col_a": (0.05, 0.08),
            "col_b": (0.25, 0.08),
            "col_c": (0.45, 0.08),
            "col_d": (0.65, 0.08),
        }
        prof = classify_table_structure(
            table_name="nport_holdings",
            records_count=20,
            col_samples=col_samples,
            is_dense_grid=False,
        )
        assert prof.structure_type == TableStructureType.TRUE_2D_TABLE
        assert prof.num_columns == 4
        assert prof.reason == "multiple_independent_columns_and_colinear_fields"

    def test_co_linear_fields_returns_true_2d_table(self) -> None:
        """Tables with 3 rails and fields co-linear on the same visual line classify as TRUE_2D_TABLE."""
        col_samples = {
            "col_1": (0.10, 0.10),
            "col_2": (0.35, 0.10),
            "col_3": (0.60, 0.10),
        }
        # Two fields per record on the exact same line (y=0.20, dy=0.0)
        resolved_boxes = {
            "sched[0]": [
                BBox(x=0.10, y=0.20, width=0.10, height=0.010, page=1),
                BBox(x=0.35, y=0.20, width=0.10, height=0.010, page=1),
            ],
            "sched[1]": [
                BBox(x=0.10, y=0.25, width=0.10, height=0.010, page=1),
                BBox(x=0.35, y=0.25, width=0.10, height=0.010, page=1),
            ],
        }
        prof = classify_table_structure(
            table_name="sched",
            records_count=2,
            col_samples=col_samples,
            record_resolved_boxes=resolved_boxes,
            is_dense_grid=False,
        )
        assert prof.structure_type == TableStructureType.TRUE_2D_TABLE
        assert prof.avg_colinear_pairs >= 1.0

    def test_vertically_stacked_cards_returns_linear_record(self) -> None:
        """Card layouts where fields within each record are stacked vertically classify as LINEAR_RECORD."""
        # Fields within each record are stacked vertically on different lines
        resolved_boxes = {
            "creditors[0]": [
                BBox(x=0.10, y=0.10, width=0.20, height=0.010, page=1),
                BBox(x=0.10, y=0.13, width=0.20, height=0.010, page=1),
                BBox(x=0.10, y=0.16, width=0.20, height=0.010, page=1),
            ],
            "creditors[1]": [
                BBox(x=0.10, y=0.25, width=0.20, height=0.010, page=1),
                BBox(x=0.10, y=0.28, width=0.20, height=0.010, page=1),
                BBox(x=0.10, y=0.31, width=0.20, height=0.010, page=1),
            ],
        }
        prof = classify_table_structure(
            table_name="creditors",
            records_count=2,
            col_samples={"name": (0.10, 0.20)},
            record_resolved_boxes=resolved_boxes,
            is_dense_grid=False,
        )
        assert prof.structure_type == TableStructureType.LINEAR_RECORD
        assert prof.avg_colinear_pairs == 0.0
        assert prof.avg_vertical_pairs >= 1.0
        assert prof.reason == "linear_stream_or_vertically_stacked_cards"

    def test_single_column_rail_returns_linear_record(self) -> None:
        """Arrays with at most 1 observable column rail classify as LINEAR_RECORD."""
        prof = classify_table_structure(
            table_name="foreign_taxes",
            records_count=5,
            col_samples={"tax_amount": (0.25, 0.10)},
            is_dense_grid=False,
        )
        assert prof.structure_type == TableStructureType.LINEAR_RECORD
        assert prof.num_columns == 1

    def test_ambiguous_sparse_columns_returns_unknown(self) -> None:
        """Ambiguous 3-rail layouts without verified co-linear evidence fall back to UNKNOWN."""
        col_samples = {
            "fld_a": (0.10, 0.08),
            "fld_b": (0.35, 0.08),
            "fld_c": (0.60, 0.08),
        }
        prof = classify_table_structure(
            table_name="sparse_table",
            records_count=3,
            col_samples=col_samples,
            record_resolved_boxes={},
            is_dense_grid=False,
        )
        assert prof.structure_type == TableStructureType.UNKNOWN
        assert prof.reason == "ambiguous_sparse_columns"


def _make_token(text: str, x: float, y: float, w: float, h: float, page: int = 1) -> DocumentToken:
    return DocumentToken(
        text=text,
        bbox=BBox(x=x, y=y, width=w, height=h, page=page),
        page=page,
        char_index_in_page=0,
    )


class TestSameLineRecoveryWithMaxRightBoundary:
    """Test suite for max_right_boundary protection in extend_same_line_tokens."""

    def test_clamps_at_max_right_boundary(self) -> None:
        """extend_same_line_tokens halts before max_right_boundary."""
        tokens = [
            _make_token("Acme", 0.10, 0.20, 0.05, 0.01),
            _make_token("Corporation", 0.16, 0.20, 0.08, 0.01),
            _make_token("International", 0.25, 0.20, 0.10, 0.01),
        ]
        start_box = BBox(0.10, 0.20, 0.05, 0.01, page=1)

        # Without boundary: recovers all 3 tokens up to x=0.35
        unconstrained = extend_same_line_tokens(
            cand_bbox=start_box,
            reference_text="Acme",
            target_value="Acme Corporation International",
            line_tokens=tokens,
            confidence=0.90,
        )
        assert unconstrained.width > 0.20

        # With boundary at x=0.24 (e.g. adjacent column starts at x=0.25)
        # Token "International" at x=0.25 exceeds 0.24, so it is strictly stopped
        constrained = extend_same_line_tokens(
            cand_bbox=start_box,
            reference_text="Acme",
            target_value="Acme Corporation International",
            line_tokens=tokens,
            confidence=0.90,
            max_right_boundary=0.24,
        )
        assert constrained.x + constrained.width <= 0.24 + 1e-6
        assert constrained.width >= start_box.width * 1.25
