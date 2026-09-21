"""EXP-017R Structure-Aware Table and Array Structural Classifier.

Provides generic, observable geometric classification of repeating records and tables
into TRUE_2D_TABLE (multi-column grids), LINEAR_RECORD (1D sequences or vertically-stacked cards),
and UNKNOWN (ambiguous layouts requiring conservative protection).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tonerhound.geometry.coordinates import BBox


class TableStructureType(str, enum.Enum):
    """Generic structural classification of an array or table."""

    TRUE_2D_TABLE = "TRUE_2D_TABLE"     # Multi-column grid; multiple fields side-by-side on same row
    LINEAR_RECORD = "LINEAR_RECORD"     # 1D sequence or card layout; at most 1 field per visual line
    UNKNOWN = "UNKNOWN"                 # Indeterminate structure; conservative fallback to protected


@dataclass(frozen=True)
class StructureProfile:
    """Detailed structural diagnostic profile for an array or table."""

    structure_type: TableStructureType
    num_records: int
    num_columns: int
    avg_colinear_pairs: float
    avg_vertical_pairs: float
    is_dense_grid: bool
    reason: str


def classify_table_structure(
    table_name: str,
    records_count: int,
    col_samples: Mapping[str, Any] | None = None,
    record_anchors: Mapping[str, tuple[int, float, float, Any, Any]] | None = None,
    record_resolved_boxes: Mapping[str, Sequence[BBox]] | None = None,
    is_dense_grid: bool = False,
    row_leaves_map: Mapping[int, Sequence[Any]] | None = None,
) -> StructureProfile:
    """Classifies array/table structure using purely observable geometric signals.

    Signals evaluated:
    1. Sample size: Arrays with < 2 records lack structural evidence -> UNKNOWN.
    2. Field density: Single-field records cannot contain competing columns -> LINEAR_RECORD.
    3. Grid regularity: Verified dense tabular grid from observable row pitch regularity -> TRUE_2D_TABLE.
    4. Observable column rails: Distinct horizontal column positions derived from physical tokens.
    5. Baseline co-linearity: Whether field pairs within records share a visual baseline vs stack vertically.

    Zero reliance on document IDs, benchmark-specific templates, or semantic field names.
    """
    if records_count < 2:
        return StructureProfile(
            structure_type=TableStructureType.UNKNOWN,
            num_records=records_count,
            num_columns=0,
            avg_colinear_pairs=0.0,
            avg_vertical_pairs=0.0,
            is_dense_grid=is_dense_grid,
            reason="insufficient_records_for_inference",
        )

    # Signal 1: Single-field records cannot have horizontal co-linear fields
    max_fields_per_record = 0
    if row_leaves_map:
        max_fields_per_record = max((len(rleaves) for rleaves in row_leaves_map.values()), default=0)
        if max_fields_per_record == 1:
            return StructureProfile(
                structure_type=TableStructureType.LINEAR_RECORD,
                num_records=records_count,
                num_columns=1,
                avg_colinear_pairs=0.0,
                avg_vertical_pairs=0.0,
                is_dense_grid=False,
                reason="single_field_1d_sequence",
            )

    # Signal 2: Verified dense grid from observable row pitch regularity
    if is_dense_grid:
        num_cols = len(col_samples) if col_samples else 0
        return StructureProfile(
            structure_type=TableStructureType.TRUE_2D_TABLE,
            num_records=records_count,
            num_columns=num_cols,
            avg_colinear_pairs=2.0,
            avg_vertical_pairs=0.0,
            is_dense_grid=True,
            reason="verified_dense_tabular_grid",
        )

    # Signal 3: Observable horizontal column rails
    distinct_cols: list[tuple[str, float]] = []
    if col_samples:
        for fld, val in col_samples.items():
            if isinstance(val, (tuple, list)) and len(val) >= 2 and isinstance(val[0], (int, float)):
                distinct_cols.append((fld, float(val[0])))
            elif isinstance(val, list) and val and isinstance(val[0], (tuple, list)):
                if len(val) >= max(2, int(records_count * 0.15)):
                    med_x = sorted(s[0] for s in val)[len(val) // 2]
                    distinct_cols.append((fld, float(med_x)))

    distinct_cols.sort(key=lambda item: item[1])
    merged_rails: list[float] = []
    for _fld, x in distinct_cols:
        if not merged_rails or (x - merged_rails[-1] > 0.035):
            merged_rails.append(x)
    num_rails = len(merged_rails)

    # Signal 4: Co-linear vs vertically-stacked fields within records
    colinear_counts: list[int] = []
    vertical_counts: list[int] = []
    if record_resolved_boxes:
        for rk, boxes in record_resolved_boxes.items():
            if not rk.startswith(f"{table_name}[") or len(boxes) < 2:
                continue
            c_pairs = 0
            v_pairs = 0
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    b1 = boxes[i]
                    b2 = boxes[j]
                    if b1.page == b2.page:
                        dy = abs(b1.y - b2.y)
                        if dy <= max(0.012, max(b1.height, b2.height) * 0.85):
                            dx = abs(b1.x - b2.x)
                            if dx >= 0.035:
                                c_pairs += 1
                        else:
                            v_pairs += 1
            colinear_counts.append(c_pairs)
            vertical_counts.append(v_pairs)

    avg_colinear = sum(colinear_counts) / len(colinear_counts) if colinear_counts else 0.0
    avg_vert = sum(vertical_counts) / len(vertical_counts) if vertical_counts else 0.0

    # Decision Matrix:
    # 1. TRUE_2D_TABLE / GRID:
    #    Multiple distinct columns (>= 3 rails) AND multiple fields sharing rows (avg_colinear >= 0.80)
    #    OR >= 4 stable columns side-by-side
    if (num_rails >= 3 and avg_colinear >= 0.80) or num_rails >= 4:
        return StructureProfile(
            structure_type=TableStructureType.TRUE_2D_TABLE,
            num_records=records_count,
            num_columns=num_rails,
            avg_colinear_pairs=avg_colinear,
            avg_vertical_pairs=avg_vert,
            is_dense_grid=False,
            reason="multiple_independent_columns_and_colinear_fields",
        )

    # 2. LINEAR_RECORD / 1D_SEQUENCE:
    #    - Single column track (num_rails <= 1)
    #    - OR vertically stacked card records (avg_colinear < 0.35 and avg_vert >= 0.80)
    #    - OR 2 columns but strictly vertically decoupled (avg_colinear < 0.25)
    if (num_rails <= 1) or (num_rails <= 2 and avg_colinear < 0.35) or (avg_colinear < 0.20 and avg_vert >= 1.0):
        return StructureProfile(
            structure_type=TableStructureType.LINEAR_RECORD,
            num_records=records_count,
            num_columns=num_rails,
            avg_colinear_pairs=avg_colinear,
            avg_vertical_pairs=avg_vert,
            is_dense_grid=False,
            reason="linear_stream_or_vertically_stacked_cards",
        )

    # 3. UNKNOWN / AMBIGUOUS: Conservative fallback to protected behavior
    return StructureProfile(
        structure_type=TableStructureType.UNKNOWN,
        num_records=records_count,
        num_columns=num_rails,
        avg_colinear_pairs=avg_colinear,
        avg_vertical_pairs=avg_vert,
        is_dense_grid=False,
        reason="ambiguous_sparse_columns",
    )
