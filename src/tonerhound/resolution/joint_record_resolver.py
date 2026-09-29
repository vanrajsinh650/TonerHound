"""Joint Row/Record Resolution and Structural Context Scoring (EXP-028B2).

Provides extraction-aware structural disambiguation for tabular and multi-field records:
1. Identifies distinctive anchors for each record (high-entropy key fields).
2. Establishes structural record context: page, row vertical bounds, and column corridors.
3. Resolves all fields within the record jointly, constraining search to the record corridor.
4. Distinguishes repeated values across rows and columns using horizontal column rails and vertical row alignment.
5. Gates ambiguous candidates when structural evidence is insufficient.
6. Supports controlled experiment ablation modes A through F.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentToken


class ResolverMode(str, Enum):
    """Ablation modes for Phase 4 controlled experiments."""
    BASELINE = "A"          # Pure text match, no structural context
    ROW_ANCHOR_ONLY = "B"   # Page and row vertical confinement only
    COLUMN_ONLY = "C"       # Page and column corridor only
    ROW_AND_COLUMN = "D"    # Page, row, and column context
    JOINT_RECORD = "E"      # Joint multi-field record resolution + sibling co-linearity
    AMBIGUITY_GATED = "F"   # Mode E + abstention / ambiguity gating


@dataclass
class RecordFieldLeaf:
    """A leaf field belonging to a structured record."""
    path: str
    field_name: str
    value: Any
    page_hint: int | None = None
    context: str | None = None
    parent_record_path: str = ""


@dataclass
class StructuralRecord:
    """Represents the resolved structural geometry of a table row or record."""
    table_name: str
    row_index: int
    record_path: str
    page: int | None = None
    anchor_field: str | None = None
    anchor_bbox: BBox | None = None
    row_y_center: float | None = None
    row_height: float = 0.012
    is_anchored: bool = False
    resolved_fields: dict[str, BBox] = field(default_factory=dict)
    field_order: list[str] = field(default_factory=list)


@dataclass
class FieldResolutionResult:
    """Outcome of resolving a single field within a structured record."""
    field_name: str
    field_path: str
    candidate: MatchCandidate | None
    score: float
    breakdown: dict[str, float]
    is_ambiguous: bool = False
    is_abstained: bool = False


# High-frequency boilerplate tokens that are poor anchors for records
BOILERPLATE_ANCHORS = frozenset({
    "COMMON STOCK", "COM", "ORDINARY SHARES", "CLASS A", "CLASS B",
    "CASH", "GENERAL", "SUPPORT", "OPERATING", "SOLE", "SHARED",
    "NONE", "TRUE", "FALSE", "YES", "NO", "N/A", "NA", "NULL",
    "CALL", "PUT", "TOTAL", "SUBTOTAL", "0", "1", "2", "3",
    "NAME ON FILE", "ADDRESS ON FILE", "SEE ATTACHED", "UNKNOWN",
})


class JointRecordResolver:
    """Coordinates joint row-and-column resolution for structured document records."""

    def __init__(
        self,
        column_corridor_tolerance: float = 0.03,
        row_height_tolerance_factor: float = 1.35,
        enable_ambiguity_gating: bool = True,
        ambiguity_margin: float = 0.05,
        mode: ResolverMode = ResolverMode.JOINT_RECORD,
    ) -> None:
        self.column_corridor_tolerance = column_corridor_tolerance
        self.row_height_tolerance_factor = row_height_tolerance_factor
        self.enable_ambiguity_gating = enable_ambiguity_gating
        self.ambiguity_margin = ambiguity_margin
        self.mode = mode

    def score_candidate(
        self,
        candidate: MatchCandidate,
        field_name: str,
        record: StructuralRecord,
        column_corridor: tuple[float, float] | None = None,
        all_column_corridors: dict[str, tuple[float, float]] | None = None,
        is_numeric: bool = False,
        mode: ResolverMode | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Score a candidate against the structural context of the record.

        Answers: "Does this candidate belong to the same structural record as the other requested fields?"
        """
        active_mode = mode or self.mode
        c_box = candidate.bbox
        scores: dict[str, float] = {}

        # Mode A: Baseline (pure text similarity, ignore geometry)
        if active_mode == ResolverMode.BASELINE:
            sim = getattr(candidate, "raw_similarity", 1.0)
            scores["text_sim"] = sim * 10.0
            return scores["text_sim"], scores

        # 1. Page Context (Active in B, C, D, E, F)
        if record.page is not None:
            if candidate.page == record.page:
                scores["page"] = 15.0
            else:
                scores["page"] = -25.0
        else:
            scores["page"] = 0.0

        # 2. Row Vertical Confinement (Active in B, D, E, F)
        use_row = active_mode in (
            ResolverMode.ROW_ANCHOR_ONLY,
            ResolverMode.ROW_AND_COLUMN,
            ResolverMode.JOINT_RECORD,
            ResolverMode.AMBIGUITY_GATED,
        )
        if use_row and record.row_y_center is not None:
            c_yc = c_box.y + c_box.height / 2.0
            delta_y = abs(c_yc - record.row_y_center)
            eff_h = max(0.008, record.row_height * self.row_height_tolerance_factor)

            if delta_y <= eff_h * 0.60:
                # Close to row center line: exponential decay
                sigma_y = max(0.004, eff_h * 0.50)
                scores["row"] = 12.0 * math.exp(-(delta_y**2) / (2.0 * sigma_y**2))
            elif delta_y <= eff_h * 1.25:
                # Within row bounds: linear decline
                scores["row"] = 6.0 * (1.0 - (delta_y / (eff_h * 1.25)))
            else:
                # Outside row corridor: strong penalty proportional to distance
                excess = delta_y - eff_h
                scores["row"] = -15.0 * min(2.0, excess / (eff_h + 1e-4))
        else:
            scores["row"] = 0.0

        # 3. Column Rail & Corridor Alignment (Active in C, D, E, F)
        use_col = active_mode in (
            ResolverMode.COLUMN_ONLY,
            ResolverMode.ROW_AND_COLUMN,
            ResolverMode.JOINT_RECORD,
            ResolverMode.AMBIGUITY_GATED,
        )
        c_xc = c_box.x + c_box.width / 2.0
        c_x0 = c_box.x
        c_x1 = c_box.x + c_box.width

        if use_col and column_corridor is not None:
            col_x0, col_x1 = column_corridor
            col_w = max(0.010, col_x1 - col_x0)

            # Check overlap with designated column corridor
            overlap = max(0.0, min(c_x1, col_x1) - max(c_x0, col_x0))
            containment = overlap / max(1e-4, c_box.width)

            if is_numeric:
                # Numeric fields align right rail
                delta_rail = abs(c_x1 - col_x1)
            else:
                # Text fields align left rail
                delta_rail = abs(c_x0 - col_x0)

            sigma_x = max(0.008, col_w * 0.40)
            rail_score = math.exp(-(delta_rail**2) / (2.0 * sigma_x**2))

            if containment >= 0.50 or delta_rail <= 0.025:
                scores["column"] = 10.0 * (0.6 * rail_score + 0.4 * containment)
            else:
                # Outside assigned column
                dist_outside = max(0.0, col_x0 - c_x1, c_x0 - col_x1)
                scores["column"] = -10.0 * min(1.5, dist_outside / 0.040)
        elif use_col and all_column_corridors:
            # We don't have this field's column, but check if candidate falls inside another column
            in_other_col = False
            for other_fld, (ox0, ox1) in all_column_corridors.items():
                if other_fld != field_name:
                    if ox0 <= c_xc <= ox1:
                        in_other_col = True
                        break
            if in_other_col:
                scores["column"] = -5.0
            else:
                scores["column"] = 0.0
        else:
            scores["column"] = 0.0

        # 4. Same-Row Sibling Proximity & Co-Linearity (Active in E, F)
        use_sibling = active_mode in (ResolverMode.JOINT_RECORD, ResolverMode.AMBIGUITY_GATED)
        if use_sibling and record.resolved_fields and record.page is not None and candidate.page == record.page:
            sib_scores = []
            for sf_name, sf_box in record.resolved_fields.items():
                sf_yc = sf_box.y + sf_box.height / 2.0
                dy = abs(c_box.y + c_box.height / 2.0 - sf_yc)
                # Strong co-linearity bonus if on identical baseline
                if dy <= max(0.006, record.row_height * 0.40):
                    sib_scores.append(8.0 * math.exp(-(dy**2) / (2.0 * 0.004**2)))
                elif dy <= record.row_height * 0.90:
                    sib_scores.append(4.0)
                else:
                    sib_scores.append(-4.0)
            scores["sibling"] = max(sib_scores) if sib_scores else 0.0
        else:
            scores["sibling"] = 0.0

        # 5. Horizontal Reading-Order Consistency (Active in E, F)
        if use_sibling and record.field_order and record.resolved_fields and candidate.page == record.page:
            order_score = self._score_horizontal_order(candidate, field_name, record)
            scores["reading_order"] = order_score
        else:
            scores["reading_order"] = 0.0

        sim = getattr(candidate, "raw_similarity", 1.0)
        total_score = sum(scores.values()) + (sim * 2.0)
        return total_score, scores

    def _score_horizontal_order(
        self,
        candidate: MatchCandidate,
        field_name: str,
        record: StructuralRecord,
    ) -> float:
        """Check whether candidate preserves left-to-right field ordering with respect to resolved siblings."""
        if field_name not in record.field_order:
            return 0.0
        cur_idx = record.field_order.index(field_name)
        c_x = candidate.bbox.x

        score = 0.0
        for sib_name, sib_box in record.resolved_fields.items():
            if sib_name not in record.field_order:
                continue
            sib_idx = record.field_order.index(sib_name)
            if sib_idx < cur_idx:
                # Sibling should be to the left of candidate
                if sib_box.x + sib_box.width <= c_x + 0.01:
                    score += 1.5
                elif sib_box.x > c_x:
                    score -= 3.0
            elif sib_idx > cur_idx:
                # Sibling should be to the right of candidate
                if c_x + candidate.bbox.width <= sib_box.x + 0.01:
                    score += 1.5
                elif c_x > sib_box.x:
                    score -= 3.0
        return score

    def rank_record_candidates(
        self,
        candidates: list[MatchCandidate],
        field_name: str,
        record: StructuralRecord,
        column_corridor: tuple[float, float] | None = None,
        all_column_corridors: dict[str, tuple[float, float]] | None = None,
        is_numeric: bool = False,
        mode: ResolverMode | None = None,
    ) -> list[tuple[MatchCandidate, float, dict[str, float]]]:
        """Rank candidates using structural record scoring."""
        scored = []
        for cand in candidates:
            score, breakdown = self.score_candidate(
                candidate=cand,
                field_name=field_name,
                record=record,
                column_corridor=column_corridor,
                all_column_corridors=all_column_corridors,
                is_numeric=is_numeric,
                mode=mode,
            )
            scored.append((cand, score, breakdown))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def identify_record_anchor(
        self,
        fields: list[RecordFieldLeaf],
        candidates_by_field: dict[str, list[MatchCandidate]],
    ) -> tuple[str | None, MatchCandidate | None]:
        """Identify the most distinctive anchor candidate for a record."""
        best_anchor_field: str | None = None
        best_anchor_cand: MatchCandidate | None = None
        best_anchor_score = -1.0

        for fld in fields:
            val = fld.value
            if val is None or isinstance(val, bool):
                continue
            val_str = str(val).strip()
            cands = candidates_by_field.get(fld.field_name, [])
            if not cands:
                continue

            clean_u = val_str.upper()
            if clean_u in BOILERPLATE_ANCHORS:
                continue

            # Entropy calculation: length and character variety
            is_digit_only = val_str.replace(".", "").replace(",", "").replace("-", "").isdigit()
            is_code = bool(re.match(r"^[0-9A-Z]{6,12}$", clean_u))
            
            cand_pages = {c.page for c in cands}
            # An anchor must be unambiguous (exactly 1 candidate or clearly dominant)
            if len(cands) > 1 and not (is_code and len(cand_pages) == 1):
                continue

            if is_code:
                priority = 100.0 + len(val_str)
            elif not is_digit_only and len(val_str) >= 6:
                priority = 50.0 + min(30.0, len(val_str))
            elif is_digit_only and len(val_str) >= 8:
                priority = 40.0
            else:
                priority = 10.0 + min(10.0, len(val_str))

            if priority > best_anchor_score:
                best_anchor_score = priority
                best_anchor_field = fld.field_name
                best_anchor_cand = cands[0]

        return best_anchor_field, best_anchor_cand

    def resolve_record(
        self,
        record: StructuralRecord,
        fields: list[RecordFieldLeaf],
        candidates_by_field: dict[str, list[MatchCandidate]],
        column_corridors: dict[str, tuple[float, float]] | None = None,
        mode: ResolverMode | None = None,
    ) -> dict[str, FieldResolutionResult]:
        """Jointly resolve all fields belonging to a single structured record."""
        active_mode = mode or self.mode
        corridors = column_corridors or {}

        # Track field order in record for reading-order consistency
        record.field_order = [f.field_name for f in fields]

        # Step 1: Discover anchor if record is not anchored
        if not record.is_anchored and active_mode != ResolverMode.BASELINE:
            anc_fld, anc_cand = self.identify_record_anchor(fields, candidates_by_field)
            if anc_cand is not None and anc_fld is not None:
                record.page = anc_cand.page
                record.anchor_field = anc_fld
                record.anchor_bbox = anc_cand.bbox
                record.row_y_center = anc_cand.bbox.y + anc_cand.bbox.height / 2.0
                record.row_height = max(0.008, anc_cand.bbox.height)
                record.is_anchored = True
                record.resolved_fields[anc_fld] = anc_cand.bbox

        # Step 2: Sort fields to resolve: anchored first, then text, then numbers
        def _field_priority(f: RecordFieldLeaf) -> int:
            if f.field_name == record.anchor_field:
                return 0
            val = f.value
            is_num = isinstance(val, (int, float)) and not isinstance(val, bool)
            return 2 if is_num else 1

        sorted_fields = sorted(fields, key=_field_priority)
        results: dict[str, FieldResolutionResult] = {}

        for fld in sorted_fields:
            fname = fld.field_name
            fpath = fld.path
            cands = candidates_by_field.get(fname, [])
            is_num = isinstance(fld.value, (int, float)) and not isinstance(fld.value, bool)
            col_corr = corridors.get(fname)

            if not cands:
                results[fname] = FieldResolutionResult(
                    field_name=fname,
                    field_path=fpath,
                    candidate=None,
                    score=0.0,
                    breakdown={},
                    is_ambiguous=False,
                    is_abstained=False,
                )
                continue

            ranked = self.rank_record_candidates(
                candidates=cands,
                field_name=fname,
                record=record,
                column_corridor=col_corr,
                all_column_corridors=corridors,
                is_numeric=is_num,
                mode=active_mode,
            )

            top_cand, top_score, breakdown = ranked[0]
            is_ambiguous = False
            is_abstained = False

            if len(ranked) >= 2:
                sec_cand, sec_score, _ = ranked[1]
                margin = top_score - sec_score
                if margin < self.ambiguity_margin:
                    is_ambiguous = True

            # Ambiguity Gating in Mode F: Abstain when structural signals fail to distinguish candidates
            if active_mode == ResolverMode.AMBIGUITY_GATED and self.enable_ambiguity_gating:
                if is_ambiguous:
                    is_abstained = True

            if not is_abstained:
                record.resolved_fields[fname] = top_cand.bbox

            results[fname] = FieldResolutionResult(
                field_name=fname,
                field_path=fpath,
                candidate=None if is_abstained else top_cand,
                score=top_score,
                breakdown=breakdown,
                is_ambiguous=is_ambiguous,
                is_abstained=is_abstained,
            )

        return results
