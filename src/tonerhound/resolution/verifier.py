"""Candidate verification and false-grounding reduction layer for TonerHound.

Implements Steps 4, 5, 6, 7, 8 of EXP-003:
- Strict verification pipeline after candidate ranking
- Field-type specialization (boolean, number, currency, date, identifier, table cell)
- Score margin and abstention gating (calibrated threshold)
- Bounding box plausibility and geometry checks
- Table row alignment and spatial consistency
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import ProvenanceStatus
from tonerhound.normalization.normalizers import (
    detect_checkbox_state,
    is_date_equal,
    is_number_equal,
    parse_date_value,
    parse_numeric_value,
)


class FieldType(str, Enum):
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    CURRENCY = "currency"
    DATE = "date"
    IDENTIFIER = "identifier"
    TABLE_CELL = "table_cell"
    STRING = "string"
    UNKNOWN = "unknown"


def infer_field_type(field_path: str, value: Any) -> FieldType:
    """Classify field type from path name and Python value type."""
    if isinstance(value, bool):
        return FieldType.BOOLEAN
    if isinstance(value, (int, float)):
        return FieldType.NUMERIC

    val_str = str(value).strip() if value is not None else ""
    path_lower = field_path.lower()

    # Check for boolean keywords in path or value
    if any(k in path_lower for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no")):
        if val_str.lower() in ("true", "false", "yes", "no", "0", "1"):
            return FieldType.BOOLEAN

    # Check for date keywords or valid date
    if any(k in path_lower for k in ("date", "time", "day", "month", "year", "dob", "period")):
        if parse_date_value(val_str) is not None:
            return FieldType.DATE

    # Check for currency
    if re.search(r"[$€£¥]", val_str) or any(k in path_lower for k in ("price", "amount", "total", "subtotal", "tax", "fee", "balance", "cost")):
        if parse_numeric_value(val_str) is not None:
            return FieldType.CURRENCY

    # Check for numbers
    if parse_numeric_value(val_str) is not None and not re.search(r"[a-zA-Z]", val_str):
        return FieldType.NUMERIC

    # Check for identifier
    if any(k in path_lower for k in ("id", "ssn", "number", "no", "code", "ein", "zip", "phone")):
        return FieldType.IDENTIFIER

    # Check for table cell
    if "[" in field_path and "]" in field_path:
        return FieldType.TABLE_CELL

    return FieldType.STRING


@dataclass
class VerificationDecision:
    is_accepted: bool
    status: ProvenanceStatus
    confidence: float
    bbox: BBox | None
    page: int | None
    matched_text: str | None
    reason: str


class CandidateVerifier:
    """Verifies top-ranked candidate against semantic, geometric, and score margin rules."""

    def __init__(
        self,
        score_margin_threshold: float = 0.05,
        min_bbox_dimension: float = 0.001,
        max_scalar_width: float = 0.85,
        max_scalar_height: float = 0.25,
    ) -> None:
        self.score_margin_threshold = score_margin_threshold
        self.min_bbox_dimension = min_bbox_dimension
        self.max_scalar_width = max_scalar_width
        self.max_scalar_height = max_scalar_height

    def verify(
        self,
        field: str,
        value: Any,
        top_candidate: MatchCandidate | None,
        scored_candidates: list[tuple[MatchCandidate, float]],
        field_context: str | None = None,
        page_hint: int | None = None,
        row_anchor: tuple[int, float, float, Any] | None = None,
    ) -> VerificationDecision:
        """Run all verification filters on the top candidate."""
        if top_candidate is None or not scored_candidates:
            return VerificationDecision(
                is_accepted=False,
                status=ProvenanceStatus.NOT_FOUND,
                confidence=0.0,
                bbox=None,
                page=None,
                matched_text=None,
                reason="No candidate generated",
            )

        field_type = infer_field_type(field, value)
        top_cand, top_score = scored_candidates[0]

        # 1. Bounding Box Geometric Plausibility Check
        b = top_cand.bbox
        if b.width < self.min_bbox_dimension or b.height < self.min_bbox_dimension:
            return VerificationDecision(
                is_accepted=False,
                status=ProvenanceStatus.NOT_FOUND,
                confidence=0.0,
                bbox=None,
                page=None,
                matched_text=None,
                reason=f"Degenerate bbox dimensions ({b.width:.4f}x{b.height:.4f})",
            )

        if b.x < -0.05 or b.y < -0.05 or (b.x + b.width) > 1.05 or (b.y + b.height) > 1.05:
            return VerificationDecision(
                is_accepted=False,
                status=ProvenanceStatus.NOT_FOUND,
                confidence=0.0,
                bbox=None,
                page=None,
                matched_text=None,
                reason=f"BBox coordinates out of document bounds: {b.to_coco()}",
            )

        if field_type in (FieldType.NUMERIC, FieldType.CURRENCY, FieldType.BOOLEAN):
            if b.width > self.max_scalar_width or b.height > self.max_scalar_height:
                return VerificationDecision(
                    is_accepted=False,
                    status=ProvenanceStatus.NOT_FOUND,
                    confidence=0.0,
                    bbox=None,
                    page=None,
                    matched_text=None,
                    reason=f"Over-wide/tall bbox for scalar field: {b.to_coco()}",
                )

        # 2. Page Hint Consistency Check
        if page_hint is not None and top_cand.page != page_hint:
            # If any candidate exists on the hint page, reject candidates on other pages
            hint_page_cands = [c for c, _s in scored_candidates if c.page == page_hint]
            if hint_page_cands:
                return VerificationDecision(
                    is_accepted=False,
                    status=ProvenanceStatus.AMBIGUOUS,
                    confidence=0.4,
                    bbox=None,
                    page=None,
                    matched_text=None,
                    reason=f"Top candidate on page {top_cand.page} conflicts with page hint {page_hint}",
                )

        # 3. Table Row Alignment Check
        if row_anchor is not None:
            anc_page, anc_cy, anc_h, _anc_box = row_anchor
            if top_cand.page != anc_page:
                return VerificationDecision(
                    is_accepted=False,
                    status=ProvenanceStatus.NOT_FOUND,
                    confidence=0.0,
                    bbox=None,
                    page=None,
                    matched_text=None,
                    reason=f"Candidate on page {top_cand.page} conflicts with row anchor on page {anc_page}",
                )

            cand_cy = b.y + b.height / 2.0
            row_tol = max(0.018, anc_h * 1.6)
            if abs(cand_cy - anc_cy) > row_tol:
                return VerificationDecision(
                    is_accepted=False,
                    status=ProvenanceStatus.NOT_FOUND,
                    confidence=0.0,
                    bbox=None,
                    page=None,
                    matched_text=None,
                    reason=f"Candidate at y={cand_cy:.4f} outside row tolerance of anchor y={anc_cy:.4f} (tol={row_tol:.4f})",
                )

        # 4. Field Type Semantic Compatibility Check
        if field_type == FieldType.BOOLEAN:
            cb_state = detect_checkbox_state(top_cand.matched_text)
            if cb_state is None and top_cand.matched_text.lower() not in ("true", "false", "yes", "no"):
                # Candidate is arbitrary text, not a checkbox or boolean token
                return VerificationDecision(
                    is_accepted=False,
                    status=ProvenanceStatus.NOT_FOUND,
                    confidence=0.0,
                    bbox=None,
                    page=None,
                    matched_text=None,
                    reason=f"Boolean field matched non-checkbox text '{top_cand.matched_text}'",
                )

        elif field_type in (FieldType.NUMERIC, FieldType.CURRENCY):
            target_num = parse_numeric_value(value)
            token_num = parse_numeric_value(top_cand.matched_text)
            if target_num is not None and token_num is not None:
                if not is_number_equal(target_num, token_num):
                    return VerificationDecision(
                        is_accepted=False,
                        status=ProvenanceStatus.NOT_FOUND,
                        confidence=0.0,
                        bbox=None,
                        page=None,
                        matched_text=None,
                        reason=f"Numeric value mismatch: target={target_num} vs token={token_num}",
                    )

        elif field_type == FieldType.DATE:
            target_date = parse_date_value(value)
            if target_date is not None:
                if not is_date_equal(target_date, top_cand.matched_text):
                    return VerificationDecision(
                        is_accepted=False,
                        status=ProvenanceStatus.NOT_FOUND,
                        confidence=0.0,
                        bbox=None,
                        page=None,
                        matched_text=None,
                        reason=f"Date value mismatch: target={target_date} vs token='{top_cand.matched_text}'",
                    )

        # 5. Score Margin & Abstention Check (Step 6)
        if len(scored_candidates) > 1:
            sec_cand, sec_score = scored_candidates[1]
            score_margin = top_score - sec_score

            if score_margin < self.score_margin_threshold:
                # Check spatial separation: do they refer to the same physical region?
                same_page = (top_cand.page == sec_cand.page)
                same_region = same_page and (top_cand.bbox.iou(sec_cand.bbox) >= 0.50)

                if not same_region:
                    # Indistinguishable candidates in distinct spatial regions -> ABSTAIN!
                    return VerificationDecision(
                        is_accepted=False,
                        status=ProvenanceStatus.AMBIGUOUS,
                        confidence=0.50,
                        bbox=None,
                        page=None,
                        matched_text=None,
                        reason=(
                            f"Ambiguous candidates: top score {top_score:.4f} vs second {sec_score:.4f} "
                            f"(margin {score_margin:.4f} < {self.score_margin_threshold}) across distinct regions"
                        ),
                    )

        # Passed all verification filters!
        status_map = {
            "exact": ProvenanceStatus.EXACT,
            "normalized_number": ProvenanceStatus.NORMALIZED,
            "normalized_date": ProvenanceStatus.NORMALIZED,
            "fuzzy": ProvenanceStatus.FUZZY,
        }
        status = status_map.get(top_cand.match_type, ProvenanceStatus.EXACT)
        score_diff = (top_score - scored_candidates[1][1]) if len(scored_candidates) > 1 else 1.0
        confidence = min(0.99, max(0.70, 0.75 + (score_diff * 0.20)))

        return VerificationDecision(
            is_accepted=True,
            status=status,
            confidence=confidence,
            bbox=top_cand.bbox,
            page=top_cand.page,
            matched_text=top_cand.matched_text,
            reason="Passed all candidate verification criteria",
        )
