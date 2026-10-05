"""Failure Microscope Field Classifier.

Classifies the causal root mechanism of grounding failures across ExtractBench
evaluations in TonerHound. Adheres strictly to the required failure taxonomy and
causal tracing design principles.

TAXONOMY:
- NON_TEXT_BOOLEAN_GROUNDING
- NO_TEXT_AT_GOLD_REGION
- DATE_INDEX_MISS
- REAL_INDEXING_MISS
- NORMALIZATION_MISMATCH
- VERIFICATION_REJECTION
- BBOX_RECONSTRUCTION
- HYPHENATION
- PAGE_ROUTING
- WRONG_ROW
- WRONG_COLUMN
- WRONG_PAGE
- BBOX_TOO_NARROW
- BBOX_TOO_WIDE
- TOKEN_SLICING
- TOP_K_TRUNCATION
- DEDUPLICATION_COLLAPSE
- MULTI_LINE_SPLIT
- OTHER

AUDIT-ONLY TAXONOMY (excluded from production failure rankings):
- AUDIT_CLASSIFICATION_ERROR
- ALREADY_RESOLVED
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentToken, ExtractionInput, ResolutionResult
from tonerhound.normalization.normalizers import (
    clean_currency_and_numbers,
    normalize_unicode_and_case,
    parse_date_value,
    parse_numeric_value,
)
from tonerhound.resolution.resolver import EvidenceResolver

# Official taxonomy definitions
PRODUCTION_FAILURE_CLASSES = (
    "NON_TEXT_BOOLEAN_GROUNDING",
    "NO_TEXT_AT_GOLD_REGION",
    "DATE_INDEX_MISS",
    "REAL_INDEXING_MISS",
    "NORMALIZATION_MISMATCH",
    "VERIFICATION_REJECTION",
    "BBOX_RECONSTRUCTION",
    "HYPHENATION",
    "PAGE_ROUTING",
    "WRONG_ROW",
    "WRONG_COLUMN",
    "WRONG_PAGE",
    "BBOX_TOO_NARROW",
    "BBOX_TOO_WIDE",
    "TOKEN_SLICING",
    "TOP_K_TRUNCATION",
    "DEDUPLICATION_COLLAPSE",
    "MULTI_LINE_SPLIT",
    "OTHER",
)

AUDIT_ONLY_CLASSES = (
    "AUDIT_CLASSIFICATION_ERROR",
    "ALREADY_RESOLVED",
)

ALL_TAXONOMY_CLASSES = PRODUCTION_FAILURE_CLASSES + AUDIT_ONLY_CLASSES

_BOOL_FIELD_KEYWORDS = (
    "_box",
    "checkbox",
    "is_",
    "has_",
    "flag",
    "_yes",
    "_no",
    "final",
    "amended",
    "general",
    "domestic",
    "contributed",
    "schedule_8812",
    "box_13",
    "signature",
    "signed",
)


def compute_iou_xywh(
    box1: Sequence[float] | None, box2: Sequence[float] | None
) -> float:
    """Compute IoU between two [x, y, w, h] boxes with boundary checks."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


@dataclass
class FieldClassificationResult:
    """Structured field-level record conforming to Section 10 schema."""
    document_id: str
    field_path: str
    gold_value: Any
    predicted_value: Any
    gold_page: int | None
    predicted_page: int | None
    gold_bbox: list[float] | None
    predicted_bbox: list[float] | None
    iou: float
    candidate_count: int
    grounding_success: bool
    failure_class: str
    evidence: dict[str, Any] = field(default_factory=dict)
    experiment_run: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "field_path": self.field_path,
            "gold_value": self.gold_value,
            "predicted_value": self.predicted_value,
            "gold_page": self.gold_page,
            "predicted_page": self.predicted_page,
            "gold_bbox": self.gold_bbox,
            "predicted_bbox": self.predicted_bbox,
            "iou": round(self.iou, 4),
            "candidate_count": self.candidate_count,
            "grounding_success": self.grounding_success,
            "failure_class": self.failure_class,
            "evidence": self.evidence,
            "experiment_run": self.experiment_run,
        }


class FailureMicroscopeClassifier:
    """Classifies the causal root cause of field grounding failures."""

    def __init__(self, doc_index: DocumentIndex | None = None, resolver: EvidenceResolver | None = None) -> None:
        self.doc_index = doc_index
        self.resolver = resolver

    @staticmethod
    def is_boolean_target(value: Any, field_path: str = "") -> bool:
        """Check whether value represents a boolean / checkbox target.

        Critical regression protection: isinstance(value, bool) MUST be checked
        first because bool is a subclass of int in Python.
        """
        if isinstance(value, bool):
            return True

        # Explicitly reject ints and floats from boolean identification
        if isinstance(value, (int, float)):
            return False

        if isinstance(value, str):
            val_clean = value.strip().lower()
            if val_clean in ("true", "false", "yes", "no"):
                # If field context suggests boolean, or pure boolean string
                if any(k in field_path.lower() for k in _BOOL_FIELD_KEYWORDS):
                    return True
                if val_clean in ("true", "false"):
                    return True

        fp_lower = field_path.lower()
        if any(k in fp_lower for k in ("_box", "checkbox")):
            return True

        return False

    @staticmethod
    def get_tokens_in_region(
        doc_index: DocumentIndex | None, page: int | None, bbox: Sequence[float] | None
    ) -> list[DocumentToken]:
        """Find all document tokens overlapping the specified bbox on page."""
        if doc_index is None or page is None or bbox is None or len(bbox) != 4:
            return []

        page_obj = doc_index.get_page(page)
        if not page_obj or not page_obj.tokens:
            return []

        gx, gy, gw, gh = bbox
        gx2 = gx + gw
        gy2 = gy + gh

        matching: list[DocumentToken] = []
        for tok in page_obj.tokens:
            tb = tok.bbox
            tx, ty, tw, th = tb.x, tb.y, tb.width, tb.height
            tx2 = tx + tw
            ty2 = ty + th

            # Overlap check
            if tx < gx2 and tx2 > gx and ty < gy2 and ty2 > gy:
                matching.append(tok)

        return matching

    def classify_field(
        self,
        document_id: str,
        field_path: str,
        gold_value: Any,
        predicted_value: Any,
        gold_page: int | None,
        predicted_page: int | None,
        gold_bbox: Sequence[float] | None,
        predicted_bbox: Sequence[float] | None,
        candidates: list[MatchCandidate] | None = None,
        resolver_result: ResolutionResult | None = None,
        experiment_run: str = "production",
        is_audit_run: bool = False,
    ) -> FieldClassificationResult:
        """Trace causal failure mechanism and return structured record."""
        gold_b = list(gold_bbox) if gold_bbox else None
        pred_b = list(predicted_bbox) if predicted_bbox else None

        # Calculate prediction IoU
        iou = 0.0
        same_page = (
            gold_page is not None
            and predicted_page is not None
            and gold_page == predicted_page
        )
        if same_page and gold_b and pred_b:
            iou = compute_iou_xywh(gold_b, pred_b)

        # Grounding success definition: page match, IoU >= 0.50
        grounding_success = same_page and (iou >= 0.50)

        # Collect evidence container
        evidence: dict[str, Any] = {
            "gold_page": gold_page,
            "predicted_page": predicted_page,
            "iou": round(iou, 4),
            "same_page": same_page,
        }

        # 0. Check ALREADY_RESOLVED
        if grounding_success:
            return FieldClassificationResult(
                document_id=document_id,
                field_path=field_path,
                gold_value=gold_value,
                predicted_value=predicted_value,
                gold_page=gold_page,
                predicted_page=predicted_page,
                gold_bbox=gold_b,
                predicted_bbox=pred_b,
                iou=iou,
                candidate_count=len(candidates) if candidates else 0,
                grounding_success=True,
                failure_class="ALREADY_RESOLVED",
                evidence=evidence,
                experiment_run=experiment_run,
            )

        # If not resolved, trace failure root cause
        cands = list(candidates) if candidates is not None else []
        evidence["candidate_count"] = len(cands)

        # 1. Non-text boolean check (Critical EXP-035 Protection)
        is_bool = self.is_boolean_target(gold_value, field_path)
        evidence["is_boolean_target"] = is_bool

        tokens_in_gold = self.get_tokens_in_region(self.doc_index, gold_page, gold_b)
        evidence["tokens_in_gold_count"] = len(tokens_in_gold)
        tokens_text = " ".join(t.text for t in tokens_in_gold).strip()
        evidence["tokens_in_gold_text"] = tokens_text

        if is_bool:
            # Check if boolean target text exists literally in text layer
            val_str = str(gold_value).strip().lower()
            if not tokens_in_gold or val_str not in tokens_text.lower():
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=len(cands),
                    grounding_success=False,
                    failure_class="NON_TEXT_BOOLEAN_GROUNDING",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )

        # 2. Check text layer absence at gold region
        if not tokens_in_gold:
            # Gold coordinates contain no text in PDF text layer
            return FieldClassificationResult(
                document_id=document_id,
                field_path=field_path,
                gold_value=gold_value,
                predicted_value=predicted_value,
                gold_page=gold_page,
                predicted_page=predicted_page,
                gold_bbox=gold_b,
                predicted_bbox=pred_b,
                iou=iou,
                candidate_count=len(cands),
                grounding_success=False,
                failure_class="NO_TEXT_AT_GOLD_REGION",
                evidence=evidence,
                experiment_run=experiment_run,
            )

        # 3. Analyze candidate pool if candidates were generated
        if cands:
            # Check candidate IoUs on gold page
            cand_ious: list[tuple[float, MatchCandidate]] = []
            for c in cands:
                cp = c.bbox.page if hasattr(c.bbox, "page") else getattr(c, "page", None)
                if cp == gold_page and gold_b:
                    cb = (
                        [c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height]
                        if hasattr(c.bbox, "x")
                        else list(c.bbox)
                    )
                    c_iou = compute_iou_xywh(cb, gold_b)
                    cand_ious.append((c_iou, c))

            max_cand_iou = max((ci[0] for ci in cand_ious), default=0.0)
            evidence["max_candidate_iou"] = round(max_cand_iou, 4)

            # Case 3A: A candidate with IoU >= 0.50 exists in pool!
            if max_cand_iou >= 0.50:
                # Why did pipeline fail to ground?
                if resolver_result and not resolver_result.is_grounded:
                    return FieldClassificationResult(
                        document_id=document_id,
                        field_path=field_path,
                        gold_value=gold_value,
                        predicted_value=predicted_value,
                        gold_page=gold_page,
                        predicted_page=predicted_page,
                        gold_bbox=gold_b,
                        predicted_bbox=pred_b,
                        iou=iou,
                        candidate_count=len(cands),
                        grounding_success=False,
                        failure_class="VERIFICATION_REJECTION",
                        evidence=evidence,
                        experiment_run=experiment_run,
                    )

                # Selected a different candidate on wrong page/row/column
                if predicted_page is not None and gold_page is not None and predicted_page != gold_page:
                    return FieldClassificationResult(
                        document_id=document_id,
                        field_path=field_path,
                        gold_value=gold_value,
                        predicted_value=predicted_value,
                        gold_page=gold_page,
                        predicted_page=predicted_page,
                        gold_bbox=gold_b,
                        predicted_bbox=pred_b,
                        iou=iou,
                        candidate_count=len(cands),
                        grounding_success=False,
                        failure_class="WRONG_PAGE",
                        evidence=evidence,
                        experiment_run=experiment_run,
                    )

                # If on same page, check row vs column vs table disambiguation
                if "[" in field_path and "]" in field_path and pred_b and gold_b:
                    # Table/array element: check vertical vs horizontal offset
                    y_diff = abs(pred_b[1] - gold_b[1])
                    x_diff = abs(pred_b[0] - gold_b[0])
                    if y_diff > gold_b[3] * 0.5:
                        return FieldClassificationResult(
                            document_id=document_id,
                            field_path=field_path,
                            gold_value=gold_value,
                            predicted_value=predicted_value,
                            gold_page=gold_page,
                            predicted_page=predicted_page,
                            gold_bbox=gold_b,
                            predicted_bbox=pred_b,
                            iou=iou,
                            candidate_count=len(cands),
                            grounding_success=False,
                            failure_class="WRONG_ROW",
                            evidence=evidence,
                            experiment_run=experiment_run,
                        )
                    if x_diff > gold_b[2] * 0.5:
                        return FieldClassificationResult(
                            document_id=document_id,
                            field_path=field_path,
                            gold_value=gold_value,
                            predicted_value=predicted_value,
                            gold_page=gold_page,
                            predicted_page=predicted_page,
                            gold_bbox=gold_b,
                            predicted_bbox=pred_b,
                            iou=iou,
                            candidate_count=len(cands),
                            grounding_success=False,
                            failure_class="WRONG_COLUMN",
                            evidence=evidence,
                            experiment_run=experiment_run,
                        )

                # Check top-K truncation or deduplication collapse
                if len(cands) > 10:
                    cand_hit_indices = [i for i, ci in enumerate(cand_ious) if ci[0] >= 0.50]
                    if cand_hit_indices and cand_hit_indices[0] >= 5:
                        return FieldClassificationResult(
                            document_id=document_id,
                            field_path=field_path,
                            gold_value=gold_value,
                            predicted_value=predicted_value,
                            gold_page=gold_page,
                            predicted_page=predicted_page,
                            gold_bbox=gold_b,
                            predicted_bbox=pred_b,
                            iou=iou,
                            candidate_count=len(cands),
                            grounding_success=False,
                            failure_class="TOP_K_TRUNCATION",
                            evidence=evidence,
                            experiment_run=experiment_run,
                        )

                # Default selection mismatch
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=len(cands),
                    grounding_success=False,
                    failure_class="OTHER",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )

            # Case 3B: Candidates exist on other page(s), but none on gold page
            cands_on_gold_page = [
                c
                for c in cands
                if (c.bbox.page if hasattr(c.bbox, "page") else getattr(c, "page", None))
                == gold_page
            ]
            if not cands_on_gold_page:
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=len(cands),
                    grounding_success=False,
                    failure_class="PAGE_ROUTING",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )

            # Case 3C: Candidates exist on gold page, but all IoU < 0.50
            best_cand_on_page = max(
                cands_on_gold_page,
                key=lambda c: compute_iou_xywh(
                    [c.bbox.x, c.bbox.y, c.bbox.width, c.bbox.height]
                    if hasattr(c.bbox, "x")
                    else list(c.bbox),
                    gold_b,
                ),
            )
            bcb = (
                [best_cand_on_page.bbox.x, best_cand_on_page.bbox.y, best_cand_on_page.bbox.width, best_cand_on_page.bbox.height]
                if hasattr(best_cand_on_page.bbox, "x")
                else list(best_cand_on_page.bbox)
            )
            evidence["best_cand_bbox"] = bcb

            cand_w = bcb[2]
            gold_w = gold_b[2] if gold_b and gold_b[2] > 0 else 1.0
            w_ratio = cand_w / gold_w
            evidence["cand_gold_width_ratio"] = round(w_ratio, 4)

            if w_ratio < 0.70:
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=len(cands),
                    grounding_success=False,
                    failure_class="BBOX_TOO_NARROW",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )
            elif w_ratio > 1.35:
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=len(cands),
                    grounding_success=False,
                    failure_class="BBOX_TOO_WIDE",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )
            else:
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=len(cands),
                    grounding_success=False,
                    failure_class="BBOX_RECONSTRUCTION",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )

        # 4. Zero candidates collected (Retrieval Failure Analysis)
        val_str = str(gold_value) if gold_value is not None else ""
        norm_val = normalize_unicode_and_case(val_str).text.strip()
        page_obj = self.doc_index.get_page(gold_page) if (self.doc_index and gold_page) else None
        page_text = page_obj.text if page_obj else ""

        # Multi-line
        if "\n" in val_str:
            return FieldClassificationResult(
                document_id=document_id,
                field_path=field_path,
                gold_value=gold_value,
                predicted_value=predicted_value,
                gold_page=gold_page,
                predicted_page=predicted_page,
                gold_bbox=gold_b,
                predicted_bbox=pred_b,
                iou=iou,
                candidate_count=0,
                grounding_success=False,
                failure_class="MULTI_LINE_SPLIT",
                evidence=evidence,
                experiment_run=experiment_run,
            )

        # Date miss
        parsed_date = parse_date_value(val_str)
        if parsed_date is not None:
            return FieldClassificationResult(
                document_id=document_id,
                field_path=field_path,
                gold_value=gold_value,
                predicted_value=predicted_value,
                gold_page=gold_page,
                predicted_page=predicted_page,
                gold_bbox=gold_b,
                predicted_bbox=pred_b,
                iou=iou,
                candidate_count=0,
                grounding_success=False,
                failure_class="DATE_INDEX_MISS",
                evidence=evidence,
                experiment_run=experiment_run,
            )

        # Hyphenation
        if "-" in val_str and any(part in page_text for part in val_str.split("-") if len(part) >= 3):
            return FieldClassificationResult(
                document_id=document_id,
                field_path=field_path,
                gold_value=gold_value,
                predicted_value=predicted_value,
                gold_page=gold_page,
                predicted_page=predicted_page,
                gold_bbox=gold_b,
                predicted_bbox=pred_b,
                iou=iou,
                candidate_count=0,
                grounding_success=False,
                failure_class="HYPHENATION",
                evidence=evidence,
                experiment_run=experiment_run,
            )

        # Normalization mismatch
        # Negative numbers formatted parenthetically e.g. (100.00)
        is_numeric = isinstance(gold_value, (int, float)) and not isinstance(gold_value, bool)
        if is_numeric and gold_value < 0:
            return FieldClassificationResult(
                document_id=document_id,
                field_path=field_path,
                gold_value=gold_value,
                predicted_value=predicted_value,
                gold_page=gold_page,
                predicted_page=predicted_page,
                gold_bbox=gold_b,
                predicted_bbox=pred_b,
                iou=iou,
                candidate_count=0,
                grounding_success=False,
                failure_class="NORMALIZATION_MISMATCH",
                evidence=evidence,
                experiment_run=experiment_run,
            )

        if tokens_in_gold:
            gold_tokens_text = " ".join(t.text for t in tokens_in_gold)
            evidence["gold_tokens_text"] = gold_tokens_text
            # Check subtoken or slicing
            clean_nv = clean_currency_and_numbers(norm_val)
            clean_gt = clean_currency_and_numbers(normalize_unicode_and_case(gold_tokens_text).text)
            if clean_nv and clean_nv in clean_gt and clean_nv != clean_gt:
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=0,
                    grounding_success=False,
                    failure_class="TOKEN_SLICING",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )

            # Normalization mismatch on punctuation/spacing
            words_in_val = [w.strip(" ,.;:'\"()[]{}") for w in norm_val.split() if w.strip(" ,.;:'\"()[]{}")]
            words_in_gt = [t.text.strip(" ,.;:'\"()[]{}") for t in tokens_in_gold if t.text.strip(" ,.;:'\"()[]{}")]
            if words_in_val and any(w in words_in_gt for w in words_in_val):
                return FieldClassificationResult(
                    document_id=document_id,
                    field_path=field_path,
                    gold_value=gold_value,
                    predicted_value=predicted_value,
                    gold_page=gold_page,
                    predicted_page=predicted_page,
                    gold_bbox=gold_b,
                    predicted_bbox=pred_b,
                    iou=iou,
                    candidate_count=0,
                    grounding_success=False,
                    failure_class="NORMALIZATION_MISMATCH",
                    evidence=evidence,
                    experiment_run=experiment_run,
                )

            # Text exists on page at gold box, but matcher completely failed to index/retrieve it
            return FieldClassificationResult(
                document_id=document_id,
                field_path=field_path,
                gold_value=gold_value,
                predicted_value=predicted_value,
                gold_page=gold_page,
                predicted_page=predicted_page,
                gold_bbox=gold_b,
                predicted_bbox=pred_b,
                iou=iou,
                candidate_count=0,
                grounding_success=False,
                failure_class="REAL_INDEXING_MISS",
                evidence=evidence,
                experiment_run=experiment_run,
            )

        # Fallback
        return FieldClassificationResult(
            document_id=document_id,
            field_path=field_path,
            gold_value=gold_value,
            predicted_value=predicted_value,
            gold_page=gold_page,
            predicted_page=predicted_page,
            gold_bbox=gold_b,
            predicted_bbox=pred_b,
            iou=iou,
            candidate_count=0,
            grounding_success=False,
            failure_class="OTHER",
            evidence=evidence,
            experiment_run=experiment_run,
        )
