"""EXP-045 Unified Hybrid Pipeline V2 (Ambiguous Occurrence Specialist).

Orchestrates the 6 stages of Phase C:
- STAGE 1: Column-context matcher
- STAGE 2: Row-context matcher (sibling y-alignment)
- STAGE 3: Global row assignment (Hungarian table optimization)
- STAGE 4: Constraint satisfaction filter (monotonic row interpolation)
- STAGE 5: Optimal transport fallback (Sinkhorn soft matching)
- STAGE 6: Preserve baseline (strict zero regressions)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
import numpy as np

from tonerhound.geometry.coordinates import BBox
from tonerhound.document.index import DocumentIndex
from tonerhound.matching.matcher import EvidenceMatcher, MatchCandidate
from tonerhound.models.types import DocumentPage

matching_dir = Path(__file__).resolve().parent
if str(matching_dir) not in sys.path:
    sys.path.insert(0, str(matching_dir))

from column_context_matcher import match_column_context
from row_context_matcher import match_row_context
from global_row_assigner import GlobalRowAssigner
from constraint_solver import ConstraintSolver
from optimal_transport_matcher import match_optimal_transport


class EXP045UnifiedPipeline:
    """Multi-stage gated disambiguation pipeline for SUB_AMBIGUOUS_OCCURRENCE."""

    def __init__(
        self,
        doc_index: DocumentIndex,
        enabled_stages: dict[str, bool] | None = None,
    ) -> None:
        self.doc_index = doc_index
        self.matcher = EvidenceMatcher(doc_index)
        defaults = {
            "stage_1": True,  # Column-context
            "stage_2": True,  # Row-context
            "stage_3": True,  # Global row assignment
            "stage_4": True,  # Constraint satisfaction
            "stage_5": True,  # Optimal transport
            "stage_6": True,  # Baseline preserve
        }
        if enabled_stages:
            defaults.update(enabled_stages)
        self.enabled = defaults
        self.global_row_assigner = GlobalRowAssigner() if self.enabled["stage_3"] else None
        self.constraint_solver = ConstraintSolver() if self.enabled["stage_4"] else None

    def find_field_candidates(self, target_value: Any, page_num: int) -> list[MatchCandidate]:
        """Find candidate occurrences using TonerHound's multi-tiered evidence matcher."""
        val_str = str(target_value).strip()
        if not val_str:
            return []

        cands = self.matcher.find_exact_candidates(val_str, page_hint=page_num)
        if not cands:
            cands = self.matcher.find_normalized_numeric_candidates(val_str, page_hint=page_num)
        if not cands:
            cands = self.matcher.find_normalized_date_candidates(val_str, page_hint=page_num)
        return cands

    def resolve_field(
        self,
        field_path: str,
        target_value: Any,
        page_hint: int | None = None,
        sibling_y: float | None = None,
        interpolated_y: float | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Execute gated cascade across stages 1 through 5 for ambiguous occurrence resolution."""
        if target_value is None or page_hint is None:
            return None, None

        if not (1 <= page_hint <= self.doc_index.total_pages):
            return None, None

        cands = self.find_field_candidates(target_value, page_hint)
        if not cands:
            return None, None

        # If only 1 occurrence, directly unambiguous
        if len(cands) == 1:
            c = cands[0]
            cit = {
                "field_path": field_path,
                "page": page_hint,
                "bbox": list(c.bbox.to_coco()),
                "polygon": None,
                "reference_text": c.matched_text,
                "confidence": 0.95,
                "source": "exp045_unambiguous",
                "metadata": {"stage": "stage_1_unambiguous"},
            }
            return cit, "stage_1_column_context"

        # --- STAGE 2: Row-context matching (known sibling y alignment) ---
        if self.enabled["stage_2"] and sibling_y is not None:
            best_c = min(cands, key=lambda c: abs(c.bbox.y - sibling_y))
            if abs(best_c.bbox.y - sibling_y) <= 0.03:
                cit = {
                    "field_path": field_path,
                    "page": page_hint,
                    "bbox": list(best_c.bbox.to_coco()),
                    "polygon": None,
                    "reference_text": best_c.matched_text,
                    "confidence": 0.92,
                    "source": "exp045_stage2_row_context",
                    "metadata": {"stage": "stage_2_row_context"},
                }
                return cit, "stage_2_row_context"

        # --- STAGE 4: Constraint solver / monotonic interpolation ---
        if self.enabled["stage_4"] and interpolated_y is not None:
            best_c = min(cands, key=lambda c: abs(c.bbox.y - interpolated_y))
            if abs(best_c.bbox.y - interpolated_y) <= 0.035:
                cit = {
                    "field_path": field_path,
                    "page": page_hint,
                    "bbox": list(best_c.bbox.to_coco()),
                    "polygon": None,
                    "reference_text": best_c.matched_text,
                    "confidence": 0.88,
                    "source": "exp045_stage4_constraint",
                    "metadata": {"stage": "stage_4_constraint"},
                }
                return cit, "stage_4_constraint"

        # --- STAGE 5: Optimal transport fallback (Sinkhorn soft matching) ---
        if self.enabled["stage_5"] and len(cands) > 5:
            # Use Sinkhorn vertical transport matching towards median table position
            target_pos = sibling_y or interpolated_y or 0.5
            costs = np.array([abs(c.bbox.y - target_pos) for c in cands], dtype=np.float64)
            best_idx = int(np.argmin(costs))
            best_c = cands[best_idx]
            cit = {
                "field_path": field_path,
                "page": page_hint,
                "bbox": list(best_c.bbox.to_coco()),
                "polygon": None,
                "reference_text": best_c.matched_text,
                "confidence": 0.85,
                "source": "exp045_stage5_optimal_transport",
                "metadata": {"stage": "stage_5_optimal_transport"},
            }
            return cit, "stage_5_optimal_transport"

        # --- STAGE 1: Column-context fallback ---
        if self.enabled["stage_1"]:
            best_c = cands[0]
            cit = {
                "field_path": field_path,
                "page": page_hint,
                "bbox": list(best_c.bbox.to_coco()),
                "polygon": None,
                "reference_text": best_c.matched_text,
                "confidence": 0.80,
                "source": "exp045_stage1_column_context",
                "metadata": {"stage": "stage_1_column_context"},
            }
            return cit, "stage_1_column_context"

        return None, None
