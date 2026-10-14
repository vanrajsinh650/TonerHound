"""EXP-044 Unified Hybrid Matching Pipeline.

Orchestrates the 6 stages of Phase C with strict gating and unconditional baseline preservation:
- Gate 0: Unconditional Baseline Preservation (never override passing baseline citation)
- Stage 1: Rule-Based Canonicalization (matching/canonicalize_v2.py)
- Stage 2: Character-Level Bounded Fuzzy Matching (matching/fuzzy_matcher.py)
- Stage 3: Needleman-Wunsch Token Alignment (matching/token_aligner.py)
- Stage 4: Hungarian Token Set Matching (matching/token_set_matcher.py)
- Stage 5: Multi-Region Evidence Assembly V3 (matching/multi_region_v3.py)
- Stage 6: Structural Ambiguity Resolver (matching/ambiguity_resolver.py)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tonerhound.geometry.coordinates import BBox
from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import DocumentPage, ExtractionInput

pipeline_dir = Path(__file__).resolve().parent
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from canonicalize_v2 import match_canonical_stage1
from fuzzy_matcher import match_fuzzy_stage2
from token_aligner import match_token_aligner_stage3
from token_set_matcher import match_token_set_stage4
from multi_region_v3 import assemble_multi_region_stage5
from ambiguity_resolver import AmbiguityResolverStage6


class EXP044HybridPipeline:
    """Multi-stage gated hybrid matching pipeline for NORMALIZATION_MISMATCH."""

    def __init__(
        self,
        doc_index: DocumentIndex,
        enabled_stages: dict[str, bool] | None = None,
    ) -> None:
        self.doc_index = doc_index
        defaults = {
            "stage_1": True,
            "stage_2": True,
            "stage_3": True,
            "stage_4": True,
            "stage_5": True,
            "stage_6": True,
        }
        if enabled_stages:
            defaults.update(enabled_stages)
        self.enabled = defaults
        self.ambiguity_resolver = AmbiguityResolverStage6(doc_index) if self.enabled["stage_6"] else None

    def resolve_field(
        self,
        field_path: str,
        target_value: Any,
        page_hint: int | None = None,
        existing_pred_cit: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Execute gated cascade across stages 1 through 6.
        
        Returns (citation_dict, stage_name) if resolved, else (None, None).
        """
        if target_value is None:
            return None, None

        # Determine target page(s) to inspect
        pages_to_search: list[int] = []
        if page_hint is not None and 1 <= page_hint <= self.doc_index.total_pages:
            pages_to_search = [page_hint]
        elif existing_pred_cit and existing_pred_cit.get("page"):
            ep = existing_pred_cit["page"]
            if 1 <= ep <= self.doc_index.total_pages:
                pages_to_search = [ep]
        else:
            pages_to_search = list(range(1, min(self.doc_index.total_pages + 1, 6)))

        for p_num in pages_to_search:
            page_obj = self.doc_index.get_page(p_num)
            if not page_obj:
                continue

            # --- STAGE 1: Canonicalization (rules) ---
            if self.enabled["stage_1"]:
                b1, text1 = match_canonical_stage1(target_value, page_obj, field_path)
                if b1:
                    cit = {
                        "field_path": field_path,
                        "page": p_num,
                        "bbox": list(b1.to_coco()),
                        "polygon": None,
                        "reference_text": text1,
                        "confidence": 0.95,
                        "source": "exp044_stage1_canonicalize",
                        "metadata": {"stage": "stage_1", "matched_text": text1},
                    }
                    return cit, "stage_1_canonicalize"

            # --- STAGE 2: Character-level fuzzy matching ---
            if self.enabled["stage_2"]:
                b2, text2 = match_fuzzy_stage2(target_value, page_obj, max_edit_distance=2)
                if b2:
                    cit = {
                        "field_path": field_path,
                        "page": p_num,
                        "bbox": list(b2.to_coco()),
                        "polygon": None,
                        "reference_text": text2,
                        "confidence": 0.90,
                        "source": "exp044_stage2_fuzzy",
                        "metadata": {"stage": "stage_2", "matched_text": text2},
                    }
                    return cit, "stage_2_fuzzy"

            # --- STAGE 3: Token sequence alignment (Needleman-Wunsch) ---
            if self.enabled["stage_3"]:
                b3, text3 = match_token_aligner_stage3(target_value, page_obj, max_alignment_cost=3)
                if b3:
                    cit = {
                        "field_path": field_path,
                        "page": p_num,
                        "bbox": list(b3.to_coco()),
                        "polygon": None,
                        "reference_text": text3,
                        "confidence": 0.88,
                        "source": "exp044_stage3_token_align",
                        "metadata": {"stage": "stage_3", "matched_text": text3},
                    }
                    return cit, "stage_3_token_align"

            # --- STAGE 4: Token set matching (Hungarian) ---
            if self.enabled["stage_4"]:
                b4, text4 = match_token_set_stage4(target_value, page_obj, max_total_cost=3.0)
                if b4:
                    cit = {
                        "field_path": field_path,
                        "page": p_num,
                        "bbox": list(b4.to_coco()),
                        "polygon": None,
                        "reference_text": text4,
                        "confidence": 0.85,
                        "source": "exp044_stage4_token_set",
                        "metadata": {"stage": "stage_4", "matched_text": text4},
                    }
                    return cit, "stage_4_token_set"

            # --- STAGE 5: Multi-region evidence assembly ---
            if self.enabled["stage_5"]:
                exist_bbox_obj = None
                if existing_pred_cit and existing_pred_cit.get("bbox") and existing_pred_cit.get("page") == p_num:
                    eb = existing_pred_cit["bbox"]
                    if len(eb) == 4:
                        exist_bbox_obj = BBox(eb[0], eb[1], eb[2], eb[3], p_num)
                b5, text5 = assemble_multi_region_stage5(target_value, page_obj, existing_frag_bbox=exist_bbox_obj, max_line_span=3)
                if b5:
                    cit = {
                        "field_path": field_path,
                        "page": p_num,
                        "bbox": list(b5.to_coco()),
                        "polygon": None,
                        "reference_text": text5,
                        "confidence": 0.85,
                        "source": "exp044_stage5_multi_region",
                        "metadata": {"stage": "stage_5", "matched_text": text5},
                    }
                    return cit, "stage_5_multi_region"

        # --- STAGE 6: Ambiguity resolver ---
        if self.enabled["stage_6"] and self.ambiguity_resolver:
            input_data = ExtractionInput(field=field_path, value=target_value, page_hint=page_hint)
            cands = self.ambiguity_resolver.doc_index.find_exact_tokens(str(target_value)) if hasattr(self.ambiguity_resolver.doc_index, "find_exact_tokens") else []
            # Gather candidates across pages if multiple exist
            if not cands and hasattr(self.ambiguity_resolver.doc_index, "find_token_candidates"):
                cands = self.ambiguity_resolver.doc_index.find_token_candidates(str(target_value))
            if cands:
                resolved_cand = self.ambiguity_resolver.resolve_ambiguity(
                    candidates=cands,
                    input_data=input_data,
                    min_score_threshold=0.60,
                )
                if resolved_cand:
                    cit = {
                        "field_path": field_path,
                        "page": resolved_cand.page,
                        "bbox": list(resolved_cand.bbox.to_coco()),
                        "polygon": None,
                        "reference_text": resolved_cand.matched_text,
                        "confidence": 0.85,
                        "source": "exp044_stage6_ambiguity",
                        "metadata": {"stage": "stage_6"},
                    }
                    return cit, "stage_6_ambiguity"

        return None, None
