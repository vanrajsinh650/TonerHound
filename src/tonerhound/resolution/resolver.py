"""Contextual evidence resolution engine for TonerHound.

Integrates multi-tier matching with spatial disambiguation,
multi-line region recovery, ambiguity gating, and provenance assignment.
"""

from __future__ import annotations

import math
from typing import Any

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.matching.matcher import EvidenceMatcher, MatchCandidate
from tonerhound.models.types import (
    ExtractionInput,
    ProvenanceStatus,
    ResolutionResult,
)
from tonerhound.normalization.normalizers import normalize_unicode_and_case


class EvidenceResolver:
    """Independent evidence resolution engine."""

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self.matcher = EvidenceMatcher(index)

    def resolve(self, extraction: ExtractionInput) -> ResolutionResult:
        """Resolve physical evidence for an extracted field."""
        field = extraction.field
        value = extraction.value
        evidence_text = extraction.evidence_text
        context = extraction.field_context or field
        page_hint = extraction.page_hint

        # Handle null / empty extraction
        if value is None or (isinstance(value, str) and not value.strip()):
            return ResolutionResult(
                field=field,
                value=value,
                status=ProvenanceStatus.EXACT,
                page=None,
                bbox=None,
                confidence=1.0,
                explanation="Null / empty field",
            )

        # Step 1: Candidate Generation across tiers
        candidates: list[MatchCandidate] = []

        # Tier 1a: Exact match on evidence_text (if provided)
        if evidence_text:
            candidates.extend(self.matcher.find_exact_candidates(evidence_text, page_hint=page_hint))

        # Tier 1b: If value is numeric, prioritize normalized numeric matching over raw float str()
        is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not candidates and is_num:
            candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=page_hint))

        # Tier 1c: Normalized date match
        if not candidates and isinstance(value, str):
            candidates.extend(self.matcher.find_normalized_date_candidates(value, page_hint=page_hint))

        # Tier 1d: Exact match on stringified value
        if not candidates and isinstance(value, (str, int, float)):
            candidates.extend(self.matcher.find_exact_candidates(str(value), page_hint=page_hint))

        # Tier 2: Normalized numeric match for strings that might be formatted numbers
        if not candidates and isinstance(value, str):
            candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=page_hint))

        # Tier 3: Fuzzy sequence alignment fallback
        if not candidates:
            query = evidence_text if evidence_text else str(value)
            candidates.extend(self.matcher.find_fuzzy_candidates(query, threshold=0.82, page_hint=page_hint))

        # If still no candidates found on page_hint, relax page_hint to search all pages
        if not candidates and page_hint is not None:
            if evidence_text:
                candidates.extend(self.matcher.find_exact_candidates(evidence_text, page_hint=None))
            if not candidates and isinstance(value, (int, float, str)):
                candidates.extend(self.matcher.find_normalized_numeric_candidates(value, page_hint=None))
            if not candidates and isinstance(value, str):
                candidates.extend(self.matcher.find_normalized_date_candidates(value, page_hint=None))

        # If zero candidates found: Check for derived vs not_found
        if not candidates:
            is_derived = self._is_likely_derived(field, value)
            return ResolutionResult(
                field=field,
                value=value,
                status=ProvenanceStatus.DERIVED if is_derived else ProvenanceStatus.NOT_FOUND,
                page=None,
                bbox=None,
                confidence=0.0,
                explanation=f"Evidence could not be physically sighted ({'derived calculation' if is_derived else 'not found'})",
            )

        # Step 2: Spatial & Contextual Disambiguation
        if len(candidates) == 1:
            best_candidate = candidates[0]
            return self._build_result(field, value, best_candidate, confidence=0.95, explanation="Single unambiguous match")

        # Multi-candidate disambiguation
        scored_candidates = self._score_candidates_with_context(candidates, context)
        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        top_cand, top_score = scored_candidates[0]
        second_cand, second_score = scored_candidates[1]

        # Ambiguity Gating: if indistinguishable, refuse to guess!
        score_diff = top_score - second_score
        if score_diff < 0.05 and abs(top_score - second_score) < 1e-4:
            # Check if both candidates point to essentially the same bounding box
            if top_cand.page == second_cand.page and top_cand.bbox.iou(second_cand.bbox) >= 0.8:
                # Same physical region, take top
                return self._build_result(field, value, top_cand, confidence=0.90, explanation="Overlapping duplicate candidates")

            return ResolutionResult(
                field=field,
                value=value,
                status=ProvenanceStatus.AMBIGUOUS,
                page=None,
                bbox=None,
                confidence=0.5,
                explanation=f"Multiple indistinguishable candidates found ({len(candidates)} occurrences)",
            )

        # Calibrate confidence based on score margin
        confidence = min(0.99, max(0.60, 0.70 + (score_diff * 0.25)))
        return self._build_result(field, value, top_cand, confidence=confidence, explanation="Contextually disambiguated")

    def _score_candidates_with_context(
        self,
        candidates: list[MatchCandidate],
        context: str,
    ) -> list[tuple[MatchCandidate, float]]:
        """Score each candidate based on spatial proximity to context label."""
        norm_context = normalize_unicode_and_case(context).text.strip()
        context_words = [w for w in norm_context.split() if len(w) > 1]

        scored: list[tuple[MatchCandidate, float]] = []

        for cand in candidates:
            score = 1.0
            page = self.index.get_page(cand.page)
            if not page:
                scored.append((cand, score))
                continue

            # Look for context label tokens on the same page
            context_label_boxes: list[BBox] = []
            for line in page.lines:
                line_norm = normalize_unicode_and_case(line.text).text
                if any(w in line_norm for w in context_words):
                    context_label_boxes.append(line.bbox)

            if context_label_boxes:
                # Find nearest context label
                min_dist = float("inf")
                same_line_bonus = 0.0

                cand_cx = cand.bbox.x + cand.bbox.width / 2.0
                cand_cy = cand.bbox.y + cand.bbox.height / 2.0

                for lbox in context_label_boxes:
                    lbl_cx = lbox.x + lbox.width / 2.0
                    lbl_cy = lbox.y + lbox.height / 2.0

                    # Euclidean distance in normalized space
                    dist = math.sqrt((cand_cx - lbl_cx) ** 2 + (cand_cy - lbl_cy) ** 2)
                    min_dist = min(min_dist, dist)

                    # Check same visual horizontal band (same line)
                    if abs(cand_cy - lbl_cy) < (cand.bbox.height * 1.2):
                        # Candidate is to the right of the label (standard form/invoice key-value pair)
                        if cand.bbox.x >= lbox.x:
                            same_line_bonus = max(same_line_bonus, 3.0)
                        else:
                            same_line_bonus = max(same_line_bonus, 1.5)

                # Distance score: closer is better
                dist_score = 1.0 / (1.0 + min_dist * 5.0)
                score += (dist_score * 2.0) + same_line_bonus

            # Bonus for exact match type over normalized or fuzzy
            if cand.match_type == "exact":
                score += 0.5
            elif cand.match_type.startswith("normalized"):
                score += 0.3

            scored.append((cand, score))

        return scored

    def _build_result(
        self,
        field: str,
        value: Any,
        cand: MatchCandidate,
        confidence: float,
        explanation: str,
    ) -> ResolutionResult:
        """Construct ResolutionResult with line-aware geometry."""
        # Determine provenance status
        if cand.is_multi_line:
            status = ProvenanceStatus.MULTI_REGION
            # Group tokens by line
            lines_dict: dict[int, list[BBox]] = {}
            for t in cand.tokens:
                lines_dict.setdefault(t.line_index, []).append(t.bbox)
            regions = [union_bbox_list(boxes) for boxes in lines_dict.values()]
            regions = [r for r in regions if r is not None]
        elif cand.match_type == "exact":
            status = ProvenanceStatus.EXACT
            regions = [cand.bbox]
        elif cand.match_type.startswith("normalized"):
            status = ProvenanceStatus.NORMALIZED
            regions = [cand.bbox]
        else:
            status = ProvenanceStatus.FUZZY
            regions = [cand.bbox]

        return ResolutionResult(
            field=field,
            value=value,
            status=status,
            page=cand.page,
            bbox=cand.bbox,
            regions=regions,
            confidence=confidence,
            matched_text=cand.matched_text,
            explanation=explanation,
        )

    def _is_likely_derived(self, field: str, value: Any) -> bool:
        """Heuristic to detect mathematically or semantically derived values."""
        derived_keywords = {"total", "sum", "subtotal", "tax_rate", "average", "ratio", "percent", "count"}
        field_lower = field.lower()
        return any(kw in field_lower for kw in derived_keywords)
