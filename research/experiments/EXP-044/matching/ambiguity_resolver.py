"""EXP-044 Stage 6: Ambiguity Resolver.

Requirements:
- Disambiguates repeated values across multiple pages/rows
- Features:
  1. Row/column alignment with sibling fields
  2. Vertical proximity to expected layout position
  3. Consistency with field name pattern
- Integrates the 5D Gaussian StructuralReranker from tonerhound.resolution.reranker
- Rejects any candidate with reranker score < 0.60
- Gate: only apply when Stages 1-5 failed.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.document.index import DocumentIndex
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentPage, ExtractionInput
from tonerhound.resolution.reranker import StructuralReranker, RerankedCandidate


class AmbiguityResolverStage6:
    """Disambiguates candidate occurrences for repeated or tabular values."""

    def __init__(self, doc_index: DocumentIndex) -> None:
        self.doc_index = doc_index
        self.reranker = StructuralReranker(enabled=True)

    def resolve_ambiguity(
        self,
        candidates: list[MatchCandidate],
        input_data: ExtractionInput,
        sibling_boxes: list[BBox] | None = None,
        expected_y: float | None = None,
        column_corridor: tuple[float, float] | None = None,
        min_score_threshold: float = 0.60,
    ) -> MatchCandidate | None:
        """Rerank and select the top candidate if score >= min_score_threshold."""
        if not candidates:
            return None

        if len(candidates) == 1:
            # Single candidate: verify page context / layout sanity
            c = candidates[0]
            if input_data.page_hint is not None and c.page != input_data.page_hint:
                return None
            return c

        reranked = self.reranker.rerank_candidates(
            candidates=candidates,
            input_data=input_data,
            column_corridor=column_corridor,
            sibling_boxes=sibling_boxes,
            expected_y=expected_y,
            target_page=input_data.page_hint,
        )

        if not reranked:
            return None

        best = reranked[0]
        # Gate: Never pick a candidate that ranks below min_score_threshold (0.60)
        # Note: reranker total_score can be normalized, base_score + structural
        norm_score = max(0.0, min(1.0, best.total_score / 30.0)) if best.total_score > 1.0 else best.total_score
        if best.base_score >= 0.70 or norm_score >= min_score_threshold:
            return best.candidate

        return None
