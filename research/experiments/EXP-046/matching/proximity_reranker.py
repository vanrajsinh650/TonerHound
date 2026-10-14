"""Stage 17: Token Proximity Reranker.

Reranks candidates when multiple matches exist using 5D Gaussian / structural reranker:
- Same-page preference
- Same-row preference
- Vertical position relative to expected / interpolated y
- Sibling field alignment
- Row-context feature
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.resolution.reranker import StructuralReranker, RerankedCandidate


@dataclass
class ProximityReranker:
    structural_reranker: StructuralReranker = None

    def __post_init__(self):
        if self.structural_reranker is None:
            self.structural_reranker = StructuralReranker()

    def rerank_candidates(
        self,
        candidates: Sequence[MatchCandidate | dict[str, Any]],
        expected_page: int | None = None,
        expected_y: float | None = None,
        sibling_boxes: Sequence[BBox | tuple[float, float, float, float]] | None = None,
        row_band_y: tuple[float, float] | None = None,
    ) -> list[tuple[Any, float]]:
        """Rerank candidates based on spatial proximity and structural context.
        
        Returns:
            Sorted list of (candidate, composite_score) descending.
        """
        if not candidates:
            return []

        scored: list[tuple[Any, float]] = []

        for c in candidates:
            # Extract basic geometry
            if isinstance(c, MatchCandidate):
                c_page = c.page
                c_bbox = c.bbox
                c_score = getattr(c, "confidence", 0.5)
            elif isinstance(c, dict):
                c_page = c.get("page", 0)
                bb = c.get("bounding_box") or c.get("bbox") or [0, 0, 0, 0]
                c_bbox = BBox(bb[0], bb[1], bb[2], bb[3]) if len(bb) == 4 else BBox(0, 0, 0, 0)
                c_score = c.get("confidence", 0.5)
            else:
                continue

            score = c_score * 10.0

            # 1. Page match
            if expected_page is not None:
                if c_page == expected_page:
                    score += 15.0
                else:
                    page_dist = abs(c_page - expected_page)
                    score -= page_dist * 5.0

            # 2. Row band confinement
            if row_band_y is not None:
                y_min, y_max = row_band_y
                cy = c_bbox.center_y
                if y_min <= cy <= y_max:
                    score += 10.0
                else:
                    dist_band = min(abs(cy - y_min), abs(cy - y_max))
                    score += 10.0 * math.exp(-0.5 * (dist_band / 20.0) ** 2)

            # 3. Expected vertical position
            if expected_y is not None:
                dy = abs(c_bbox.center_y - expected_y)
                # Gaussian vertical score
                score += 8.0 * math.exp(-0.5 * (dy / 30.0) ** 2)

            # 4. Sibling field horizontal/vertical alignment
            if sibling_boxes:
                sib_scores = []
                for sb in sibling_boxes:
                    if isinstance(sb, (list, tuple)) and len(sb) == 4:
                        s_box = BBox(sb[0], sb[1], sb[2], sb[3])
                    elif isinstance(sb, BBox):
                        s_box = sb
                    else:
                        continue

                    # Same horizontal row line
                    dy_sib = abs(c_bbox.center_y - s_box.center_y)
                    if dy_sib <= max(c_bbox.height, s_box.height) * 0.75:
                        sib_scores.append(8.0)
                    else:
                        sib_scores.append(8.0 * math.exp(-0.5 * (dy_sib / 25.0) ** 2))
                if sib_scores:
                    score += max(sib_scores)

            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
