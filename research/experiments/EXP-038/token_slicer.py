r"""EXP-038 Fix 7: Token Slicing & Column Bleed Refinement.

Refines bounding boxes that bleed into adjacent table columns:
- Problem: In wide tables, token bounding boxes extend into adjacent numeric columns,
  causing predicted bbox width to be 2-3x wider than ground truth ($IoU < 0.50$).
- Solution: Proportionally slice bounding box based on exact character span of the target value.
- Safety:
  * Never applied to already-passing citations ($IoU \ge 0.50$).
  * Preserves vertical coordinates ($y_0, y_1$) completely.
  * Only narrows when predicted bbox is substantially wider than expected text length.
"""

from __future__ import annotations

from typing import Sequence

from tonerhound.geometry.coordinates import BBox


class TokenSlicer:
    """Refines wide bounding boxes to exact character sub-spans."""

    @staticmethod
    def refine_bbox_width(
        cand_bbox: Sequence[float],
        matched_text: str,
        target_value: str,
    ) -> list[float]:
        """Proportionally slice candidate bbox [x, y, w, h] to match target_value span."""
        if not cand_bbox or len(cand_bbox) != 4:
            return list(cand_bbox)

        x, y, w, h = cand_bbox
        m_len = len(matched_text.strip())
        t_len = len(target_value.strip())

        if m_len <= t_len or m_len == 0 or t_len == 0:
            return list(cand_bbox)

        # Ratio check: only slice if matched text is at least 1.4x longer than target
        if m_len < t_len * 1.4:
            return list(cand_bbox)

        idx = matched_text.lower().find(target_value.lower())
        if idx == -1:
            # Check prefix / suffix
            if matched_text.lower().startswith(target_value.lower()[:3]):
                idx = 0
            else:
                return list(cand_bbox)

        start_ratio = idx / float(m_len)
        span_ratio = t_len / float(m_len)

        refined_x = x + (w * start_ratio)
        refined_w = w * span_ratio

        return [refined_x, y, refined_w, h]
