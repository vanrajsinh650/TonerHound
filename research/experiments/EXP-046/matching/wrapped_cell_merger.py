"""EXP-046 Stage 9: Universal Wrapped Cell Merger.

Detects abnormally tall table rows (>2x median row height) and multi-line cell fragments,
merging vertically overlapping or consecutive text lines into unified logical rows.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def is_numeric_cell(text: str) -> bool:
    cleaned = text.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "").strip()
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


class WrappedCellMerger:
    """Merges wrapped cell lines into unified record bounding boxes."""

    def __init__(self, height_multiplier_threshold: float = 1.8) -> None:
        self.height_multiplier_threshold = height_multiplier_threshold

    def merge_wrapped_lines(
        self, tokens: Sequence[DocumentToken]
    ) -> list[dict[str, Any]]:
        """Group tokens into logical table rows, merging wrapped lines."""
        if not tokens:
            return []

        # Sort tokens top-to-bottom, left-to-right
        sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y, t.bbox.x))

        # Estimate median token height
        heights = [t.bbox.height for t in sorted_tokens if t.bbox.height > 0]
        if not heights:
            return []
        med_h = float(np.median(heights))

        # Form initial horizontal line clusters
        lines: list[list[DocumentToken]] = []
        curr_line = [sorted_tokens[0]]
        last_y = sorted_tokens[0].bbox.y

        for t in sorted_tokens[1:]:
            if abs(t.bbox.y - last_y) <= med_h * 0.7:
                curr_line.append(t)
            else:
                lines.append(curr_line)
                curr_line = [t]
                last_y = t.bbox.y
        if curr_line:
            lines.append(curr_line)

        # Merge adjacent lines where line 1 has text description and line 2 has numbers/continuation
        merged_records: list[dict[str, Any]] = []
        idx = 0
        while idx < len(lines):
            l1 = lines[idx]
            l1_text = " ".join(t.text for t in l1)
            l1_has_num = any(is_numeric_cell(t.text) for t in l1)

            if idx + 1 < len(lines):
                l2 = lines[idx + 1]
                l2_text = " ".join(t.text for t in l2)
                l2_has_num = any(is_numeric_cell(t.text) for t in l2)
                v_gap = l2[0].bbox.y - (l1[0].bbox.y + l1[0].bbox.height)

                # Merge if tight vertical gap and l1 is description while l2 contains value or continuation
                if v_gap <= med_h * 1.5 and ((not l1_has_num and l2_has_num) or (l1_has_num and not l2_has_num)):
                    combo_tokens = l1 + l2
                    rx0 = min(t.bbox.x for t in combo_tokens)
                    ry0 = min(t.bbox.y for t in combo_tokens)
                    rx1 = max(t.bbox.x + t.bbox.width for t in combo_tokens)
                    ry1 = max(t.bbox.y + t.bbox.height for t in combo_tokens)
                    merged_records.append({
                        "tokens": combo_tokens,
                        "text": f"{l1_text} {l2_text}",
                        "bbox": [rx0, ry0, rx1 - rx0, ry1 - ry0],
                        "merged": True,
                    })
                    idx += 2
                    continue

            # Single line record
            rx0 = min(t.bbox.x for t in l1)
            ry0 = min(t.bbox.y for t in l1)
            rx1 = max(t.bbox.x + t.bbox.width for t in l1)
            ry1 = max(t.bbox.y + t.bbox.height for t in l1)
            merged_records.append({
                "tokens": l1,
                "text": l1_text,
                "bbox": [rx0, ry0, rx1 - rx0, ry1 - ry0],
                "merged": False,
            })
            idx += 1

        return merged_records

    def match_wrapped_target(
        self, target_value: Any, page_obj: DocumentPage
    ) -> tuple[BBox | None, str | None]:
        """Search merged row records for target value."""
        target_str = str(target_value).strip().lower()
        if not target_str:
            return None, None

        records = self.merge_wrapped_lines(page_obj.tokens)
        for r in records:
            if target_str in r["text"].lower():
                # Locate specific matching tokens in record
                m_tokens = [
                    t for t in r["tokens"]
                    if target_str in t.text.strip().lower() or t.text.strip().lower() in target_str
                ]
                if m_tokens:
                    rx0 = min(t.bbox.x for t in m_tokens)
                    ry0 = min(t.bbox.y for t in m_tokens)
                    rw = max(t.bbox.x + t.bbox.width for t in m_tokens) - rx0
                    rh = max(t.bbox.y + t.bbox.height for t in m_tokens) - ry0
                    return BBox(rx0, ry0, rw, rh, page_obj.page_number), " ".join(t.text for t in m_tokens)
        return None, None
