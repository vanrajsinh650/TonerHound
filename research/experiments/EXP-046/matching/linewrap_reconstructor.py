"""Stage 26: Line-Wrap Reconstructor.

Reconstructs logical text entities across line wraps:
- Detect trailing hyphen (strip hyphen and join with first token of next line)
- Detect trailing comma/period (join if next line starts with lowercase)
- Detect unmatched open parenthesis (join across subsequent lines until closed)
"""

from __future__ import annotations

import re
from typing import Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import VisualLine, DocumentToken


def should_join_lines(line_a_text: str, line_b_text: str) -> tuple[bool, str]:
    """Determines if line_a and line_b should be joined and how."""
    s_a = line_a_text.strip()
    s_b = line_b_text.strip()

    if not s_a or not s_b:
        return False, ""

    # 1. Trailing hyphen (hyphenation wrap)
    if s_a.endswith("-") or s_a.endswith("—") or s_a.endswith("–"):
        clean_a = s_a[:-1].strip()
        joined = f"{clean_a}{s_b}"
        return True, joined

    # 2. Open parenthesis wrap
    open_count = s_a.count("(") - s_a.count(")")
    if open_count > 0:
        joined = f"{s_a} {s_b}"
        return True, joined

    # 3. Trailing comma/period continuation if next line starts with lowercase
    if (s_a.endswith(",") or s_a.endswith(".")) and s_b[0].islower():
        joined = f"{s_a} {s_b}"
        return True, joined

    return False, ""


def reconstruct_wrapped_lines(
    lines: Sequence[VisualLine],
) -> list[tuple[str, BBox]]:
    """Merges consecutive wrapped lines into combined entities with union bboxes."""
    if not lines:
        return []

    results: list[tuple[str, BBox]] = []
    i = 0
    n = len(lines)

    while i < n:
        curr_line = lines[i]
        curr_text = curr_line.text.strip()
        curr_box = curr_line.bbox

        j = i + 1
        while j < n:
            next_line = lines[j]
            next_text = next_line.text.strip()
            # Check vertical proximity
            v_gap = next_line.bbox.y0 - curr_box.y1
            if v_gap > 2.0 * max(10.0, curr_box.height):
                break

            should_join, joined_text = should_join_lines(curr_text, next_text)
            if should_join:
                curr_text = joined_text
                # Union bboxes
                curr_box = BBox.from_xyxy(
                    min(curr_box.x0, next_line.bbox.x0),
                    min(curr_box.y0, next_line.bbox.y0),
                    max(curr_box.x1, next_line.bbox.x1),
                    max(curr_box.y1, next_line.bbox.y1),
                )
                j += 1
            else:
                break

        results.append((curr_text, curr_box))
        i = j

    return results
