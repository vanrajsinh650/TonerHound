"""EXP-039 Phase D: Multi-Line Evidence Assembly & Hyphenation Joiner.

Target classes: HYPHENATION (7,623 fields) + MULTI_LINE_SPLIT (563 fields)
Theoretical ceiling: +1.20 pp
Realistic target: +0.5 to +1.0 pp

Assembles multi-line values (addresses, wrapped descriptions) and hyphenated words
split across line breaks into a unified bounding box.
"""

from __future__ import annotations

from typing import Any

from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


def union_bbox(bboxes: list[tuple[float, float, float, float] | list[float]]) -> tuple[float, float, float, float]:
    """Compute the union bounding box from a list of (x, y, w, h) bboxes."""
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[0] + b[2] for b in bboxes)
    y_max = max(b[1] + b[3] for b in bboxes)
    return (round(x_min, 6), round(y_min, 6), round(x_max - x_min, 6), round(y_max - y_min, 6))


class MultiLineAssembler:
    """Detects and merges multi-line visual text spans and hyphenated line-breaks."""

    def __init__(self, doc_index: Any) -> None:
        self.doc_index = doc_index

    def assemble_multiline_bbox(
        self,
        gold_value: str,
        page_hint: int | None = None,
        max_lines: int = 5,
    ) -> tuple[tuple[float, float, float, float] | None, str | None, int | None]:
        """Assemble multi-line tokens matching gold_value into a unified bounding box.
        
        Returns:
            (union_bbox_tuple, matched_str, page_num) or (None, None, None)
        """
        lines = [line.strip() for line in gold_value.split("\n") if line.strip()]
        if len(lines) <= 1:
            return None, None, None

        target_pages = [page_hint] if page_hint is not None else [p.page_number for p in self.doc_index.pages]

        for p_num in target_pages:
            page = self.doc_index.get_page(p_num)
            if not page:
                continue

            matched_lines_bboxes: list[tuple[float, float, float, float]] = []
            matched_line_indices: list[int] = []

            for target_line in lines:
                norm_target = target_line.lower()
                line_found = False
                for vl in page.lines:
                    vl_text = vl.text.strip().lower()
                    if norm_target in vl_text or vl_text in norm_target:
                        matched_lines_bboxes.append(vl.bbox.to_coco())
                        matched_line_indices.append(vl.line_index)
                        line_found = True
                        break

            if len(matched_lines_bboxes) >= 2 and len(matched_lines_bboxes) == len(lines):
                # Check line spread constraint
                if max(matched_line_indices) - min(matched_line_indices) <= max_lines + 2:
                    ub = union_bbox(matched_lines_bboxes)
                    return ub, " ".join(lines), p_num

        return None, None, None

    def join_hyphenated(
        self,
        gold_value: str,
        page_hint: int | None = None,
    ) -> tuple[tuple[float, float, float, float] | None, str | None, int | None]:
        """Detect tokens split across lines with trailing hyphens and join them.
        
        Returns:
            (union_bbox_tuple, joined_str, page_num) or (None, None, None)
        """
        norm_gold = gold_value.strip().lower()
        if not norm_gold or " " in norm_gold:
            return None, None, None

        target_pages = [page_hint] if page_hint is not None else [p.page_number for p in self.doc_index.pages]

        for p_num in target_pages:
            page = self.doc_index.get_page(p_num)
            if not page or len(page.tokens) < 2:
                continue

            tokens = page.tokens
            for i in range(len(tokens) - 1):
                t1 = tokens[i]
                t2 = tokens[i + 1]

                if t1.text.endswith("-") and len(t1.text) > 1:
                    joined = (t1.text[:-1] + t2.text).lower()
                    if joined == norm_gold:
                        ub = union_bbox([t1.bbox.to_coco(), t2.bbox.to_coco()])
                        return ub, t1.text[:-1] + t2.text, p_num

        return None, None, None
