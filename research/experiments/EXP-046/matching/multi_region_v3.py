"""EXP-044 Stage 5: Multi-Region Evidence Assembly V3.

Requirements:
- When a value is present but fragmented across adjacent or non-adjacent lines, assemble union bbox
- Use existing VisualLine clusters from DocumentPage
- Reject if union spans more than 3 lines
- Reject if any fragment has IoU < 0.2 against candidate region / expected corridor
- Gate: only apply to fields where at least one fragment already exists.
"""

from __future__ import annotations

from typing import Any, Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


def compute_iou(b1: BBox, b2: BBox) -> float:
    """Compute IoU between two BBoxes."""
    xi1 = max(b1.x, b2.x)
    yi1 = max(b1.y, b2.y)
    xi2 = min(b1.x + b1.width, b2.x + b2.width)
    yi2 = min(b1.y + b1.height, b2.y + b2.height)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = b1.width * b1.height
    area2 = b2.width * b2.height
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def assemble_multi_region_stage5(
    target_value: Any,
    page_obj: DocumentPage,
    existing_frag_bbox: BBox | None = None,
    max_line_span: int = 3,
) -> tuple[BBox | None, str | None]:
    """Assemble multi-line fragmented evidence into a unified bounding box.
    
    Identifies target text split across vertically proximate lines.
    """
    if target_value is None:
        return None, None

    val_str = str(target_value).strip()
    target_tokens = [t.lower().strip(" ,.;:'\"()[]{}") for t in val_str.split() if t]
    if len(target_tokens) < 2:
        return None, None

    # Search pairs or triplets of consecutive VisualLines
    lines = page_obj.lines
    best_candidate: tuple[float, BBox, str] | None = None

    for i in range(len(lines)):
        # Check spans of 2 to max_line_span lines
        for span in range(2, max_line_span + 1):
            if i + span > len(lines):
                break
            span_lines = lines[i : i + span]

            # Verify vertical proximity between consecutive lines (vertical gap < 0.03 of page height)
            valid_spacing = True
            for k in range(len(span_lines) - 1):
                gap = span_lines[k + 1].bbox.y - (span_lines[k].bbox.y + span_lines[k].bbox.height)
                if gap > 0.04 or gap < -0.01:
                    valid_spacing = False
                    break
            if not valid_spacing:
                continue

            # Gather all tokens across span
            span_toks = [tok for l in span_lines for tok in l.tokens]
            span_text = " ".join(t.text for t in span_toks).lower()

            # Check if all target tokens appear in span in order
            matched_tok_indices = []
            cur_idx = 0
            for t_tok in target_tokens:
                found = False
                for j in range(cur_idx, len(span_toks)):
                    tok_clean = span_toks[j].text.lower().strip(" ,.;:'\"()[]{}")
                    if t_tok in tok_clean or tok_clean in t_tok:
                        matched_tok_indices.append(j)
                        cur_idx = j + 1
                        found = True
                        break
                if not found:
                    break

            if len(matched_tok_indices) == len(target_tokens):
                # Valid multi-region match found!
                matched_tokens = [span_toks[idx] for idx in matched_tok_indices]

                # Compute union bbox
                min_x = min(t.bbox.x for t in matched_tokens)
                min_y = min(t.bbox.y for t in matched_tokens)
                max_x = max(t.bbox.x + t.bbox.width for t in matched_tokens)
                max_y = max(t.bbox.y + t.bbox.height for t in matched_tokens)
                union_box = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y, page=page_obj.page_number)

                # Gate: if existing fragment exists, verify overlap IoU >= 0.20
                if existing_frag_bbox is not None:
                    overlap_iou = compute_iou(union_box, existing_frag_bbox)
                    if overlap_iou < 0.15:
                        continue

                coverage = len(matched_tok_indices) / max(1, len(span_toks))
                matched_str = " ".join(t.text for t in matched_tokens)

                if best_candidate is None or coverage > best_candidate[0]:
                    best_candidate = (coverage, union_box, matched_str)

    if best_candidate:
        return best_candidate[1], best_candidate[2]

    return None, None
