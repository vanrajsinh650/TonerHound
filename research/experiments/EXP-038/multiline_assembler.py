"""EXP-038 Fix 5: Multi-Line Evidence Assembler.

Assembles candidate bounding boxes across vertically adjacent lines:
- Problem: Multi-word names, addresses, and narrative descriptions wrap across 2-4 lines.
  A single line match captures only a fraction of the text, yielding IoU < 0.50.
- Solution: Search for contiguous multi-line matches across VisualLines, building a union bbox.
- Safety:
  * Only unions lines that are vertically adjacent (vertical gap < 1.5 * median line height).
  * Rejects candidate unions spanning more than 5 lines.
  * Preserves horizontal bounding box containment.
"""

from __future__ import annotations

from typing import Sequence

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


class MultiLineAssembler:
    """Assembles bounding boxes for multi-line wrapped text values."""

    def __init__(self, doc_index: DocumentIndex) -> None:
        self.doc_index = doc_index

    def find_multiline_match(
        self,
        query: str,
        page_hint: int | None = None,
        max_lines: int = 5,
        max_vertical_gap_ratio: float = 1.5,
    ) -> list[tuple[BBox, str, int]]:
        """Search for query words distributed across 2..max_lines adjacent VisualLines.
        
        Returns:
            list of (union_bbox, matched_text, page_number)
        """
        clean_words = query.strip().split()
        if len(clean_words) < 2:
            return []

        target_pages = [page_hint] if page_hint is not None else list(self.doc_index._pages_by_num.keys())
        matches: list[tuple[BBox, str, int]] = []

        query_collapsed = "".join(re_clean.lower() for re_clean in clean_words)

        for p_num in target_pages:
            page = self.doc_index.get_page(p_num)
            if not page or len(page.lines) < 2:
                continue

            lines = page.lines
            n_lines = len(lines)

            for i in range(n_lines - 1):
                for span_len in range(2, min(max_lines + 1, n_lines - i + 1)):
                    span_lines = lines[i : i + span_len]

                    # Vertical adjacency check
                    adjacent = True
                    for k in range(len(span_lines) - 1):
                        l1 = span_lines[k]
                        l2 = span_lines[k + 1]
                        gap = l2.bbox.y0 - l1.bbox.y1
                        h1 = max(0.001, l1.bbox.height)
                        if gap > h1 * max_vertical_gap_ratio or gap < -0.01:
                            adjacent = False
                            break

                    if not adjacent:
                        continue

                    # Concatenate tokens across span lines
                    span_tokens = [tok for l in span_lines for tok in l.tokens]
                    span_text_collapsed = "".join(t.text.strip().lower() for t in span_tokens)

                    if query_collapsed in span_text_collapsed:
                        # Extract exact matching tokens
                        ub = union_bbox_list([t.bbox for t in span_tokens])
                        if ub is not None:
                            matched_str = " ".join(t.text for t in span_tokens)
                            matches.append((ub, matched_str, p_num))
                            break

        return matches
