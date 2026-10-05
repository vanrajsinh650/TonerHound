"""EXP-038 Fix 8: Hyphenation Joiner across Visual Lines.

Recovers words hyphenated across line wraps:
- Problem: Words split across lines with trailing hyphens (e.g., 'Consoli-' on line 1, 'dated' on line 2).
- Solution: When exact lookup fails, identify trailing hyphen on line N, join with leading token on line N+1.
- Safety:
  * Only joins if hyphen is at the terminal end of the line.
  * Only joins if the next line's leading token begins with a lowercase letter (standard word hyphenation).
  * Emits union bounding box of the two token fragments.
"""

from __future__ import annotations

from typing import Sequence

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine


class HyphenationJoiner:
    """Detects and joins line-terminal hyphenated word tokens."""

    def __init__(self, doc_index: DocumentIndex) -> None:
        self.doc_index = doc_index

    def find_hyphenated_word(
        self,
        query: str,
        page_hint: int | None = None,
    ) -> list[tuple[BBox, str, int]]:
        """Search for word split by terminal hyphen across adjacent lines.
        
        Returns:
            list of (union_bbox, joined_text, page_number)
        """
        clean_query = query.strip().lower()
        if len(clean_query) < 4 or " " in clean_query:
            return []

        target_pages = [page_hint] if page_hint is not None else list(self.doc_index._pages_by_num.keys())
        results: list[tuple[BBox, str, int]] = []

        for p_num in target_pages:
            page = self.doc_index.get_page(p_num)
            if not page or len(page.lines) < 2:
                continue

            lines = page.lines
            for i in range(len(lines) - 1):
                l1 = lines[i]
                l2 = lines[i + 1]

                if not l1.tokens or not l2.tokens:
                    continue

                t_last = l1.tokens[-1]
                t_first = l2.tokens[0]

                # Check if t_last ends with hyphen
                last_txt = t_last.text.strip()
                if not last_txt.endswith(("-", "‐", "–")):
                    continue

                stem1 = last_txt.rstrip("-‐–").lower()
                stem2 = t_first.text.strip().lower()

                # Joined candidate
                joined = stem1 + stem2
                if joined == clean_query:
                    ub = union_bbox_list([t_last.bbox, t_first.bbox])
                    if ub is not None:
                        results.append((ub, joined, p_num))

        return results
