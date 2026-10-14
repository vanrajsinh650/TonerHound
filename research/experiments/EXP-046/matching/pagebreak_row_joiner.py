"""EXP-046 Stage 12: Page-Break Row Joiner.

Stitches records split across page boundaries (e.g. entry description starting
at the bottom of page N and completing at the top of page N+1).
"""

from __future__ import annotations

from typing import Any, Sequence
from rapidfuzz import fuzz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


class PageBreakRowJoiner:
    """Detects and reconciles table rows split across consecutive page boundaries."""

    def __init__(self, header_similarity_threshold: float = 80.0) -> None:
        self.header_similarity_threshold = header_similarity_threshold

    def is_continued_header(self, text_candidate: str, primary_header: str) -> bool:
        """Check if candidate line is a continued table header."""
        sim = fuzz.ratio(text_candidate.strip().lower(), primary_header.strip().lower())
        return sim >= self.header_similarity_threshold

    def join_split_page_boundary(
        self,
        target_value: Any,
        page_n: DocumentPage,
        page_n_plus_1: DocumentPage,
    ) -> tuple[BBox | None, str | None, int | None]:
        """Check if target value spans bottom of page N and top of page N+1."""
        target_str = str(target_value).strip().lower()
        if not target_str:
            return None, None, None

        # Collect bottom tokens of page N (y > 0.85)
        bottom_tokens = [t for t in page_n.tokens if t.bbox.y >= 0.85]
        # Collect top tokens of page N+1 (y <= 0.25)
        top_tokens = [t for t in page_n_plus_1.tokens if t.bbox.y <= 0.25]

        # Check if target is in bottom of page N
        for t in bottom_tokens:
            if target_str in t.text.strip().lower():
                b = t.bbox
                return BBox(b.x, b.y, b.width, b.height, page_n.page_number), t.text, page_n.page_number

        # Check if target is in top of page N+1
        for t in top_tokens:
            if target_str in t.text.strip().lower():
                b = t.bbox
                return BBox(b.x, b.y, b.width, b.height, page_n_plus_1.page_number), t.text, page_n_plus_1.page_number

        # Check combined text across boundary
        bottom_text = " ".join(t.text for t in bottom_tokens)
        top_text = " ".join(t.text for t in top_tokens)
        combo = f"{bottom_text} {top_text}"
        if target_str in combo.lower():
            # If target appears in composite, attribute to the page with higher token overlap
            m_top = [t for t in top_tokens if t.text.strip().lower() in target_str]
            if m_top:
                rx0 = min(t.bbox.x for t in m_top)
                ry0 = min(t.bbox.y for t in m_top)
                rw = max(t.bbox.x + t.bbox.width for t in m_top) - rx0
                rh = max(t.bbox.y + t.bbox.height for t in m_top) - ry0
                return BBox(rx0, ry0, rw, rh, page_n_plus_1.page_number), " ".join(t.text for t in m_top), page_n_plus_1.page_number

        return None, None, None
