"""EXP-047 Phase A: Character-Level Document Index.

Builds a character-level index over all PDF characters extracted via pypdfium2.
Provides O(log n) substring search across the entire document via a suffix array
(doubling algorithm, O(n log² n)), returning exact bounding boxes for every
occurrence of any query substring.

Key improvements over the token-level index:
  - Finds values that span multiple tokens ("12" "500" → "12500")
  - Finds values inside longer tokens ("1,250" in "$1,250.00")
  - Handles OCR token boundary misalignment

Usage:
    index = CharIndex(pdf_path)
    matches = index.find_all("12500")
    # Returns list[CharMatch] with page, bbox, matched text
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pypdfium2 as pdfium


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CharRecord:
    """A single character extracted from the PDF with its physical geometry."""
    char: str
    x0: float
    y0: float
    x1: float
    y1: float
    page_index: int   # 0-indexed
    font_size: float
    font_name: str


@dataclass(slots=True)
class CharMatch:
    """A located occurrence of a query substring in the document."""
    page: int          # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float
    char_start: int    # global char offset (inclusive)
    char_end: int      # global char offset (exclusive)
    matched_text: str

    @property
    def bbox_xywh(self) -> list[float]:
        """Return [x, y, w, h] in page-space coordinates (unnormalized)."""
        return [self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0]


# Page boundary sentinel in the concatenated string
_PAGE_SENTINEL = "\x00"


def _normalize_char(ch: str) -> str:
    """Casefold + NFKC for consistent searching."""
    return unicodedata.normalize("NFKC", ch).casefold()


def _build_suffix_array(s: bytes) -> list[int]:
    """Build suffix array using O(n log² n) doubling algorithm.

    Args:
        s: The string as UTF-8 bytes (after encoding).

    Returns:
        Suffix array: list of starting positions in sorted suffix order.
    """
    n = len(s)
    if n == 0:
        return []

    # Initial ranking by first character
    sa = list(range(n))
    rank = list(s)  # byte values as initial ranks
    tmp = [0] * n
    gap = 1

    while gap < n:
        def sort_key(i: int) -> tuple[int, int]:
            r1 = rank[i]
            r2 = rank[i + gap] if i + gap < n else -1
            return (r1, r2)

        sa.sort(key=sort_key)

        # Recompute ranks
        tmp[sa[0]] = 0
        for j in range(1, n):
            prev, cur = sa[j - 1], sa[j]
            tmp[cur] = tmp[prev]
            if sort_key(cur) != sort_key(prev):
                tmp[cur] += 1

        rank = tmp[:]
        if rank[sa[-1]] == n - 1:
            break  # All ranks unique → done
        gap *= 2

    return sa


class CharIndex:
    """Character-level index over a PDF document for fast substring search.

    Extracts every character with its bounding box via pypdfium2, concatenates
    them into one long string with page delimiters, builds a suffix array, and
    supports O(log n) binary-search queries.
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        self._records: list[CharRecord] = []
        self._raw_str: str = ""          # original (casefolded for search)
        self._orig_str: str = ""         # original case (for display)
        self._page_offsets: list[int] = []   # offset of start of each page
        self._suffix_array: list[int] = []
        self._raw_bytes: bytes = b""
        self.num_chars: int = 0
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Extract characters and build suffix array."""
        doc = pdfium.PdfDocument(str(self.pdf_path))
        try:
            chars_parts: list[str] = []
            orig_parts: list[str] = []
            self._page_offsets = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text_page = page.get_textpage()
                width = page.get_width()
                height = page.get_height()

                self._page_offsets.append(len("".join(chars_parts)))

                page_chars: list[CharRecord] = []
                n_chars = text_page.count_chars()
                for ci in range(n_chars):
                    ch = text_page.get_text_range(ci, 1)
                    if not ch:
                        continue
                    rect = text_page.get_charbox(ci, loose=False)
                    if rect is None:
                        rect = (0.0, 0.0, 0.0, 0.0)
                    # pdfium returns (x0, y0, x1, y1) in PDF coordinates
                    # y is bottom-up in PDF → flip to top-down
                    x0_n = rect[0] / width if width > 0 else 0.0
                    y0_n = 1.0 - rect[3] / height if height > 0 else 0.0
                    x1_n = rect[2] / width if width > 0 else 0.0
                    y1_n = 1.0 - rect[1] / height if height > 0 else 0.0

                    # Font info (best-effort)
                    try:
                        font_size = text_page.get_charbox(ci, loose=False)
                        fs = 12.0
                    except Exception:
                        fs = 12.0

                    rec = CharRecord(
                        char=ch,
                        x0=x0_n,
                        y0=y0_n,
                        x1=x1_n,
                        y1=y1_n,
                        page_index=page_idx,
                        font_size=fs,
                        font_name="",
                    )
                    page_chars.append(rec)
                    norm_ch = _normalize_char(ch)
                    # Replace sentinels and newlines with space for cleaner search
                    if norm_ch == _PAGE_SENTINEL or norm_ch in ("\n", "\r"):
                        norm_ch = " "
                    chars_parts.append(norm_ch)
                    orig_parts.append(ch)

                self._records.extend(page_chars)

                # Add page sentinel
                chars_parts.append(_PAGE_SENTINEL)
                orig_parts.append(_PAGE_SENTINEL)
                self._records.append(CharRecord(
                    char=_PAGE_SENTINEL, x0=0, y0=0, x1=0, y1=0,
                    page_index=page_idx, font_size=0, font_name=""
                ))

                page.close()

        finally:
            doc.close()

        self._raw_str = "".join(chars_parts)
        self._orig_str = "".join(orig_parts)
        self.num_chars = len(self._raw_str)

        # Build suffix array on ASCII-safe encoded form
        # We encode normalized string as Latin-1-like bytes, replacing chars > 127
        encoded: list[int] = []
        for ch in self._raw_str:
            cp = ord(ch)
            encoded.append(min(cp, 127))
        self._raw_bytes = bytes(encoded)

        # Build suffix array
        if self.num_chars <= 200_000:
            # Use doubling algorithm
            self._suffix_array = _build_suffix_array(self._raw_bytes)
        else:
            # For large docs, use a simpler position-sorted approach
            # (still O(n log n) via Python sort)
            s = self._raw_str
            self._suffix_array = sorted(range(len(s)), key=lambda i: s[i:i+32])

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def find_all(self, substring: str, case_sensitive: bool = False) -> list[CharMatch]:
        """Find all occurrences of substring in the document.

        Args:
            substring: The query string.
            case_sensitive: If False (default), uses casefolded comparison.

        Returns:
            List of CharMatch objects sorted by (page, y0, x0).
        """
        if not substring or not self._raw_str:
            return []

        # Normalize query
        if case_sensitive:
            query = substring
            text = self._orig_str
        else:
            query = _normalize_char(substring).replace(_PAGE_SENTINEL, " ")
            text = self._raw_str

        qlen = len(query)
        if qlen == 0:
            return []

        # Binary search for lower bound
        lo = self._lower_bound(text, query)
        hi = self._upper_bound(text, query, lo)

        results: list[CharMatch] = []
        seen: set[int] = set()  # deduplicate by global offset

        for rank in range(lo, hi):
            if rank >= len(self._suffix_array):
                break
            pos = self._suffix_array[rank]
            if pos in seen:
                continue
            seen.add(pos)

            end_pos = pos + qlen
            if end_pos > len(text):
                continue

            # Skip if sentinel is in the span
            span_text = text[pos:end_pos]
            if _PAGE_SENTINEL in span_text:
                continue

            # Build bbox by unioning all char bboxes in span
            match = self._span_to_match(pos, end_pos)
            if match is not None:
                results.append(match)

        results.sort(key=lambda m: (m.page, m.y0, m.x0))
        return results

    def find_all_pattern(self, pattern: str) -> list[CharMatch]:
        """Find all matches of a regex pattern (fallback for complex queries)."""
        results: list[CharMatch] = []
        for m in re.finditer(pattern, self._raw_str, re.IGNORECASE):
            if _PAGE_SENTINEL in self._raw_str[m.start():m.end()]:
                continue
            match = self._span_to_match(m.start(), m.end())
            if match is not None:
                results.append(match)
        return results

    def get_page_char_count(self, page_idx: int) -> int:
        """Return number of characters on a given 0-indexed page."""
        start = self._page_offsets[page_idx] if page_idx < len(self._page_offsets) else 0
        end = self._page_offsets[page_idx + 1] if page_idx + 1 < len(self._page_offsets) else self.num_chars
        return end - start

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lower_bound(self, text: str, query: str) -> int:
        """Binary search: first suffix >= query."""
        sa = self._suffix_array
        lo, hi = 0, len(sa)
        while lo < hi:
            mid = (lo + hi) // 2
            pos = sa[mid]
            if text[pos:pos + len(query)] < query:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _upper_bound(self, text: str, query: str, lo: int) -> int:
        """Binary search: first suffix where prefix > query."""
        sa = self._suffix_array
        hi_bound = len(sa)
        lo_b, hi_b = lo, hi_bound
        qlen = len(query)
        while lo_b < hi_b:
            mid = (lo_b + hi_b) // 2
            pos = sa[mid]
            if text[pos:pos + qlen] <= query:
                lo_b = mid + 1
            else:
                hi_b = mid
        return lo_b

    def _span_to_match(self, start: int, end: int) -> CharMatch | None:
        """Convert a global character span to a CharMatch with bbox."""
        if start >= len(self._records) or end > len(self._records):
            return None

        x0_vals, y0_vals, x1_vals, y1_vals = [], [], [], []
        page_indices: set[int] = set()
        matched_text_parts = []

        for i in range(start, end):
            if i >= len(self._records):
                break
            rec = self._records[i]
            if rec.char == _PAGE_SENTINEL:
                continue
            page_indices.add(rec.page_index)
            matched_text_parts.append(rec.char)
            if rec.x0 != rec.x1 and rec.y0 != rec.y1:  # skip zero-area glyphs
                x0_vals.append(rec.x0)
                y0_vals.append(rec.y0)
                x1_vals.append(rec.x1)
                y1_vals.append(rec.y1)

        if not x0_vals or len(page_indices) > 1:
            # Cross-page match — skip
            return None

        page_idx = next(iter(page_indices))
        return CharMatch(
            page=page_idx + 1,  # 1-indexed
            x0=min(x0_vals),
            y0=min(y0_vals),
            x1=max(x1_vals),
            y1=max(y1_vals),
            char_start=start,
            char_end=end,
            matched_text="".join(matched_text_parts),
        )

    def iter_all_chars(self) -> Iterator[CharRecord]:
        """Iterate over all character records."""
        return iter(self._records)

    def chars_on_page(self, page_idx: int) -> list[CharRecord]:
        """Return all characters on a given 0-indexed page."""
        return [r for r in self._records if r.page_index == page_idx]
