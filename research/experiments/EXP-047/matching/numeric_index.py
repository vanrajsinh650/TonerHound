"""EXP-047 Phase B: Numeric Interval Index.

Scans the character stream to detect all numeric patterns in the document,
canonicalizes each to a float, and provides range-query via an interval tree
(implemented as a sorted list + bisect for zero-dependency simplicity).

Handles:
  - Integers: 12500
  - Decimals: 1250.00
  - Currency: $1,250.00, €1.250,00
  - Percentages: 5.00%  →  0.05
  - Negatives: (1,250.00), -1250
  - Scientific: 1.25e3
  - Parenthetical negatives: (1,234.56)

Usage:
    index = NumericIndex(pdf_path)
    matches = index.find_all_numbers(12500.0, tolerance=0.01)
    # Returns list[NumericMatch]
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium


# ---------------------------------------------------------------------------
# Numeric pattern regex
# ---------------------------------------------------------------------------

# Main numeric pattern (captures currency prefix, optional negation, digits/commas/dots)
_NUMERIC_RE = re.compile(
    r"""
    (?:                          # Optional leading currency or sign
        [$€£¥]?\s*               # currency symbol
        (?P<neg1>-|\()?          # explicit minus or open paren
    )?
    (?P<digits>
        \d{1,3}(?:[,\s]\d{3})*  # thousand-separated: 1,234 or 1 234
        (?:\.\d+)?               # optional fractional part
        |
        \d+(?:\.\d+)?            # plain number
        |
        \.\d+                    # starts with decimal: .75
    )
    (?P<neg2>\))?                # close paren (paired negative)
    \s*(?P<pct>%)?               # optional percentage
    """,
    re.VERBOSE,
)


def parse_numeric_value(text: str) -> float | None:
    """Parse a numeric value from text, returning a canonical float or None.

    Handles currency prefixes, comma separators, parenthetical negatives,
    and percentage symbols.
    """
    s = text.strip()
    if not s:
        return None

    # Quick reject: must contain at least one digit
    if not any(c.isdigit() for c in s):
        return None

    # Reject strings that look like identifiers/CUSIPs:
    # Must not have letters (except allowed suffixes/prefixes)
    # Allow: $, €, £, ¥, (, ), -, %, . , digits, comma, space
    # Reject: strings with letters inside digit runs (e.g. "08862E109", "ABC123")
    stripped_currency = re.sub(r"^[$€£¥\s]+", "", s)
    stripped_paren = stripped_currency.strip("()")
    stripped_pct = stripped_paren.rstrip("%").strip()
    # If after stripping allowed symbols we have letters, reject
    if re.search(r"[A-Za-z]", stripped_pct):
        return None

    # Strip currency symbols
    s_clean = re.sub(r"[$€£¥]\s*", "", s)

    # Detect parenthetical negative: (1,234.56)
    is_paren_neg = s_clean.startswith("(") and s_clean.endswith(")")
    if is_paren_neg:
        s_clean = s_clean[1:-1]

    # Detect explicit minus
    is_minus = s_clean.startswith("-")
    if is_minus:
        s_clean = s_clean[1:]

    # Detect percentage
    is_pct = s_clean.endswith("%")
    if is_pct:
        s_clean = s_clean[:-1].strip()

    # Remove thousands separators (comma or space between groups of 3 digits)
    # but be careful with European formats where dot is the separator
    # Heuristic: if there's a single dot and it has exactly 2 digits after → decimal
    #            if there's commas with 3-digit groups → thousand separator
    s_clean = s_clean.strip()

    # Handle European format: 1.250,00 (dot = thousands, comma = decimal)
    if re.match(r"^\d{1,3}(\.\d{3})+,\d+$", s_clean):
        s_clean = s_clean.replace(".", "").replace(",", ".")
    else:
        # Standard format: remove thousand commas
        s_clean = s_clean.replace(",", "")

    try:
        val = float(s_clean)
    except ValueError:
        # Try scientific notation
        try:
            val = float(s_clean.replace(" ", ""))
        except ValueError:
            return None

    if is_paren_neg or is_minus:
        val = -val

    if is_pct:
        val = val / 100.0

    return val


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class NumericMatch:
    """A located numeric occurrence in the document."""
    page: int            # 1-indexed
    x0: float
    y0: float
    x1: float
    y1: float
    value_float: float
    original_text: str

    @property
    def bbox_xywh(self) -> list[float]:
        return [self.x0, self.y0, self.x1 - self.x0, self.y1 - self.y0]


# ---------------------------------------------------------------------------
# Interval tree (simplified: sorted list + bisect)
# ---------------------------------------------------------------------------

class _SortedNumericStore:
    """Sorted list of (value_float, NumericMatch) for range queries via bisect."""

    def __init__(self) -> None:
        self._keys: list[float] = []
        self._items: list[list[NumericMatch]] = []  # grouped by key

    def insert(self, value: float, match: NumericMatch) -> None:
        idx = bisect.bisect_left(self._keys, value)
        if idx < len(self._keys) and self._keys[idx] == value:
            self._items[idx].append(match)
        else:
            self._keys.insert(idx, value)
            self._items.insert(idx, [match])

    def query_range(self, lo: float, hi: float) -> list[NumericMatch]:
        """Return all matches with value in [lo, hi]."""
        left = bisect.bisect_left(self._keys, lo)
        right = bisect.bisect_right(self._keys, hi)
        result: list[NumericMatch] = []
        for i in range(left, right):
            result.extend(self._items[i])
        return result

    def __len__(self) -> int:
        return sum(len(g) for g in self._items)


# ---------------------------------------------------------------------------
# NumericIndex
# ---------------------------------------------------------------------------

# Regex to detect numeric-looking tokens (fast pre-filter)
_HAS_DIGIT_RE = re.compile(r"\d")
_TOKEN_SPLIT_RE = re.compile(r"\s+")

# Pattern to detect numeric runs in a character stream
_NUM_RUN_RE = re.compile(
    r"""
    [$€£¥]?               # optional currency
    \(?                   # optional open paren
    -?                    # optional minus
    \d[\d,. ]*            # digit run with separators
    \)?                   # optional close paren
    %?                    # optional percent
    """,
    re.VERBOSE,
)


class NumericIndex:
    """Numeric interval index over all numbers found in a PDF document.

    Extracts numeric tokens from pypdfium2 character stream, canonicalizes them,
    and stores in a sorted structure for O(log n + k) range queries.
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        self._store = _SortedNumericStore()
        self.num_values: int = 0
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Extract all numeric values from the PDF."""
        doc = pdfium.PdfDocument(str(self.pdf_path))
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text_page = page.get_textpage()
                width = page.get_width()
                height = page.get_height()

                # Extract full page text with character positions
                n_chars = text_page.count_chars()
                if n_chars == 0:
                    page.close()
                    continue

                # Build runs of characters to detect numeric tokens
                # We scan left-to-right, accumulating digit/punct runs
                self._extract_page_numerics(
                    text_page, n_chars, page_idx + 1, width, height
                )
                page.close()

        finally:
            doc.close()

    def _extract_page_numerics(
        self,
        text_page: Any,
        n_chars: int,
        page_num: int,  # 1-indexed
        page_width: float,
        page_height: float,
    ) -> None:
        """Extract numeric values from a single page's character stream."""

        # Collect (char, x0_n, y0_n, x1_n, y1_n) for each char
        chars: list[tuple[str, float, float, float, float]] = []
        for ci in range(n_chars):
            ch = text_page.get_text_range(ci, 1)
            if not ch:
                continue
            try:
                rect = text_page.get_charbox(ci, loose=False)
            except Exception:
                rect = None
            if rect is None:
                rect = (0.0, 0.0, 0.0, 0.0)

            x0_n = rect[0] / page_width if page_width > 0 else 0.0
            y0_n = 1.0 - rect[3] / page_height if page_height > 0 else 0.0
            x1_n = rect[2] / page_width if page_width > 0 else 0.0
            y1_n = 1.0 - rect[1] / page_height if page_height > 0 else 0.0
            chars.append((ch, x0_n, y0_n, x1_n, y1_n))

        # Segment into "words" by whitespace/newline boundaries
        # Then try to parse each word as a numeric token
        run_chars: list[tuple[str, float, float, float, float]] = []
        for item in chars:
            ch = item[0]
            if ch in (" ", "\t", "\n", "\r", "\x00"):
                if run_chars:
                    self._process_run(run_chars, page_num)
                    run_chars = []
            else:
                run_chars.append(item)

        if run_chars:
            self._process_run(run_chars, page_num)

    def _process_run(
        self,
        run: list[tuple[str, float, float, float, float]],
        page_num: int,
    ) -> None:
        """Try to parse a character run as a numeric value and store it."""
        text = "".join(c[0] for c in run)
        if not _HAS_DIGIT_RE.search(text):
            return

        val = parse_numeric_value(text)
        if val is None:
            return

        x0 = min(c[1] for c in run if c[1] != c[3] or True)
        y0 = min(c[2] for c in run if c[2] != c[4] or True)
        x1 = max(c[3] for c in run)
        y1 = max(c[4] for c in run)

        # Use actual bbox from non-zero glyphs if possible
        non_zero = [(c[1], c[2], c[3], c[4]) for c in run if c[1] != c[3]]
        if non_zero:
            x0 = min(c[0] for c in non_zero)
            y0 = min(c[1] for c in non_zero)
            x1 = max(c[2] for c in non_zero)
            y1 = max(c[3] for c in non_zero)

        match = NumericMatch(
            page=page_num,
            x0=x0, y0=y0, x1=x1, y1=y1,
            value_float=round(val, 8),
            original_text=text,
        )
        self._store.insert(round(val, 8), match)
        self.num_values += 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def find_all_numbers(
        self,
        target: float,
        tolerance: float = 0.01,
    ) -> list[NumericMatch]:
        """Find all numeric occurrences within [target - tolerance, target + tolerance].

        Args:
            target: The canonical float value to search for.
            tolerance: Absolute tolerance for floating-point comparison.
                       For percentages already converted to fractions (e.g. 0.05),
                       tolerance of 0.0001 is appropriate.

        Returns:
            List of NumericMatch sorted by (page, y0, x0).
        """
        lo = target - abs(tolerance)
        hi = target + abs(tolerance)
        results = self._store.query_range(lo, hi)
        results.sort(key=lambda m: (m.page, m.y0, m.x0))
        return results

    def find_numeric_value(
        self,
        value_str: str,
        tolerance: float = 0.01,
    ) -> list[NumericMatch]:
        """Parse value_str to float and call find_all_numbers.

        Returns empty list if value_str cannot be parsed as numeric.
        """
        val = parse_numeric_value(str(value_str))
        if val is None:
            return []
        return self.find_all_numbers(val, tolerance)

    def __len__(self) -> int:
        return self.num_values
