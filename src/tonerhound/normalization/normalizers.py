"""Reversible text, numeric, currency, and date normalization for TonerHound.

Crucial architectural invariant:
Every normalization must preserve an exact character mapping back to the
original source text and atom offsets, so that matching in normalized space
directly recovers the physical bounding boxes of the underlying glyphs.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from dateutil import parser as date_parser

_CHAR_REPLACEMENTS: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201a": ",",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2010": "-",
    "\u2011": "-",
    "\u00ad": "",  # soft hyphen
    "\u00a0": " ",  # non-breaking space
    "\u200b": "",  # zero-width space
    "\ufeff": "",  # BOM
}

_CURRENCY_SYMBOLS = re.compile(r"[$€£¥₹]")
_CURRENCY_CODES = re.compile(r"\b(USD|EUR|GBP|JPY|INR|CAD|AUD|CHF)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """A normalized string accompanied by a mapping to original character offsets."""

    text: str
    # char_map[i] gives the index in the original string that generated text[i]
    char_map: list[int]
    original_text: str

    def span_to_original_range(self, start: int, end: int) -> tuple[int, int]:
        """Convert a [start, end) span in normalized text to [orig_start, orig_end)."""
        if not self.char_map or start >= len(self.char_map):
            return (0, 0)
        orig_start = self.char_map[start]
        # End is exclusive
        last_norm_idx = min(end - 1, len(self.char_map) - 1)
        orig_end = self.char_map[last_norm_idx] + 1
        return (orig_start, max(orig_start, orig_end))


def normalize_unicode_and_case(source: str) -> NormalizedText:
    """Normalize Unicode (NFKC + smart quotes/ligatures) and casefold, preserving char_map."""
    norm_chars: list[str] = []
    char_map: list[int] = []

    for orig_idx, ch in enumerate(source):
        # Step 1: glyph mapping
        replaced = _CHAR_REPLACEMENTS.get(ch, ch)
        # Step 2: NFKC decomposition
        decomposed = unicodedata.normalize("NFKC", replaced)
        # Step 3: casefold
        folded = decomposed.casefold()

        for folded_ch in folded:
            norm_chars.append(folded_ch)
            char_map.append(orig_idx)

    return NormalizedText(text="".join(norm_chars), char_map=char_map, original_text=source)


def normalize_whitespace(norm_input: NormalizedText) -> NormalizedText:
    """Collapse consecutive whitespace and trim, preserving char_map."""
    src_text = norm_input.text
    src_map = norm_input.char_map

    new_chars: list[str] = []
    new_map: list[int] = []
    in_whitespace = False

    # Skip leading whitespace
    start_idx = 0
    while start_idx < len(src_text) and src_text[start_idx].isspace():
        start_idx += 1

    end_idx = len(src_text)
    while end_idx > start_idx and src_text[end_idx - 1].isspace():
        end_idx -= 1

    for idx in range(start_idx, end_idx):
        ch = src_text[idx]
        if ch.isspace():
            if not in_whitespace:
                new_chars.append(" ")
                new_map.append(src_map[idx])
                in_whitespace = True
        else:
            new_chars.append(ch)
            new_map.append(src_map[idx])
            in_whitespace = False

    return NormalizedText(
        text="".join(new_chars),
        char_map=new_map,
        original_text=norm_input.original_text,
    )


def clean_currency_and_numbers(text: str) -> str:
    """Extract clean numeric representation from formatted currency or numbers."""
    cleaned = text.strip()
    # Check parenthesized negative: (100.00) -> -100.00
    negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1].strip()
        negative = True

    # Strip currency symbols and codes
    cleaned = _CURRENCY_SYMBOLS.sub("", cleaned)
    cleaned = _CURRENCY_CODES.sub("", cleaned).strip()

    # Strip commas in numbers: 1,450.00 -> 1450.00
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace(" ", "")

    if negative and not cleaned.startswith("-"):
        cleaned = f"-{cleaned}"

    return cleaned


def parse_numeric_value(value: Any) -> float | None:
    """Parse numeric value into float, ignoring formatting, commas, currency symbols."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    cleaned = clean_currency_and_numbers(str(value))
    cleaned = re.sub(r"^[~≈]", "", cleaned).strip()
    cleaned = cleaned.rstrip("%")

    try:
        return float(cleaned)
    except ValueError:
        return None


def is_number_equal(v1: Any, v2: Any, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    """Check numeric equivalence with tolerance."""
    n1 = parse_numeric_value(v1)
    n2 = parse_numeric_value(v2)
    if n1 is None or n2 is None:
        return False
    return math.isclose(n1, n2, rel_tol=rel_tol, abs_tol=abs_tol)


_DATE_PATTERNS = (
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),
    re.compile(r"\b(?:[A-Za-z]{3,9}\.?\s+)?\d{1,2},?\s+[A-Za-z]{3,9}\.?\s+\d{4}\b"),
    re.compile(r"\b(?:[A-Za-z]{3,9}\.?\s+)?[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}\b"),
    re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}\b"),
)


def parse_date_value(value: Any) -> date | None:
    """Parse date into standard datetime.date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = date_parser.parse(text, fuzzy=True)
        return parsed.date()
    except (ValueError, OverflowError, TypeError):
        return None


def is_date_equal(d1: Any, d2: Any) -> bool:
    """Check if two date representations resolve to the same calendar date."""
    parsed1 = parse_date_value(d1)
    parsed2 = parse_date_value(d2)
    if parsed1 is None or parsed2 is None:
        return False
    return parsed1 == parsed2
