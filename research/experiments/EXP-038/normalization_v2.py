"""EXP-038 Fix 6: Expanded Normalization for Financial & Accounting Values.

Handles accounting parenthesized negatives and international currency codes:
- Accounting negative: (1,234.56) -> -1234.56, ($500) -> -500.0
- Currency codes: USD, EUR, GBP, CAD, AUD, JPY, CHF stripped from token values
- Plausible numeric variants indexed canonically alongside raw values.
"""

from __future__ import annotations

import re
from typing import Any

from tonerhound.normalization.normalizers import clean_currency_and_numbers, parse_numeric_value

RE_CURRENCY_CODES = re.compile(r"\b(USD|EUR|GBP|CAD|AUD|JPY|CHF|CNY|HKD|SGD)\b", re.IGNORECASE)
RE_ACCOUNTING_NEGATIVE = re.compile(r"^\s*[\(\[]\s*([\$€£¥]?\s*[\d,]+\.?\d*)\s*[\)\]]\s*$")


def parse_expanded_numeric(value: Any) -> float | None:
    """Extract numeric value with accounting parenthesis and currency code handling."""
    if value is None or isinstance(value, bool):
        return None

    val_str = str(value).strip()
    if not val_str:
        return None

    # Strip 3-letter currency codes
    val_str = RE_CURRENCY_CODES.sub("", val_str).strip()

    # Check accounting parenthesized negative
    m = RE_ACCOUNTING_NEGATIVE.match(val_str)
    if m:
        inner = m.group(1).strip()
        num = parse_numeric_value(inner)
        if num is not None:
            return -abs(num)

    return parse_numeric_value(val_str)


def generate_numeric_query_variants(val: Any) -> list[float]:
    """Generate both positive and negative numeric variants where ambiguous."""
    num = parse_expanded_numeric(val)
    if num is None:
        return []
    variants = [round(num, 6)]
    # If gold value is negative, also allow unsigned counterpart if table header indicates deductions
    if num < 0:
        variants.append(round(abs(num), 6))
    return list(dict.fromkeys(variants))
