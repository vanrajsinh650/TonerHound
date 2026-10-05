"""EXP-041 Phase E: Low-Hanging Fruit - Geometric Punctuation & Dash-as-Zero Variants.

Applies exact-string literal variants (NO semantic normalization):
- E.1: Trailing punctuation variants: 1,234. -> ["1,234.", "1,234", "1234."]
- E.2: Dash-as-zero variants: - -> ['0', '0.00', '0.0', '$-', '($-)']
"""

from __future__ import annotations


def strip_trailing_punct_variants(value: str) -> list[str]:
    """Generate exact-string variants without semantic distortion.

    Examples:
        "1,234." -> ["1,234.", "1,234", "1234"]
        "64,216," -> ["64,216,", "64,216", "64216"]
    """
    if not value:
        return []

    val_str = str(value).strip()
    variants = [val_str]

    stripped = val_str.rstrip(".,;:")
    if stripped != val_str:
        variants.append(stripped)

    no_commas = stripped.replace(",", "")
    if no_commas != stripped:
        variants.append(no_commas)

    # Return deduplicated while preserving order
    seen = set()
    result = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result


def dash_to_zero_variants(value: str) -> list[str]:
    """For financial documents where a dash (- or —) represents zero or nil."""
    if not value:
        return []

    val_str = str(value).strip()
    if val_str in ("-", "—", "–", "―"):
        return ["0", "0.00", "0.0", "$-", "($-)", "-", "—"]

    return [val_str]
