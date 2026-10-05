"""EXP-040 Phase C & E.2: Normalization Variants Generator.

Targets NORMALIZATION_MISMATCH (largest subclass: SUB_PERCENT_DECIMAL, 37,850 fields,
plus SUB_CURRENCY_PREFIX, SUB_TRAILING_PUNCT, SUB_DASH_AS_ZERO, and parenthetical negatives).
Generates alternative text variants for matching target values on document pages.
"""

from __future__ import annotations

import re
from typing import Any


def generate_normalization_variants(value: Any, field_path: str = "") -> list[str]:
    """Generate all plausible surface text representations of a target field value."""
    if value is None:
        return []

    val_str = str(value).strip()
    if not val_str:
        return []

    variants: set[str] = {val_str}
    fp_lower = field_path.lower()

    # 1. Percent vs Decimal (SUB_PERCENT_DECIMAL)
    if val_str.endswith("%"):
        try:
            num = float(re.sub(r"[^\d.]", "", val_str))
            dec_val = num / 100.0
            variants.add(f"{dec_val}")
            variants.add(f"{dec_val:.2f}")
            variants.add(f"{dec_val:.4f}")
            variants.add(f"{num:g}")
        except ValueError:
            pass
    else:
        # Check if float between 0 and 1 or field indicates percent/rate
        try:
            val_float = float(re.sub(r"[^\d.-]", "", val_str))
            if 0.0 < abs(val_float) <= 1.0 or any(k in fp_lower for k in ("_percent", "_pct", "rate", "interest", "ratio")):
                pct_val = val_float * 100.0
                variants.add(f"{pct_val:g}%")
                variants.add(f"{pct_val:.1f}%")
                variants.add(f"{pct_val:.2f}%")
                variants.add(f"{int(round(pct_val))}%")
                variants.add(f"{pct_val:g}")
        except ValueError:
            pass

    # 2. Currency Codes & Symbols (SUB_CURRENCY_PREFIX)
    # Strip currency codes: USD, EUR, GBP, JPY, CAD, AUD
    value_no_code = re.sub(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", "", val_str, flags=re.I).strip()
    if value_no_code and value_no_code != val_str:
        variants.add(value_no_code)

    # Strip symbols: $, €, £, ¥, ₹
    value_no_sym = re.sub(r"[\$€£¥₹]", "", value_no_code or val_str).strip()
    if value_no_sym and value_no_sym != val_str:
        variants.add(value_no_sym)

    # Commas stripped
    val_no_commas = value_no_sym.replace(",", "")
    if val_no_commas and val_no_commas != value_no_sym:
        variants.add(val_no_commas)

    # If numeric, add comma-separated format
    try:
        fval = float(val_no_commas)
        if abs(fval) >= 1000.0:
            if fval.is_integer():
                variants.add(f"{int(fval):,}")
            else:
                variants.add(f"{fval:,.2f}")
                variants.add(f"{fval:,.1f}")
    except ValueError:
        pass

    # 3. Parenthetical Accounting Negatives (e.g. -1234.56 <-> (1,234.56))
    if val_str.startswith("-"):
        pos_str = val_str[1:].strip()
        variants.add(f"({pos_str})")
        # With commas
        try:
            fpos = float(pos_str.replace(",", ""))
            variants.add(f"({fpos:,.2f})")
            variants.add(f"({fpos:,.0f})")
            variants.add(f"({fpos:g})")
            variants.add(f"$({fpos:,.2f})")
            variants.add(f"(${fpos:,.2f})")
        except ValueError:
            pass
    elif val_str.startswith("(") and val_str.endswith(")"):
        inner = val_str[1:-1].strip()
        variants.add(f"-{inner}")
        variants.add(inner)
        variants.add(inner.replace(",", ""))

    # 4. Trailing Punctuation (SUB_TRAILING_PUNCT)
    stripped_punct = val_str.rstrip(".,;:")
    if stripped_punct != val_str:
        variants.add(stripped_punct)
        variants.add(stripped_punct.replace(",", ""))

    # 5. Dash as Zero (SUB_DASH_AS_ZERO)
    if val_str in ("0", "0.0", "0.00", "0.000", "0.0000"):
        variants.update(["-", "—", "–", "$-", "($-)", "$0", "$0.00", "nil", "none"])

    # 6. Tilde / Approximation
    if val_str.startswith(("~", "≈", "≃")):
        no_tilde = re.sub(r"^[~≈≃]\s*", "", val_str).strip()
        variants.add(no_tilde)
        variants.add(no_tilde.replace(",", ""))

    return [v for v in variants if v]
