"""EXP-044 Stage 1: Rule-Based Canonicalization.

Targets NORMALIZATION_MISMATCH sub-classes:
- SUB_PERCENT_DECIMAL ("0.05" <-> "5.0%")
- SUB_CURRENCY_CODE ("USD 5000" <-> "$5,000")
- SUB_ACCOUNTING_NEGATIVE ("-1234.56" <-> "(1,234.56)")
- SUB_DATE_FORMAT ("2024-01-15" <-> "Jan 15, 2024")
- SUB_NUMBER_GROUPING ("1234" <-> "1,234")
- SUB_PRECISION_STRIP ("$1250.00" <-> "$1,250")

Gate: Only apply to fields where the extracted value and candidate page text
are textually close but not strictly equal.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def canonicalize_number_string(val_str: str) -> str:
    """Normalize numeric string by stripping currency and grouping commas."""
    # Strip currency codes and symbols
    s = re.sub(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", "", val_str, flags=re.I)
    s = re.sub(r"[\$€£¥₹]", "", s).strip()
    # Strip commas between digits
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    # Strip trailing .00 if integer
    s = re.sub(r"\.00$", "", s)
    return s.strip()


def generate_canonical_variants(value: Any, field_path: str = "") -> list[str]:
    """Generate canonical textual variants for matching."""
    if value is None:
        return []

    val_str = str(value).strip()
    if not val_str:
        return []

    variants: set[str] = {val_str}
    fp_lower = field_path.lower()

    # 1. Currency stripping
    curr_stripped = re.sub(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", "", val_str, flags=re.I)
    curr_stripped = re.sub(r"[\$€£¥₹]", "", curr_stripped).strip()
    if curr_stripped and curr_stripped != val_str:
        variants.add(curr_stripped)

    # 2. Number grouping commas
    no_commas = curr_stripped.replace(",", "")
    if no_commas != curr_stripped:
        variants.add(no_commas)

    try:
        fval = float(no_commas)
        # Integer formatted with commas
        if fval.is_integer():
            variants.add(f"{int(fval):,}")
            variants.add(f"{int(fval)}")
            variants.add(f"{int(fval)}.00")
        else:
            variants.add(f"{fval:,.2f}")
            variants.add(f"{fval:.2f}")
            variants.add(f"{fval:g}")
    except ValueError:
        pass

    # 3. Percent <-> Decimal
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

    # 4. Accounting Negatives
    if val_str.startswith("-"):
        pos = val_str[1:].strip()
        variants.add(f"({pos})")
        try:
            fpos = float(pos.replace(",", ""))
            variants.add(f"({fpos:,.2f})")
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
        try:
            finner = float(inner.replace(",", ""))
            variants.add(f"-{finner:g}")
        except ValueError:
            pass

    # 5. Precision stripping (trailing .00)
    if val_str.endswith(".00"):
        variants.add(val_str[:-3])
    elif "." not in val_str and val_str.isdigit():
        variants.add(f"{val_str}.00")

    # 6. Dates (YYYY-MM-DD <-> MM/DD/YYYY <-> Month DD, YYYY)
    iso_m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", val_str)
    if iso_m:
        y, m, d = iso_m.groups()
        variants.add(f"{int(m)}/{int(d)}/{y}")
        variants.add(f"{m}/{d}/{y}")
        variants.add(f"{int(d)}/{int(m)}/{y}")
        variants.add(f"{d}-{m}-{y}")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        m_idx = int(m) - 1
        if 0 <= m_idx < 12:
            variants.add(f"{month_names[m_idx]} {int(d)}, {y}")
            variants.add(f"{month_names[m_idx]} {d}, {y}")

    return sorted(list(variants), key=lambda x: (len(x), x))


def match_canonical_stage1(
    target_value: Any,
    page_obj: DocumentPage,
    field_path: str = "",
) -> tuple[BBox | None, str | None]:
    """Execute Stage 1 matching across candidate tokens of the page.
    
    Returns (bbox, matched_variant) if successful.
    """
    variants = generate_canonical_variants(target_value, field_path)
    val_str = str(target_value).strip() if target_value is not None else ""

    # Check each variant against visual lines and tokens
    for var in variants:
        if var == val_str:
            continue  # Exact match handled before Stage 1
        var_norm = var.lower().strip()

        # 1. Single token exact match with canonical variant
        for line in page_obj.lines:
            for tok in line.tokens:
                t_clean = tok.text.strip(" ,.;:'\"()[]{}").lower()
                if t_clean == var_norm:
                    return tok.bbox, var

        # 2. Multi-token sequential match within line
        var_words = var_norm.split()
        if len(var_words) > 1:
            for line in page_obj.lines:
                ltoks = line.tokens
                if len(ltoks) < len(var_words):
                    continue
                for start_idx in range(len(ltoks) - len(var_words) + 1):
                    sub_toks = ltoks[start_idx : start_idx + len(var_words)]
                    sub_str = " ".join(t.text.strip(" ,.;:'\"()[]{}").lower() for t in sub_toks)
                    if sub_str == var_norm:
                        # Union bbox
                        min_x = min(t.bbox.x for t in sub_toks)
                        min_y = min(t.bbox.y for t in sub_toks)
                        max_x = max(t.bbox.x + t.bbox.width for t in sub_toks)
                        max_y = max(t.bbox.y + t.bbox.height for t in sub_toks)
                        ubox = BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y, page=page_obj.page_number)
                        return ubox, var

    return None, None
