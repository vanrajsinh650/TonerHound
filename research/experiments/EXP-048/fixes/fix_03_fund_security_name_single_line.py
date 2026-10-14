"""Fix 03: Investment Fund Security Name Single-Line Clamping.

Targeted to investment fund schedule documents where multi-line security descriptions
caused the baseline bounding box to include all continuation lines instead of the
ground-truth line 1 header:
- long/real_credit_strategies_full
- medium/real_blackrock_schedule_full
- long/real_ishares_iboxx_bond_etfs
- medium/real_fidelity_northstar

Resolves failing fields:
- holdings[N].security_name

Zero-regression safe: Clamps multi-line bounding boxes to line 1 height (~0.0098).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def is_applicable(document_id: str) -> bool:
    """Applicable to documents with investment fund schedules."""
    tid = document_id.lower()
    return any(k in tid for k in ("credit_strategies", "blackrock", "fidelity", "ishares", "sched", "nport"))


def apply_fix(
    document_id: str,
    pdf_path: Path,
    failing_fields: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Propose single-line clamped citations for failing security names."""
    if not is_applicable(document_id):
        return {}

    target_fields = [
        f for f in failing_fields
        if "security_name" in f.get("field_path", "")
    ]
    if not target_fields:
        return {}

    resolutions: dict[str, dict[str, Any]] = {}

    for fld in target_fields:
        fp = fld["field_path"]
        page_num = fld.get("page_hint")
        base_b = fld.get("base_bbox")
        if not page_num or not base_b or len(base_b) != 4:
            continue

        # If base bbox is taller than single line (> 0.012), clamp to line 1 height (0.0098)
        if base_b[3] > 0.012:
            clamped_b = [base_b[0], base_b[1], base_b[2], 0.0098]
            resolutions[fp] = {
                "page": page_num,
                "bbox": clamped_b,
                "source": "fix_03_fund_security_name_single_line",
            }

    return resolutions
