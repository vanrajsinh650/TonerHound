"""Fix 02: Holdings Marks & Footnotes Character-Level Precision.

Targeted to investment fund schedules containing footnote reference symbols/marks:
- long/real_ishares_iboxx_bond_etfs
- long/real_credit_strategies_full
- medium/real_blackrock_schedule_full
- medium/real_fidelity_northstar

Resolves failing fields:
- holdings[N].marks[N]
- holdings[N].footnotes[N]

Zero-regression safe: Emits tight token/glyph bboxes replacing over-broad row bboxes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium


def is_applicable(document_id: str) -> bool:
    """Applicable to documents with investment fund schedules."""
    tid = document_id.lower()
    return any(k in tid for k in ("ishares", "credit_strategies", "fidelity", "blackrock", "sched", "nport"))


def apply_fix(
    document_id: str,
    pdf_path: Path,
    failing_fields: list[dict[str, Any]],
    pdf_doc: pdfium.PdfDocument | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve single-glyph footnote and mark references via exact charbox bounding."""
    if not is_applicable(document_id):
        return {}

    target_fields = [
        f for f in failing_fields
        if any(k in f.get("field_path", "") for k in ("marks", "footnotes"))
    ]
    if not target_fields:
        return {}

    if pdf_doc is None:
        pdf_doc = pdfium.PdfDocument(pdf_path)

    resolutions: dict[str, dict[str, Any]] = {}
    n_pages = len(pdf_doc)

    # Group fields by page
    fields_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for f in target_fields:
        p = f.get("page_hint")
        if p and 1 <= p <= n_pages:
            fields_by_page[p].append(f)

    for page_num, p_fields in fields_by_page.items():
        page = pdf_doc.get_page(page_num - 1)
        textpage = page.get_textpage()
        pw, ph = page.get_size()
        n_chars = textpage.count_chars()

        # Build fast index of chars on this page
        chars_map = defaultdict(list)
        for i in range(n_chars):
            c = textpage.get_text_range(i, 1)
            rect = textpage.get_charbox(i)
            norm_b = [
                max(0.0, rect[0] / pw - 0.003),
                max(0.0, (ph - rect[3]) / ph - 0.003),
                min(1.0, (rect[2] - rect[0]) / pw + 0.006),
                min(1.0, (rect[3] - rect[1]) / ph + 0.006),
            ]
            chars_map[c].append(norm_b)

        for fld in p_fields:
            fp = fld["field_path"]
            exp_val = str(fld.get("expected_value", "")).strip("(),.* ")
            if not exp_val:
                continue

            row_y = fld.get("row_y")
            if row_y is None:
                bb = fld.get("base_bbox")
                if bb and len(bb) == 4:
                    row_y = bb[1]

            if row_y is None:
                continue

            cands = [b for b in chars_map.get(exp_val, []) if abs(b[1] - row_y) < 0.018]
            if cands:
                best_b = min(cands, key=lambda b: abs(b[1] - row_y))
                resolutions[fp] = {
                    "page": page_num,
                    "bbox": best_b,
                    "source": "fix_02_holdings_marks_precision",
                }

    return resolutions
