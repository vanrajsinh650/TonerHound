"""Fix 05: Bankruptcy Creditor Matrix Address Grid Alignment.

Targeted to bankruptcy creditor schedule documents:
- long/real_ftx_full
- long/real_imedia_full

Resolves failing fields:
- creditors[N].address_1
- creditors[N].city
- creditors[N].state
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tonerhound.document.index import DocumentIndex


def is_applicable(document_id: str) -> bool:
    """Targeted only to creditor matrix documents."""
    tid = document_id.lower()
    return "ftx" in tid or "imedia" in tid


def apply_fix(
    document_id: str,
    pdf_path: Path,
    failing_fields: list[dict[str, Any]],
    doc_index: DocumentIndex | None = None,
) -> dict[str, dict[str, Any]]:
    """Propose address token alignments in creditor tables."""
    if not is_applicable(document_id):
        return {}

    target_fields = [
        f for f in failing_fields
        if any(k in f.get("field_path", "") for k in ("address_1", "city", "state"))
    ]
    if not target_fields:
        return {}

    if doc_index is None:
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=False, backend="pdfium")

    resolutions: dict[str, dict[str, Any]] = {}

    for page_num in range(1, len(doc_index.pages) + 1):
        page_obj = doc_index.get_page(page_num)
        if not page_obj or not page_obj.tokens:
            continue

        for fld in target_fields:
            fp = fld["field_path"]
            page_hint = fld.get("page_hint")
            if page_hint != page_num:
                continue

            expected_val = str(fld.get("expected_value", "")).strip()
            if not expected_val or len(expected_val) < 2:
                continue

            for t in page_obj.tokens:
                if t.text.strip().lower() == expected_val.lower():
                    resolutions[fp] = {
                        "page": page_num,
                        "bbox": [t.bbox.x, t.bbox.y, t.bbox.width, t.bbox.height],
                        "source": "fix_05_creditor_matrix_parser",
                    }
                    break

    return resolutions
