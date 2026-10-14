"""Fix 04: SEC Form 13F Title of Class Grid Rail.

Targeted to SEC 13F filings:
- long/sec_13f_0010_renaissance_technologies
- long/sec_13f_0026_brown_brothers_harriman
- medium/13f__nalls_sherbakoff_2025q4

Resolves failing fields:
- holdings[N].title_of_class
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tonerhound.document.index import DocumentIndex


def is_applicable(document_id: str) -> bool:
    """Targeted only to SEC 13F regulatory filing documents."""
    return "13f" in document_id.lower()


def apply_fix(
    document_id: str,
    pdf_path: Path,
    failing_fields: list[dict[str, Any]],
    doc_index: DocumentIndex | None = None,
) -> dict[str, dict[str, Any]]:
    """Propose title of class citations within column 2 x-rail (0.20 <= x <= 0.32)."""
    if not is_applicable(document_id):
        return {}

    target_fields = [
        f for f in failing_fields
        if "title_of_class" in f.get("field_path", "")
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

        toks_title = [t for t in page_obj.tokens if 0.20 <= t.bbox.x <= 0.32 and t.bbox.y >= 0.25]

        for fld in target_fields:
            fp = fld["field_path"]
            page_hint = fld.get("page_hint")
            if page_hint != page_num:
                continue

            expected_val = str(fld.get("expected_value", "")).strip().upper()

            for t in toks_title:
                if t.text.strip().upper() == expected_val:
                    resolutions[fp] = {
                        "page": page_num,
                        "bbox": [t.bbox.x, t.bbox.y, t.bbox.width, t.bbox.height],
                        "source": "fix_04_sec13f_title_of_class",
                    }
                    break

    return resolutions
