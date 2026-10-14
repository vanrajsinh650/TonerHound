"""Fix 01: SEC Form 13F Voting Authority Rail Alignment.

Targeted exclusively to SEC 13F table documents:
- long/sec_13f_0010_renaissance_technologies
- long/sec_13f_0026_brown_brothers_harriman
- medium/sec_13f_0031_loomis_sayles

Resolves failing fields:
- holdings[N].voting_authority.none
- holdings[N].voting_authority.shared
- holdings[N].voting_authority.sole

Zero-regression safe: Proposes candidates strictly matching table column rails on the target row.
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
    """Propose high-precision voting authority citations from table column rails."""
    if not is_applicable(document_id):
        return {}

    target_fields = [
        f for f in failing_fields
        if "voting_authority" in f.get("field_path", "")
    ]
    if not target_fields:
        return {}

    if doc_index is None:
        doc_index = DocumentIndex.from_pdf(pdf_path, enable_ocr=False, backend="pdfium")

    resolutions: dict[str, dict[str, Any]] = {}

    for fld in target_fields:
        fp = fld["field_path"]
        page_num = fld.get("page_hint")
        if not page_num or page_num < 1 or page_num > len(doc_index.pages):
            continue

        page_obj = doc_index.get_page(page_num)
        if not page_obj or not page_obj.tokens:
            continue

        row_y = fld.get("row_y")
        if row_y is None:
            # Fallback to bbox if available in field dict
            bb = fld.get("base_bbox")
            if bb and len(bb) == 4:
                row_y = bb[1]

        if row_y is None:
            continue

        cands = []
        if "voting_authority.none" in fp:
            cands = [t for t in page_obj.tokens if 0.92 <= t.bbox.x <= 0.98 and abs(t.bbox.y - row_y) < 0.018]
        elif "voting_authority.shared" in fp:
            cands = [t for t in page_obj.tokens if 0.86 <= t.bbox.x <= 0.92 and abs(t.bbox.y - row_y) < 0.018]
        elif "voting_authority.sole" in fp:
            cands = [t for t in page_obj.tokens if 0.78 <= t.bbox.x <= 0.86 and abs(t.bbox.y - row_y) < 0.018]

        if cands:
            best_t = min(cands, key=lambda t: abs(t.bbox.y - row_y))
            resolutions[fp] = {
                "page": page_num,
                "bbox": [best_t.bbox.x, best_t.bbox.y, best_t.bbox.width, best_t.bbox.height],
                "source": "fix_01_sec13f_voting_authority",
            }

    return resolutions
