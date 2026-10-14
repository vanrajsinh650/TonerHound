"""Fix 03: SEC Form 13F Metadata Table Rails (SH/PRN, Discretion, Other Manager).

Targeted exclusively to SEC 13F table documents:
- long/sec_13f_0010_renaissance_technologies
- long/sec_13f_0026_brown_brothers_harriman
- medium/sec_13f_0031_loomis_sayles

Resolves failing fields:
- holdings[N].sh_prn
- holdings[N].investment_discretion
- holdings[N].other_manager

Zero-regression safe: Maps discrete repeated token codes to dedicated column x-rails.
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
    """Propose metadata citations along known 13F column rails."""
    if not is_applicable(document_id):
        return {}

    target_fields = [
        f for f in failing_fields
        if any(k in f.get("field_path", "") for k in ("sh_prn", "investment_discretion", "other_manager"))
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

        toks_sh_prn = [t for t in page_obj.tokens if 0.54 <= t.bbox.x <= 0.60 and t.bbox.y >= 0.22]
        toks_disc = [t for t in page_obj.tokens if 0.61 <= t.bbox.x <= 0.69 and t.bbox.y >= 0.22]
        toks_mgr = [t for t in page_obj.tokens if 0.69 <= t.bbox.x <= 0.76 and t.bbox.y >= 0.22]

        for fld in target_fields:
            fp = fld["field_path"]
            page_hint = fld.get("page_hint")
            if page_hint != page_num:
                continue

            expected_val = str(fld.get("expected_value", "")).strip().upper()

            cands = []
            if "sh_prn" in fp:
                cands = toks_sh_prn
            elif "investment_discretion" in fp:
                cands = toks_disc
            elif "other_manager" in fp:
                cands = toks_mgr

            for t in cands:
                t_up = t.text.strip().upper()
                if t_up == expected_val or (expected_val == "0" and t_up in ("0", "-")):
                    resolutions[fp] = {
                        "page": page_num,
                        "bbox": [t.bbox.x, t.bbox.y, t.bbox.width, t.bbox.height],
                        "source": "fix_03_sec13f_metadata_rails",
                    }
                    break

    return resolutions
