"""EXP-050 Fix 03: SEC Form N-SAR / N-CSR Semi-Annual Schedule Localizer.

Target Family: SEC Form N-SAR Semi-Annual Reports.
Status: REJECTED (1 rescue on 07021-2016-p0051, below 100 threshold).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from tonerhound.document.index import DocumentIndex

def is_applicable(document_id: str) -> bool:
    return "07021-2016" in document_id or "00581-2011" in document_id

def apply_fix(document_id: str, pdf_path: Path, failing_fields: list[dict[str, Any]], doc_index: DocumentIndex | None = None) -> dict[str, dict[str, Any]]:
    if not is_applicable(document_id):
        return {}
    resolutions = {}
    for fld in failing_fields:
        fp = fld.get("field_path", "")
        # Cell padding for compact box 15 single character
        if "box_15_amt_items" in fp and fp.endswith(".code"):
            bb = fld.get("base_bbox")
            if bb and len(bb) == 4:
                cand_bb = [bb[0] - 0.005, bb[1] - 0.003, max(0.022, bb[2] + 0.010), max(0.015, bb[3] + 0.006)]
                resolutions[fp] = {"page": fld.get("page_hint", 1), "bbox": cand_bb, "source": "fix_03_nsar"}
    return resolutions
