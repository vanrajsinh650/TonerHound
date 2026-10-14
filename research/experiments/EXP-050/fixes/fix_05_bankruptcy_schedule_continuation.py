"""EXP-050 Fix 05: Bankruptcy Schedule Continuation Localizer.

Target Family: Bankruptcy Schedules (clean / short non-corrupted documents).
Status: REJECTED (0 rescues, below 100 threshold).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from tonerhound.document.index import DocumentIndex

def is_applicable(document_id: str) -> bool:
    return "dd1155" in document_id.lower() or "sf1449" in document_id.lower()

def apply_fix(document_id: str, pdf_path: Path, failing_fields: list[dict[str, Any]], doc_index: DocumentIndex | None = None) -> dict[str, dict[str, Any]]:
    if not is_applicable(document_id):
        return {}
    return {}
