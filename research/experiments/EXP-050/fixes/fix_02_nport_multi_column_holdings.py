"""EXP-050 Fix 02: SEC Form N-PORT Multi-Column Portfolio Localizer.

Target Family: SEC Form N-PORT Portfolio Investments.
Status: REJECTED (0 rescues, below 100 threshold).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
from tonerhound.document.index import DocumentIndex

def is_applicable(document_id: str) -> bool:
    return "nport" in document_id.lower()

def apply_fix(document_id: str, pdf_path: Path, failing_fields: list[dict[str, Any]], doc_index: DocumentIndex | None = None) -> dict[str, dict[str, Any]]:
    # Scoped strictly to N-PORT tabular holdings
    if not is_applicable(document_id):
        return {}
    return {}
