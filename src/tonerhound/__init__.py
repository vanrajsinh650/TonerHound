"""TonerHound: Document Evidence Grounding & Provenance Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, CoordinateFrame, union_bbox_list
from tonerhound.models.types import (
    DocumentPage,
    DocumentToken,
    ExtractionInput,
    ProvenanceStatus,
    ResolutionResult,
    VisualLine,
)
from tonerhound.resolution.resolver import EvidenceResolver

__all__ = [
    "BBox",
    "CoordinateFrame",
    "DocumentIndex",
    "DocumentPage",
    "DocumentToken",
    "EvidenceResolver",
    "ExtractionInput",
    "ProvenanceStatus",
    "ResolutionResult",
    "VisualLine",
    "resolve",
    "union_bbox_list",
]


def resolve(
    document: str | Path | bytes | BinaryIO | DocumentIndex,
    extraction: ExtractionInput | dict[str, Any] | list[ExtractionInput | dict[str, Any]],
) -> ResolutionResult | list[ResolutionResult]:
    """Resolve physical evidence and bounding boxes for extracted document fields.

    Parameters:
        document: PDF path, bytes, file stream, or pre-built DocumentIndex.
        extraction: ExtractionInput object, dictionary, or list thereof.

    Returns:
        ResolutionResult or list[ResolutionResult] with status, page, bbox, and confidence.
    """
    if isinstance(document, DocumentIndex):
        index = document
    else:
        index = DocumentIndex.from_pdf(document)

    resolver = EvidenceResolver(index)

    # Handle batch extraction
    if isinstance(extraction, list):
        results: list[ResolutionResult] = []
        for item in extraction:
            inp = _coerce_extraction_input(item)
            results.append(resolver.resolve(inp))
        return results

    # Handle single extraction
    inp = _coerce_extraction_input(extraction)
    return resolver.resolve(inp)


def _coerce_extraction_input(data: ExtractionInput | dict[str, Any]) -> ExtractionInput:
    if isinstance(data, ExtractionInput):
        return data
    if isinstance(data, dict):
        return ExtractionInput(
            field=str(data.get("field", "unnamed_field")),
            value=data.get("value"),
            evidence_text=data.get("evidence_text"),
            field_context=data.get("field_context"),
            page_hint=data.get("page_hint"),
        )
    raise TypeError(f"Expected ExtractionInput or dict, got {type(data)}")
