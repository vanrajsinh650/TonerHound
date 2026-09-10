"""ExtractBench Benchmark Adapter for TonerHound.

Converts arbitrary document extraction payloads into the official ExtractBench
citation and output representations, grounded with TonerHound's evidence resolver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tonerhound.document.index import DocumentIndex
from tonerhound.models.types import ExtractionInput
from tonerhound.resolution.resolver import EvidenceResolver


class ExtractBenchAdapter:
    """Adapter to resolve ExtractBench extraction payloads to official FieldCitations."""

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self.resolver = EvidenceResolver(index)

    def ground_extracted_data(
        self,
        extracted_data: dict[str, Any] | list[dict[str, Any]],
        example_id: str = "example_0",
        pipeline_name: str = "tonerhound",
    ) -> dict[str, Any]:
        """Ground extracted data fields and return official ExtractBench payload dictionary."""
        leaves = list(_flatten_leaves(extracted_data))
        citations: list[dict[str, Any]] = []

        for path, value in leaves:
            if value is None:
                continue

            # Context derived from path leaf name
            field_name = path.rsplit(".", 1)[-1].split("[", 1)[0]
            context = field_name.replace("_", " ")

            inp = ExtractionInput(
                field=path,
                value=value,
                field_context=context,
            )
            res = self.resolver.resolve(inp)

            if res.is_grounded and res.page is not None and res.bbox is not None:
                citation = {
                    "field_path": path,
                    "page": res.page,
                    "bbox": res.bbox.to_coco(),
                    "reference_text": res.matched_text,
                    "confidence": res.confidence,
                    "source": "tonerhound",
                }
                citations.append(citation)

        return {
            "task_type": "extract",
            "example_id": example_id,
            "pipeline_name": pipeline_name,
            "extracted_data": extracted_data,
            "field_citations": citations,
        }


def _flatten_leaves(
    data: Any, prefix: str = ""
) -> list[tuple[str, Any]]:
    """Recursively flatten dictionary / array data into (path, value) pairs."""
    items: list[tuple[str, Any]] = []

    if isinstance(data, Mapping):
        for k, v in data.items():
            child_path = f"{prefix}.{k}" if prefix else str(k)
            items.extend(_flatten_leaves(v, child_path))
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        for idx, item in enumerate(data):
            child_path = f"{prefix}[{idx}]"
            items.extend(_flatten_leaves(item, child_path))
    else:
        if prefix:
            items.append((prefix, data))

    return items
