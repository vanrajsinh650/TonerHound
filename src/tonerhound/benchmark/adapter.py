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
        leaves = _flatten_leaves_with_context(extracted_data)
        citations: list[dict[str, Any]] = []

        # Detect any systematic offset between logical source_page and physical PDF pages
        doc_offset = self._detect_page_offset(leaves)

        # Track resolved page hints per record path
        record_page_hints: dict[str, int] = {}

        # Pass 1: Resolve high-entropy string anchors to establish record page hints
        for path, value, page_hint, context, parent_record_path in leaves:
            if value is None:
                continue
            if page_hint is not None:
                calibrated_page = max(1, min(len(self.index.pages), page_hint + doc_offset))
                record_page_hints[parent_record_path] = calibrated_page
                continue

            # If it's a distinctive string (> 5 chars, not a pure number/date), try resolving page
            val_str = str(value).strip()
            if len(val_str) > 5 and not val_str.replace(".", "").replace(",", "").isdigit():
                inp = ExtractionInput(
                    field=path,
                    value=value,
                    field_context=context,
                )
                res = self.resolver.resolve(inp)
                if res.is_grounded and res.page is not None and res.confidence >= 0.8:
                    record_page_hints[parent_record_path] = res.page

        # Pass 2: Full resolution with record page hints and enriched sibling context
        for path, value, page_hint, context, parent_record_path in leaves:
            if value is None:
                continue

            # Inherit page hint from record if not directly present
            effective_page_hint = (
                max(1, min(len(self.index.pages), page_hint + doc_offset))
                if page_hint is not None
                else record_page_hints.get(parent_record_path)
            )

            inp = ExtractionInput(
                field=path,
                value=value,
                field_context=context,
                page_hint=effective_page_hint,
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

    def _detect_page_offset(
        self,
        leaves: list[tuple[str, Any, int | None, str | None, str]],
    ) -> int:
        """Calibrate offset between logical source_page and physical PDF pages."""
        votes = {0: 0, 1: 0, -1: 0}
        total_pages = len(self.index.pages)
        checked = 0

        for _path, value, page_hint, _context, _record_path in leaves:
            if page_hint is None:
                continue
            val_str = str(value).strip()
            if len(val_str) >= 4 and not val_str.replace(".", "").replace(",", "").isdigit():
                checked += 1
                for offset in (0, 1, -1):
                    target_p = page_hint + offset
                    if 1 <= target_p <= total_pages and self.index.search_exact(val_str, page=target_p):
                        votes[offset] += 1
                if checked >= 20:
                    break

        # If zero candidates match on nominal page, but offset +1 or -1 has strong consensus
        if votes[0] == 0 and votes[1] >= 2 and votes[1] > votes[-1]:
            return 1
        if votes[0] == 0 and votes[-1] >= 2 and votes[-1] > votes[1]:
            return -1
        return 0


def _flatten_leaves_with_context(
    data: Any,
    prefix: str = "",
    parent_record_path: str = "",
) -> list[tuple[str, Any, int | None, str | None, str]]:
    """Recursively flatten data into (path, value, page_hint, context, parent_record_path)."""
    items: list[tuple[str, Any, int | None, str | None, str]] = []

    if isinstance(data, Mapping):
        # Check for explicit page hint inside the mapping
        page_hint = None
        for pkey in ("source_page", "page", "page_number", "page_no", "page_num"):
            if pkey in data and isinstance(data[pkey], int):
                page_hint = data[pkey]
                break

        # Collect salient sibling strings for context
        salient_siblings = []
        for k, v in data.items():
            if k in ("source_page", "page", "page_number"):
                continue
            if isinstance(v, str) and 2 <= len(v) <= 40:
                salient_siblings.append(v)

        sibling_context = " ".join(salient_siblings[:3]) if salient_siblings else None

        for k, v in data.items():
            child_path = f"{prefix}.{k}" if prefix else str(k)
            # Derive field name context
            field_name = k.replace("_", " ")
            context = f"{sibling_context} {field_name}".strip() if sibling_context else field_name
            record_path = prefix or "root"

            if isinstance(v, (Mapping, Sequence)) and not isinstance(v, (str, bytes, bytearray)):
                items.extend(_flatten_leaves_with_context(v, child_path, record_path))
            else:
                items.append((child_path, v, page_hint, context, record_path))

    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        for idx, item in enumerate(data):
            child_path = f"{prefix}[{idx}]"
            record_path = child_path
            if isinstance(item, (Mapping, Sequence)) and not isinstance(item, (str, bytes, bytearray)):
                items.extend(_flatten_leaves_with_context(item, child_path, record_path))
            else:
                items.append((child_path, item, None, None, prefix))
    else:
        if prefix:
            items.append((prefix, data, None, None, parent_record_path))

    return items
