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

    def __init__(
        self,
        index: DocumentIndex,
        enable_structural_disambiguation: bool = True,
        enable_verification: bool = True,
        score_margin_threshold: float = 0.05,
        enable_bbox_precision: bool = True,
    ) -> None:
        self.index = index
        self.enable_structural_disambiguation = enable_structural_disambiguation
        self.enable_verification = enable_verification
        self.score_margin_threshold = score_margin_threshold
        self.enable_bbox_precision = enable_bbox_precision
        self.resolver = EvidenceResolver(
            index,
            enable_verification=enable_verification,
            score_margin_threshold=score_margin_threshold,
        )

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
        # record_anchors: record_path -> (page, y_center, line_height, bbox)
        record_anchors: dict[str, tuple[int, float, float, Any]] = {}

        if self.enable_structural_disambiguation:
            # Group leaves by parent_record_path
            records: dict[str, list[tuple[str, Any, int | None, str | None, str]]] = {}
            for leaf in leaves:
                parent_rec = leaf[4]
                records.setdefault(parent_rec, []).append(leaf)

            # Pass 1: For each record, find high-confidence anchor fields
            for parent_rec, fields in records.items():
                # Exclude root/empty parent record paths from row anchoring!
                if parent_rec in ("", "root"):
                    continue

                for path, value, page_hint, context, _rec in fields:
                    if value is None or isinstance(value, bool):
                        continue
                    val_str = str(value).strip()
                    if val_str.lower() in ("true", "false", "yes", "no"):
                        continue
                    calibrated_p = (
                        max(1, min(len(self.index.pages), page_hint + doc_offset))
                        if page_hint is not None
                        else None
                    )
                    # Salient anchor candidate
                    is_anchor_cand = (
                        any(kw in path.lower() for kw in ("issuer", "owner", "entity", "company", "name", "number", "subject", "id", "code", "desc", "title", "item"))
                        or (len(val_str) >= 4 and not val_str.replace(".", "").replace(",", "").isdigit())
                    )
                    if is_anchor_cand:
                        inp = ExtractionInput(
                            field=path,
                            value=value,
                            field_context=context,
                            page_hint=calibrated_p,
                        )
                        res = self.resolver.resolve(inp)
                        if res.is_grounded and res.page is not None and res.bbox is not None and res.confidence >= 0.75:
                            yc = res.bbox.y + res.bbox.height / 2.0
                            record_anchors[parent_rec] = (res.page, yc, res.bbox.height, res.bbox)
                            record_page_hints[parent_rec] = res.page
                            break

                # If still not anchored, try any unique field that resolves with single match
                if parent_rec not in record_anchors:
                    for path, value, page_hint, context, _rec in fields:
                        if value is None or isinstance(value, bool):
                            continue
                        val_str = str(value).strip()
                        if val_str.lower() in ("true", "false", "yes", "no"):
                            continue
                        calibrated_p = (
                            max(1, min(len(self.index.pages), page_hint + doc_offset))
                            if page_hint is not None
                            else None
                        )
                        inp = ExtractionInput(
                            field=path,
                            value=value,
                            field_context=context,
                            page_hint=calibrated_p,
                        )
                        res = self.resolver.resolve(inp)
                        if res.is_grounded and res.page is not None and res.bbox is not None and res.confidence >= 0.85:
                            yc = res.bbox.y + res.bbox.height / 2.0
                            record_anchors[parent_rec] = (res.page, yc, res.bbox.height, res.bbox)
                            record_page_hints[parent_rec] = res.page
                            break
        else:
            # Baseline Pass 1
            for path, value, page_hint, context, parent_record_path in leaves:
                if value is None:
                    continue
                if page_hint is not None:
                    calibrated_page = max(1, min(len(self.index.pages), page_hint + doc_offset))
                    record_page_hints[parent_record_path] = calibrated_page
                    continue
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

        # Pass 2: Full resolution with record page hints and row constraints
        for path, value, page_hint, context, parent_record_path in leaves:
            if value is None:
                continue

            calibrated_p = (
                max(1, min(len(self.index.pages), page_hint + doc_offset))
                if page_hint is not None
                else None
            )
            anchor = record_anchors.get(parent_record_path)
            effective_page_hint = calibrated_p or (anchor[0] if anchor else record_page_hints.get(parent_record_path))

            # Row-anchored resolution when structural disambiguation is active
            if self.enable_structural_disambiguation and anchor is not None:
                anc_page, anc_cy, anc_h, _anc_box = anchor
                is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
                cands = []
                if is_num:
                    cands.extend(self.resolver.matcher.find_normalized_numeric_candidates(value, page_hint=anc_page))
                if isinstance(value, str):
                    cands.extend(self.resolver.matcher.find_exact_candidates(value, page_hint=anc_page))
                    if not cands:
                        cands.extend(self.resolver.matcher.find_normalized_date_candidates(value, page_hint=anc_page))
                        cands.extend(self.resolver.matcher.find_normalized_numeric_candidates(value, page_hint=anc_page))
                elif not cands and isinstance(value, (int, float)):
                    cands.extend(self.resolver.matcher.find_exact_candidates(str(value), page_hint=anc_page))

                row_cands = []
                row_tol = max(0.018, anc_h * 1.6)
                for c in cands:
                    if c.page == anc_page:
                        c_cy = c.bbox.y + c.bbox.height / 2.0
                        dist_y = abs(c_cy - anc_cy)
                        if dist_y <= row_tol:
                            row_cands.append((c, dist_y))

                if row_cands:
                    row_cands.sort(key=lambda item: item[1])
                    best_cand = row_cands[0][0]

                    is_valid = True
                    if self.resolver.verifier is not None:
                        scored_row = [(c[0], 5.0 - c[1]) for c in row_cands]
                        decision = self.resolver.verifier.verify(
                            field=path,
                            value=value,
                            top_candidate=best_cand,
                            scored_candidates=scored_row,
                            field_context=context,
                            page_hint=anc_page,
                            row_anchor=anchor,
                        )
                        is_valid = decision.is_accepted

                    if is_valid:
                        box = best_cand.bbox
                        if self.enable_bbox_precision:
                            box = box.align_to_line_height(0.018)
                        citation = {
                            "field_path": path,
                            "page": best_cand.page,
                            "bbox": box.to_coco(),
                            "reference_text": best_cand.matched_text,
                            "confidence": 0.95,
                            "source": "tonerhound",
                        }
                        citations.append(citation)
                        continue

            inp = ExtractionInput(
                field=path,
                value=value,
                field_context=context,
                page_hint=effective_page_hint,
            )
            res = self.resolver.resolve(inp)

            if res.is_grounded and res.page is not None and res.bbox is not None:
                box = res.bbox
                if self.enable_bbox_precision:
                    box = box.align_to_line_height(0.018)
                citation = {
                    "field_path": path,
                    "page": res.page,
                    "bbox": box.to_coco(),
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
        offsets_to_test = range(-2, 5)
        votes = {o: 0 for o in offsets_to_test}
        total_pages = len(self.index.pages)
        checked = 0

        for _path, value, page_hint, _context, _record_path in leaves:
            if page_hint is None:
                continue
            val_str = str(value).strip()
            if len(val_str) >= 5 and not val_str.replace(".", "").replace(",", "").isdigit():
                checked += 1
                for offset in offsets_to_test:
                    target_p = page_hint + offset
                    if 1 <= target_p <= total_pages and self.index.search_exact(val_str, page=target_p):
                        votes[offset] += 1
                if checked >= 20:
                    break

        best_offset, best_votes = max(votes.items(), key=lambda x: x[1])
        if best_votes >= 2 and best_votes > votes.get(0, 0):
            return best_offset
        return 0


def _flatten_leaves_with_context(
    data: Any,
    prefix: str = "",
    parent_record_path: str = "",
    parent_page_hint: int | None = None,
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
        if page_hint is None:
            page_hint = parent_page_hint

        # Collect salient sibling strings for context
        salient_siblings = []
        for k, v in data.items():
            if k in ("source_page", "page", "page_number", "page_no", "page_num"):
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
                items.extend(_flatten_leaves_with_context(v, child_path, record_path, page_hint))
            else:
                items.append((child_path, v, page_hint, context, record_path))

    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        for idx, item in enumerate(data):
            child_path = f"{prefix}[{idx}]"
            record_path = parent_record_path if parent_record_path not in ("", "root") else child_path
            if isinstance(item, (Mapping, Sequence)) and not isinstance(item, (str, bytes, bytearray)):
                items.extend(_flatten_leaves_with_context(item, child_path, record_path, parent_page_hint))
            else:
                items.append((child_path, item, parent_page_hint, None, record_path))
    else:
        if prefix:
            items.append((prefix, data, parent_page_hint, None, parent_record_path))

    return items
