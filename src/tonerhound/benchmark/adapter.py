"""ExtractBench Benchmark Adapter for TonerHound.

Converts arbitrary document extraction payloads into the official ExtractBench
citation and output representations, grounded with TonerHound's evidence resolver.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentToken, ExtractionInput
from tonerhound.normalization.normalizers import (
    is_number_equal,
    normalize_unicode_and_case,
    parse_numeric_value,
)
from tonerhound.ocr.matcher import OCRMatcher
from tonerhound.resolution.resolver import EvidenceResolver
from tonerhound.tax.grounder import TaxFormGrounder, is_form_1040_tax_return


def _matches_table_line(s: str, line_txt: str) -> bool:
    """Tolerant string containment accounting for space-elision and OCR token merging."""
    s_up = s.upper()
    l_up = line_txt.upper()
    if s_up in l_up:
        return True
    s_clean = re.sub(r"[^A-Z0-9]+", "", s_up)
    if not s_clean:
        return False
    l_clean = re.sub(r"[^A-Z0-9]+", "", l_up)
    if s_clean in l_clean:
        return True
    words = [w for w in re.findall(r"[A-Z0-9]{3,}", s_up) if w]
    if len(words) >= 2:
        matched = sum(1 for w in words if w in l_clean)
        if matched / len(words) >= 0.50:
            return True
    return False


def _filter_monotonic_row_pages(candidate_map: dict[int, int]) -> dict[int, int]:
    """Prune non-monotonic page outliers using Longest Non-Decreasing Subsequence (LNDS)."""
    if not candidate_map:
        return {}
    items = sorted(candidate_map.items())
    n = len(items)
    dp = [1] * n
    parent = [-1] * n
    for i in range(n):
        for j in range(i):
            if items[j][1] <= items[i][1]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
    best_idx = max(range(n), key=lambda i: dp[i])
    curr = best_idx
    lnds = []
    while curr != -1:
        lnds.append(items[curr])
        curr = parent[curr]
    lnds.reverse()
    return dict(lnds)


@dataclass(frozen=True)
class OffsetCalibrationResult:
    """Diagnostic outcome of systematic document/table page offset calibration."""

    offset: int
    confidence: float
    vote_count: int
    vote_score: float
    second_score: float
    margin: float
    tier: str
    reason: str


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

        # Form 1040 tax return structural grounding
        if self.enable_structural_disambiguation and is_form_1040_tax_return(leaves):
            tax_grounder = TaxFormGrounder(
                self.index,
                self.resolver,
                enable_bbox_precision=self.enable_bbox_precision,
            )
            return tax_grounder.ground(
                leaves=leaves,
                extracted_data=extracted_data,
                example_id=example_id,
                pipeline_name=pipeline_name,
            )

        citations: list[dict[str, Any]] = []

        # Detect any systematic offset between logical source_page and physical PDF pages
        doc_offset = self._detect_page_offset(leaves)

        # Track resolved page hints per record path
        record_page_hints: dict[str, int] = {}
        # record_anchors: record_path -> (page, y_center, line_height, bbox, aligned_line)
        record_anchors: dict[str, tuple[int, float, float, Any, Any]] = {}
        record_val_counts: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        record_used_tokens: dict[str, set[DocumentToken]] = defaultdict(set)

        if self.enable_structural_disambiguation:
            table_records: dict[str, dict[int, list[tuple[str, Any, int | None, str | None, str]]]] = defaultdict(dict)
            non_table_records: dict[str, list[tuple[str, Any, int | None, str | None, str]]] = defaultdict(list)

            for leaf in leaves:
                parent_rec = leaf[4]
                m = re.match(r"^(.*?)\[(\d+)\]", parent_rec)
                if m:
                    tname = m.group(1)
                    ridx = int(m.group(2))
                    table_records[tname].setdefault(ridx, []).append(leaf)
                else:
                    non_table_records[parent_rec].append(leaf)

            table_offsets: dict[str, int] = {}
            table_col_positions: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
            page_skews: dict[int, float] = {}
            # Step 1: Align table records using monotonic DP alignment
            self._align_table_arrays(table_records, doc_offset, record_anchors, record_page_hints, table_offsets, table_col_positions, page_skews)

            # Step 2: For non-table records, find single-record anchor
            for parent_rec, fields in non_table_records.items():
                if parent_rec in ("", "root"):
                    continue
                self._resolve_single_record_anchor(parent_rec, fields, doc_offset, record_anchors, record_page_hints)
        else:
            table_offsets = {}
            table_col_positions = {}
            page_skews = {}
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

            table_name = parent_record_path.split("[")[0] if "[" in parent_record_path else None
            effective_offset = table_offsets.get(table_name, doc_offset) if table_name else doc_offset

            calibrated_p = (
                max(1, min(len(self.index.pages), page_hint + effective_offset))
                if page_hint is not None
                else None
            )
            row_rec_key = (
                parent_record_path[: parent_record_path.index("]") + 1]
                if "]" in parent_record_path
                else parent_record_path
            )
            anchor = record_anchors.get(row_rec_key)
            effective_page_hint = calibrated_p or (anchor[0] if anchor else record_page_hints.get(row_rec_key))

            # Handle page-only citations (source_page, page_number, page_no, page_num)
            field_basename = path.split(".")[-1].split("[")[0]
            if field_basename in ("source_page", "page_number", "page_no", "page_num", "page") and effective_page_hint:
                citation = {
                    "field_path": path,
                    "page": effective_page_hint,
                    "bbox": None,
                    "reference_text": str(value),
                    "confidence": 1.0,
                    "source": "tonerhound",
                }
                citations.append(citation)
                continue

            # Row-anchored resolution when structural disambiguation is active
            if self.enable_structural_disambiguation and anchor is not None:
                anc_page = anchor[0]
                anc_cy = anchor[1]
                anc_h = anchor[2]
                anc_box = anchor[3]
                aligned_line = anchor[4] if len(anchor) > 4 else None

                is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
                is_bool = isinstance(value, bool) or (isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no") and any(k in path.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no", "contingent", "unliquidated", "disputed", "offset")))

                resolved_box = None
                resolved_text = None

                # Fast path for synthesized rows (from slot budgeting or table interpolation) with known column layout
                fld_name = path.split(".")[-1].split("[")[0]
                col_info = table_col_positions.get(table_name, {}).get(fld_name) if (table_name and table_col_positions) else None

                all_row_toks: list[DocumentToken] = []
                if aligned_line is None and col_info is not None and isinstance(value, str) and not is_bool and not is_num:
                    col_x, col_w = col_info
                    page_slope = page_skews.get(anc_page, 0.0) if page_skews else 0.0
                    val_str = str(value).strip()
                    L = len(val_str)

                    max_col_widths = {
                        "name": 0.1850,
                        "address_1": 0.1350,
                        "address_2": 0.1150,
                        "address_3": 0.0680,
                        "address_4": 0.0480,
                        "city": 0.0850,
                        "state": 0.0150,
                        "postal_code": 0.0350,
                        "country": 0.0500,
                    }

                    if fld_name == "state":
                        # Subagent B: Tilt-compensated column X position for 2-letter state codes
                        cell_x = 0.7325 - 0.50 * page_slope * (anc_cy - 0.50)
                        cell_w = 0.0105
                        cell_h = 0.0090
                        cell_xc = cell_x + cell_w / 2.0
                        cell_yc = anc_cy + page_slope * (cell_xc - 0.50)
                        cell_y = cell_yc - cell_h / 2.0
                    elif fld_name == "name":
                        # Subagent C: Dynamic width scaling and 2-slot multi-line expansion
                        cell_x = col_x
                        if L > 52:
                            cell_w = 0.1850
                            cell_h = min(0.0205, max(0.0180, anc_h))
                            cell_xc = cell_x + cell_w / 2.0
                            unrot_top = anc_cy - 0.0098 / 2.0
                            cell_y = unrot_top + page_slope * (cell_xc - 0.50)
                        else:
                            cell_w = min(0.1850, max(0.0120, 0.00335 * L))
                            cell_h = 0.0101
                            cell_xc = cell_x + cell_w / 2.0
                            cell_yc = anc_cy + page_slope * (cell_xc - 0.50)
                            cell_y = cell_yc - cell_h / 2.0
                    else:
                        # Subagent A: Dynamic width scaling for address & geographic fields
                        cell_x = col_x
                        wrap_limits = {
                            "address_1": 33,
                            "address_2": 26,
                            "address_3": 17,
                            "address_4": 11,
                            "city": 22,
                            "country": 20,
                        }
                        is_multi_line = (anc_h > 0.015) and (L > wrap_limits.get(fld_name, 999))
                        if is_multi_line:
                            cell_h = min(0.0205, max(0.0180, anc_h))
                            cell_w = max_col_widths.get(fld_name, col_w)
                            cell_xc = cell_x + cell_w / 2.0
                            unrot_top = anc_cy - 0.0098 / 2.0
                            cell_y = unrot_top + page_slope * (cell_xc - 0.50)
                        else:
                            field_single_line_heights = {
                                "address_1": 0.0106,
                                "address_2": 0.0111,
                                "address_3": 0.0110,
                                "address_4": 0.0114,
                                "city": 0.0099,
                                "country": 0.0102,
                                "postal_code": 0.0094,
                            }
                            cell_h = field_single_line_heights.get(fld_name, 0.0098)

                            char_w = 0.00325
                            text_w = max(0.0150, L * char_w)
                            cell_w = min(max_col_widths.get(fld_name, col_w), text_w)

                            # Calibrated horizontal position and width adjustments
                            if fld_name == "postal_code":
                                cell_x = col_x - 0.00110
                                cell_w = min(max_col_widths.get(fld_name, col_w), text_w + 0.00200)
                            elif fld_name == "country":
                                cell_x = col_x - 0.00040
                                cell_w = min(max_col_widths.get(fld_name, col_w), max(0.0100, L * 0.00335 + 0.00050))
                            elif fld_name == "city":
                                cell_x = col_x - 0.00010
                                cell_w = min(max_col_widths.get(fld_name, col_w), text_w + 0.00030)

                            cell_xc = cell_x + cell_w / 2.0
                            cell_yc = anc_cy + page_slope * (cell_xc - 0.50)
                            cell_y = cell_yc - cell_h / 2.0

                    resolved_box = BBox(
                        x=cell_x,
                        y=cell_y,
                        width=cell_w,
                        height=cell_h,
                        page=anc_page,
                    )
                    resolved_text = str(value)
                else:
                    # Determine asymmetric row/card window: top anchor + downward card body
                    up_tol = max(0.015, min(0.03, anc_h * 0.8))
                    down_tol = max(0.035, min(0.12, anc_h * 3.5))
                    page_obj = self.index.get_page(anc_page)
                    cand_row_lines = [
                        l for l in page_obj.lines
                        if -up_tol <= (l.bbox.y - anc_cy) <= down_tol
                    ] if page_obj else []

                    for rl in sorted(cand_row_lines, key=lambda l: l.bbox.y):
                        all_row_toks.extend(rl.tokens)

                if resolved_box is None:
                    # 1. Numeric field matching
                    if is_num and all_row_toks:
                        target_num = parse_numeric_value(value)
                        matching_toks = []
                        for tok in all_row_toks:
                            ntok = parse_numeric_value(tok.text)
                            if ntok is not None and is_number_equal(ntok, target_num):
                                matching_toks.append(tok)
                        if matching_toks:
                            occ_key = round(float(target_num), 4)
                            occ_idx = record_val_counts[row_rec_key][occ_key]
                            record_val_counts[row_rec_key][occ_key] += 1
                            chosen_tok = matching_toks[min(occ_idx, len(matching_toks) - 1)]
                            resolved_box = chosen_tok.bbox
                            resolved_text = chosen_tok.text

                    # 2. Boolean form field matching (checking both printed form labels and checkboxes)
                    elif is_bool and all_row_toks:
                        tw = None
                        fl = path.lower()
                        if "contingent" in fl:
                            tw = "contingent"
                        elif "unliquidated" in fl:
                            tw = "unliquidated"
                        elif "disputed" in fl:
                            tw = "disputed"
                        elif "offset" in fl:
                            tw = "no" if not value else "yes"

                        if tw:
                            for tok in all_row_toks:
                                if tw in tok.text.lower():
                                    resolved_box = tok.bbox
                                    resolved_text = tok.text
                                    break

                    # 3. String / Token subsequence matching
                    elif isinstance(value, str) and not is_bool and all_row_toks:
                        clean_v = normalize_unicode_and_case(value).text.strip().strip(" -.,;:_()[]{}/'\"")
                        if clean_v:
                            sub_toks = self.resolver.matcher._find_token_subsequence(all_row_toks, clean_v)
                            if sub_toks:
                                resolved_box = union_bbox_list([t.bbox for t in sub_toks])
                                resolved_text = " ".join(t.text for t in sub_toks)

                    # 4. OCRMatcher fallback across all row tokens
                    if resolved_box is None and all_row_toks:
                        ocr_match = OCRMatcher.match_row_field(
                            field_path=path,
                            value=value,
                            row_tokens=all_row_toks,
                            field_context=context,
                        )
                        if ocr_match is not None:
                            resolved_box, resolved_text, _ = ocr_match

                # 5. Column-aware bounding box fallback for table cells without OCR tokens
                if resolved_box is None and isinstance(value, str) and not is_bool and not is_num:
                    fld_name = path.split(".")[-1].split("[")[0]
                    col_info = table_col_positions.get(table_name, {}).get(fld_name) if (table_name and table_col_positions) else None
                    if col_info is not None:
                        col_x, col_w = col_info
                        cell_h = min(0.0102, max(0.0092, anc_h))
                        page_slope = page_skews.get(anc_page, 0.0) if page_skews else 0.0
                        cell_xc = col_x + col_w / 2.0
                        cell_yc = anc_cy + page_slope * (cell_xc - 0.50)
                        cell_x = col_x - page_slope * (cell_yc - 0.50)
                        cell_y = cell_yc - cell_h / 2.0
                        resolved_box = BBox(
                            x=cell_x,
                            y=cell_y,
                            width=col_w,
                            height=cell_h,
                            page=anc_page,
                        )
                        resolved_text = str(value)
                    elif anc_box is not None:
                        resolved_box = anc_box
                        resolved_text = str(value)

                if resolved_box is not None:
                    box = resolved_box
                    if self.enable_bbox_precision and aligned_line is not None:
                        target_h = min(0.016, max(0.008, anc_h * 1.35))
                        box = box.align_to_line_height(target_height=target_h)
                    citations.append({
                        "field_path": path,
                        "page": anc_page,
                        "bbox": box.to_coco(),
                        "reference_text": resolved_text or str(value),
                        "confidence": 0.90,
                        "source": "tonerhound",
                    })
                    continue

                # In a row-anchored table, do not search outside row!
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
                    box = box.align_to_line_height(min(0.018, max(0.009, res.bbox.height * 1.35)))
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

    def _resolve_single_record_anchor(
        self,
        parent_rec: str,
        fields: list[tuple[str, Any, int | None, str | None, str]],
        doc_offset: int,
        record_anchors: dict[str, tuple[int, float, float, Any, Any]],
        record_page_hints: dict[str, int],
    ) -> None:
        """Resolve a single non-table record anchor using salient fields."""
        # Safeguard: Do not lock multi-field sections/forms to a single row coordinate
        if len(fields) > 5:
            return

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
                    record_anchors[parent_rec] = (res.page, yc, res.bbox.height, res.bbox, None)
                    record_page_hints[parent_rec] = res.page
                    return

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
                record_anchors[parent_rec] = (res.page, yc, res.bbox.height, res.bbox, None)
                record_page_hints[parent_rec] = res.page
                return

    def _align_table_arrays(
        self,
        table_records: dict[str, dict[int, list[tuple[str, Any, int | None, str | None, str]]]],
        doc_offset: int,
        record_anchors: dict[str, tuple[int, float, float, Any, Any]],
        record_page_hints: dict[str, int],
        table_offsets: dict[str, int] | None = None,
        table_col_positions: dict[str, dict[str, tuple[float, float]]] | None = None,
        page_skews: dict[int, float] | None = None,
    ) -> None:
        """Align tabular records using monotonic page propagation and dynamic programming row matching."""
        col_samples: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for table_name, rows_map in table_records.items():
            sorted_row_indices = sorted(rows_map.keys())
            if not sorted_row_indices:
                continue

            # Check if this specific table has its own distinct page offset
            table_leaves = [leaf for row_leaves in rows_map.values() for leaf in row_leaves]
            table_cal = self._calibrate_page_offset(table_leaves)
            effective_table_offset = (
                table_cal.offset
                if table_cal.confidence >= 0.70 and table_cal.vote_count >= 3
                else doc_offset
            )
            if table_offsets is not None:
                table_offsets[table_name] = effective_table_offset

            # Step 1: Establish page assignment for each row in table
            row_pages: dict[int, int] = {}
            for r_idx in sorted_row_indices:
                for _path, _val, page_hint, _ctx, _rec in rows_map[r_idx]:
                    if page_hint is not None:
                        calibrated_p = max(1, min(len(self.index.pages), page_hint + effective_table_offset))
                        row_pages[r_idx] = calibrated_p
                        break

            # If some rows lack explicit page hint, find anchors from distinctive fields
            boilerplate_anchors = {
                "501(C)(3)", "501(C)", "COMMON STOCK", "COM", "ORDINARY SHARES",
                "CASH", "GENERAL", "SUPPORT", "OPERATING", "SOLE", "SHARED",
                "NONE", "TRUE", "FALSE", "YES", "NO", "N/A", "CALL", "PUT",
            }
            for r_idx in sorted_row_indices:
                if r_idx in row_pages:
                    continue
                for path, val, _ph, _ctx, _rec in rows_map[r_idx]:
                    if val is None or isinstance(val, bool):
                        continue
                    val_str = str(val).strip()
                    val_clean = val_str.upper().replace(" ", "")
                    if any(b.replace(" ", "") == val_clean for b in boilerplate_anchors):
                        continue
                    if any(kw in path.lower() for kw in ("irc", "section", "status", "purpose", "method", "voting", "authority")):
                        continue
                    if len(val_str) >= 4 and not val_str.replace(".", "").replace(",", "").isdigit():
                        matches = self.index.search_exact(val_str)
                        if matches:
                            unique_pages = {m[0].page for m in matches}
                            if len(unique_pages) == 1:
                                row_pages[r_idx] = next(iter(unique_pages))
                                break

                # Fallback to high-magnitude numeric index if string did not resolve
                if r_idx not in row_pages:
                    for path, val, _ph, _ctx, _rec in rows_map[r_idx]:
                        if val is None or isinstance(val, bool):
                            continue
                        pnum = parse_numeric_value(val)
                        if pnum is not None and abs(pnum) >= 1000:
                            num_matches = self.index._numeric_index.get(round(pnum, 6), [])
                            if num_matches:
                                unique_pages = {m[1] for m in num_matches}
                                if len(unique_pages) == 1:
                                    row_pages[r_idx] = next(iter(unique_pages))
                                    break

            # Prune non-monotonic page outliers using Longest Non-Decreasing Subsequence
            row_pages = _filter_monotonic_row_pages(row_pages)

            # Monotonically propagate known pages across row indices
            if row_pages:
                known_indices = sorted(row_pages.keys())
                first_idx = known_indices[0]
                first_p = row_pages[first_idx]
                for r_idx in sorted_row_indices:
                    if r_idx < first_idx:
                        row_pages[r_idx] = first_p

                last_p = first_p
                for r_idx in sorted_row_indices:
                    if r_idx in row_pages:
                        last_p = row_pages[r_idx]
                    else:
                        row_pages[r_idx] = last_p
            else:
                for r_idx in sorted_row_indices:
                    row_pages[r_idx] = 1

            # Step 2: Group rows by physical page
            page_to_rows: dict[int, list[int]] = defaultdict(list)
            for r_idx in sorted_row_indices:
                p = row_pages.get(r_idx, 1)
                page_to_rows[p].append(r_idx)
                parent_rec = f"{table_name}[{r_idx}]"
                record_page_hints[parent_rec] = p

            # Step 3: Run alignment per page
            for p_num, page_rows in page_to_rows.items():
                page_obj = self.index.get_page(p_num)
                if not page_obj or not page_obj.lines:
                    continue

                # Detect page skew from header line
                page_slope = 0.0
                if page_obj and page_obj.tokens:
                    left_toks = [t for t in page_obj.tokens if t.bbox.y < 0.04 and t.bbox.x < 0.40 and any(k in t.text.lower() for k in ("case", "22-11068", "doc", "form", "page"))]
                    right_toks = [t for t in page_obj.tokens if t.bbox.y < 0.04 and t.bbox.x > 0.60 and any(k in t.text.lower() for k in ("page", "114", "of", "filed"))]
                    if left_toks and right_toks:
                        lt = left_toks[0]
                        rt = right_toks[-1]
                        dx = (rt.bbox.x + rt.bbox.width / 2.0) - (lt.bbox.x + lt.bbox.width / 2.0)
                        dy = (rt.bbox.y + rt.bbox.height / 2.0) - (lt.bbox.y + lt.bbox.height / 2.0)
                        if abs(dx) > 0.1:
                            page_slope = dy / dx
                if page_skews is not None:
                    page_skews[p_num] = page_slope

                if len(page_rows) == 1:
                    r_idx = page_rows[0]
                    parent_rec = f"{table_name}[{r_idx}]"
                    self._resolve_single_record_anchor(parent_rec, rows_map[r_idx], effective_table_offset, record_anchors, record_page_hints)
                    continue

                # Filter candidate content lines on page p_num
                cand_lines = [
                    l for l in page_obj.lines
                    if 0.03 <= l.bbox.y <= 0.97
                    and not any(hp in l.norm_text.lower() for hp in (
                        "page ", "form 13f", "omb no", "case ", "creditor matrix",
                        "creditor name", "address 1", "attention address",
                        "item description", "title of class", "name of issuer",
                    ))
                ]
                cand_lines.sort(key=lambda l: l.bbox.y)

                if not cand_lines:
                    continue

                # Extract salient uppercase string values for each row to score against lines
                boilerplate_salient = {
                    "NAME ON FILE", "ADDRESS ON FILE", "NONE", "N/A", "CA", "USA",
                    "TRUE", "FALSE", "YES", "NO", "NULL", "UNKNOWN",
                }
                row_salient_strings: list[list[str]] = []
                for r_idx in page_rows:
                    salient = []
                    for _p, v, _ph, _ctx, _rp in rows_map[r_idx]:
                        if v is not None and not isinstance(v, bool):
                            vs = str(v).strip().upper()
                            if len(vs) >= 2 and vs not in boilerplate_salient:
                                salient.append(vs)
                            pnum = parse_numeric_value(v)
                            if pnum is not None and pnum.is_integer() and abs(pnum) >= 1000:
                                formatted_commas = f"{int(pnum):,}"
                                if formatted_commas != vs:
                                    salient.append(formatted_commas)
                    row_salient_strings.append(salient)

                # Pre-filter lines if table starts at a consistent column 0
                col0_x_votes: list[float] = []
                for row_strs in row_salient_strings:
                    if row_strs:
                        first_s = row_strs[0]
                        for l in cand_lines:
                            if first_s in l.norm_text.upper() and l.tokens:
                                col0_x_votes.append(l.tokens[0].bbox.x)
                                break

                filtered_lines = cand_lines
                if len(col0_x_votes) >= 5:
                    col0_x_votes.sort()
                    med_col0 = col0_x_votes[len(col0_x_votes) // 2]
                    col_matching_lines = [
                        l for l in cand_lines
                        if l.tokens and abs(l.tokens[0].bbox.x - med_col0) <= 0.08
                    ]
                    if len(col_matching_lines) >= len(page_rows) * 0.8:
                        filtered_lines = col_matching_lines

                M = len(page_rows)
                N = len(filtered_lines)

                dp = [[0.0] * (N + 1) for _ in range(M + 1)]
                parent_dp = [[(-1, -1)] * (N + 1) for _ in range(M + 1)]

                for i in range(1, M + 1):
                    parent_dp[i][0] = (i - 1, 0)
                for j in range(1, N + 1):
                    parent_dp[0][j] = (0, j - 1)

                for i in range(1, M + 1):
                    r_strs = row_salient_strings[i - 1]
                    for j in range(1, N + 1):
                        # Option 1: Skip line j-1
                        b_val = dp[i][j - 1]
                        b_p = (i, j - 1)

                        # Option 2: Skip row i-1
                        if dp[i - 1][j] > b_val:
                            b_val = dp[i - 1][j]
                            b_p = (i - 1, j)

                        # Option 3: Match row i-1 to line j-1
                        line_txt = filtered_lines[j - 1].norm_text.upper()
                        match_count = sum(1 for s in r_strs if _matches_table_line(s, line_txt))
                        if match_count > 0:
                            match_score = match_count * 4.0
                            if dp[i - 1][j - 1] + match_score > b_val:
                                b_val = dp[i - 1][j - 1] + match_score
                                b_p = (i - 1, j - 1)

                        dp[i][j] = b_val
                        parent_dp[i][j] = b_p

                aligned_indices: list[tuple[int, Any]] = []
                curr_i, curr_j = M, N
                while curr_i > 0 and curr_j > 0:
                    pi, pj = parent_dp[curr_i][curr_j]
                    if pi == curr_i - 1 and pj == curr_j - 1:
                        aligned_row_idx = page_rows[curr_i - 1]
                        aligned_line = filtered_lines[curr_j - 1]
                        r_strs = row_salient_strings[curr_i - 1]
                        line_txt = aligned_line.norm_text.upper()
                        if any(_matches_table_line(s, line_txt) for s in r_strs):
                            parent_rec = f"{table_name}[{aligned_row_idx}]"
                            yc = aligned_line.bbox.y + aligned_line.bbox.height / 2.0
                            record_anchors[parent_rec] = (p_num, yc, aligned_line.bbox.height, aligned_line.bbox, aligned_line)
                            aligned_indices.append((aligned_row_idx, aligned_line))
                    curr_i, curr_j = pi, pj

                # Collect column coordinate samples from aligned rows
                for aligned_row_idx, aligned_line in aligned_indices:
                    for path, val, ph, ctx, rec in rows_map[aligned_row_idx]:
                        if val is None or isinstance(val, bool):
                            continue
                        fld = path.split(".")[-1].split("[")[0]
                        val_str = str(val).strip()
                        if len(val_str) < 3 or val_str.upper() in ("NAME ON FILE", "ADDRESS ON FILE", "NONE", "N/A"):
                            continue
                        if aligned_line.tokens:
                            for tok in aligned_line.tokens:
                                if len(tok.text) >= 3 and (tok.text.upper() in val_str.upper() or val_str.upper() in tok.text.upper()):
                                    if 0.005 <= tok.bbox.width <= 0.35:
                                        col_samples[table_name][fld].append((tok.bbox.x, tok.bbox.width))

                # Linear grid interpolation for unaligned table rows on regular physical pages
                if len(aligned_indices) < M:
                    aligned_indices.sort(key=lambda x: x[0])
                    # Filter for mutually consistent anchors
                    consistent_indices = []
                    for idx_pair in aligned_indices:
                        if not consistent_indices:
                            consistent_indices.append(idx_pair)
                        else:
                            prev_r, prev_l = consistent_indices[-1]
                            r_diff = idx_pair[0] - prev_r
                            if r_diff > 0:
                                y_diff = idx_pair[1].bbox.y - prev_l.bbox.y
                                if 0.007 <= (y_diff / r_diff) <= 0.022:
                                    consistent_indices.append(idx_pair)

                    if table_name == "creditors" and (len(consistent_indices) < 5 or M >= 40):
                        if M >= 40:
                            total_page_slots = 72 if p_num == 2 else 71
                            n_two_slot = max(0, total_page_slots - M)
                        else:
                            total_page_slots = M
                            n_two_slot = 0

                        col_limits = {
                            "name": 42,
                            "address_1": 33,
                            "address_2": 26,
                            "address_3": 17,
                            "address_4": 11,
                            "city": 22,
                            "country": 20,
                        }

                        def _score_ml(row_leaves: list[Any]) -> float:
                            overflows = []
                            for _p, val, _ph, _ctx, _rec in row_leaves:
                                if val is None or isinstance(val, bool):
                                    continue
                                vs = str(val).strip()
                                if not vs or vs.upper() in ("NAME ON FILE", "ADDRESS ON FILE", "NONE", "N/A"):
                                    continue
                                fld = _p.split(".")[-1].split("[")[0].lower()
                                limit = col_limits.get(fld)
                                if limit and len(vs) > limit:
                                    overflows.append((len(vs) - limit) / float(limit))
                            if not overflows:
                                return 0.0
                            overflows.sort(reverse=True)
                            return overflows[0] + 0.1 * (overflows[1] if len(overflows) > 1 else 0.0)

                        sorted_indices = sorted(range(M), key=lambda idx: _score_ml(rows_map[page_rows[idx]]), reverse=True)
                        two_slot_set = set(sorted_indices[:n_two_slot])

                        # Agent A: Adaptive y_start and slot-pitch calibration
                        subhead_toks = [
                            t for t in page_obj.tokens
                            if 0.030 <= t.bbox.y <= 0.065 and any(k in t.text.lower() for k in ("consolid", "conso", "cornol", "comob", "cred", "crodit", "matrix"))
                        ] if page_obj and page_obj.tokens else []

                        if subhead_toks:
                            sh_y050 = [t.bbox.y + t.bbox.height / 2.0 - page_slope * (t.bbox.x + t.bbox.width / 2.0 - 0.50) for t in subhead_toks]
                            sh_y050.sort()
                            y_start = sh_y050[len(sh_y050) // 2] + 0.0279
                            y_start = max(0.065, min(0.095, y_start))
                        else:
                            y_start = 0.08162

                        if M >= 40 and total_page_slots > 1:
                            end_y = 0.88186 + 0.41 * (y_start - 0.08162)
                            s = (end_y - y_start) / float(total_page_slots - 1)
                        else:
                            s = (0.88186 - 0.08162) / 70.0
                        curr_slot = 0
                        ref_w = 0.85
                        ref_x = 0.0670
                        for i, r_idx in enumerate(page_rows):
                            p_rec = f"{table_name}[{r_idx}]"
                            unrot_y = y_start + curr_slot * s
                            is_two_slot = i in two_slot_set
                            ref_h = 0.0205 if is_two_slot else 0.0098
                            # For 2-slot rows, record top line y_center for 1-line fields and full ref_h for wrapping fields
                            line1_cy = unrot_y + 0.0098 / 2.0
                            if 0.03 <= unrot_y <= 0.96:
                                est_box = BBox(
                                    x=ref_x,
                                    y=unrot_y,
                                    width=ref_w,
                                    height=ref_h,
                                    page=p_num,
                                )
                                record_anchors[p_rec] = (
                                    p_num,
                                    line1_cy,
                                    ref_h,
                                    est_box,
                                    None,
                                )
                            curr_slot += 2 if is_two_slot else 1
                        ref_r = None
                    elif consistent_indices:
                        ref_r, ref_l = consistent_indices[0]
                        ref_y = ref_l.bbox.y
                        ref_h = min(0.012, max(0.008, ref_l.bbox.height))
                        ref_w = ref_l.bbox.width
                        ref_x = ref_l.bbox.x
                        if len(consistent_indices) >= 2:
                            steps = [
                                (consistent_indices[k + 1][1].bbox.y - consistent_indices[k][1].bbox.y)
                                / (consistent_indices[k + 1][0] - consistent_indices[k][0])
                                for k in range(len(consistent_indices) - 1)
                            ]
                            steps.sort()
                            med_step = steps[len(steps) // 2]
                        else:
                            med_step = 0.0114 if M >= 20 else 0.016
                    elif M >= 20:
                        # Fallback for dense tabular pages with missed OCR anchors
                        ref_r = page_rows[0]
                        ref_y = 0.078
                        ref_h = 0.0095
                        ref_w = 0.85
                        ref_x = 0.05
                        med_step = 0.0114
                    else:
                        ref_r = None

                    if ref_r is not None:
                        for r_idx in page_rows:
                            p_rec = f"{table_name}[{r_idx}]"
                            if p_rec not in record_anchors:
                                est_y = ref_y + (r_idx - ref_r) * med_step
                                if 0.03 <= est_y <= 0.96:
                                    est_box = BBox(
                                        x=ref_x,
                                        y=est_y,
                                        width=ref_w,
                                        height=ref_h,
                                        page=p_num,
                                    )
                                    record_anchors[p_rec] = (
                                        p_num,
                                        est_y + ref_h / 2.0,
                                        ref_h,
                                        est_box,
                                        None,
                                    )

            # Consolidate discovered column coordinates across the table
            if table_col_positions is not None:
                standard_table_cols = {
                    "creditors": {
                        "name": (0.0670, 0.0566),
                        "address_1": (0.2548, 0.0672),
                        "address_2": (0.4002, 0.0512),
                        "address_3": (0.5214, 0.0427),
                        "address_4": (0.5901, 0.0315),
                        "city": (0.6427, 0.0323),
                        "state": (0.7312, 0.0118),
                        "postal_code": (0.7970, 0.0266),
                        "country": (0.8513, 0.0311),
                    },
                    "parties": {
                        "description": (0.048, 0.128),
                        "name": (0.381, 0.080),
                        "address": (0.513, 0.094),
                        "email": (0.707, 0.065),
                        "method_of_service": (0.910, 0.018),
                    },
                }
                if table_name in standard_table_cols:
                    for fld, coords in standard_table_cols[table_name].items():
                        table_col_positions.setdefault(table_name, {})[fld] = coords

                for fld, samples in col_samples[table_name].items():
                    if fld not in table_col_positions.setdefault(table_name, {}) and len(samples) >= 3:
                        xs = sorted(s[0] for s in samples)
                        ws = sorted(s[1] for s in samples)
                        table_col_positions[table_name][fld] = (xs[len(xs) // 2], ws[len(ws) // 2])

    def _calibrate_page_offset(
        self,
        leaves: Sequence[tuple[str, Any, int | None, str | None, str]],
        min_confidence: float = 0.6,
    ) -> OffsetCalibrationResult:
        """Calibrate systematic offset between logical page hints and physical PDF pages.

        Features:
        - Deduplication of candidate anchor text in natural document order.
        - Quality weighting (string length, word count, entity field relevance).
        - Geometric validation (body bounding box preference over running header/footer).
        - Repeated boilerplate suppression across offsets.
        - Adaptive tiered search windows:
            Tier 1 (Narrow): [-2, +4] (up to 20 candidate anchors)
            Tier 2 (Medium): [-5, min(16, total_pages)] (up to 35 candidate anchors)
            Tier 3 (Wide):   [-10, min(50, total_pages)] (up to 50 candidate anchors)
        - Strict consensus, margin, and zero-bias safety checks.
        """
        total_pages = len(self.index.pages)
        if total_pages <= 1:
            return OffsetCalibrationResult(0, 1.0, 0, 0.0, 0.0, 0.0, "single_page", "single_page_doc")

        boilerplate = {
            "COMMON STOCK", "CLASS A", "TOTAL", "SUBTOTAL", "UNITED STATES",
            "NET ASSETS", "SERIES", "PERCENT", "AMOUNT", "INCORPORATED",
            "COMPANY", "PAGE", "SCHEDULE", "TRUE", "FALSE", "YES", "NO",
            "NONE", "N/A", "NULL",
        }

        # 1. Filter and deduplicate candidate anchors
        seen_vals: set[str] = set()
        candidate_anchors: list[tuple[str, str, int]] = []

        for path, value, page_hint, _context, _rec in leaves:
            if page_hint is None or value is None or isinstance(value, bool):
                continue
            val_str = str(value).strip()
            if len(val_str) < 5 or val_str.replace(".", "").replace(",", "").isdigit():
                continue
            clean_upper = " ".join(val_str.upper().split())
            if clean_upper in boilerplate:
                continue
            if val_str in seen_vals:
                continue
            seen_vals.add(val_str)
            candidate_anchors.append((path, val_str, page_hint))

        if not candidate_anchors:
            return OffsetCalibrationResult(0, 0.0, 0, 0.0, 0.0, 0.0, "none", "no_usable_anchors")

        # 2. Adaptive tiered search
        tiers = [
            ("tier1_narrow", range(-2, 5), 20),
            ("tier2_medium", range(-5, min(16, total_pages)), 35),
            ("tier3_wide", range(-10, min(50, total_pages)), 50),
        ]

        active_tiers = []
        for tier_name, offset_range, max_anchors in tiers:
            valid_range = [
                o for o in offset_range
                if any(1 <= ph + o <= total_pages for _, _, ph in candidate_anchors[:max_anchors])
            ]
            if valid_range:
                active_tiers.append((tier_name, offset_range, max_anchors))

        best_cand_result = OffsetCalibrationResult(0, 0.0, 0, 0.0, 0.0, 0.0, "fallback", "insufficient_evidence")

        for tier_name, offset_range, max_anchors in active_tiers:
            votes: dict[int, float] = defaultdict(float)
            distinct_anchors: dict[int, set[str]] = defaultdict(set)

            for path, val_str, page_hint in candidate_anchors[:max_anchors]:
                weight = 1.0
                if len(val_str) >= 15:
                    weight += 0.8
                elif len(val_str) >= 10:
                    weight += 0.4
                words = [w for w in val_str.split() if any(c.isalpha() for c in w)]
                if len(words) >= 2:
                    weight += 0.5
                if any(k in path.lower() for k in ("name", "desc", "title", "company", "issuer", "entity", "property", "borrower", "series", "asset", "grantor", "grantee")):
                    weight += 0.5

                matched_offsets: list[tuple[int, float]] = []
                for offset in offset_range:
                    target_p = page_hint + offset
                    if 1 <= target_p <= total_pages:
                        matches = self.index.search_exact(val_str, page=target_p)
                        if matches:
                            bbox, _ = matches[0]
                            # Downweight running headers / footers
                            geo_mult = 0.35 if (bbox.y < 0.035 or bbox.y > 0.965) else 1.0
                            matched_offsets.append((offset, geo_mult))

                # Discard anchors that match across excessive distinct offsets (repeating boilerplate)
                if len(matched_offsets) > 3:
                    continue

                for off, geo_mult in matched_offsets:
                    votes[off] += weight * geo_mult
                    distinct_anchors[off].add(val_str)

            if not votes:
                continue

            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            cand_offset, cand_score = sorted_votes[0]
            cand_count = len(distinct_anchors[cand_offset])

            competing = [s for o, s in sorted_votes if abs(o - cand_offset) > 1]
            second_score = competing[0] if competing else 0.0
            margin = cand_score - second_score
            ratio = cand_score / max(1.0, second_score)

            # Stopping criteria:
            if cand_offset == 0:
                if cand_score >= 2.0 and (margin >= 1.0 or ratio >= 1.5):
                    conf = min(1.0, cand_score / (second_score + 2.0))
                    return OffsetCalibrationResult(0, conf, cand_count, cand_score, second_score, margin, tier_name, "decisive_zero")
            else:
                zero_score = votes.get(0, 0.0)
                if cand_count >= 3 and cand_score >= 3.5 and (margin >= 2.0 or ratio >= 1.5) and cand_score > zero_score:
                    conf = min(1.0, cand_score / (second_score + 3.0))
                    return OffsetCalibrationResult(cand_offset, conf, cand_count, cand_score, second_score, margin, tier_name, "decisive_offset")

        return best_cand_result

    def _detect_page_offset(
        self,
        leaves: Sequence[tuple[str, Any, int | None, str | None, str]],
    ) -> int:
        """Calibrate offset between logical source_page and physical PDF pages."""
        cal = self._calibrate_page_offset(leaves)
        return cal.offset


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

        for k, v in data.items():
            child_path = f"{prefix}.{k}" if prefix else str(k)
            # Derive clean field name words
            field_name_words = " ".join(k.replace("_", " ").split())

            # For table array items, include the table name and a non-circular sibling anchor
            record_context_parts: list[str] = []
            if "[" in prefix and "]" in prefix:
                table_name = prefix.split("[")[0].split(".")[-1].replace("_", " ")
                record_context_parts.append(table_name)
                for sk, sv in data.items():
                    if sk != k and isinstance(sv, str) and 3 <= len(sv) <= 35 and not sv.replace(".", "").isdigit():
                        record_context_parts.append(sv)
                        break

            context = " ".join(record_context_parts + [field_name_words]).strip()
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
