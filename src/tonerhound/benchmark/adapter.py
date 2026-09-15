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
from tonerhound.geometry.coordinates import union_bbox_list
from tonerhound.models.types import DocumentToken, ExtractionInput
from tonerhound.normalization.normalizers import (
    is_number_equal,
    normalize_unicode_and_case,
    parse_numeric_value,
)
from tonerhound.resolution.resolver import EvidenceResolver


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
            # Step 1: Align table records using monotonic DP alignment
            self._align_table_arrays(table_records, doc_offset, record_anchors, record_page_hints, table_offsets)

            # Step 2: For non-table records, find single-record anchor
            for parent_rec, fields in non_table_records.items():
                if parent_rec in ("", "root"):
                    continue
                self._resolve_single_record_anchor(parent_rec, fields, doc_offset, record_anchors, record_page_hints)
        else:
            table_offsets = {}
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
                is_bool = isinstance(value, bool) or (isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no") and any(k in path.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no")))

                resolved_box = None
                resolved_text = None

                # Fast path: search directly on aligned_line tokens when aligned_line is available
                if aligned_line and aligned_line.tokens:
                    toks = aligned_line.tokens
                    if is_num:
                        target_num = parse_numeric_value(value)
                        matching_toks = []
                        for tok in toks:
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
                    elif is_bool:
                        bool_matches = self.resolver.matcher.find_boolean_candidates(
                            value, field_name=path, page_hint=anc_page, field_context=context
                        )
                        for bc in bool_matches:
                            bc_cy = bc.bbox.y + bc.bbox.height / 2.0
                            if abs(bc_cy - anc_cy) <= max(0.015, anc_h * 1.8):
                                resolved_box = bc.bbox
                                resolved_text = bc.matched_text
                                break
                    elif isinstance(value, str):
                        norm_v = normalize_unicode_and_case(value).text.strip()
                        clean_v = norm_v.strip(" -.,;:_()[]{}/'\"")
                        if clean_v:
                            sub_toks = self.resolver.matcher._find_token_subsequence(toks, clean_v)
                            if sub_toks:
                                resolved_box = union_bbox_list([t.bbox for t in sub_toks])
                                resolved_text = " ".join(t.text for t in sub_toks)

                if resolved_box is not None:
                    box = resolved_box
                    if self.enable_bbox_precision:
                        target_h = min(0.016, max(0.008, anc_h * 1.35))
                        box = box.align_to_line_height(target_height=target_h)
                    citation = {
                        "field_path": path,
                        "page": anc_page,
                        "bbox": box.to_coco(),
                        "reference_text": resolved_text or str(value),
                        "confidence": 0.95,
                        "source": "tonerhound",
                    }
                    citations.append(citation)
                    continue

                # Fallback: check adjacent lines within row_tol
                row_tol = max(0.038, anc_h * 3.5)
                page_obj = self.index.get_page(anc_page)
                if page_obj:
                    cand_row_lines = [
                        l for l in page_obj.lines
                        if abs((l.bbox.y + l.bbox.height / 2.0) - anc_cy) <= row_tol
                    ]
                    for rl in cand_row_lines:
                        if is_num:
                            target_num = parse_numeric_value(value)
                            matching_toks = []
                            for tok in rl.tokens:
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
                                break
                        elif isinstance(value, str) and not is_bool:
                            norm_v = normalize_unicode_and_case(value).text.strip()
                            clean_v = norm_v.strip(" -.,;:_()[]{}/'\"")
                            if clean_v:
                                sub_toks = self.resolver.matcher._find_token_subsequence(rl.tokens, clean_v)
                                if sub_toks:
                                    resolved_box = union_bbox_list([t.bbox for t in sub_toks])
                                    resolved_text = " ".join(t.text for t in sub_toks)
                                    break

                    # Multi-line token subsequence check across cand_row_lines
                    if resolved_box is None and isinstance(value, str) and not is_bool and len(cand_row_lines) > 1:
                        norm_v = normalize_unicode_and_case(value).text.strip()
                        clean_v = norm_v.strip(" -.,;:_()[]{}/'\"")
                        if clean_v:
                            all_row_toks = []
                            for rl in sorted(cand_row_lines, key=lambda l: l.bbox.y):
                                all_row_toks.extend(rl.tokens)
                            sub_toks = self.resolver.matcher._find_token_subsequence(all_row_toks, clean_v)
                            if sub_toks:
                                resolved_box = union_bbox_list([t.bbox for t in sub_toks])
                                resolved_text = " ".join(t.text for t in sub_toks)

                    if resolved_box is not None:
                        box = resolved_box
                        if self.enable_bbox_precision:
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
    ) -> None:
        """Align tabular records using monotonic page propagation and dynamic programming row matching."""
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
                row_salient_strings: list[list[str]] = []
                for r_idx in page_rows:
                    salient = []
                    for _p, v, _ph, _ctx, _rp in rows_map[r_idx]:
                        if v is not None and not isinstance(v, bool):
                            vs = str(v).strip().upper()
                            if len(vs) >= 2:
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

                dp = [[-1e9] * (N + 1) for _ in range(M + 1)]
                parent_dp = [[(-1, -1)] * (N + 1) for _ in range(M + 1)]

                for j in range(N + 1):
                    dp[0][j] = 0.0

                for i in range(1, M + 1):
                    r_strs = row_salient_strings[i - 1]
                    for j in range(1, N + 1):
                        b_val = dp[i][j - 1]
                        b_p = (i, j - 1)

                        line_txt = filtered_lines[j - 1].norm_text.upper()
                        match_count = sum(1 for s in r_strs if s in line_txt)
                        match_score = (match_count * 4.0) if match_count > 0 else 0.05

                        if dp[i - 1][j - 1] + match_score > b_val:
                            b_val = dp[i - 1][j - 1] + match_score
                            b_p = (i - 1, j - 1)

                        dp[i][j] = b_val
                        parent_dp[i][j] = b_p

                curr_i, curr_j = M, N
                while curr_i > 0 and curr_j > 0:
                    pi, pj = parent_dp[curr_i][curr_j]
                    if pi == curr_i - 1 and pj == curr_j - 1:
                        aligned_row_idx = page_rows[curr_i - 1]
                        aligned_line = filtered_lines[curr_j - 1]
                        parent_rec = f"{table_name}[{aligned_row_idx}]"
                        yc = aligned_line.bbox.y + aligned_line.bbox.height / 2.0
                        record_anchors[parent_rec] = (p_num, yc, aligned_line.bbox.height, aligned_line.bbox, aligned_line)
                    curr_i, curr_j = pi, pj

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
