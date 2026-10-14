"""EXP-045 Stage 7: Array Index -> Physical Table Row Mapper.

Fast, deterministic mapping of schema array index notation (e.g. `holdings[4003].security_name`)
to the physical table row across multi-page table flows.
Incorporates:
- Visual line-wrap merging
- Intermediate section sub-header and issuer filtering
- Column header and subtotal disambiguation
- Multi-column (2-column) page reading order flow
to eliminate row accumulation drift before index lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pymupdf
from rapidfuzz import fuzz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


@dataclass
class TableRow:
    """Represents a single detected logical row in a document table."""
    page_index: int              # 1-indexed page number
    row_index_in_page: int       # 0-indexed position on page
    row_index_global: int        # sequential 0-indexed position across table data rows
    y_top: float                 # normalized top y
    y_bottom: float              # normalized bottom y
    bbox: list[float]            # [x, y, w, h] normalized
    is_header: bool = False
    is_continued_header: bool = False
    is_data: bool = True
    cells_text: list[str] = field(default_factory=list)


@dataclass
class TableStructure:
    """Represents an assembled multi-page or single-page table."""
    document_id: str
    table_id: str
    header_row_count: int
    page_span: list[int]         # [start_page, end_page] (1-indexed)
    rows: list[TableRow]
    data_row_count: int
    data_row_index_offset: int = 0
    fallback_reasons: list[str] = field(default_factory=list)


def is_numeric_token(s: str) -> bool:
    """Check if string token represents a numeric value."""
    if not isinstance(s, str):
        return False
    cleaned = s.replace("$", "").replace(",", "").replace("%", "").replace("(", "").replace(")", "").strip()
    if not cleaned:
        return False
    # Exclude multiplier notation like '(000)'
    if cleaned == "000" and ("(" in s or "$" in s):
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def has_large_or_currency_number(cells: Sequence[str]) -> bool:
    """Check if any cell contains a large quantity, par value, or currency amount."""
    for c in cells:
        if not isinstance(c, str):
            continue
        cleaned = c.replace("$", "").replace(",", "").strip()
        if "," in c:
            try:
                val = float(cleaned)
                if val > 10:
                    return True
            except ValueError:
                pass
        else:
            try:
                val = float(cleaned)
                if val >= 100:
                    return True
            except ValueError:
                pass
    return False


class RowIndexMapper:
    """Extracts table layout, tracks multi-page flows, and maps schema array indices to rows."""

    _ARRAY_PATH_PATTERN = re.compile(r"^([a-zA-Z0-9_.]+)\[(\d+)\]\.(.+)$")

    _ARRAY_PAGE_KEYWORDS: dict[str, list[str]] = {
        "holdings": [
            "schedule of investments", "portfolio of investments", "holdings",
            "statement of net assets", "statement of investments",
        ],
        "checks": ["check register", "checks issued", "check", "disbursement"],
        "grants": ["grants", "assistance", "grant"],
        "entities": ["specially designated", "sanctions", "entities", "sdn"],
        "creditors": ["creditor", "claim", "schedule e/f", "creditors"],
        "transactions": ["transactions", "capital gains", "part iv", "sales"],
    }

    _ARRAY_HEADER_KEYWORDS: dict[str, list[str]] = {
        "holdings": ["security", "shares", "par", "par(000)", "value", "investment", "market value", "($000)"],
        "checks": ["check", "date", "payee", "amount", "check nbr", "check number", "credit memo"],
        "grants": ["recipient", "amount", "purpose", "grant"],
        "entities": ["name", "address", "id", "remarks", "program"],
        "creditors": ["creditor", "claim", "amount", "address"],
        "transactions": ["description", "date", "price", "cost", "sales price", "gain", "gross"],
    }

    def __init__(self, pdf_path: str | Path, document_id: str = "") -> None:
        self.pdf_path = Path(pdf_path)
        self.document_id = document_id
        self._array_tables: dict[str, TableStructure] = {}
        self._default_table_structure: TableStructure | None = None

    @property
    def table_structure(self) -> TableStructure | None:
        if self._default_table_structure is None:
            self._default_table_structure = self.get_table_structure_for_array("holdings")
        return self._default_table_structure

    @table_structure.setter
    def table_structure(self, val: TableStructure | None) -> None:
        self._default_table_structure = val

    def parse_array_field_path(self, field_path: str) -> tuple[str, int, str] | None:
        """Parse schema field path into (array_name, index, sub_field)."""
        m = self._ARRAY_PATH_PATTERN.match(field_path)
        if not m:
            return None
        return m.group(1), int(m.group(2)), m.group(3)

    def get_table_structure_for_array(self, array_name: str) -> TableStructure | None:
        """Retrieve or build the TableStructure for a specific array name."""
        # Normalize base array name (e.g. 'form_8949.part1_short_term_transactions' -> 'transactions')
        base_name = array_name.split(".")[-1]
        for key in self._ARRAY_PAGE_KEYWORDS:
            if key in base_name.lower():
                base_name = key
                break

        if base_name in self._array_tables:
            return self._array_tables[base_name]

        ts = self._build_table_structure_for_array(base_name)
        if ts:
            self._array_tables[base_name] = ts
            if not self.table_structure or base_name == "holdings":
                self.table_structure = ts
        return ts

    def _build_table_structure_for_array(self, array_name: str) -> TableStructure | None:
        """Detect tables across document pages and assemble global row sequence."""
        if not self.pdf_path.exists():
            return None

        try:
            doc = pymupdf.open(self.pdf_path)
        except Exception:
            return None

        title_keywords = self._ARRAY_PAGE_KEYWORDS.get(array_name, [array_name])
        header_keywords = self._ARRAY_HEADER_KEYWORDS.get(
            array_name, ["security", "shares", "par", "value", "amount", "date"]
        )

        # Step 1: Scan pages to locate table candidate range
        matched_pages: list[int] = []
        for p_idx in range(len(doc)):
            page_text = doc[p_idx].get_text().lower()
            if any(tk in page_text for tk in title_keywords):
                matched_pages.append(p_idx)

        # Fallback: if no title keyword matched, inspect all pages
        target_page_indices = matched_pages if matched_pages else list(range(len(doc)))

        all_data_rows: list[TableRow] = []
        page_span = [target_page_indices[0] + 1, target_page_indices[-1] + 1] if target_page_indices else [1, len(doc)]

        for p_idx in target_page_indices:
            page_num = p_idx + 1
            page = doc[p_idx]
            pw, ph = max(1.0, page.rect.width), max(1.0, page.rect.height)

            # Check 2-column layout detection
            tabs_full = page.find_tables(strategy="text")
            is_two_col = False
            tabs_list = [tabs_full]

            if tabs_full.tables:
                t0 = tabs_full.tables[0]
                hdr_names = [str(n).lower() for n in t0.header.names] if t0.header else []
                if len(hdr_names) >= 6 and any(hdr_names.count(k) >= 2 for k in ["security", "shares", "value", "par"]):
                    is_two_col = True
                    col_rects = [pymupdf.Rect(0, 0, pw / 2.0, ph), pymupdf.Rect(pw / 2.0, 0, pw, ph)]
                    tabs_list = [page.find_tables(clip=c, strategy="text") for c in col_rects]

            # If not detected by full header, check dual column clipped find_tables
            if not is_two_col and pw > 500:
                t_left_test = page.find_tables(clip=pymupdf.Rect(0, 0, pw / 2.0, ph), strategy="text")
                t_right_test = page.find_tables(clip=pymupdf.Rect(pw / 2.0, 0, pw, ph), strategy="text")
                if t_left_test.tables and t_right_test.tables:
                    hl = [str(n).lower() for n in t_left_test.tables[0].header.names] if t_left_test.tables[0].header else []
                    hr = [str(n).lower() for n in t_right_test.tables[0].header.names] if t_right_test.tables[0].header else []
                    if any(k in " ".join(hl) for k in header_keywords) and any(k in " ".join(hr) for k in header_keywords):
                        is_two_col = True
                        tabs_list = [t_left_test, t_right_test]

            for tabs in tabs_list:
                if not tabs.tables:
                    continue

                for t_idx, t in enumerate(tabs.tables):
                    in_table = False
                    current_desc_lines: list[str] = []
                    current_bboxes: list[tuple[float, float, float, float]] = []

                    raw_rows = t.extract()
                    for r_idx, r in enumerate(raw_rows):
                        non_empty = [c.strip() for c in r if c and c.strip()]
                        if not non_empty:
                            continue

                        full_text = " ".join(non_empty)
                        compact_text = "".join(non_empty).lower()
                        has_num = any(is_numeric_token(c) for c in non_empty)
                        has_large_num = has_large_or_currency_number(non_empty)

                        # Filter 1: Column Header detection (contains header keywords but lacks numeric values)
                        if not has_num and any(h in compact_text for h in header_keywords):
                            in_table = True
                            current_desc_lines.clear()
                            current_bboxes.clear()
                            continue

                        if not in_table:
                            continue

                        # Filter 2: Subtotals / Totals
                        if full_text.lower().startswith(("total", "subtotal", "cost", "net assets")) or (
                            len(non_empty) == 1 and is_numeric_token(non_empty[0])
                        ):
                            current_desc_lines.clear()
                            current_bboxes.clear()
                            continue

                        # Filter 3: Intermediate Section Sub-Headers (e.g. percentages or sector titles without data values)
                        if (
                            "%" in compact_text
                            or any(
                                compact_text.startswith(kw)
                                for kw in [
                                    "corporatebonds", "municipalbonds", "commonstocks",
                                    "preferredstocks", "governmentobligations", "asset-backed",
                                    "otherassets",
                                ]
                            )
                        ) and not has_large_num:
                            current_desc_lines.clear()
                            current_bboxes.clear()
                            continue

                        r_box = t.rows[r_idx].bbox

                        # Feature: Visual Line-Wrap Merging
                        if not has_large_num:
                            # Accumulate wrapped multi-line description
                            current_desc_lines.append(full_text)
                            current_bboxes.append(r_box)
                        else:
                            # Anchor line with values: merge preceding description lines into unified TableRow
                            all_boxes = current_bboxes + [r_box]
                            rx0 = min(b[0] for b in all_boxes)
                            ry0 = min(b[1] for b in all_boxes)
                            rx1 = max(b[2] for b in all_boxes)
                            ry1 = max(b[3] for b in all_boxes)

                            norm_bbox = [
                                round(rx0 / pw, 5),
                                round(ry0 / ph, 5),
                                round((rx1 - rx0) / pw, 5),
                                round((ry1 - ry0) / ph, 5),
                            ]
                            y_top = norm_bbox[1]
                            y_bottom = norm_bbox[1] + norm_bbox[3]

                            merged_row = TableRow(
                                page_index=page_num,
                                row_index_in_page=len(all_data_rows),
                                row_index_global=len(all_data_rows),
                                y_top=y_top,
                                y_bottom=y_bottom,
                                bbox=norm_bbox,
                                is_header=False,
                                is_continued_header=False,
                                is_data=True,
                                cells_text=current_desc_lines + non_empty,
                            )
                            all_data_rows.append(merged_row)
                            current_desc_lines.clear()
                            current_bboxes.clear()

        doc.close()

        # Re-index sequential global row indices
        for idx, r in enumerate(all_data_rows):
            r.row_index_global = idx

        ts = TableStructure(
            document_id=self.document_id,
            table_id=f"table_{array_name}",
            header_row_count=1,
            page_span=page_span,
            rows=all_data_rows,
            data_row_count=len(all_data_rows),
            data_row_index_offset=0,
        )
        return ts

    def map_schema_index_to_row(
        self,
        schema_index: int,
        array_name: str = "holdings",
        known_good_offset: int | None = None,
    ) -> TableRow | None:
        """Map schema index N to physical TableRow."""
        ts = self.get_table_structure_for_array(array_name)
        if not ts or not ts.rows:
            return None

        offset = known_good_offset if known_good_offset is not None else ts.data_row_index_offset
        target_idx = offset + schema_index

        if target_idx < 0 or target_idx >= ts.data_row_count:
            return None

        return ts.rows[target_idx]

    def filter_candidates_by_row(
        self,
        row: TableRow,
        candidates: Sequence[Any],
        tolerance: float = 0.008,
    ) -> list[Any]:
        """Row-constrained candidate filter."""
        filtered = []
        for cand in candidates:
            cand_page = getattr(cand, "page", None)
            if cand_page is not None and cand_page != row.page_index:
                continue

            cand_bbox = getattr(cand, "bbox", None)
            if cand_bbox is None:
                continue

            cy = cand_bbox.y + (cand_bbox.height / 2.0)
            cx = cand_bbox.x + (cand_bbox.width / 2.0)
            if (row.y_top - tolerance) <= cy <= (row.y_bottom + tolerance):
                if (row.bbox[0] - 0.04) <= cx <= (row.bbox[0] + row.bbox[2] + 0.04):
                    filtered.append(cand)

        filtered.sort(key=lambda c: getattr(c.bbox, "x", 0.0))
        return filtered

    def resolve_field_via_row_mapping(
        self,
        field_path: str,
        target_value: Any,
        page_obj: DocumentPage | None,
        existing_tokens: Sequence[DocumentToken] | None = None,
    ) -> tuple[BBox | None, str | None, str | None]:
        """Resolve citation bbox using row index mapping."""
        parsed = self.parse_array_field_path(field_path)
        if not parsed:
            return None, None, "not_array_pattern"

        array_name, schema_idx, sub_field = parsed
        mapped_row = self.map_schema_index_to_row(schema_idx, array_name=array_name)
        if not mapped_row:
            return None, None, "out_of_bounds_or_no_table"

        if page_obj and page_obj.page_number != mapped_row.page_index:
            return None, None, "different_page"

        target_str = str(target_value).strip().lower()
        if not target_str:
            return None, None, "empty_target"

        tokens = existing_tokens if existing_tokens is not None else (page_obj.tokens if page_obj else [])
        row_candidates = []

        cleaned_target = target_str.replace("$", "").replace(",", "").replace("%", "").strip()

        for tok in tokens:
            tb = tok.bbox
            cy = tb.y + (tb.height / 2.0)
            cx = tb.x + (tb.width / 2.0)

            # Strict spatial constraint: must fall within mapped row's y-window and x-column span
            if (mapped_row.y_top - 0.006) <= cy <= (mapped_row.y_bottom + 0.006):
                if (mapped_row.bbox[0] - 0.04) <= cx <= (mapped_row.bbox[0] + mapped_row.bbox[2] + 0.04):
                    t_text = tok.text.strip().lower()
                    cleaned_tok = t_text.replace("$", "").replace(",", "").replace("%", "").strip()

                    # Exact token match
                    if cleaned_target == cleaned_tok or target_str == t_text:
                        row_candidates.append(tok)
                    # Substring or high similarity
                    elif cleaned_target and cleaned_target in cleaned_tok:
                        row_candidates.append(tok)
                    elif fuzz.ratio(target_str, t_text) >= 80:
                        row_candidates.append(tok)

        if not row_candidates:
            return None, None, "no_matching_token_in_row"

        rx = min(t.bbox.x for t in row_candidates)
        ry = min(t.bbox.y for t in row_candidates)
        rw = max(t.bbox.x + t.bbox.width for t in row_candidates) - rx
        rh = max(t.bbox.y + t.bbox.height for t in row_candidates) - ry
        matched_text = " ".join(t.text for t in row_candidates)

        return BBox(rx, ry, rw, rh, mapped_row.page_index), matched_text, None
