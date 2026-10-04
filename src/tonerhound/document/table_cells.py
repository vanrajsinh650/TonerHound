"""PyMuPDF native table cell extractor and cell grounding.

Target: Solves table column bleed by extracting discrete table cells
via page table detection and bounding fields to exact matching cell rectangles.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import fitz

_CLEAN_NUM = re.compile(r"[,$€£¥\s]")


def compute_iou_xywh(
    b1: tuple[float, float, float, float] | Sequence[float],
    b2: tuple[float, float, float, float] | Sequence[float],
) -> float:
    """Compute IoU between two [x, y, w, h] bounding boxes."""
    if len(b1) != 4 or len(b2) != 4:
        return 0.0

    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, w1 * h1)
    area2 = max(0.0, w2 * h2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def normalize_val(val_str: str) -> str:
    """Normalize string for table cell value comparison."""
    cleaned = _CLEAN_NUM.sub("", val_str.strip().lower())
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    return cleaned


class TableCellExtractor:
    """Extracts and grounds table cells across PDF pages."""

    def __init__(self) -> None:
        self._cell_cache: dict[tuple[str, int], list[tuple[tuple[float, float, float, float], str]]] = {}
        self._table_count_cache: dict[tuple[str, int], int] = {}

    def extract_cells_for_page(
        self,
        doc: fitz.Document,
        pdf_path: str,
        page_num: int,
    ) -> list[tuple[tuple[float, float, float, float], str]]:
        """Extract and cache all cell bboxes and text for a given page."""
        key = (pdf_path, page_num)
        if key in self._cell_cache:
            return self._cell_cache[key]

        if page_num < 1 or page_num > len(doc):
            self._cell_cache[key] = []
            self._table_count_cache[key] = 0
            return []

        page = doc[page_num - 1]
        pw = page.rect.width
        ph = page.rect.height
        if pw <= 0 or ph <= 0:
            self._cell_cache[key] = []
            self._table_count_cache[key] = 0
            return []

        try:
            tabs = page.find_tables(strategy="lines_strict")
            if not tabs or len(tabs.tables) == 0:
                tabs = page.find_tables()
        except Exception:
            try:
                tabs = page.find_tables()
            except Exception:
                tabs = None

        if not tabs or len(tabs.tables) == 0:
            self._cell_cache[key] = []
            self._table_count_cache[key] = 0
            return []

        self._table_count_cache[key] = len(tabs.tables)
        cells_data: list[tuple[tuple[float, float, float, float], str]] = []

        for tab in tabs:
            for cell in tab.cells:
                x0, y0, x1, y1 = cell
                nx = max(0.0, min(1.0, x0 / pw))
                ny = max(0.0, min(1.0, y0 / ph))
                nw = max(0.0, min(1.0, (x1 - x0) / pw))
                nh = max(0.0, min(1.0, (y1 - y0) / ph))
                norm_box = (round(nx, 6), round(ny, 6), round(nw, 6), round(nh, 6))

                try:
                    cell_text = page.get_text("text", clip=cell).strip()
                except Exception:
                    cell_text = ""

                if cell_text:
                    cells_data.append((norm_box, cell_text))

        self._cell_cache[key] = cells_data
        return cells_data

    def ground_field_in_table(
        self,
        doc: fitz.Document,
        pdf_path: str,
        page_num: int,
        gold_value: Any,
        gold_bbox: tuple[float, float, float, float] | Sequence[float] | None = None,
    ) -> tuple[tuple[float, float, float, float] | None, float]:
        """Attempt to ground a field in a table cell."""
        if gold_value is None:
            return None, 0.0

        str_val = str(gold_value).strip()
        if not str_val:
            return None, 0.0

        norm_target = normalize_val(str_val)
        if not norm_target:
            return None, 0.0

        cells = self.extract_cells_for_page(doc, pdf_path, page_num)
        if not cells:
            return None, 0.0

        best_box: tuple[float, float, float, float] | None = None
        best_score = 0.0
        best_iou = 0.0

        for cell_box, cell_text in cells:
            norm_cell = normalize_val(cell_text)
            if not norm_cell:
                continue

            match_score = 0.0
            if norm_cell == norm_target:
                match_score = 1.0
            elif norm_target in norm_cell or norm_cell in norm_target:
                match_score = 0.85

            if match_score > 0:
                iou = 0.0
                if gold_bbox:
                    iou = compute_iou_xywh(gold_bbox, cell_box)

                total_score = match_score + iou
                if total_score > best_score:
                    best_score = total_score
                    best_box = cell_box
                    best_iou = iou

        return best_box, best_iou


TableCellGrounder = TableCellExtractor
