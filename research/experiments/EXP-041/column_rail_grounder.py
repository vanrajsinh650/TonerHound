"""EXP-041 Phase A: Column Rail Constraint Grounder.

Target class: REAL_INDEXING_MISS / SUB_COLUMN_DRIFT (7,804 fields).
Realistic gain: +0.99 pp.

In dense financial schedules, identical values (USD, 0.00, 1,000) drift horizontally
into adjacent column cells. Restricts candidate selection to the expected vertical column rail.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
import fitz

from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentToken


def extract_column_rails(tab: Any, pw: float) -> list[tuple[float, float]]:
    """Extract normalized (x_left, x_right) column rails from a PyMuPDF Table object."""
    if not tab or not tab.cells:
        return []

    pw = float(pw) if pw > 0 else 1.0
    x_coords = set()
    for cell in tab.cells:
        x0 = round(cell[0] / pw, 4)
        x1 = round(cell[2] / pw, 4)
        if 0.0 <= x0 <= 1.0:
            x_coords.add(x0)
        if 0.0 <= x1 <= 1.0:
            x_coords.add(x1)

    sorted_x = sorted(x_coords)
    columns: list[tuple[float, float]] = []
    for i in range(len(sorted_x) - 1):
        if sorted_x[i + 1] - sorted_x[i] > 0.01:
            columns.append((sorted_x[i], sorted_x[i + 1]))

    return columns


def infer_column_from_field_name(field_name: str, tab: Any) -> int | None:
    """Map field name to column index using keyword matching against table header text."""
    if not tab:
        return None

    header_names = [str(n).lower() for n in tab.header.names] if getattr(tab, "header", None) and tab.header.names else []
    fn_lower = field_name.lower()

    mappings = {
        "par_currency": ["currency", "par", "ccy"],
        "par_thousands": ["par", "thousands", "principal", "amount"],
        "coupon_percent": ["coupon", "rate", "interest", "%"],
        "maturity_date": ["maturity", "date", "due"],
        "shares": ["shares", "contracts"],
        "value": ["value", "fair", "market", "cost", "basis"],
        "tranche_class": ["class", "tranche", "series"],
        "principal": ["principal", "amount", "par"],
        "description": ["description", "security", "issuer", "name"],
    }

    for key, keywords in mappings.items():
        if key in fn_lower:
            for i, col_name in enumerate(header_names):
                if any(kw in col_name for kw in keywords):
                    return i

    return None


class ColumnRailGrounder:
    """Restricts candidate tokens to expected column rails on tabular pages."""

    def __init__(self) -> None:
        self._table_cache: dict[tuple[str, int], list[Any]] = {}

    def ground_field_in_column(
        self,
        page: fitz.Page,
        pdf_path: str,
        page_num: int,
        field_name: str,
        gold_value: Any,
        already_grounded_neighbors: dict[str, Sequence[float]] | None = None,
    ) -> list[float] | None:
        """Ground a field by filtering candidate tokens to its expected column rail."""
        if gold_value is None or page is None:
            return None

        val_str = str(gold_value).strip()
        if not val_str:
            return None

        pw = float(page.rect.width) if page.rect.width > 0 else 1.0
        ph = float(page.rect.height) if page.rect.height > 0 else 1.0

        cache_key = (pdf_path, page_num)
        if cache_key not in self._table_cache:
            try:
                tables = page.find_tables(strategy="lines_strict")
                self._table_cache[cache_key] = tables.tables if tables else []
            except Exception:
                self._table_cache[cache_key] = []

        tabs = self._table_cache[cache_key]
        if not tabs:
            return None

        tab = max(tabs, key=lambda t: len(t.cells))
        columns = extract_column_rails(tab, pw)
        if not columns:
            return None

        col_idx = infer_column_from_field_name(field_name, tab)
        if col_idx is None or col_idx >= len(columns):
            return None

        x_left, x_right = columns[col_idx]

        # Extract words from page
        raw_words = page.get_text("words")
        val_clean = val_str.lower()

        candidates: list[list[float]] = []
        for w in raw_words:
            x0, y0, x1, y1, text, bno, lno, wno = w
            if text.strip().lower() == val_clean or val_clean == text.strip().lower().replace(",", ""):
                norm_x = x0 / pw
                norm_y = y0 / ph
                norm_w = (x1 - x0) / pw
                norm_h = (y1 - y0) / ph
                cx = norm_x + norm_w / 2.0

                # Check column rail boundary
                if (x_left - 0.01) <= cx <= (x_right + 0.01):
                    candidates.append([round(norm_x, 6), round(norm_y, 6), round(norm_w, 6), round(norm_h, 6)])

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates: use row anchor from already-grounded neighbors
        if already_grounded_neighbors:
            valid_ys = [b[1] for b in already_grounded_neighbors.values() if b and len(b) >= 2]
            if valid_ys:
                expected_y = sum(valid_ys) / len(valid_ys)
                best = min(candidates, key=lambda c: abs(c[1] - expected_y))
                return best

        return candidates[0]
