"""EXP-047 Phase C: Table Localizer.

Detects table page ranges for array fields (holdings[N], creditors[N], etc.)
and computes position priors: for field holdings[N].shares, what is the
expected y-position (normalized fraction 0–1 within the table) and x-position
(column fraction)?

Algorithm:
1. Detect table headers on each page (keyword matching against schema field names).
2. Cluster visual lines by y-coordinate to determine row heights.
3. Compute expected_y = row_index / total_rows, expected_x = column_index / n_cols.

Usage:
    loc = TableLocalizer(pdf_path)
    prior = loc.get_position_prior("holdings[42].shares")
    # Returns PositionPrior(expected_y, expected_x, confidence, page_range)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PositionPrior:
    """Expected (y, x) position of a field within its table."""
    expected_y: float        # Normalized 0–1 within table body
    expected_x: float        # Normalized 0–1 (column fraction)
    confidence: float        # 0–1: how well-localized is this table?
    page_start: int          # First page of table (1-indexed)
    page_end: int            # Last page of table (1-indexed)
    total_rows: int          # Estimated data row count
    column_index: int        # 0-indexed column position in table
    n_columns: int           # Estimated number of columns


@dataclass
class TableZone:
    """A detected table region in the document."""
    table_name: str           # e.g. "holdings", "creditors"
    page_start: int           # 1-indexed
    page_end: int             # 1-indexed
    header_bbox: list[float]  # [x, y, w, h] normalized, on page_start
    body_y_start: float       # normalized y where data rows start (on page_start)
    body_y_end: float         # normalized y where data rows end (on page_end)
    estimated_rows: int
    column_headers: list[str] = field(default_factory=list)
    column_x_centers: list[float] = field(default_factory=list)  # normalized x


# ---------------------------------------------------------------------------
# Array field path parser
# ---------------------------------------------------------------------------

_ARRAY_FIELD_RE = re.compile(
    r"^(?P<table>[a-zA-Z0-9_.]+)\[(?P<idx>\d+)\]\.(?P<subfield>.+)$"
)


def parse_array_field(field_path: str) -> tuple[str, int, str] | None:
    """Parse 'holdings[42].shares' → ('holdings', 42, 'shares')."""
    m = _ARRAY_FIELD_RE.match(field_path)
    if m:
        return m.group("table"), int(m.group("idx")), m.group("subfield")
    return None


# ---------------------------------------------------------------------------
# Table keyword registry
# ---------------------------------------------------------------------------

# Maps schema array names → page-level header keywords
_TABLE_KEYWORDS: dict[str, list[str]] = {
    "holdings": [
        "schedule of investments", "holdings", "portfolio", "securities",
        "investments", "portfolio holdings", "schedule of portfolio",
    ],
    "creditors": [
        "creditors", "schedule of creditors", "liabilities", "accounts payable",
    ],
    "transactions": [
        "transactions", "schedule of transactions", "transaction history",
        "trade activity",
    ],
    "grants": [
        "grants", "schedule of grants", "grant recipients", "contributions",
    ],
    "recipients": [
        "recipients", "grantees", "donees",
    ],
    "investments": [
        "investments", "schedule of investments", "investment portfolio",
    ],
    "securities": [
        "securities", "security name",
    ],
    "items": [
        "line items", "items", "description",
    ],
}

# Default keywords for unknown table names
_DEFAULT_KEYWORDS = ["schedule", "list of", "table of", "detail"]

# Column header keyword → normalized subfield name
_COLUMN_HEADER_MAP: dict[str, str] = {
    "shares": ["shares", "quantity", "units", "qty", "number of shares"],
    "security_name": ["name", "security", "issuer", "description", "investment"],
    "value": ["value", "fair value", "market value", "total value"],
    "cost": ["cost", "cost basis", "amortized cost", "original cost"],
    "percentage": ["percentage", "% of", "% net", "percent"],
    "cusip": ["cusip", "cusip no"],
    "coupon": ["coupon", "rate", "interest rate"],
    "maturity": ["maturity", "maturity date", "due date"],
    "par_value": ["par", "par value", "face value", "principal"],
    "amount": ["amount", "principal amount", "total"],
    "address": ["address", "street", "city"],
    "ein": ["ein", "employer", "tax id"],
    "name": ["name", "organization", "recipient", "grantee"],
}


def _keywords_for_table(table_name: str) -> list[str]:
    """Get header keywords for a given table/array name."""
    name_lower = table_name.lower().split(".")[-1]  # handle nested: "data.holdings"
    for key, kws in _TABLE_KEYWORDS.items():
        if key in name_lower or name_lower in key:
            return kws
    return _DEFAULT_KEYWORDS + [name_lower]


def _column_index_for_subfield(subfield: str, column_headers: list[str]) -> int:
    """Find the most likely column index for a given subfield name."""
    if not column_headers:
        return 0

    subfield_lower = subfield.lower()

    # Exact match first
    for i, header in enumerate(column_headers):
        if subfield_lower == header.lower():
            return i

    # Keyword match
    for canonical, aliases in _COLUMN_HEADER_MAP.items():
        if subfield_lower == canonical or any(a in subfield_lower for a in aliases):
            # Find column header matching any alias
            for i, header in enumerate(column_headers):
                h = header.lower()
                if canonical in h or any(a in h for a in aliases):
                    return i

    # Fuzzy fallback
    best_score, best_idx = 0, 0
    for i, header in enumerate(column_headers):
        score = fuzz.partial_ratio(subfield_lower, header.lower())
        if score > best_score:
            best_score, best_idx = score, i

    return best_idx


# ---------------------------------------------------------------------------
# TableLocalizer
# ---------------------------------------------------------------------------

class TableLocalizer:
    """Detects table regions in a PDF and computes per-field position priors.

    Uses PyMuPDF for fast text block extraction.
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        self._doc: fitz.Document | None = None
        self._zones: dict[str, TableZone] = {}  # table_name → TableZone
        self._n_tables: int = 0
        self._build()

    def __del__(self) -> None:
        try:
            if self._doc:
                self._doc.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Open the PDF and detect all table zones."""
        try:
            self._doc = fitz.open(str(self.pdf_path))
        except Exception:
            return

        n_pages = len(self._doc)

        # Collect page-level text blocks
        page_blocks: list[list[dict]] = []
        for page_idx in range(n_pages):
            page = self._doc[page_idx]
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            page_blocks.append(blocks.get("blocks", []))

        # Detect table headers
        detected: dict[str, TableZone] = {}

        for page_idx, blocks in enumerate(page_blocks):
            page = self._doc[page_idx]
            pw, ph = page.rect.width, page.rect.height
            page_num = page_idx + 1

            for block in blocks:
                if block.get("type") != 0:  # text block
                    continue
                block_text = " ".join(
                    span.get("text", "")
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ).strip().lower()

                if len(block_text) < 5:
                    continue

                # Try to match known table names
                for table_name, keywords in _TABLE_KEYWORDS.items():
                    for kw in keywords:
                        if kw in block_text or fuzz.partial_ratio(kw, block_text) >= 80:
                            if table_name not in detected:
                                bbox = block.get("bbox", (0, 0, pw, ph))
                                x_n = bbox[0] / pw
                                y_n = bbox[1] / ph
                                w_n = (bbox[2] - bbox[0]) / pw
                                h_n = (bbox[3] - bbox[1]) / ph
                                detected[table_name] = TableZone(
                                    table_name=table_name,
                                    page_start=page_num,
                                    page_end=page_num,
                                    header_bbox=[x_n, y_n, w_n, h_n],
                                    body_y_start=y_n + h_n,
                                    body_y_end=1.0,
                                    estimated_rows=0,
                                )
                            break

        # Extend table zones to their end pages and estimate row counts
        for table_name, zone in detected.items():
            self._extend_zone(zone, page_blocks)

        self._zones = detected
        self._n_tables = len(detected)

    def _extend_zone(
        self,
        zone: TableZone,
        page_blocks: list[list[dict]],
    ) -> None:
        """Extend the table zone to cover all data pages and estimate row count."""
        if self._doc is None:
            return

        n_pages = len(self._doc)
        # Look for "end of table" markers up to 10 pages forward
        end_page = min(zone.page_start + 100, n_pages)
        total_rows = 0
        col_headers: list[str] = []

        for page_idx in range(zone.page_start - 1, end_page):
            if page_idx >= len(page_blocks):
                break
            page = self._doc[page_idx]
            pw, ph = page.rect.width, page.rect.height
            blocks = page_blocks[page_idx]

            # Look for data rows: blocks that are narrow and repeated
            data_y_positions: list[float] = []
            page_col_headers: list[str] = []

            for block in blocks:
                if block.get("type") != 0:
                    continue
                bbox = block.get("bbox", (0, 0, pw, ph))
                y_mid = ((bbox[1] + bbox[3]) / 2) / ph

                # Skip header area (first 15% of page typically)
                if y_mid < 0.15 and page_idx == zone.page_start - 1:
                    # Detect column headers
                    block_text = " ".join(
                        span.get("text", "")
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    ).strip()
                    if block_text:
                        page_col_headers.append(block_text)
                    continue

                # Count lines as data rows
                n_lines = sum(1 for _ in block.get("lines", []))
                data_y_positions.extend(
                    [(bbox[1] + bbox[3]) / 2 / ph] * n_lines
                )

            if not col_headers and page_col_headers:
                col_headers = page_col_headers

            if data_y_positions:
                total_rows += len(data_y_positions)
                zone.page_end = page_idx + 1
                zone.body_y_end = max(data_y_positions)

            # Check for table end markers (summary lines)
            page_full_text = " ".join(
                span.get("text", "")
                for block in blocks if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).lower()
            if any(
                marker in page_full_text
                for marker in ["total investments", "end of schedule", "subtotal", "grand total"]
            ) and page_idx > zone.page_start:
                zone.page_end = page_idx + 1
                break

        zone.estimated_rows = max(1, total_rows)
        zone.column_headers = col_headers
        zone.column_x_centers = self._detect_column_centers(zone)

    def _detect_column_centers(self, zone: TableZone) -> list[float]:
        """Estimate normalized x-centers of columns from first data page."""
        if self._doc is None or zone.page_start > len(self._doc):
            return []

        page_idx = zone.page_start - 1
        page = self._doc[page_idx]
        pw = page.rect.width

        blocks = page.get_text("dict").get("blocks", [])
        x_centers: list[float] = []

        for block in blocks:
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox", (0, 0, pw, 1))
            xc = ((bbox[0] + bbox[2]) / 2) / pw
            if xc not in x_centers:
                x_centers.append(xc)

        x_centers.sort()
        return x_centers

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_position_prior(self, field_path: str) -> PositionPrior | None:
        """Compute a position prior for an array field.

        Args:
            field_path: e.g. "holdings[42].shares"

        Returns:
            PositionPrior or None if field is not an array or no table found.
        """
        parsed = parse_array_field(field_path)
        if parsed is None:
            return None

        table_name, array_idx, subfield = parsed

        # Find matching zone
        zone = self._find_zone(table_name)
        if zone is None:
            return None

        total_rows = max(1, zone.estimated_rows)
        # Clamp array index to table size
        safe_idx = min(array_idx, total_rows - 1)
        expected_y = safe_idx / total_rows

        # Find column index
        col_idx = _column_index_for_subfield(subfield, zone.column_headers)
        n_cols = max(1, len(zone.column_x_centers))
        expected_x = col_idx / n_cols if n_cols > 1 else 0.5

        # If we have column x_centers, use them
        if zone.column_x_centers and col_idx < len(zone.column_x_centers):
            expected_x = zone.column_x_centers[col_idx]

        confidence = 0.8 if zone.estimated_rows > 10 else 0.5
        if not zone.column_headers:
            confidence *= 0.6

        return PositionPrior(
            expected_y=expected_y,
            expected_x=expected_x,
            confidence=confidence,
            page_start=zone.page_start,
            page_end=zone.page_end,
            total_rows=total_rows,
            column_index=col_idx,
            n_columns=n_cols,
        )

    def _find_zone(self, table_name: str) -> TableZone | None:
        """Find the best matching zone for a given table name."""
        name_lower = table_name.lower().split(".")[-1]
        # Direct match
        if name_lower in self._zones:
            return self._zones[name_lower]
        # Fuzzy match
        best_score, best_zone = 0, None
        for key, zone in self._zones.items():
            score = fuzz.ratio(name_lower, key)
            if score > best_score:
                best_score, best_zone = score, zone
        if best_score >= 70:
            return best_zone
        return None

    def get_all_zones(self) -> list[TableZone]:
        return list(self._zones.values())

    @property
    def n_tables(self) -> int:
        return self._n_tables

    def fields_with_priors(self, field_paths: list[str]) -> int:
        """Count how many of the given field paths have position priors."""
        return sum(1 for fp in field_paths if self.get_position_prior(fp) is not None)
