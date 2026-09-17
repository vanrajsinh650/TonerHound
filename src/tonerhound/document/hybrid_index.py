"""Hybrid document indexing backend for TonerHound.

Combines high-speed layout-aware parsing (LiteParse) on digital PDF pages
with TonerHound's custom PDFium + Tesseract OCR pipeline (domain calibrations,
repair_ocr_text, PSM 11 sparse fallback) on scanned or corrupted pages.

Fully implements the DocumentIndex interface so ExtractBenchAdapter and all
downstream resolvers consume it with zero changes to adapter logic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pypdfium2 as pdfium

from tonerhound.document.index import (
    DocumentIndex,
    _cluster_tokens_into_lines,
    _extract_tokens_from_ocr,
    _extract_tokens_from_page,
)
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine
from tonerhound.normalization.normalizers import _CHAR_REPLACEMENTS

try:
    from liteparse import LiteParse
    _LITEPARSE_AVAILABLE = True
except ImportError:
    LiteParse = None  # type: ignore[assignment, misc]
    _LITEPARSE_AVAILABLE = False


@dataclass(slots=True)
class LayoutBlockHint:
    """Layout block hint describing structural elements (table, paragraph, heading, etc.)."""

    kind: str  # e.g. "table", "paragraph", "heading", "figure", "list_item", "header", "footer"
    bbox: BBox
    page: int
    text: str | None = None
    rows: list[list[str]] | None = None
    cell_bboxes: list[list[BBox]] | None = None
    id: str | None = None
    level: int | None = None

    @property
    def is_table(self) -> bool:
        return self.kind.lower() == "table"

    @property
    def is_paragraph(self) -> bool:
        return self.kind.lower() == "paragraph"

    @property
    def is_heading(self) -> bool:
        return self.kind.lower() in ("heading", "header", "title")


def _convert_rect_to_bbox(
    rect: Any,
    page_width: float,
    page_height: float,
    page_num: int,
) -> BBox:
    """Convert a top-left point rect (from LiteParse) to a normalized [0, 1] BBox."""
    if rect is not None and page_width > 0 and page_height > 0:
        rx = getattr(rect, "x", 0.0)
        ry = getattr(rect, "y", 0.0)
        rw = getattr(rect, "width", 0.0)
        rh = getattr(rect, "height", 0.0)
        nx = max(0.0, min(1.0, rx / page_width))
        ny = max(0.0, min(1.0, ry / page_height))
        nw = max(0.0, min(1.0 - nx, rw / page_width))
        nh = max(0.0, min(1.0 - ny, rh / page_height))
        return BBox(x=nx, y=ny, width=nw, height=nh, page=page_num)
    return BBox(x=0.0, y=0.0, width=1.0, height=1.0, page=page_num)


def _extract_tokens_and_blocks_from_liteparse_page(
    lp_page: Any,
    page_num: int,
    page_width: float,
    page_height: float,
) -> tuple[list[DocumentToken], list[VisualLine], list[LayoutBlockHint]]:
    """Extract tokens, lines, and layout block hints from a LiteParse ParsedPage."""
    layout_blocks: list[LayoutBlockHint] = []
    raw_blocks = getattr(lp_page, "blocks", None) or []

    for b_idx, b in enumerate(raw_blocks):
        b_kind = getattr(b, "kind", "block") or "block"
        b_rect = getattr(b, "bbox", None)
        block_bbox = _convert_rect_to_bbox(b_rect, page_width, page_height, page_num)

        b_text = getattr(b, "text", None)
        b_id = getattr(b, "id", None)
        b_level = getattr(b, "level", None)

        # Table rows and cell bboxes
        rows_text: list[list[str]] | None = None
        cell_bboxes: list[list[BBox]] | None = None
        raw_rows = getattr(b, "rows", None)
        if raw_rows is not None:
            rows_text = []
            cell_bboxes = []
            for row in raw_rows:
                r_text: list[str] = []
                r_boxes: list[BBox] = []
                for cell in row:
                    r_text.append(getattr(cell, "text", "") or "")
                    c_rect = getattr(cell, "bbox", None)
                    r_boxes.append(_convert_rect_to_bbox(c_rect, page_width, page_height, page_num))
                rows_text.append(r_text)
                cell_bboxes.append(r_boxes)

        hint = LayoutBlockHint(
            kind=b_kind,
            bbox=block_bbox,
            page=page_num,
            text=b_text,
            rows=rows_text,
            cell_bboxes=cell_bboxes,
            id=b_id,
            level=b_level,
        )
        layout_blocks.append(hint)

    # Word token extraction
    raw_tokens: list[DocumentToken] = []
    char_idx = 0

    for item in getattr(lp_page, "text_items", []):
        words = getattr(item, "words", None)
        if words:
            for w in words:
                w_text = getattr(w, "text", "") or ""
                clean = unicodedata.normalize("NFKC", w_text.strip())
                for k, v in _CHAR_REPLACEMENTS.items():
                    clean = clean.replace(k, v)
                if not clean:
                    continue

                tb = _convert_rect_to_bbox(w, page_width, page_height, page_num)
                if tb.width <= 0.0 or tb.height <= 0.0:
                    continue

                raw_tokens.append(
                    DocumentToken(
                        text=clean,
                        bbox=tb,
                        page=page_num,
                        char_index_in_page=char_idx,
                    )
                )
                char_idx += len(clean) + 1
        else:
            item_text = getattr(item, "text", "") or ""
            item_words = item_text.split()
            if not item_words:
                continue

            ix = max(0.0, min(1.0, getattr(item, "x", 0.0) / page_width))
            iy = max(0.0, min(1.0, getattr(item, "y", 0.0) / page_height))
            iw = max(0.0, min(1.0 - ix, getattr(item, "width", 0.0) / page_width))
            ih = max(0.0, min(1.0 - iy, getattr(item, "height", 0.0) / page_height))

            if len(item_words) == 1:
                clean = unicodedata.normalize("NFKC", item_words[0].strip())
                for k, v in _CHAR_REPLACEMENTS.items():
                    clean = clean.replace(k, v)
                if clean and iw > 0 and ih > 0:
                    raw_tokens.append(
                        DocumentToken(
                            text=clean,
                            bbox=BBox(x=ix, y=iy, width=iw, height=ih, page=page_num),
                            page=page_num,
                            char_index_in_page=char_idx,
                        )
                    )
                    char_idx += len(clean) + 1
            else:
                total_chars = sum(len(w) for w in item_words)
                cur_x = ix
                for iw_str in item_words:
                    clean = unicodedata.normalize("NFKC", iw_str.strip())
                    for k, v in _CHAR_REPLACEMENTS.items():
                        clean = clean.replace(k, v)
                    word_w = iw * (len(iw_str) / max(1, total_chars))
                    if clean and word_w > 0 and ih > 0:
                        raw_tokens.append(
                            DocumentToken(
                                text=clean,
                                bbox=BBox(x=cur_x, y=iy, width=min(word_w, 1.0 - cur_x), height=ih, page=page_num),
                                page=page_num,
                                char_index_in_page=char_idx,
                            )
                        )
                        char_idx += len(clean) + 1
                    cur_x += word_w

    # Associate tokens with enclosing layout blocks
    tokens_with_block: list[DocumentToken] = []
    for tok in raw_tokens:
        tcx = tok.bbox.x + tok.bbox.width / 2.0
        tcy = tok.bbox.y + tok.bbox.height / 2.0
        best_block_idx = 0
        best_area = float("inf")

        for b_i, b in enumerate(layout_blocks, start=1):
            if (
                (b.bbox.x0 - 1e-4 <= tcx <= b.bbox.x1 + 1e-4)
                and (b.bbox.y0 - 1e-4 <= tcy <= b.bbox.y1 + 1e-4)
                and (b.bbox.area < best_area)
            ):
                best_area = b.bbox.area
                best_block_idx = b_i

        tokens_with_block.append(dataclasses.replace(tok, block_index=best_block_idx))

    # Cluster into visual lines
    lines = _cluster_tokens_into_lines(tokens_with_block, page_num)

    # Set accurate line_index on tokens
    final_tokens: list[DocumentToken] = []
    for line in lines:
        updated_line_tokens = [dataclasses.replace(t, line_index=line.line_index) for t in line.tokens]
        line.tokens = updated_line_tokens
        final_tokens.extend(updated_line_tokens)

    return final_tokens, lines, layout_blocks


def _route_page_mode(
    pdf_page: Any,
    page_num: int,
    lp_page: Any | None,
    pdfium_tokens: list[DocumentToken],
    candidate_tokens: list[DocumentToken],
    candidate_blocks: list[LayoutBlockHint],
    min_digital_tokens: int = 25,
    min_digital_char_density: float = 0.35,
    force_ocr: bool = False,
    force_liteparse: bool = False,
) -> str:
    """Determine the optimal extraction mode for a page: 'ocr', 'pdfium', or 'liteparse'."""
    if force_ocr:
        return "ocr"
    if force_liteparse:
        return "liteparse"

    p_cnt = len(pdfium_tokens)
    lp_cnt = len(candidate_tokens)

    # 1. OCR / Scanned / Garbled check
    needs_ocr = False
    if p_cnt < min_digital_tokens and lp_cnt < min_digital_tokens:
        needs_ocr = True
    elif lp_page is not None:
        complexity = getattr(lp_page, "complexity", None)
        if complexity is not None:
            if getattr(complexity, "is_garbled", False) or getattr(complexity, "full_page_image", False) and lp_cnt < 50:
                needs_ocr = True
            reasons = getattr(complexity, "reasons", []) or []
            if "scanned" in reasons and lp_cnt < 50:
                needs_ocr = True

    if not needs_ocr and lp_cnt > 0 and lp_cnt < 50:
        full_text = "".join(t.text for t in candidate_tokens)
        if full_text:
            n_alnum = sum(1 for c in full_text if c.isalnum())
            n_non_space = sum(1 for c in full_text if not c.isspace())
            density = n_alnum / max(1, n_non_space)
            if density < min_digital_char_density:
                needs_ocr = True

    if needs_ocr:
        return "ocr"

    # 2. LiteParse dropped native text completely -> fallback to PDFium
    if lp_cnt < min_digital_tokens and p_cnt >= min_digital_tokens:
        return "pdfium"

    # 3. Digital Page Quality & Fragmentation Signals
    tok_ratio = (lp_cnt / max(1, p_cnt)) if p_cnt > 0 else 1.0

    # Signal A: Split decimals (e.g. ".3333333%", ".15)", ".005)")
    split_decimals = sum(
        1 for t in candidate_tokens
        if re.match(r"^[\.,]\d+", t.text)
    )

    # Signal B: Split parenthesized numbers (e.g. "$(0", "(0", "(1", "(8")
    split_parens = sum(
        1 for t in candidate_tokens
        if re.match(r"^[\$\(]+\d+$", t.text)
    )

    # Signal C: Table complexity
    total_cells = 0
    for tb in candidate_blocks:
        if tb.is_table and tb.rows:
            total_cells += sum(len(r) for r in tb.rows)

    # Signal D: Tax form / boxed schedule detection
    p_text_sample = " ".join(t.text for t in pdfium_tokens[:60]).lower()
    is_tax_form = bool(
        re.search(
            r"schedule\s*k-1|form\s*1065|form\s*1120-s|partner's\s*share|form\s*1040",
            p_text_sample,
        )
    )

    # Fine-grained routing rules:
    # Rule 1: Fixed tax form with decimal or micro-block fragmentation
    if is_tax_form and (split_decimals > 0 or tok_ratio > 1.01):
        return "pdfium"

    # Rule 2: Heavy decimal / financial parenthesis fragmentation
    if (split_decimals >= 5 or (split_decimals >= 2 and split_parens >= 2)) and tok_ratio >= 1.05:
        return "pdfium"

    # Rule 3: Moderate decimal fragmentation on non-table pages
    if split_decimals >= 2 and total_cells == 0 and tok_ratio > 1.03:
        return "pdfium"

    return "liteparse"


def _should_fallback_to_ocr(
    lp_page: Any | None,
    tokens: list[DocumentToken],
    min_digital_tokens: int = 25,
    min_digital_char_density: float = 0.35,
    force_ocr: bool = False,
    force_liteparse: bool = False,
) -> bool:
    """Determine whether a page should fall back to TonerHound's OCR pipeline."""
    if force_ocr:
        return True
    if force_liteparse:
        return False
    if lp_page is None:
        return True

    # 1. Token count threshold: pages with negligible native text
    if len(tokens) < min_digital_tokens:
        return True

    # 2. LiteParse page complexity diagnostics
    complexity = getattr(lp_page, "complexity", None)
    if complexity is not None:
        if getattr(complexity, "is_garbled", False):
            return True
        if getattr(complexity, "full_page_image", False) and len(tokens) < 50:
            return True
        reasons = getattr(complexity, "reasons", []) or []
        if "scanned" in reasons and len(tokens) < 50:
            return True

    # 3. Alphanumeric character density check (detect garbled glyph streams)
    full_text = "".join(t.text for t in tokens)
    if full_text:
        n_alnum = sum(1 for c in full_text if c.isalnum())
        n_non_space = sum(1 for c in full_text if not c.isspace())
        density = n_alnum / max(1, n_non_space)
        if density < min_digital_char_density and len(tokens) < 50:
            return True

    return False


class HybridDocumentIndex(DocumentIndex):
    """Hybrid document index combining LiteParse layout parsing with TonerHound OCR fallback.

    - Digital PDF pages: High-speed native parsing via LiteParse, providing word bounding
      boxes and structural layout blocks (tables, headings, paragraphs).
    - Corrupted / scanned pages: Seamless delegation to TonerHound's PDFium + Tesseract OCR
      pipeline with character repair (repair_ocr_text), PSM 11 sparse fallback, and domain calibrations.
    - Implements the complete DocumentIndex contract so ExtractBenchAdapter consumes it
      with zero changes to adapter logic.
    """

    INDEX_VERSION = "hybrid_v2"

    def __init__(
        self,
        pages: list[DocumentPage],
        layout_blocks: list[LayoutBlockHint] | None = None,
        page_modes: dict[int, str] | None = None,
    ) -> None:
        super().__init__(pages)
        self.layout_blocks: list[LayoutBlockHint] = []
        self.layout_blocks_by_page: dict[int, list[LayoutBlockHint]] = defaultdict(list)
        self.page_modes: dict[int, str] = dict(page_modes or {})

        # Ingest layout blocks from argument
        if layout_blocks:
            for b in layout_blocks:
                self.layout_blocks.append(b)
                self.layout_blocks_by_page[b.page].append(b)

        # Ingest layout blocks already stored on page objects
        for p in pages:
            p_blocks = getattr(p, "blocks", None) or []
            for b in p_blocks:
                if isinstance(b, LayoutBlockHint) and b not in self.layout_blocks:
                    self.layout_blocks.append(b)
                    self.layout_blocks_by_page[b.page].append(b)

    def get_layout_blocks(
        self,
        page_num: int | None = None,
        kind: str | None = None,
    ) -> list[LayoutBlockHint]:
        """Retrieve layout blocks, optionally filtered by page number and/or kind."""
        if page_num is not None:
            blocks = self.layout_blocks_by_page.get(page_num, [])
        else:
            blocks = self.layout_blocks

        if kind is not None:
            target_kind = kind.lower().strip()
            return [b for b in blocks if b.kind.lower().strip() == target_kind]
        return list(blocks)

    def get_table_blocks(self, page_num: int | None = None) -> list[LayoutBlockHint]:
        """Retrieve table layout blocks for the document or a specific page."""
        return self.get_layout_blocks(page_num=page_num, kind="table")

    def get_paragraph_blocks(self, page_num: int | None = None) -> list[LayoutBlockHint]:
        """Retrieve paragraph layout blocks for the document or a specific page."""
        return self.get_layout_blocks(page_num=page_num, kind="paragraph")

    def get_heading_blocks(self, page_num: int | None = None) -> list[LayoutBlockHint]:
        """Retrieve heading layout blocks for the document or a specific page."""
        return [b for b in self.get_layout_blocks(page_num=page_num) if b.is_heading]

    def find_blocks_at(self, page_num: int, bbox: BBox) -> list[LayoutBlockHint]:
        """Find layout blocks overlapping a specific bounding box on a page."""
        page_blocks = self.layout_blocks_by_page.get(page_num, [])
        overlapping: list[LayoutBlockHint] = []
        for b in page_blocks:
            inter = b.bbox.intersection(bbox)
            if inter is not None and inter.area > 0.0:
                overlapping.append(b)
        return overlapping

    @classmethod
    def from_pdf(
        cls,
        source: str | Path | bytes | BinaryIO,
        enable_ocr: bool = True,
        ocr_scale: float = 200.0 / 72.0,
        ocr_token_threshold: int = 25,
        min_digital_tokens: int = 25,
        min_digital_char_density: float = 0.35,
        force_ocr: bool = False,
        force_liteparse: bool = False,
        max_pages: int | None = None,
        use_cache: bool = True,
        cache_dir: Path | str = "research/cache/document_index",
    ) -> HybridDocumentIndex:
        """Load and index a PDF using the hybrid LiteParse + TonerHound OCR pipeline with persistent caching."""
        # 1. Disk Cache check
        cache_path: Path | None = None
        source_bytes: bytes | None = None

        if isinstance(source, (bytes, bytearray)):
            source_bytes = bytes(source)
        elif hasattr(source, "read") and callable(source.read):
            source_bytes = source.read()

        if use_cache:
            h = hashlib.sha256()
            if source_bytes is not None:
                h.update(source_bytes)
            elif isinstance(source, (str, Path)):
                src_path = Path(source)
                if src_path.exists():
                    with open(src_path, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
            file_hash = h.hexdigest()
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = f"{file_hash}_{enable_ocr}_{ocr_scale:.2f}_{min_digital_tokens}_{force_ocr}_{force_liteparse}_{max_pages}_{cls.INDEX_VERSION}"
            cache_path = cache_dir / f"{cache_key}.pkl"

            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached_idx = pickle.load(f)
                        if isinstance(cached_idx, HybridDocumentIndex):
                            return cached_idx
                except (pickle.PickleError, EOFError, OSError, ValueError):
                    pass

        # 2. Parse digital layer with LiteParse (if available and not force_ocr)
        lp_result: Any | None = None
        lp_pages_by_num: dict[int, Any] = {}

        if _LITEPARSE_AVAILABLE and LiteParse is not None and not force_ocr:
            try:
                lp_parser = LiteParse(
                    ocr_enabled=False,
                    emit_word_boxes=True,
                    extract_blocks=True,
                    include_complexity=True,
                    max_pages=max_pages,
                )
                lp_target = source_bytes if source_bytes is not None else str(source)
                lp_result = lp_parser.parse(lp_target)
                if lp_result and hasattr(lp_result, "pages"):
                    for p in lp_result.pages:
                        lp_pages_by_num[p.page_num] = p
            except Exception:
                # Graceful degradation on any LiteParse error
                lp_result = None
                lp_pages_by_num.clear()

        # 3. Open with PDFium for exact geometry, page count, and OCR rendering
        pdf_target = source_bytes if source_bytes is not None else source
        doc = pdfium.PdfDocument(pdf_target)
        pages: list[DocumentPage] = []
        all_layout_blocks: list[LayoutBlockHint] = []
        page_modes: dict[int, str] = {}

        # Pre-load cached baseline DocumentIndex if present to accelerate OCR fallback
        cached_baseline_doc: DocumentIndex | None = None
        if use_cache and file_hash is not None:
            base_cache_path = Path(cache_dir) / f"{file_hash}_True_{ocr_scale:.2f}_{DocumentIndex.INDEX_VERSION}.pkl"
            if base_cache_path.exists():
                try:
                    with open(base_cache_path, "rb") as f:
                        loaded = pickle.load(f)
                        if isinstance(loaded, DocumentIndex):
                            cached_baseline_doc = loaded
                except Exception:
                    cached_baseline_doc = None

        try:
            total_doc_pages = len(doc)
            num_pages_to_process = min(total_doc_pages, max_pages) if max_pages else total_doc_pages

            for page_idx in range(num_pages_to_process):
                page_num = page_idx + 1  # 1-indexed
                pdf_page = doc[page_idx]
                width, height = pdf_page.get_size()

                # Extract PDFium native tokens
                pdfium_tokens = _extract_tokens_from_page(pdf_page, page_num, width, height)

                lp_page = lp_pages_by_num.get(page_num)
                candidate_tokens: list[DocumentToken] = []
                candidate_lines: list[VisualLine] = []
                candidate_blocks: list[LayoutBlockHint] = []

                if lp_page is not None:
                    candidate_tokens, candidate_lines, candidate_blocks = (
                        _extract_tokens_and_blocks_from_liteparse_page(
                            lp_page=lp_page,
                            page_num=page_num,
                            page_width=width,
                            page_height=height,
                        )
                    )

                # 4. Decision Gate: Tri-mode routing ('ocr', 'pdfium', 'liteparse')
                selected_mode = _route_page_mode(
                    pdf_page=pdf_page,
                    page_num=page_num,
                    lp_page=lp_page,
                    pdfium_tokens=pdfium_tokens,
                    candidate_tokens=candidate_tokens,
                    candidate_blocks=candidate_blocks,
                    min_digital_tokens=min_digital_tokens,
                    min_digital_char_density=min_digital_char_density,
                    force_ocr=force_ocr,
                    force_liteparse=force_liteparse,
                )

                if selected_mode == "ocr" and enable_ocr:
                    # OCR Fallback path (PDFium bitmap render + Tesseract + repair_ocr_text + PSM 11 fallback)
                    base_page = cached_baseline_doc.get_page(page_num) if cached_baseline_doc is not None else None
                    if base_page is not None and len(base_page.tokens) > 0:
                        ocr_tokens = base_page.tokens
                        ocr_lines = base_page.lines
                    else:
                        ocr_tokens = _extract_tokens_from_ocr(
                            pdf_page,
                            page_num=page_num,
                            scale=ocr_scale,
                        )
                        ocr_lines = _cluster_tokens_into_lines(ocr_tokens, page_num)

                    page_modes[page_num] = "ocr"
                    # CRITICAL: Preserve canonical OCR token order! Never re-flatten OCR lines horizontally.
                    page_obj = DocumentPage(
                        page_number=page_num,
                        width=width,
                        height=height,
                        tokens=ocr_tokens,
                        lines=ocr_lines,
                        blocks=candidate_blocks,  # retain any structural blocks if found
                    )
                    pages.append(page_obj)
                    all_layout_blocks.extend(candidate_blocks)

                elif selected_mode == "pdfium":
                    # PDFium Native path: clusters tokens into lines, preserves atomic word tokens & reading order
                    p_lines = _cluster_tokens_into_lines(pdfium_tokens, page_num)
                    page_modes[page_num] = "pdfium"
                    page_obj = DocumentPage(
                        page_number=page_num,
                        width=width,
                        height=height,
                        tokens=pdfium_tokens,
                        lines=p_lines,
                        blocks=candidate_blocks,  # retain structural blocks if found
                    )
                    pages.append(page_obj)
                    all_layout_blocks.extend(candidate_blocks)

                else:
                    # LiteParse native digital path
                    page_modes[page_num] = "liteparse"
                    page_obj = DocumentPage(
                        page_number=page_num,
                        width=width,
                        height=height,
                        tokens=candidate_tokens,
                        lines=candidate_lines,
                        blocks=candidate_blocks,
                    )
                    pages.append(page_obj)
                    all_layout_blocks.extend(candidate_blocks)

        finally:
            doc.close()

        index = cls(
            pages=pages,
            layout_blocks=all_layout_blocks,
            page_modes=page_modes,
        )

        # 5. Write to disk cache atomically
        if cache_path is not None:
            try:
                tmp_path = cache_path.with_suffix(".tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path.replace(cache_path)
            except (pickle.PickleError, OSError, TypeError):
                pass

        return index

    @classmethod
    def from_pages(
        cls,
        pages: list[DocumentPage],
        layout_blocks: list[LayoutBlockHint] | None = None,
        page_modes: dict[int, str] | None = None,
    ) -> HybridDocumentIndex:
        """Construct directly from pre-built DocumentPages (e.g. for unit tests and fixtures)."""
        return cls(
            pages=pages,
            layout_blocks=layout_blocks,
            page_modes=page_modes,
        )


# Alias
HybridIndex = HybridDocumentIndex
