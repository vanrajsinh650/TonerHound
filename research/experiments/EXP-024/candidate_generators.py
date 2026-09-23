"""Research-only candidate generators for EXP-024 Candidate Coverage & Perception Ablation.

These generators DO NOT modify production code in src/tonerhound.
They provide research-grade candidate extraction for:
- Multi-token spans (EXP-024B)
- Independent OCR (EXP-024C)
- Table / Layout cells (EXP-024D)
- Union pool with deduplication (EXP-024E)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pypdfium2 as pdfium
import pytesseract
from PIL import Image

from tonerhound.document.index import DocumentIndex
from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.matching.matcher import MatchCandidate
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine
from tonerhound.normalization.normalizers import (
    normalize_unicode_and_case,
    parse_date_value,
    parse_numeric_value,
)


@dataclass(frozen=True, slots=True)
class ResearchCandidate:
    """Research candidate representation with explicit provenance."""
    page: int
    bbox: BBox
    matched_text: str
    normalized_text: str
    source: str
    confidence: float = 1.0
    raw_similarity: float = 1.0
    line_index: int = 0
    tokens: tuple[DocumentToken, ...] = ()


def _normalize(text: str) -> str:
    return normalize_unicode_and_case(text).text.strip()


def _reconstruct_text(tokens: Sequence[DocumentToken]) -> str:
    """Reconstruct text from contiguous tokens with proper punctuation spacing."""
    if not tokens:
        return ""
    text_parts = []
    for i, t in enumerate(tokens):
        raw = t.text
        if i == 0:
            text_parts.append(raw)
        else:
            prev = tokens[i - 1].text
            # No space before punctuation or decimal point or comma
            if raw in {",", ".", ":", ";", "%", ")", "]", "-", "/"}:
                text_parts.append(raw)
            # No space after currency or open brackets
            elif prev in {"$", "€", "£", "¥", "(", "[", "-"}:
                text_parts.append(raw)
            else:
                text_parts.append(" " + raw)
    return "".join(text_parts).strip()


class MultiTokenSpanGenerator:
    """EXP-024B: Contiguous token span candidate generator."""

    def __init__(self, index: DocumentIndex, max_span_tokens: int = 12) -> None:
        self.index = index
        self.max_span_tokens = max_span_tokens
        # Build inverted lookup for reconstructed spans
        self._page_spans: dict[int, list[ResearchCandidate]] = {}

    def extract_candidates(self, query_value: Any, page_hint: int | None = None) -> list[ResearchCandidate]:
        if query_value is None:
            return []
        q_str = str(query_value).strip()
        norm_q = _normalize(q_str)
        if not norm_q:
            return []

        q_num = parse_numeric_value(q_str)
        q_date = parse_date_value(q_str)

        pages = [page_hint] if (page_hint is not None and page_hint in self.index._pages_by_num) else [p.page_number for p in self.index.pages]
        cands: list[ResearchCandidate] = []

        for p_num in pages:
            page = self.index.get_page(p_num)
            if not page:
                continue

            for line_idx, line in enumerate(page.lines):
                tokens = line.tokens
                n_tok = len(tokens)
                if n_tok < 2:
                    continue

                for start in range(n_tok):
                    max_end = min(n_tok, start + self.max_span_tokens)
                    for end in range(start + 2, max_end + 1):
                        span_tokens = tokens[start:end]
                        span_text = _reconstruct_text(span_tokens)
                        norm_span = _normalize(span_text)

                        is_match = False
                        match_score = 0.0

                        if norm_span == norm_q:
                            is_match = True
                            match_score = 1.0
                        elif q_num is not None:
                            s_num = parse_numeric_value(span_text)
                            if s_num is not None and abs(s_num - q_num) < 1e-5:
                                is_match = True
                                match_score = 1.0
                        elif q_date is not None:
                            s_date = parse_date_value(span_text)
                            if s_date is not None and s_date == q_date:
                                is_match = True
                                match_score = 1.0

                        if is_match:
                            u_box = union_bbox_list([t.bbox for t in span_tokens])
                            cands.append(
                                ResearchCandidate(
                                    page=p_num,
                                    bbox=u_box,
                                    matched_text=span_text,
                                    normalized_text=norm_span,
                                    source="multi_token_span",
                                    confidence=match_score,
                                    raw_similarity=match_score,
                                    line_index=line_idx,
                                    tokens=tuple(span_tokens),
                                )
                            )
        return cands


class TesseractOCRGenerator:
    """EXP-024C: Independent OCR candidate generator using Tesseract."""

    def __init__(self, pdf_path: str, dpi: int = 150) -> None:
        self.pdf_path = pdf_path
        self.dpi = dpi
        self._ocr_data_by_page: dict[int, list[dict[str, Any]]] = {}
        self._doc: pdfium.PdfDocument | None = None

    def _ensure_doc(self) -> pdfium.PdfDocument:
        if self._doc is None:
            self._doc = pdfium.PdfDocument(self.pdf_path)
        return self._doc

    def _get_page_ocr(self, page_num: int) -> list[dict[str, Any]]:
        if page_num in self._ocr_data_by_page:
            return self._ocr_data_by_page[page_num]

        doc = self._ensure_doc()
        if page_num < 1 or page_num > len(doc):
            return []

        pdf_page = doc[page_num - 1]
        p_w, p_h = pdf_page.get_size()

        # Render page to image
        scale = self.dpi / 72.0
        pil_img = pdf_page.render(scale=scale).to_pil()

        # Run Tesseract
        try:
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        except Exception:
            self._ocr_data_by_page[page_num] = []
            return []

        n_boxes = len(data["text"])
        words: list[dict[str, Any]] = []
        for i in range(n_boxes):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i])
            if not txt or conf < 20:
                continue

            # Convert image pixels to PDF points
            ix = data["left"][i]
            iy = data["top"][i]
            iw = data["width"][i]
            ih = data["height"][i]

            bx = ix / scale
            by = iy / scale
            bw = iw / scale
            bh = ih / scale

            # Clamp to page
            bx = max(0.0, min(bx, p_w))
            by = max(0.0, min(by, p_h))
            bw = max(1.0, min(bw, p_w - bx))
            bh = max(1.0, min(bh, p_h - by))

            words.append({
                "text": txt,
                "norm_text": _normalize(txt),
                "bbox": BBox(x=bx, y=by, width=bw, height=bh, page=page_num),
                "conf": conf / 100.0,
                "line_num": data["line_num"][i],
            })

        self._ocr_data_by_page[page_num] = words
        return words

    def extract_candidates(self, query_value: Any, page_hint: int | None = None, total_pages: int = 1) -> list[ResearchCandidate]:
        if query_value is None:
            return []
        q_str = str(query_value).strip()
        norm_q = _normalize(q_str)
        if not norm_q:
            return []

        q_num = parse_numeric_value(q_str)
        cands: list[ResearchCandidate] = []
        pages_to_check = [page_hint] if page_hint is not None else list(range(1, total_pages + 1))

        for p_num in pages_to_check:
            words = self._get_page_ocr(p_num)
            for w in words:
                match = False
                if w["norm_text"] == norm_q:
                    match = True
                elif q_num is not None:
                    w_num = parse_numeric_value(w["text"])
                    if w_num is not None and abs(w_num - q_num) < 1e-5:
                        match = True

                if match:
                    cands.append(
                        ResearchCandidate(
                            page=p_num,
                            bbox=w["bbox"],
                            matched_text=w["text"],
                            normalized_text=w["norm_text"],
                            source="ocr_tesseract",
                            confidence=w["conf"],
                            raw_similarity=w["conf"],
                            line_index=w["line_num"],
                        )
                    )
        return cands


class TableLayoutCellGenerator:
    """EXP-024D: Table and layout cell candidate generator."""

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self._page_cells: dict[int, list[tuple[BBox, str, list[DocumentToken]]]] = {}

    def _build_page_cells(self, page_num: int) -> list[tuple[BBox, str, list[DocumentToken]]]:
        if page_num in self._page_cells:
            return self._page_cells[page_num]

        page = self.index.get_page(page_num)
        if not page or not page.lines:
            self._page_cells[page_num] = []
            return []

        # Find columns by clustering token X positions
        tokens = [t for line in page.lines for t in line.tokens]
        if not tokens:
            self._page_cells[page_num] = []
            return []

        # Table cell detection:
        # A cell consists of tokens within a tight horizontal and vertical envelope on a line,
        # or grouped contiguous tokens bounded by significant horizontal whitespace.
        cells: list[tuple[BBox, str, list[DocumentToken]]] = []
        for line in page.lines:
            ltoks = line.tokens
            if not ltoks:
                continue

            # Group tokens separated by > 1.5 average character width into distinct table cells
            current_cell_tokens = [ltoks[0]]
            for i in range(1, len(ltoks)):
                prev = ltoks[i - 1]
                curr = ltoks[i]
                gap = curr.bbox.x - (prev.bbox.x + prev.bbox.width)
                avg_char_w = prev.bbox.width / max(1, len(prev.text))

                if gap > max(8.0, 2.5 * avg_char_w):
                    # Finalize previous cell
                    c_box = union_bbox_list([t.bbox for t in current_cell_tokens])
                    c_text = _reconstruct_text(current_cell_tokens)
                    cells.append((c_box, c_text, list(current_cell_tokens)))
                    current_cell_tokens = [curr]
                else:
                    current_cell_tokens.append(curr)

            if current_cell_tokens:
                c_box = union_bbox_list([t.bbox for t in current_cell_tokens])
                c_text = _reconstruct_text(current_cell_tokens)
                cells.append((c_box, c_text, list(current_cell_tokens)))

        self._page_cells[page_num] = cells
        return cells

    def extract_candidates(self, query_value: Any, page_hint: int | None = None) -> list[ResearchCandidate]:
        if query_value is None:
            return []
        q_str = str(query_value).strip()
        norm_q = _normalize(q_str)
        if not norm_q:
            return []

        q_num = parse_numeric_value(q_str)
        cands: list[ResearchCandidate] = []
        pages = [page_hint] if (page_hint is not None and page_hint in self.index._pages_by_num) else [p.page_number for p in self.index.pages]

        for p_num in pages:
            cells = self._build_page_cells(p_num)
            for c_box, c_text, c_toks in cells:
                norm_c = _normalize(c_text)
                match = False
                if norm_c == norm_q:
                    match = True
                elif q_num is not None:
                    c_num = parse_numeric_value(c_text)
                    if c_num is not None and abs(c_num - q_num) < 1e-5:
                        match = True

                if match:
                    cands.append(
                        ResearchCandidate(
                            page=p_num,
                            bbox=c_box,
                            matched_text=c_text,
                            normalized_text=norm_c,
                            source="table_cell",
                            confidence=0.9,
                            raw_similarity=0.9,
                            tokens=tuple(c_toks),
                        )
                    )
        return cands


def deduplicate_candidate_pool(candidates: list[ResearchCandidate], iou_thresh: float = 0.80) -> list[ResearchCandidate]:
    """EXP-024E: Deduplicate candidates across sources while preserving highest confidence and source provenance."""
    if not candidates:
        return []

    # Sort by confidence descending, then source priority (baseline > spans > cells > ocr)
    source_priority = {"baseline": 4, "multi_token_span": 3, "table_cell": 2, "ocr_tesseract": 1}
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c.confidence, source_priority.get(c.source, 0)),
        reverse=True,
    )

    deduped: list[ResearchCandidate] = []
    for cand in sorted_cands:
        duplicate = False
        for kept in deduped:
            if kept.page == cand.page:
                iou = kept.bbox.iou(cand.bbox)
                if iou >= iou_thresh and (kept.normalized_text == cand.normalized_text or not cand.normalized_text):
                    duplicate = True
                    break
        if not duplicate:
            deduped.append(cand)

    return deduped
