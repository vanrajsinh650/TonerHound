"""Document indexing and text/geometry extraction for TonerHound."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO

import pypdfium2 as pdfium

from tonerhound.geometry.coordinates import BBox, union_bbox_list
from tonerhound.models.types import DocumentPage, DocumentToken, VisualLine
from tonerhound.normalization.normalizers import (
    _CHAR_REPLACEMENTS,
    NormalizedText,
    normalize_unicode_and_case,
)


class DocumentIndex:
    """Document index extracting word and line geometry from PDFs or in-memory pages.

    Constructed once per document, cached and reused for many field resolution queries.
    """

    def __init__(self, pages: list[DocumentPage]) -> None:
        self.pages = pages
        self._pages_by_num: dict[int, DocumentPage] = {p.page_number: p for p in pages}
        # Inverted index: normalized token text -> list of DocumentTokens
        self._token_index: dict[str, list[DocumentToken]] = defaultdict(list)
        # Normalized full page text with char_map to tokens
        self._page_normalized_text: dict[int, tuple[NormalizedText, list[DocumentToken]]] = {}

        self._build_indexes()

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def get_page(self, page_num: int) -> DocumentPage | None:
        """Get page by 1-indexed page number."""
        return self._pages_by_num.get(page_num)

    def _build_indexes(self) -> None:
        for page in self.pages:
            # Build token index
            for token in page.tokens:
                norm = normalize_unicode_and_case(token.text).text.strip()
                if norm:
                    self._token_index[norm].append(token)

            # Build full-page string by concatenating lines
            page_text_parts: list[str] = []
            char_to_token: list[DocumentToken] = []

            for line in page.lines:
                for token in line.tokens:
                    tok_text = token.text
                    page_text_parts.append(tok_text)
                    for _ in tok_text:
                        char_to_token.append(token)
                    # Separator between tokens
                    page_text_parts.append(" ")
                    char_to_token.append(token)
                page_text_parts.append("\n")
                if line.tokens:
                    char_to_token.append(line.tokens[-1])

            raw_page_str = "".join(page_text_parts)
            norm_page_text = normalize_unicode_and_case(raw_page_str)
            self._page_normalized_text[page.page_number] = (norm_page_text, char_to_token)

    @classmethod
    def from_pdf(cls, source: str | Path | bytes | BinaryIO) -> DocumentIndex:
        """Load and index a PDF using pypdfium2."""
        doc = pdfium.PdfDocument(source)
        pages: list[DocumentPage] = []

        try:
            for page_idx, pdf_page in enumerate(doc):
                page_num = page_idx + 1  # 1-indexed
                width, height = pdf_page.get_size()
                tokens = _extract_tokens_from_page(pdf_page, page_num, width, height)

                # Cluster tokens into visual lines by vertical alignment
                lines = _cluster_tokens_into_lines(tokens, page_num)
                pages.append(
                    DocumentPage(
                        page_number=page_num,
                        width=width,
                        height=height,
                        tokens=tokens,
                        lines=lines,
                    )
                )
        finally:
            doc.close()

        return cls(pages)

    @classmethod
    def from_pages(cls, pages: list[DocumentPage]) -> DocumentIndex:
        """Construct directly from pre-built pages (e.g. for synthetic fixtures)."""
        return cls(pages)

    def search_exact(self, query: str, page: int | None = None) -> list[tuple[BBox, str]]:
        """Search for exact substring occurrences across document pages."""
        norm_query = normalize_unicode_and_case(query).text.strip()
        if not norm_query:
            return []

        results: list[tuple[BBox, str]] = []
        target_pages = [page] if page is not None else list(self._pages_by_num.keys())

        for p_num in target_pages:
            entry = self._page_normalized_text.get(p_num)
            if not entry:
                continue
            norm_text_obj, char_to_token = entry
            text = norm_text_obj.text

            start = 0
            while True:
                idx = text.find(norm_query, start)
                if idx == -1:
                    break

                # Boundary check: ensure match does not cut into alphanumeric tokens
                is_alnum_before = idx > 0 and text[idx - 1].isalnum()
                is_alnum_after = (idx + len(norm_query) < len(text)) and text[idx + len(norm_query)].isalnum()
                if (norm_query[0].isalnum() and is_alnum_before) or (norm_query[-1].isalnum() and is_alnum_after):
                    start = idx + 1
                    continue

                # Extract matched tokens
                matched_tokens: list[DocumentToken] = []
                orig_start, orig_end = norm_text_obj.span_to_original_range(idx, idx + len(norm_query))

                for char_i in range(orig_start, min(orig_end, len(char_to_token))):
                    tok = char_to_token[char_i]
                    if tok not in matched_tokens:
                        matched_tokens.append(tok)

                if matched_tokens:
                    boxes = [t.bbox for t in matched_tokens]
                    union_box = union_bbox_list(boxes)
                    if union_box is not None:
                        matched_str = " ".join(t.text for t in matched_tokens)
                        results.append((union_box, matched_str))

                start = idx + max(1, len(norm_query))

        return results


def _extract_tokens_from_page(
    pdf_page: Any,
    page_num: int,
    page_width: float,
    page_height: float,
) -> list[DocumentToken]:
    """Extract individual words with their bounding boxes from a PDF page."""
    textpage = pdf_page.get_textpage()
    n_chars = textpage.count_chars()

    tokens: list[DocumentToken] = []
    current_word_chars: list[str] = []
    current_word_boxes: list[tuple[float, float, float, float]] = []
    current_word_start_idx = 0

    def _flush_word() -> None:
        nonlocal current_word_chars, current_word_boxes, current_word_start_idx
        if not current_word_chars:
            return
        word_text = "".join(current_word_chars)
        if not word_text.strip():
            current_word_chars = []
            current_word_boxes = []
            return

        min_x = min(b[0] for b in current_word_boxes)
        min_y = min(b[1] for b in current_word_boxes)
        max_x = max(b[2] for b in current_word_boxes)
        max_y = max(b[3] for b in current_word_boxes)

        word_bbox = BBox.from_pdf_points(
            x0=min_x,
            y0=min_y,
            x1=max_x,
            y1=max_y,
            page_width=page_width,
            page_height=page_height,
            page=page_num,
            origin_bottom_left=True,
        )
        tokens.append(
            DocumentToken(
                text=word_text,
                bbox=word_bbox,
                page=page_num,
                char_index_in_page=current_word_start_idx,
            )
        )
        current_word_chars = []
        current_word_boxes = []

    for i in range(n_chars):
        char_text = textpage.get_text_range(i, 1)
        if not char_text:
            continue

        left, bottom, right, top = textpage.get_charbox(i)

        if char_text.isspace() or char_text in ("\r", "\n"):
            _flush_word()
        else:
            if not current_word_chars:
                current_word_start_idx = i
            cleaned_char = _CHAR_REPLACEMENTS.get(char_text, char_text)
            cleaned_char = unicodedata.normalize("NFKC", cleaned_char)
            current_word_chars.append(cleaned_char)
            current_word_boxes.append((left, bottom, right, top))

    _flush_word()
    return tokens


def _cluster_tokens_into_lines(tokens: list[DocumentToken], page: int) -> list[VisualLine]:
    """Cluster tokens into horizontal visual lines based on vertical overlap."""
    if not tokens:
        return []

    # Sort tokens primarily by top (y0), then left (x0)
    sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y0, t.bbox.x0))
    line_groups: list[list[DocumentToken]] = []

    for token in sorted_tokens:
        placed = False
        for group in line_groups:
            # Check vertical overlap with line group
            group_y0 = min(t.bbox.y0 for t in group)
            group_y1 = max(t.bbox.y1 for t in group)

            overlap_y0 = max(token.bbox.y0, group_y0)
            overlap_y1 = min(token.bbox.y1, group_y1)
            overlap_h = max(0.0, overlap_y1 - overlap_y0)

            # If vertical overlap is >= 50% of the token height, belongs to same line
            if overlap_h / max(0.001, token.bbox.height) >= 0.5:
                group.append(token)
                placed = True
                break

        if not placed:
            line_groups.append([token])

    # Sort tokens within each line from left to right, and sort lines top to bottom
    visual_lines: list[VisualLine] = []
    for idx, group in enumerate(line_groups):
        group.sort(key=lambda t: t.bbox.x0)
        line_box = union_bbox_list([t.bbox for t in group])
        if line_box is not None:
            visual_lines.append(
                VisualLine(
                    tokens=group,
                    page=page,
                    line_index=idx,
                    bbox=line_box,
                )
            )

    visual_lines.sort(key=lambda l: (l.bbox.y0, l.bbox.x0))
    # Re-index line numbers
    for i, line in enumerate(visual_lines):
        line.line_index = i

    return visual_lines
