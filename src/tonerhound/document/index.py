"""Document indexing and text/geometry extraction for TonerHound."""

from __future__ import annotations

import hashlib
import pickle
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
    is_plausible_date_string,
    normalize_unicode_and_case,
    parse_date_value,
    parse_numeric_value,
)


class DocumentIndex:
    """Document index extracting word and line geometry from PDFs or in-memory pages.

    Constructed once per document, cached and reused for many field resolution queries.
    Provides sublinear inverted indexes for tokens, canonical numbers, dates, and n-grams.
    """

    INDEX_VERSION = "v3"

    def __init__(self, pages: list[DocumentPage]) -> None:
        self.pages = pages
        self._pages_by_num: dict[int, DocumentPage] = {p.page_number: p for p in pages}
        # Inverted index: normalized token text -> list of DocumentTokens
        self._token_index: dict[str, list[DocumentToken]] = defaultdict(list)
        # Inverted index: punctuation-stripped stem -> list of DocumentTokens
        self._stem_token_index: dict[str, list[DocumentToken]] = defaultdict(list)
        # Inverted index: canonical numeric float -> list of (token, page_num, line_idx)
        self._numeric_index: dict[float, list[tuple[DocumentToken, int, int]]] = defaultdict(list)
        # Inverted index: ISO date string YYYY-MM-DD -> list of (page_num, line_idx, tokens, matched_text)
        self._date_index: dict[str, list[tuple[int, int, tuple[DocumentToken, ...], str]]] = defaultdict(list)
        # Inverted index: normalized word -> list of (page_num, line_idx)
        self._lines_by_token: dict[str, list[tuple[int, int]]] = defaultdict(list)
        # Inverted index: 3-character ngrams -> set of (page_num, line_idx)
        self._ngram_to_lines: dict[str, set[tuple[int, int]]] = defaultdict(set)
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
            p_num = page.page_number

            # 1. Build line-level and token-level indexes
            for line in page.lines:
                line_idx = line.line_index
                line_toks = line.tokens
                n_toks = len(line_toks)

                for token in line_toks:
                    tok_text = token.text
                    norm = normalize_unicode_and_case(tok_text).text.strip()
                    if norm:
                        self._token_index[norm].append(token)
                        stem = norm.strip(" ,.:;-/_'\"()[]{}*&#$€£¥")
                        if stem:
                            if stem != norm:
                                self._stem_token_index[stem].append(token)
                            self._lines_by_token[stem].append((p_num, line_idx))
                            if len(stem) >= 4:
                                for k in range(len(stem) - 2):
                                    self._ngram_to_lines[stem[k : k + 3]].add((p_num, line_idx))

                    # Numeric indexing on atomic token
                    parsed_num = parse_numeric_value(tok_text)
                    if parsed_num is not None:
                        key = round(parsed_num, 6)
                        self._numeric_index[key].append((token, p_num, line_idx))

                # Adjacent token numeric pairing (e.g. "$" + "1,200" or "-" + "500")
                for i in range(n_toks - 1):
                    t1, t2 = line_toks[i], line_toks[i + 1]
                    if t1.text in ("$", "€", "£", "¥", "-", "+"):
                        combo = t1.text + t2.text
                        pnum = parse_numeric_value(combo)
                        if pnum is not None:
                            key = round(pnum, 6)
                            ub = union_bbox_list([t1.bbox, t2.bbox])
                            if ub:
                                combo_tok = DocumentToken(
                                    text=combo,
                                    bbox=ub,
                                    page=p_num,
                                    char_index_in_page=t1.char_index_in_page,
                                    line_index=line_idx,
                                )
                                self._numeric_index[key].append((combo_tok, p_num, line_idx))

                # Date indexing on visual line windows (preferring minimal token spans)
                if any(ch.isdigit() for ch in line.text):
                    found_date_spans: list[tuple[int, int]] = []
                    for w_size in range(1, min(5, n_toks + 1)):
                        for start_i in range(n_toks - w_size + 1):
                            end_i = start_i + w_size
                            if any(s >= start_i and e <= end_i for s, e in found_date_spans):
                                continue
                            window = line_toks[start_i:end_i]
                            w_text = " ".join(t.text for t in window)
                            if is_plausible_date_string(w_text):
                                d_val = parse_date_value(w_text)
                                if d_val:
                                    d_key = d_val.strftime("%Y-%m-%d")
                                    found_date_spans.append((start_i, end_i))
                                    self._date_index[d_key].append(
                                        (p_num, line_idx, tuple(window), w_text)
                                    )

            # 2. Build full-page string by concatenating lines
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
    def from_pdf(
        cls,
        source: str | Path | bytes | BinaryIO,
        enable_ocr: bool = False,
        ocr_scale: float = 200.0 / 72.0,
        ocr_token_threshold: int = 25,
        use_cache: bool = True,
        cache_dir: Path | str = "research/cache/document_index",
    ) -> DocumentIndex:
        """Load and index a PDF using pypdfium2 with optional OCR fallback and persistent disk caching."""
        cache_path: Path | None = None
        if use_cache and isinstance(source, (str, Path)):
            src_path = Path(source)
            if src_path.exists():
                cache_dir = Path(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                # Compute SHA256 of the source PDF
                h = hashlib.sha256()
                with open(src_path, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                file_hash = h.hexdigest()
                cache_path = cache_dir / f"{file_hash}_{enable_ocr}_{ocr_scale:.2f}_{cls.INDEX_VERSION}.pkl"
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            cached_idx = pickle.load(f)
                            if isinstance(cached_idx, DocumentIndex):
                                return cached_idx
                    except (pickle.PickleError, EOFError, OSError, ValueError):
                        pass

        doc = pdfium.PdfDocument(source)
        pages: list[DocumentPage] = []

        try:
            for page_idx, pdf_page in enumerate(doc):
                page_num = page_idx + 1  # 1-indexed
                width, height = pdf_page.get_size()
                tokens = _extract_tokens_from_page(pdf_page, page_num, width, height)

                # Fallback to OCR if page has negligible native text tokens
                if enable_ocr and len(tokens) < ocr_token_threshold:
                    ocr_tokens = _extract_tokens_from_ocr(
                        pdf_page,
                        page_num=page_num,
                        scale=ocr_scale,
                    )
                    if len(ocr_tokens) > len(tokens):
                        tokens = ocr_tokens

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

        index = cls(pages)

        # Write to disk cache atomically
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
    for i, line in enumerate(visual_lines):
        line.line_index = i

    return visual_lines


def _parse_ocr_data(
    data: dict[str, list[Any]],
    img_w: int,
    img_h: int,
    page_num: int,
) -> list[DocumentToken]:
    """Convert pytesseract Output.DICT data into DocumentTokens with normalized bounding boxes."""
    from tonerhound.normalization.normalizers import repair_ocr_text

    tokens: list[DocumentToken] = []
    char_idx = 0
    n_boxes = len(data.get("text", []))
    for i in range(n_boxes):
        raw_text = str(data["text"][i]).strip()
        if not raw_text:
            continue

        clean_text = repair_ocr_text(raw_text)
        if not clean_text:
            clean_text = raw_text

        # Normalized coordinates [0, 1]
        x = max(0.0, min(1.0, data["left"][i] / img_w))
        y = max(0.0, min(1.0, data["top"][i] / img_h))
        w = max(0.0, min(1.0 - x, data["width"][i] / img_w))
        h = max(0.0, min(1.0 - y, data["height"][i] / img_h))

        if w <= 0.0 or h <= 0.0:
            continue

        bbox = BBox(x=x, y=y, width=w, height=h, page=page_num)
        tokens.append(
            DocumentToken(
                text=clean_text,
                bbox=bbox,
                page=page_num,
                char_index_in_page=char_idx,
            )
        )
        char_idx += len(clean_text) + 1

    return tokens


def _extract_tokens_from_ocr(
    pdf_page: Any,
    page_num: int,
    scale: float = 200.0 / 72.0,
) -> list[DocumentToken]:
    """Render page bitmap and run Tesseract OCR to extract word bounding boxes with sparse fallback."""
    try:
        import pytesseract
    except ImportError:
        return []

    bitmap = pdf_page.render(scale=scale)
    img = bitmap.to_pil()
    img_w, img_h = img.size

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    tokens = _parse_ocr_data(data, img_w, img_h, page_num)

    # Sparse / tabular fallback: if default PSM dropped most text (< 150 tokens),
    # try PSM 11 (sparse text). If PSM 11 recovers substantially more tokens, use it.
    if len(tokens) < 150:
        try:
            data11 = pytesseract.image_to_data(img, config="--psm 11", output_type=pytesseract.Output.DICT)
            tokens11 = _parse_ocr_data(data11, img_w, img_h, page_num)
            if len(tokens11) > len(tokens) * 1.4:
                tokens = tokens11
        except Exception:
            pass

    return tokens


