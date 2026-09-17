"""LiteParse document indexing and text/geometry extraction for TonerHound."""

from __future__ import annotations

import hashlib
import pickle
import unicodedata
from pathlib import Path
from typing import Any, BinaryIO

try:
    import liteparse
except ImportError:
    liteparse = None  # type: ignore[assignment]

from tonerhound.document.index import DocumentIndex, _cluster_tokens_into_lines
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def _normalize_bbox_coords(
    x: float,
    y: float,
    width: float,
    height: float,
    page_width: float,
    page_height: float,
    page_num: int,
) -> BBox:
    """Convert LiteParse point coordinates into normalized [0.0, 1.0] BBox."""
    if page_width <= 0.0:
        page_width = 612.0
    if page_height <= 0.0:
        page_height = 792.0

    norm_x = max(0.0, min(1.0, float(x) / page_width))
    norm_y = max(0.0, min(1.0, float(y) / page_height))
    norm_w = max(0.0, min(1.0 - norm_x, max(0.0, float(width)) / page_width))
    norm_h = max(0.0, min(1.0 - norm_y, max(0.0, float(height)) / page_height))

    return BBox(x=norm_x, y=norm_y, width=norm_w, height=norm_h, page=page_num)


def _extract_tokens_from_liteparse_page(
    page: Any,
    page_num: int,
    page_width: float,
    page_height: float,
    emit_word_boxes: bool = True,
) -> list[DocumentToken]:
    """Extract DocumentTokens from LiteParse page text items, prioritizing word boxes."""
    tokens: list[DocumentToken] = []
    char_idx = 0

    text_items = getattr(page, "text_items", None) or []
    for item in text_items:
        words = getattr(item, "words", None) if emit_word_boxes else None
        if words:
            for w in words:
                raw_text = getattr(w, "text", "")
                clean = unicodedata.normalize("NFKC", raw_text.strip())
                if not clean:
                    continue

                bbox = _normalize_bbox_coords(
                    x=getattr(w, "x", 0.0),
                    y=getattr(w, "y", 0.0),
                    width=getattr(w, "width", 0.0),
                    height=getattr(w, "height", 0.0),
                    page_width=page_width,
                    page_height=page_height,
                    page_num=page_num,
                )
                tokens.append(
                    DocumentToken(
                        text=clean,
                        bbox=bbox,
                        page=page_num,
                        char_index_in_page=char_idx,
                    )
                )
                char_idx += len(clean) + 1
        else:
            raw_text = getattr(item, "text", "")
            clean = unicodedata.normalize("NFKC", raw_text.strip())
            if not clean:
                continue

            bbox = _normalize_bbox_coords(
                x=getattr(item, "x", 0.0),
                y=getattr(item, "y", 0.0),
                width=getattr(item, "width", 0.0),
                height=getattr(item, "height", 0.0),
                page_width=page_width,
                page_height=page_height,
                page_num=page_num,
            )
            tokens.append(
                DocumentToken(
                    text=clean,
                    bbox=bbox,
                    page=page_num,
                    char_index_in_page=char_idx,
                )
            )
            char_idx += len(clean) + 1

    return tokens


class LiteParseDocumentIndex(DocumentIndex):
    """Document index using LiteParse for high-throughput text and word-box extraction."""

    INDEX_VERSION = "lp_v1"

    @classmethod
    def from_pdf(
        cls,
        source: str | Path | bytes | BinaryIO,
        emit_word_boxes: bool = True,
        ocr_enabled: bool = True,
        use_cache: bool = True,
        cache_dir: Path | str = "research/cache/document_index",
        backend: str = "liteparse",
        **kwargs: Any,
    ) -> LiteParseDocumentIndex:
        """Load and index a PDF using LiteParse with disk caching.

        Parameters:
            source: PDF file path (str or Path), raw PDF bytes, or file-like BinaryIO.
            emit_word_boxes: When True, extracts word-level bounding boxes (item.words).
            ocr_enabled: Whether to enable OCR in LiteParse for scanned/image pages.
            use_cache: Whether to load and persist indices to the disk cache.
            cache_dir: Directory for storing pickled index caches.
            backend: Parser backend, defaults to "liteparse".
            **kwargs: Additional parameters passed to `liteparse.LiteParse(...)`.
        """
        if backend == "pdfium":
            return super().from_pdf(  # type: ignore[return-value]
                source=source,
                use_cache=use_cache,
                cache_dir=cache_dir,
                backend="pdfium",
                **kwargs,
            )

        if liteparse is None:
            raise ImportError(
                "The 'liteparse' package is required to use LiteParseDocumentIndex. "
                "Please install it with `pip install liteparse`."
            )

        # Read source and compute cache hash
        file_data: str | Path | bytes
        file_hash: str | None = None

        if hasattr(source, "read") and callable(source.read):
            file_data = source.read()
            if use_cache:
                file_hash = hashlib.sha256(file_data).hexdigest()
        elif isinstance(source, bytes):
            file_data = source
            if use_cache:
                file_hash = hashlib.sha256(file_data).hexdigest()
        elif isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"PDF file not found: {source}")
            file_data = src_path
            if use_cache:
                h = hashlib.sha256()
                with open(src_path, "rb") as f:
                    while chunk := f.read(65536):
                        h.update(chunk)
                file_hash = h.hexdigest()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        cache_path: Path | None = None
        if use_cache and file_hash is not None:
            cache_dir_p = Path(cache_dir)
            cache_dir_p.mkdir(parents=True, exist_ok=True)
            cache_path = (
                cache_dir_p
                / f"{file_hash}_liteparse_{emit_word_boxes}_{ocr_enabled}_{cls.INDEX_VERSION}.pkl"
            )
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached_idx = pickle.load(f)
                        if isinstance(cached_idx, DocumentIndex):
                            return cached_idx  # type: ignore[return-value]
                except (pickle.PickleError, EOFError, OSError, ValueError):
                    pass

        parser = liteparse.LiteParse(
            emit_word_boxes=emit_word_boxes,
            ocr_enabled=ocr_enabled,
            **kwargs,
        )
        try:
            parse_result = parser.parse(file_data)
        finally:
            if hasattr(parser, "close"):
                parser.close()

        pages: list[DocumentPage] = []
        for page_idx, p in enumerate(parse_result.pages):
            page_num = getattr(p, "page_num", None) or (page_idx + 1)
            width = float(getattr(p, "width", 612.0) or 612.0)
            height = float(getattr(p, "height", 792.0) or 792.0)
            if width <= 0.0:
                width = 612.0
            if height <= 0.0:
                height = 792.0

            tokens = _extract_tokens_from_liteparse_page(
                page=p,
                page_num=page_num,
                page_width=width,
                page_height=height,
                emit_word_boxes=emit_word_boxes,
            )
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

        index = cls(pages)

        if cache_path is not None:
            try:
                tmp_path = cache_path.with_suffix(".tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path.replace(cache_path)
            except (pickle.PickleError, OSError, TypeError):
                pass

        return index
