"""Deterministic Page-Level OCR Router for EXP-037.

Implements page-selective deterministic OCR routing:
- IF a page already has usable native text (len(tokens) >= min_digital_tokens):
    DO NOT OCR IT
- IF a page has no usable native text (len(tokens) < min_digital_tokens):
    Invoke deterministic OCR (Tesseract via PyMuPDF/OpenCV)

Zero AI: No LLMs, No VLMs, No Neural Networks, No Embeddings, No Cloud APIs.
Laptop Safety: Single-threaded math, memory cleanup, persistent disk caching.
"""

from __future__ import annotations

import gc
import hashlib
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Set single thread math for laptop safety
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pypdfium2 as pdfium

from tonerhound.benchmark.adapter import ExtractBenchAdapter
from tonerhound.document.index import (
    DocumentIndex,
    _cluster_tokens_into_lines,
    _extract_tokens_from_ocr,
    _extract_tokens_from_page,
)
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken, ExtractionInput, ResolutionResult
from tonerhound.resolution.resolver import EvidenceResolver


@dataclass
class OCRInstrumentationStats:
    """Instrumentation statistics required by Section 7."""
    pages_inspected: int = 0
    pages_with_native_text: int = 0
    pages_without_native_text: int = 0
    pages_ocrd: int = 0
    pages_skipped_from_ocr: int = 0
    ocr_runtime_sec: float = 0.0
    ocr_failures: int = 0
    ocr_empty_pages: int = 0
    ocr_tokens_extracted: int = 0
    ocr_candidates_generated: int = 0
    ocr_grounded_fields: int = 0
    ocr_rescued_fields: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_inspected": self.pages_inspected,
            "pages_with_native_text": self.pages_with_native_text,
            "pages_without_native_text": self.pages_without_native_text,
            "pages_ocrd": self.pages_ocrd,
            "pages_skipped_from_ocr": self.pages_skipped_from_ocr,
            "ocr_runtime_sec": round(self.ocr_runtime_sec, 3),
            "ocr_failures": self.ocr_failures,
            "ocr_empty_pages": self.ocr_empty_pages,
            "ocr_tokens_extracted": self.ocr_tokens_extracted,
            "ocr_candidates_generated": self.ocr_candidates_generated,
            "ocr_grounded_fields": self.ocr_grounded_fields,
            "ocr_rescued_fields": self.ocr_rescued_fields,
        }


class DeterministicOCRRouter:
    """Page-selective deterministic OCR router and DocumentIndex builder."""

    ROUTER_VERSION = "exp037_v1"

    def __init__(
        self,
        min_digital_tokens: int = 25,
        ocr_scale: float = 200.0 / 72.0,
        cache_dir: Path | str = "research/cache/exp037_ocr",
    ) -> None:
        self.min_digital_tokens = min_digital_tokens
        self.ocr_scale = ocr_scale
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = OCRInstrumentationStats()

    def get_document_index(
        self,
        pdf_path: Path | str,
        use_cache: bool = True,
    ) -> tuple[DocumentIndex, dict[int, str]]:
        """Index a document with page-selective OCR routing.
        
        Returns:
            (DocumentIndex, dict mapping page_number -> 'native' | 'ocr')
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Compute file hash for caching
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        file_hash = h.hexdigest()

        cache_file = (
            self.cache_dir
            / f"{file_hash}_{self.min_digital_tokens}_{self.ocr_scale:.2f}_{self.ROUTER_VERSION}.pkl"
        )

        if use_cache and cache_file.exists():
            try:
                with open(cache_file, "rb") as cf:
                    cached_data = pickle.load(cf)
                    if isinstance(cached_data, tuple) and len(cached_data) == 2:
                        idx, modes = cached_data
                        if isinstance(idx, DocumentIndex):
                            return idx, modes
            except Exception:
                pass

        # Build with page-selective routing
        doc = pdfium.PdfDocument(pdf_path)
        pages: list[DocumentPage] = []
        page_modes: dict[int, str] = {}

        try:
            for page_idx, pdf_page in enumerate(doc):
                page_num = page_idx + 1
                self.stats.pages_inspected += 1
                width, height = pdf_page.get_size()

                # Step 1: Extract native digital text tokens
                native_tokens = _extract_tokens_from_page(pdf_page, page_num, width, height)

                if len(native_tokens) >= self.min_digital_tokens:
                    # Page has usable native digital text -> DO NOT OCR
                    self.stats.pages_with_native_text += 1
                    self.stats.pages_skipped_from_ocr += 1
                    page_modes[page_num] = "native"
                    lines = _cluster_tokens_into_lines(native_tokens, page_num)
                    pages.append(
                        DocumentPage(
                            page_number=page_num,
                            width=width,
                            height=height,
                            tokens=native_tokens,
                            lines=lines,
                        )
                    )
                else:
                    # Page has negligible or zero native text -> Trigger OCR
                    self.stats.pages_without_native_text += 1
                    self.stats.pages_ocrd += 1
                    t0_ocr = time.perf_counter()
                    try:
                        ocr_tokens = _extract_tokens_from_ocr(
                            pdf_page,
                            page_num=page_num,
                            scale=self.ocr_scale,
                        )
                        dt_ocr = time.perf_counter() - t0_ocr
                        self.stats.ocr_runtime_sec += dt_ocr

                        if len(ocr_tokens) > 0:
                            self.stats.ocr_tokens_extracted += len(ocr_tokens)
                            final_tokens = ocr_tokens
                            page_modes[page_num] = "ocr"
                        else:
                            self.stats.ocr_empty_pages += 1
                            final_tokens = native_tokens
                            page_modes[page_num] = "empty"
                    except Exception as e:
                        dt_ocr = time.perf_counter() - t0_ocr
                        self.stats.ocr_runtime_sec += dt_ocr
                        self.stats.ocr_failures += 1
                        final_tokens = native_tokens
                        page_modes[page_num] = "error"

                    lines = _cluster_tokens_into_lines(final_tokens, page_num)
                    pages.append(
                        DocumentPage(
                            page_number=page_num,
                            width=width,
                            height=height,
                            tokens=final_tokens,
                            lines=lines,
                        )
                    )
        finally:
            doc.close()

        doc_index = DocumentIndex(pages)

        # Cache atomically
        if use_cache:
            try:
                temp_file = cache_file.with_suffix(".tmp")
                with open(temp_file, "wb") as tf:
                    pickle.dump((doc_index, page_modes), tf)
                temp_file.replace(cache_file)
            except Exception:
                pass

        return doc_index, page_modes
