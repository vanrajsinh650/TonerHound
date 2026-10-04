"""Unit tests for OCRNoiseTolerantIndex."""

from __future__ import annotations

import pytest
from tonerhound.document.index import DocumentIndex
from tonerhound.document.ocr_noise_index import OCRNoiseTolerantIndex
from tonerhound.geometry.coordinates import BBox
from tonerhound.models.types import DocumentPage, DocumentToken


def test_ocr_noise_index_query() -> None:
    token1 = DocumentToken(
        text="Comsolidated",
        bbox=BBox(x=0.1, y=0.1, width=0.2, height=0.05, page=1),
        page=1,
        char_index_in_page=0,
    )
    page1 = DocumentPage(
        page_number=1,
        width=100.0,
        height=100.0,
        tokens=[token1],
        lines=[],
    )

    doc_index = DocumentIndex(pages=[page1])
    doc_index.page_modes = {1: "ocr"}
    ocr_index = OCRNoiseTolerantIndex(doc_index)

    assert ocr_index.is_ocr_page(1)
    res = ocr_index.query("Consolidated", page_hint=1, max_distance=1)
    assert len(res) == 1
    assert res[0][0].text == "Comsolidated"
    assert res[0][1] >= 0.75
