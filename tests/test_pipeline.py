"""Unit tests for DeterministicResolutionPipeline."""

from __future__ import annotations

import fitz
import pytest
from tonerhound.resolution.pipeline import (
    RESOLUTION_PRIORITY,
    DeterministicResolutionPipeline,
)


def test_resolution_priority_contract() -> None:
    assert RESOLUTION_PRIORITY[0] == "baseline_preserve"
    assert "date_literal_variants" in RESOLUTION_PRIORITY
    assert "hungarian_table_assigner" in RESOLUTION_PRIORITY
    assert "visual_checkbox_provider" in RESOLUTION_PRIORITY


def test_pipeline_baseline_preservation(tmp_path: pytest.TempPathFactory) -> None:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page(width=500, height=500)
    page.insert_text(fitz.Point(100, 100), "Existing Value")
    doc.save(str(pdf_path))
    doc.close()

    pipeline = DeterministicResolutionPipeline(pdf_path)
    existing = {
        "field_path": "title",
        "page": 1,
        "bbox": [0.1, 0.1, 0.2, 0.05],
    }
    gold_box = [0.1, 0.1, 0.2, 0.05]

    cit, stage = pipeline.resolve_field(
        field_path="title",
        gold_value="Existing Value",
        page_hint=1,
        existing_citation=existing,
        gold_bbox=gold_box,
    )
    pipeline.close()

    assert stage == "baseline_preserve"
    assert cit == existing
