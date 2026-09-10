"""Official ExtractBench evaluator harness for TonerHound."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure reference ExtractBench src is in Python path for official evaluation
_REF_EXTRACTBENCH = Path(__file__).resolve().parent.parent.parent.parent / "research" / "reference" / "ExtractBench" / "src"
if _REF_EXTRACTBENCH.exists() and str(_REF_EXTRACTBENCH) not in sys.path:
    sys.path.insert(0, str(_REF_EXTRACTBENCH))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    compute_unified_evidence_metrics,
)


def evaluate_prediction(
    expected_output: dict[str, Any],
    extracted_data: dict[str, Any],
    field_rules: list[Any],
    field_citations: list[dict[str, Any]],
    data_schema: dict[str, Any],
    bbox_iou_threshold: float = 0.50,
) -> dict[str, float | None]:
    """Execute official ExtractBench unified evaluation and extract headline metrics."""
    metrics = compute_unified_evidence_metrics(
        expected_output=expected_output,
        extracted_data=extracted_data,
        field_rules=field_rules,
        field_citations=field_citations,
        data_schema=data_schema,
        bbox_iou_threshold=bbox_iou_threshold,
    )

    results: dict[str, float | None] = {}
    for m in metrics:
        results[m.metric_name] = m.value

    return {
        "value_f1": results.get("extract_unified_value_f1"),
        "value_precision": results.get("extract_unified_value_precision"),
        "value_recall": results.get("extract_unified_value_recall"),
        "word_grounding_f1": results.get("extract_unified_grounded_f1"),
        "word_grounding_precision": results.get("extract_unified_grounded_precision"),
        "word_grounding_recall": results.get("extract_unified_grounded_recall"),
        "page_grounding_f1": results.get("extract_unified_page_f1"),
        "page_grounding_precision": results.get("extract_unified_page_precision"),
        "page_grounding_recall": results.get("extract_unified_page_recall"),
    }
