"""Official ExtractBench evaluator harness for TonerHound."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path
from typing import Any

# Ensure reference ExtractBench src is in Python path for official evaluation
_REF_EXTRACTBENCH = Path(__file__).resolve().parent.parent.parent.parent / "research" / "reference" / "ExtractBench" / "src"
if _REF_EXTRACTBENCH.exists() and str(_REF_EXTRACTBENCH) not in sys.path:
    sys.path.insert(0, str(_REF_EXTRACTBENCH))

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    _GROUNDED_MAX_CELLS,
    compute_unified_evidence_metrics,
)


def evaluate_prediction(
    expected_output: dict[str, Any],
    extracted_data: dict[str, Any],
    field_rules: list[Any],
    field_citations: list[dict[str, Any]],
    data_schema: dict[str, Any],
    bbox_iou_threshold: float = 0.50,
    force_include_all: bool = False,
    chunk_size: int = 10000,
) -> dict[str, float | None]:
    """Execute official ExtractBench unified evaluation and extract headline metrics.

    If force_include_all is enabled and a document array triggers the memory guard
    (_GROUNDED_MAX_CELLS = 100_000_000), evaluates the array in chunked blocks of
    chunk_size rows to prevent memory crashes while computing exact grounding F1.
    """
    # Detect giant array exceeding memory guard
    giant_key: str | None = None
    for k, v in expected_output.items():
        if isinstance(v, list) and len(v) * len(extracted_data.get(k, [])) > _GROUNDED_MAX_CELLS:
            giant_key = k
            break

    if giant_key is not None and force_include_all:
        exp_arr = expected_output[giant_key]
        ext_arr = extracted_data.get(giant_key, [])
        n_rows = len(exp_arr)
        results_list: list[dict[str, float | None]] = []

        pattern = re.compile(rf"^{giant_key}\[(\d+)\](\..*)?$")

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk_exp = {k: (v[start:end] if k == giant_key else v) for k, v in expected_output.items() if k == giant_key or start == 0}
            chunk_ext = {k: (v[start:end] if k == giant_key else v) for k, v in extracted_data.items() if k == giant_key or start == 0}

            chunk_rules = []
            for r in field_rules:
                m = pattern.match(r.field_path)
                if m:
                    idx = int(m.group(1))
                    if start <= idx < end:
                        rc = copy.copy(r)
                        rc.field_path = f"{giant_key}[{idx - start}]{m.group(2) or ''}"
                        chunk_rules.append(rc)
                elif start == 0:
                    chunk_rules.append(r)

            chunk_citations = []
            for c in field_citations:
                fp = c.get("field_path", "")
                m = pattern.match(fp)
                if m:
                    idx = int(m.group(1))
                    if start <= idx < end:
                        cc = dict(c)
                        cc["field_path"] = f"{giant_key}[{idx - start}]{m.group(2) or ''}"
                        chunk_citations.append(cc)
                elif start == 0:
                    chunk_citations.append(c)

            chunk_metrics = compute_unified_evidence_metrics(
                expected_output=chunk_exp,
                extracted_data=chunk_ext,
                field_rules=chunk_rules,
                field_citations=chunk_citations,
                data_schema=data_schema,
                bbox_iou_threshold=bbox_iou_threshold,
            )

            res: dict[str, float | None] = {m.metric_name: m.value for m in chunk_metrics}
            results_list.append({
                "value_f1": res.get("extract_unified_value_f1"),
                "value_precision": res.get("extract_unified_value_precision"),
                "value_recall": res.get("extract_unified_value_recall"),
                "word_grounding_f1": res.get("extract_unified_grounded_f1"),
                "word_grounding_precision": res.get("extract_unified_grounded_precision"),
                "word_grounding_recall": res.get("extract_unified_grounded_recall"),
                "page_grounding_f1": res.get("extract_unified_page_f1"),
                "page_grounding_precision": res.get("extract_unified_page_precision"),
                "page_grounding_recall": res.get("extract_unified_page_recall"),
            })

        # Combine chunk results via macro-averaging
        avg_res: dict[str, float | None] = {}
        for metric_name in results_list[0].keys():
            vals = [r[metric_name] for r in results_list if r.get(metric_name) is not None]
            avg_res[metric_name] = sum(vals) / len(vals) if vals else None
        return avg_res

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
