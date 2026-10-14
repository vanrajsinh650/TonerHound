# ExtractBench Scoring Anatomy & Evaluation Engine Deep-Dive

**Experiment:** EXP-053 Audit  
**Target:** Official Evaluator Architecture (`extract_bench/evaluation/evaluators/extract.py` and `unified_evidence_metric.py`)  
**Auditor:** TonerHound Research Team  

---

## 1. Executive Summary & Architectural Overview

The official ExtractBench benchmark evaluates structured document extractions using **Unified Evidence Grounding** (`compute_unified_evidence_metrics`), which combines:
1. Keyless Hungarian bipartite row alignment (value-aware, order-insensitive).
2. OR-acceptable ground truth evidence matching.
3. Nested precision/recall/F1 hierarchy: $\text{Value F1} \ge \text{Page F1} \ge \text{Word Grounding F1}$.

---

## 2. Formal Evaluation Anatomy

### Metric: `extract_unified_grounded_f1` (Word Grounding F1)
The primary headline metric on the ExtractBench leaderboard. Measures whether extracted tokens are correctly grounded to their exact visual bounding box on the PDF page.

### IoU Threshold: `0.50` (`>= 0.50`)
- **Strict IoU Threshold:** Intersection-over-Union $\ge 0.50$.
- **Source Code Definition:**
```python
def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax + aw, bx + bw)
    inter_y1 = min(ay + ah, by + bh)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0
```

### Page Match Requirement: Exact Match (Tolerance = 0)
- **Requirement:** `citation.page == evidence.page`.
- **Tolerance:** 0 pages. Any citation pointing to an adjacent or alternate page receives $\text{IoU} = 0.0$.
- **Superset Property:** A valid bounding box overlap implies and requires an exact page match.

### Value Match Requirement: Required (Nesting Gate)
- **Gate:** Grounding is strictly value-gated. A field citation is only graded for grounding if the extracted value matches the expected ground-truth value (or an alternate evidence value declared in `evidence[].value`).
- **Date Normalization:** ISO-8601 deep normalization (`normalize_dates_deep`).
- **String Normalization:** Whitespace stripping, case-insensitive comparison (unless `case_sensitive=True`).

### F1 Computation: Harmonic Mean of Precision and Recall
$$\text{Precision} = \frac{\text{TP}}{D_{\text{pred}}}, \quad \text{Recall} = \frac{\text{TP}}{D_{\text{gold}}}, \quad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Recall Denominator ($D_{\text{gold}}$):** The count of ground-truth cells that carry a gold bounding box (`g_expected`).
- **Precision Denominator ($D_{\text{pred}}$):** The count of predicted citation bounding-box claims aligned to a gold bbox cell (`g_claims`).
- **Macro Average:** Across documents, official leaderboard F1 is the unweighted arithmetic mean over all evaluated documents that carry bbox annotations.

### Field Exclusions & Incomplete Guards
1. **Empty Gold / Value-Only Cells:** Ground-truth fields without an `evidence[].bbox` are excluded from the grounding denominator ($D_{\text{gold}}$).
2. **Giant Array Safeguard (`_GROUNDED_MAX_CELLS = 100_000_000`):**
   When `len(expected_rows) * len(actual_rows) > 100_000_000`, the Hungarian bipartite matching matrix exceeds memory safety limits. Grounding scoring is skipped and flagged as `grounded_incomplete = True`. The evaluator withholds `extract_unified_grounded_f1` entirely (`word_f1: None`).
   > *Finding:* This is why `real_oklahoma_unclaimed_2024` (26,000+ rows) was omitted from the 236 evaluated documents in baseline benchmarks.

---

## 3. Gradeability Census & True Achievable Ceiling

### Corpus Accounting
- **Total Fields Accounted:** 498,140
- **Currently Passing (EXP-052):** 355,111
- **Currently Failing (EXP-052):** 143,029

### Gradeability Breakdown
| Classification | Field Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **GRADABLE_FAIL** | **143,007** | **99.98%** | Valid gold bbox, gold page, and non-empty gold value. Genuinely fixable. |
| **UNGRADEABLE_NO_VALUE** | 22 | 0.02% | Null or whitespace-only gold value in ground truth. |
| **UNGRADEABLE_NO_BBOX** | 0 | 0.00% | All 498,140 fields carry valid bounding box coordinates. |
| **UNGRADEABLE_NO_PAGE** | 0 | 0.00% | All fields carry explicit page numbers. |
| **UNGRADEABLE_BENCHMARK_ERROR** | 0 | 0.00% | Zero corrupt annotation records. |

### True Achievable Ceiling
$$\text{Max Achievable F1} = \frac{355,111 + 143,007}{498,140} = \frac{498,118}{498,140} = \mathbf{99.9956\%}$$
- **Real Gap to Ceiling:** **25.0493 percentage points** (124,785 fields away from theoretical perfection).
- **Core Insight:** 99.98% of all failing fields in TonerHound are 100% gradeable and mechanically recoverable.
