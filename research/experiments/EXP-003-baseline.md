# EXP-003 Baseline Verification Report

## Executive Summary
This document establishes the verified baseline for **EXP-003**, auditing the benchmark results from **EXP-002E** against the official ExtractBench evaluation harness and metrics.

- **Baseline System**: TonerHound EXP-002E (OCR Fallback + Two-Pass Structural Row & Record Anchoring)
- **Official Leader**: LlamaExtract Agentic Plus (Word Grounding F1: **46.43%**)
- **TonerHound EXP-002E Word Grounding F1**: **60.43%** (Macro-average over the 5 bounding-box-bearing documents in the local experimental suite)
- **ExtractBench Repository Commit**: `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`
- **Official Evaluator**: `extract_bench.evaluation.metrics.extract.unified_evidence_metric.compute_unified_evidence_metrics`

> [!WARNING]
> TonerHound EXP-002E's 60.43% is evaluated on our local 6-case experimental suite (5 bbox-bearing documents). It is an encouraging local result, **NOT** yet an official leaderboard victory. The official benchmark evaluation must be reported separately.

---

## 1. Evaluator and Invariant Verification

### Evaluator Code Path
The evaluator used by TonerHound (`src/tonerhound/benchmark/evaluator.py`) directly invokes:
```python
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (
    compute_unified_evidence_metrics,
)
```
with bounding box IoU threshold `bbox_iou_threshold = 0.50`.

### Invariants Confirmed

1. **Word Grounding F1 is the Harmonic Mean of Precision and Recall**:
   Within each document:
   $$\text{Word Grounding F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Verification across all test cases shows exact mathematical equivalence (residual $\Delta < 10^{-15}$).

2. **False-Grounding Rate**:
   Defined as the rate of incorrect grounding claims among gradeable predictions:
   $$\text{False-Grounding Rate} = 1.0 - \text{Word Grounding Precision}$$
   In EXP-002E, this stands at **36.56%** across the suite, indicating that more than 1 in 3 grounded predictions are erroneous or misaligned.

3. **Macro-Averaged Aggregation**:
   The suite average of **60.43%** is an **unweighted macro-average** across the 5 documents that contain ground truth bounding boxes. The document `medium/veralto_earnings_deck_q4fy25` contains 0 ground truth bounding boxes and, in accordance with ExtractBench specifications (`c.g_expected == 0`), emits `None` and is excluded from the grounded dataset average.

---

## 2. Per-Document Verified Baseline (EXP-002E)

| Document | Word F1 | Word Precision | Word Recall | False Grounding | Ambiguity Rate | Citations / GT Boxes | Grounding Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `real_sm0801_eco_full` (66p) | **66.17%** | 67.68% | 64.73% | 32.32% | 0.00% | 7,302 / 6,657 | 105.80s |
| `real_pueblo_oct_2025` (11p) | **99.05%** | 99.12% | 98.99% | 0.88% | 0.00% | 4,263 / 3,757 | 104.26s |
| `veralto_earnings_deck_q4fy25` (41p) | *N/A (no GT)* | *N/A* | *N/A* | 0.00% | 0.00% | 18 / 0 | 4.28s |
| `W14-Atascosa SWD Well No. 4` (1p) | **37.59%** | 50.00% | 30.12% | 50.00% | 39.76% | 50 / 83 | 3.46s |
| `bianco-2024` (10p) | **10.85%** | 11.43% | 10.32% | 88.57% | 9.68% | 140 / 155 | 4.95s |
| `real_wyo_Goshen_2024` (10p) | **88.51%** | 88.97% | 88.05% | 11.03% | 19.02% | 796 / 983 | 5.75s |
| **MACRO AVERAGE (5 BBox Docs)** | **60.43%** | **63.44%** | **58.44%** | **36.56%** | **13.69%** | **12,551 / 11,635** | **224.24s** |

---

## 3. ExtractBench Evaluator Alignment

- **ExtractBench commit**: `94ceac15d457881b3d6f1c0f35c15bdea6af4b95`
- **Metric Definitions Reference**: `extract_bench/analysis/metric_definitions.py` specifies:
  - "Per-document micro F1; slice scores are unweighted means over documents."
  - "Word-level grounding. A cell counts only when its value is accepted AND its predicted citation box overlaps an accepted evidence box for that field at IoU >= 0.5 on the correct page."
  - "Precision counts only gradeable claims (citations on cells aligned to box-bearing ground truth); recall counts ground-truth cells carrying a verified box."
  - "Documents whose ground truth has no verified boxes emit no score at all rather than a zero, so this averages over box-bearing documents only."
- **Leaderboard Comparison**:
  - Official Leader: `LlamaExtract Agentic Plus`
  - Leader Word Grounding F1: `46.43%`
  - Leader Page Grounding F1: `84.92%`
  - Leader Value F1: `95.59%`
  - TonerHound Local EXP-002E Word Grounding F1: `60.43%` (+14.00 pp on local suite)

---

## 4. Key Failure Points in EXP-002E
1. **High False Grounding Rate (36.56%)**: 1 in 3 citations are incorrect.
2. **Scanned Documents Lagging**:
   - `bianco-2024`: 10.85% Word F1, 88.57% false grounding rate.
   - `W14-Atascosa`: 37.59% Word F1, 50.00% false grounding rate.
3. **Missing Verification Stage**: The resolver currently picks the highest scoring candidate without score-margin abstention or spatial plausibility verification.

This verified baseline will serve as the benchmark against which EXP-003 innovations are evaluated.
