# Benchmark Plan: Experimental Protocol & Reproduction Ladder

## 1. Primary Benchmark Objective

The primary objective is to experimentally demonstrate that **TonerHound** outperforms the current ExtractBench word-level grounding leader:

- **Current Reference Leader**: LlamaExtract Agentic Plus
- **Current Reference Score**: **46.43% Word Grounding F1**
- **Target Progression**:
  - Baseline reproduction
  - 46.43% + (Baseline Victory)
  - 60.00% + (Meaningful Improvement)
  - 70.00% + (Strong Improvement)
  - 80.00% + (Major Result)

---

## 2. Evaluation Protocol & Ground Truth Integrity

To guarantee 100% scientific validity:
1. **Zero Evaluator Modifications**: We will evaluate TonerHound predictions directly using ExtractBench's official `compute_unified_evidence_metrics` (`unified_evidence_metric.py`).
2. **Zero Ground Truth Modifications**: No test cases, expected outputs, or evidence lists will be altered.
3. **IoU Threshold**: Strict COCO IoU threshold of **0.50** (`bbox_iou_threshold=0.50`) on 1-indexed pages.
4. **Target Metrics**:
   - `extract_unified_grounded_f1` (Word-level grounding F1)
   - `extract_unified_grounded_precision`
   - `extract_unified_grounded_recall`
   - `extract_unified_page_f1` (Page-level grounding F1)
   - `extract_unified_value_f1` (Value extraction F1)
   - `accepted_grounding_accuracy` (Correctness rate on groundings where status is NOT `ambiguous` or `not_found`)

---

## 3. Controlled A/B Experimental Ladder

As mandated by Section 28 (A/B Testing Rule), we will never combine multiple major changes at once. Each experiment changes exactly ONE component, measured against the baseline.

| Experiment ID | Component Tested | Hypothesis | Baseline F1 | Target Delta |
|:---|:---|:---|:---:|:---:|
| **EXP-001** | **Exact Token Matching** | Direct exact search of extracted value tokens against native PDF character atoms. | 0.00% (Native LLM) | +30.0% |
| **EXP-002** | **Reversible Normalization** | Currency, date, whitespace, punctuation, and numeric format normalizations with original character span tracking. | EXP-001 | +10.0% |
| **EXP-003** | **Spatial Context Disambiguation** | When multiple identical candidate values exist, rank using proximity to `field_context` label and reading order. | EXP-002 | +8.0% |
| **EXP-004** | **Multi-line Segment Grouping** | Wrap multi-line spans into individual line bounding boxes instead of a single inflated rectangle. | EXP-003 | +5.0% |
| **EXP-005** | **Fuzzy Sequence Alignment** | Levenshtein / Smith-Waterman character fallback for OCR noise and hyphenation breaks. | EXP-004 | +4.0% |
| **EXP-006** | **Table & Column Structure** | Align extracted row records using column header and table bounding boxes. | EXP-005 | +5.0% |
| **EXP-007** | **Provenance Gating & Calibration** | Refuse to output false boxes for `derived` or `ambiguous` values, achieving near 100% accepted precision. | EXP-006 | Precision Boost |

---

## 4. Failure Analysis Protocol

For every experiment:
1. All misses (`iou < 0.50` or wrong page) are logged to `research/failures/`.
2. Categorized into standardized failure classes:
   - `WRONG_OCCURRENCE`: Duplicate value chosen wrongly.
   - `NORMALIZATION_FAILURE`: Format mismatch between value and document text.
   - `WRAPPED_LINE_IOU_MISS`: Single big box caused IoU < 0.50.
   - `OCR_CORRUPTION`: Character dropped or substituted.
   - `DERIVED_VALUE`: Value calculated by LLM with no literal source.
   - `AMBIGUOUS`: Multiple identical candidates with identical contexts.
3. Every pull request or experiment note records:
   - Experiment ID
   - Hypothesis
   - Metric before vs after
   - Failure impact
   - Latency impact
   - Decision (Adopt / Revert)
