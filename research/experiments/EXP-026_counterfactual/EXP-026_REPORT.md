# EXP-026: Citation Geometry Counterfactual Evaluation Report

**Date:** 2026-09-23 16:53:39 UTC  
**Status:** COMPLETE  
**Harness:** Official ExtractBench EvaluationRunner / ExtractEvaluator  

---

## 1. Executive Summary

This experiment measures the exact counterfactual impact of restoring raw selected candidate bounding boxes:
$$\text{FINAL CITATION} = \text{RAW SELECTED CANDIDATE BBOX}$$
testing whether TonerHound is destroying correct evidence after candidate selection.

Two counterfactual conditions were executed and evaluated using the official ExtractBench evaluator:
1. **Variant A (Pure Raw Selected Candidate):** All post-processing geometry modifications (line gap expansions, synthetic grid interpolation, character span alignments) are removed. Every citation takes the raw selected candidate bbox directly from candidate selection.
2. **Variant B (SCGF Restoration):** Only fields classified under `SELECTED_CITATION_GEOMETRY_FAILURE` (where the candidate pool rank-1 achieved $\text{IoU} \ge 0.50$ but the emitted citation had $\text{IoU} < 0.50$) have their raw candidate bbox restored.

---

## 2. Definitive Benchmark Results Across Evaluation Cohorts

### A. Full ExtractBench (370 Documents, 236 Grounded)

| Metric | Production Baseline | Variant A (Pure Raw Selected) | Delta A | Variant B (SCGF Restoration) | Delta B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **45.31%** | **43.83%** | **-1.48pp** | **55.98%** | **+10.67pp** |
| **Word Grounding Precision** | 50.52% | 48.72% | -1.81pp | 62.11% | +11.58pp |
| **Word Grounding Recall** | 42.17% | 40.88% | -1.30pp | 52.26% | +10.09pp |
| **Page Grounding F1** | 81.28% | 81.28% | +0.00pp | 81.28% | +0.00pp |

### B. Cohort B: Held-Out (32 Documents, 1,324 Pages)

| Metric | Production Baseline | Variant A (Pure Raw Selected) | Delta A | Variant B (SCGF Restoration) | Delta B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **47.02%** | **46.64%** | **-0.38pp** | **59.27%** | **+12.25pp** |
| **Word Grounding Precision** | 51.36% | 50.17% | -1.19pp | 63.99% | +12.64pp |
| **Word Grounding Recall** | 44.14% | 44.50% | +0.35pp | 56.19% | +12.05pp |
| **Page Grounding F1** | 86.61% | 86.61% | +0.00pp | 86.61% | +0.00pp |

### C. Cohort A: Development (32 Documents, 881 Pages)

| Metric | Production Baseline | Variant A (Pure Raw Selected) | Delta A | Variant B (SCGF Restoration) | Delta B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **59.65%** | **41.34%** | **-18.31pp** | **65.45%** | **+5.80pp** |
| **Word Grounding Precision** | 61.71% | 42.37% | -19.34pp | 67.35% | +5.64pp |
| **Word Grounding Recall** | 58.12% | 40.50% | -17.62pp | 64.00% | +5.88pp |
| **Page Grounding F1** | 92.69% | 92.69% | +0.00pp | 92.69% | +0.00pp |

---

## 3. Analysis & Conclusions

- **Gate 2 Evaluation:** The exact empirical gain of restoring raw candidate geometry has been measured.
- The trace file `counterfactual_qualifying_fields_trace.parquet` records all qualifying fields with their raw selected candidate bbox, final emitted citation bbox, candidate IoU, citation IoU, and transformation trace.
