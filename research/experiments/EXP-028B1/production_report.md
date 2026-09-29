# EXP-028B1: True Text Path Productionization Report

**Date:** 2026-09-29 09:29:42 UTC  
**Status:** COMPLETE  
**Harness:** Official ExtractBench EvaluationRunner / ExtractEvaluator  

---

## 1. Executive Summary

This experiment measures the actual end-to-end production **Word Grounding F1** after integrating the 10 information-loss fixes discovered in **EXP-028B0**.

### Primary Comparison

| System | Word Grounding F1 | Word Precision | Word Recall | Page Grounding F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **EXP-026 Production Baseline** | **55.98%** | 62.11% | 52.26% | 81.28% | Production Baseline |
| **EXP-028B0 True Text Oracle** | **75.12%** | 81.23% | 71.26% | 83.85% | Oracle (Upper Bound) |
| **EXP-028B1 Production** | **56.05%** | **61.73%** | **52.57%** | **81.66%** | **Official Production B1** |

- **Delta vs EXP-026 Baseline:** +0.07pp
- **Remaining Oracle-to-Production Gap:** 19.07pp
- **Remaining Gap to 90% Target:** 33.95pp

---

## 2. Cohort Breakdowns

### A. Development vs. Held-Out Cohorts

| Cohort | Word Grounding F1 | Word Precision | Word Recall | Page Grounding F1 | Document Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cohort A (Development)** | 65.33% | 66.88% | 64.13% | 92.50% | 32 |
| **Cohort B (Held-Out)** | 59.36% | 63.78% | 56.52% | 85.44% | 32 |

### B. Document Length Splits

| Split | Word Grounding F1 | Word Precision | Word Recall | Page Grounding F1 | Document Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Short (<= 10 pages)** | 52.00% | 59.22% | 47.70% | 84.47% | 252 |
| **Medium (11-50 pages)** | 67.35% | 68.88% | 65.98% | 75.24% | 98 |
| **Long (> 50 pages)** | 68.81% | 69.11% | 68.53% | 77.29% | 20 |

---

## 3. Oracle-to-Production Gap Classification (Section 8)

| Failure Category | Field Count | Share of Gap | Description |
| :--- | :---: | :---: | :--- |
| **A. wrong candidate selected** | 6986 | 1.40% | Fields recoverable by oracle text spans but failing in production ranking/selection. |
| **C. repeated occurrence** | 881 | 0.18% | Fields recoverable by oracle text spans but failing in production ranking/selection. |
| **D. table row/column assignment** | 488861 | 98.14% | Fields recoverable by oracle text spans but failing in production ranking/selection. |
| **F. multiline** | 331 | 0.07% | Fields recoverable by oracle text spans but failing in production ranking/selection. |
| **G. normalization** | 1081 | 0.22% | Fields recoverable by oracle text spans but failing in production ranking/selection. |

---

## 4. Decision Gate (Section 9)

**Result:** `56.05%`  
**Decision:** The candidate generation fixes have high oracle potential (75.12%), but without gold page supervision, production candidate ranking/selection requires disambiguation.

---

## 5. Artifacts

- `results.json`: Full metrics payload matching ExtractBench schemas.
- `per_document.csv`: Detailed evaluation metrics for every document.
- `field_failure_analysis.parquet`: Field-level trace with failure classifications.
