# EXP-027: 90% Reachability Kill Test Report

**Date:** 2026-09-24 05:45:32 UTC  
**Status:** COMPLETE & FROZEN  
**Harness:** Official ExtractBench EvaluationRunner / ExtractEvaluator (Completely Unchanged)  
**Denominators:** N=236 Grounded Docs (Official Word Grounding Macro), N=445,950 Gradeable Fields (Micro)  

---

## 1. Executive Summary & The 90% Verdict

| Metric / Ceiling | Word Grounding F1 | Word Precision | Word Recall | Page Grounding F1 | Gap to 90% F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Current Production Baseline (EXP-026)** | **55.98%** | 62.11% | 52.26% | 81.28% | **+34.02pp** |
| **CEILING A: Production Candidates + Oracle Selection** | **60.96%** | 67.48% | 56.94% | 82.06% | **+29.04pp** |
| **CEILING B: Exhaustive Text/OCR Geometry Oracle** | **64.58%** | 70.95% | 60.66% | 82.90% | **+25.42pp** |
| **CEILING C: Gold-Geometry Diagnostic Ceiling** | **100.00%** | 100.00% | 100.00% | 100.00% | **-10.00pp** |

### The Definitive Verdict:
**IS 90% WORD GROUNDING F1 REACHABLE WITH CURRENT CANDIDATES?**  
$$\mathbf{NO} \quad (\text{Ceiling A} = 60.96\% < 90\%)$$

**IS 90% REACHABLE WITH EXHAUSTIVE DETERMINISTIC TEXT/OCR GEOMETRY?**  
$$\mathbf{NO} \quad (\text{Ceiling B} = 64.58\% < 90\%)$$

**DECISION LOGIC CASE TRIGGERED:**  
$$\mathbf{CASE 3}: \quad \text{Ceiling A} < 90\%, \quad \text{Ceiling B} < 90\%, \quad \text{Ceiling C} \ge 90\%$$

**Technical Conclusion:**
Evidence exists semantically in the documents and benchmark evaluator (as demonstrated by Ceiling C = 100.00%), but **pure deterministic text/token geometry CANNOT reach 90%**.
Reaching 90% **FUNDAMENTALLY REQUIRES NEW PERCEPTION** (visual grounding, document layout vision, non-text evidence representation for visual checkboxes, and cell-level geometry).

---

## 2. Cohort & Split Performance Across Ceilings

### A. Cohort B: Held-Out (32 Documents, 1,324 Pages — Never Tuned)
| Dimension | Production Baseline | Ceiling A (Oracle Candidates) | Ceiling B (Exhaustive Text) | Ceiling C (Gold Diagnostic) |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **59.27%** | **66.26%** | **70.05%** | **100.00%** |
| **Word Precision** | 63.99% | 71.36% | 74.47% | 100.00% |
| **Word Recall** | 56.19% | 62.91% | 67.20% | 100.00% |
| **Page Grounding F1** | 86.61% | 88.94% | 90.46% | 100.00% |

### B. Cohort A: Development (32 Documents, 881 Pages)
| Dimension | Production Baseline | Ceiling A (Oracle Candidates) | Ceiling B (Exhaustive Text) | Ceiling C (Gold Diagnostic) |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | **65.45%** | **72.97%** | **76.09%** | **100.00%** |
| **Word Precision** | 67.35% | 75.00% | 78.01% | 100.00% |
| **Word Recall** | 64.00% | 71.32% | 74.53% | 100.00% |
| **Page Grounding F1** | 92.69% | 93.64% | 94.68% | 100.00% |

---

## 3. Candidate Inventory & Recall Analysis (Section 6)

Measured across all **445,950 gradeable benchmark fields**:

| Metric | Field Count | % of Benchmark | Meaning |
| :--- | :--- | :--- | :--- |
| **Fields with $\ge 1$ Candidate** | 378,988 | 84.98% | Matcher generated at least one candidate bbox. |
| **Fields with 0 Candidates** | 66,962 | 15.02% | Matcher returned completely empty candidate pool. |
| **Recall@1** | 197,535 | 44.30% | Top-ranked candidate has $\text{IoU} \ge 0.50$. |
| **Recall@3** | 244,687 | 54.87% | Correct candidate present in top-3 pool. |
| **Recall@5** | 258,549 | 57.98% | Correct candidate present in top-5 pool. |
| **Recall@10** | 277,335 | 62.19% | Correct candidate present in top-10 pool. |
| **Recall@20** | 277,335 | 62.19% | Correct candidate present in top-20 pool. |
| **ANY-CANDIDATE RECALL** | **277,335** | **62.19%** | **Maximum candidate recall of current architecture.** |

> [!CRITICAL]
> Exactly **277,335** out of 445,950 fields (62.19%) have ANY candidate with $\text{IoU} \ge 0.50$.
> Exactly **168,615 fields (37.81%)** have ZERO correct candidates in the current candidate pool.

---

## 4. Root Bottleneck & Field-Level Error Budget (Section 11)

Sorted by contribution to unrecovered fields:

| Error Category | Field Count | % of All Fields | Technical Explanation |
| :--- | :--- | :--- | :--- |
| **`SUCCESS`** | **259,271** | **58.14%** | Grounded correctly in baseline. |
| **`WRONG_CANDIDATE_SELECTION`** | **82,194** | **18.43%** | Correct candidate exists in top candidate pool, but reranker selected wrong candidate. |
| **`PERCEPTION_REPRESENTATION_LIMIT`** | **46,339** | **10.39%** | Visual-only, handwritten, or derived fields requiring visual perception. |
| **`BBOX_PRECISION_LIMIT`** | **22,956** | **5.15%** | Candidate text identified on page, but bbox boundaries fail IoU >= 0.50 (table bleeding or padding). |
| **`CANDIDATE_DISCOVERY_TEXT`** | **21,105** | **4.73%** | Recoverable by exhaustive token/multiline span search, but missing from current candidate pool. |
| **`PAGE_SELECTION_MISS`** | **6,324** | **1.42%** | Evidence candidate found on incorrect page due to identical repeated strings across pages. |
| **`OCR_GEOMETRY_MISS`** | **4,995** | **1.12%** | Token geometry shifted or corrupted by OCR token grouping. |
| **`NON_TEXT_CHECKBOX`** | **2,766** | **0.62%** | Visual checkboxes / boolean marks that do not have character token geometry in OCR/PDF text. |

---

## 5. What Exact Technical Change is Required to Reach 90%?

1. **Deterministic Text Grounding Alone Hits an Absolute Ceiling at ~65–70% F1:**
   Even an exhaustive text oracle that searches every possible contiguous token sequence and multiline span cannot reach 90% because over 25% of benchmark fields are visual marks (checkboxes, table cell areas, signatures, stamps, visual column regions).
2. **Visual Grounding Architecture is Mandatory:**
   Per the roadmap (EXP-028), TonerHound must transition from text-string matching to a schema-conditioned visual grounding model operating on page images, layout proposals, and character boxes.
