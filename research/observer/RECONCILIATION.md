# Forensic Metric Reconciliation: Observer V1 vs Observer V2

**Document**: Definitive Reconciliation of ExtractBench Measurement Discrepancies  
**Author**: TonerHound Research Engineering  
**Date**: September 23, 2026  
**Scope**: Full mathematical, algorithmic, and architectural audit of the 64.84% vs 70.47% Recall@5 conflict and associated metrics.

---

## 1. The Core Contradiction

A previous forensic report (`research/benchmark_observer/report.md` and `overall.json`) reported the following headline candidate metrics:
- **Recall@1**: 59.91%
- **Recall@3**: 63.82%
- **Recall@5**: 64.84%
- **Recall@10**: 65.62%
- **Recall@20**: 66.12%

Simultaneously, the report presented this four-way case partition across all gradeable benchmark fields:
- **CASE 1 (Not in Top-5)**: 29.53% (131,683 fields)
- **CASE 2 (Top-5 but not Top-1)**: 4.16% (18,553 fields)
- **CASE 3 (Top-1 hit but bbox failure)**: 8.17% (36,443 fields)
- **CASE 4 (Top-1 pass)**: 58.14% (259,271 fields)

### The Mathematical Impossibility
Summing all cases where correct evidence is present in the Top-5 candidates yields:
$$\text{CASE 2} + \text{CASE 3} + \text{CASE 4} = 4.16\% + 8.17\% + 58.14\% = 70.47\%$$
$$\text{Or equivalently: } 100\% - \text{CASE 1} = 100\% - 29.53\% = 70.47\%$$

A system cannot simultaneously have 70.47% Top-5 candidate presence and 64.84% Recall@5 if both metrics measure the same entity under consistent definitions.

---

## 2. Root Cause Analysis: The Three Forensic Bugs

Through meticulous line-by-line inspection of `research/benchmark_observer/microscope_runner.py` and `run_full_observer.py`, we have discovered the three independent bugs that generated this discrepancy.

### Root Cause 1: The Polluted Macro Average (The Origin of "64.84%")
In `research/benchmark_observer/microscope_runner.py` line 565:
```python
"recall_at_k": r_hits[K] / total_gradeable_leaves if total_gradeable_leaves > 0 else 1.0
```
Across the 370 ExtractBench documents:
- **236 documents** carry gold bounding boxes (`total_gradeable_leaves > 0`).
- **134 documents** carry only value or page annotations without bounding boxes (`total_gradeable_leaves == 0`).

For the 134 ungrounded documents, line 565 hardcoded `recall_at_k = 1.0` (100.0%)!

Then, in `research/benchmark_observer/run_full_observer.py` line 128:
```python
topk_summary[K] = {
    "candidate_recall": mean_val([r["topk_ceilings"][K]["recall_at_k"] for r in doc_results])
}
```
`run_full_observer.py` computed an unweighted arithmetic mean across **all 370 documents**, mixing the 236 genuine grounded documents with the 134 ungrounded documents that had been assigned 100%:
$$\text{Reported Recall@5} = \frac{236 \times 44.88\% + 134 \times 100.00\%}{370} = \frac{105.91 + 134.00}{370} = \frac{239.91}{370} = 64.84\%$$

The exact same corruption produced every reported recall figure:
- **Recall@1**: $(236 \times 37.15\% + 134 \times 100\%) / 370 = \mathbf{59.91\%}$
- **Recall@3**: $(236 \times 43.28\% + 134 \times 100\%) / 370 = \mathbf{63.82\%}$
- **Recall@5**: $(236 \times 44.88\% + 134 \times 100\%) / 370 = \mathbf{64.84\%}$
- **Recall@10**: $(236 \times 46.09\% + 134 \times 100\%) / 370 = \mathbf{65.62\%}$
- **Recall@20**: $(236 \times 46.89\% + 134 \times 100\%) / 370 = \mathbf{66.12\%}$

**Conclusion**: The reported 64.84% was never a real recall metric. It was an artificial number created by padding 134 ungroundable documents with a synthetic 100% recall. The true macro Recall@5 across the 236 grounded documents is **44.88%**.

---

### Root Cause 2: Mixing Micro and Macro Aggregates
The second flaw was presentation:
- The case breakdown (29.53%, 4.16%, 8.17%, 58.14%) was calculated as a **micro-average** across all 445,950 gradeable field instances in the entire benchmark:
  $$\frac{131,683}{445,950} = 29.53\%, \quad \frac{18,553}{445,950} = 4.16\%, \quad \frac{36,443}{445,950} = 8.17\%, \quad \frac{259,271}{445,950} = 58.14\%$$
- The report placed this micro-breakdown side-by-side with the corrupted 370-document macro-average (64.84%) without declaring the different denominators or aggregation levels.

---

### Root Cause 3: Candidate Ranking Desynchronization in Large Tables
In `microscope_runner.py` lines 293–300, candidate ranking was implemented as:
```python
scored_cands = resolver._score_candidates_with_context(cands, path)
scored_cands.sort(key=lambda it: it[1], reverse=True)
ranked_cands = [c for c, _s in scored_cands]
```
This naive re-ranking discarded the actual execution context that TonerHound's production adapter (`ExtractBenchAdapter`) used to produce its citations:
1. It omitted `y_hint`, `row_corridor`, `column_corridor`, and `sibling_boxes`.
2. It failed to invoke `StructuralReranker` (EXP-011) and `FlatFormLabelReranker` (EXP-012).
3. It failed to apply geometry enhancements (`_apply_geometry_enhancements`).

In documents containing large tables with repeated values (e.g. `long/real_ftx_full_corrupted` with 26,000 creditors having `"NAME ON FILE"`), the stripped context function assigned the candidate from the first row to rank 1 for all 1,000+ subsequent rows. Consequently:
- For rows 2 through 1,000, `ranked_cands[0]` was the box from row 1, which had $IoU = 0.0$ against the gold box of row $i$.
- Therefore, `microscope_runner.py` recorded `has_hit = False` for candidate recall, recording only 30 hits out of 26,583 rows!
- BUT when assigning `CASE 4`, line 401 checked:
  ```python
  elif grounded_correct:
      case_category = "CASE 4"
  ```
  Where `grounded_correct` was evaluated on the actual selected citation produced by `adapter.ground_extracted_data` (which DID use row corridors and row pitches). Because the adapter correctly located the row, `grounded_correct` was True for 14,206 fields!

Thus, 14,206 fields in that document alone were counted as `CASE 4` (implying Top-1 pass) while simultaneously being recorded as a miss in Candidate Recall@1/3/5/10/20!

---

## 3. Disputed Metric Discrepancy Matrix

| Disputed Metric | Previous Reported Value | True Recomputed Value | Exact Definition | Numerator | Denominator | Source Code Path | Root Cause of Discrepancy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate Recall@1** | 59.91% | **37.15% (Macro)**<br>**31.11% (Micro)** | $\exists c \in \text{Top-1}: \text{IoU} \ge 0.50$ | Macro: $\sum \text{doc\_r1}$<br>Micro: 138,757 | Macro: 236 docs<br>Micro: 445,950 fields | `research/observer/run_microscope_v2.py` | 134 ungrounded docs were hardcoded to 100% and averaged into macro aggregate. |
| **Candidate Recall@3** | 63.82% | **43.28% (Macro)**<br>**39.84% (Micro)** | $\exists c \in \text{Top-3}: \text{IoU} \ge 0.50$ | Macro: $\sum \text{doc\_r3}$<br>Micro: 177,649 | Macro: 236 docs<br>Micro: 445,950 fields | `research/observer/run_microscope_v2.py` | 134 ungrounded docs hardcoded to 100% in 370-doc average. |
| **Candidate Recall@5** | 64.84% | **44.88% (Macro)**<br>**42.51% (Micro)** | $\exists c \in \text{Top-5}: \text{IoU} \ge 0.50$ | Macro: $\sum \text{doc\_r5}$<br>Micro: 189,556 | Macro: 236 docs<br>Micro: 445,950 fields | `research/observer/run_microscope_v2.py` | $(236 \times 44.88\% + 134 \times 100\%) / 370 = 64.84\%$. Corrupted macro average. |
| **Candidate Recall@10** | 65.62% | **46.09% (Macro)**<br>**45.12% (Micro)** | $\exists c \in \text{Top-10}: \text{IoU} \ge 0.50$ | Macro: $\sum \text{doc\_r10}$<br>Micro: 201,214 | Macro: 236 docs<br>Micro: 445,950 fields | `research/observer/run_microscope_v2.py` | 134 ungrounded docs hardcoded to 100% in 370-doc average. |
| **Candidate Recall@20** | 66.12% | **46.89% (Macro)**<br>**47.53% (Micro)** | $\exists c \in \text{Top-20}: \text{IoU} \ge 0.50$ | Macro: $\sum \text{doc\_r20}$<br>Micro: 211,954 | Macro: 236 docs<br>Micro: 445,950 fields | `research/observer/run_microscope_v2.py` | 134 ungrounded docs hardcoded to 100% in 370-doc average. |
| **Top-5 Presence Sum** | 70.47% | **42.51% (Micro)** | Fields where Top-5 pool contains $\text{IoU} \ge 0.50$ | 189,556 | 445,950 fields | `research/observer/run_microscope_v2.py` | CASE 4 was defined as selected citation pass, not top-1 candidate pass. |
| **Word Grounding F1** | 45.31% | **45.31%** | Official Hungarian Word F1 | $\sum \text{Word F1}_d$ | 236 grounded docs | `ExtractEvaluator` / `compute_unified_evidence_metrics` | **CONFIRMED EXACT**. Unaffected by observer bugs. |
| **Page Grounding F1** | 81.28% | **81.28%** | Official Page F1 | $\sum \text{Page F1}_d$ | 236 grounded docs | `ExtractEvaluator` / `compute_unified_evidence_metrics` | **CONFIRMED EXACT**. Unaffected by observer bugs. |
| **Value F1** | 100.00% | **100.00%** | Official Value F1 | $\sum \text{Value F1}_d$ | 370 docs | `ExtractEvaluator` / `compute_unified_evidence_metrics` | **CONFIRMED EXACT**. Perfect extraction value alignment. |

---

## 4. Definitive Answer to the Research Question

1. **Was Recall@5 = 64.84% actually correct?**
   **NO**. 64.84% was a mathematical artifact caused by hardcoding `recall_at_k = 1.0` for 134 documents that have no ground truth bounding boxes, and averaging them across 370 documents: $(236 \times 44.88\% + 134 \times 100\%) / 370 = 64.84\%$.
2. **What is the true Recall@5?**
   - **True Macro Recall@5 (N=236 grounded documents)**: **44.88%**
   - **True Micro Recall@5 (445,950 gradeable fields)**: **42.51%** (189,556 hits)
3. **What is the true baseline candidate presence ceiling?**
   The candidate pool contains geometrically valid evidence ($IoU \ge 0.50$) for:
   - **Top-1**: 37.15% (Macro) / 31.11% (Micro)
   - **Top-5**: 44.88% (Macro) / 42.51% (Micro)
   - **Top-20**: 46.89% (Macro) / 47.53% (Micro)
4. **Why did the previous report claim candidate retrieval is saturated?**
   The claim that *"Retrieval is saturated at 64.84%"* was based entirely on the corrupted ungrounded document padding. In reality, candidate retrieval in TonerHound is **NOT saturated**—more than **52% of all gradeable fields have NO geometrically valid candidate in the Top-20 candidates**. Perception and candidate coverage are the primary limiting factors.
