# Observer V2: Authoritative Benchmark Microscope Methodology

**Document Version**: Observer V2 (Frozen Research Baseline)  
**Author**: TonerHound Research Engineering  
**Date**: September 23, 2026  
**Status**: Authoritative Frozen Measurement Baseline  

---

## 1. Executive Summary

Observer V2 provides a strictly reconciled, mathematically verified measurement microscope for TonerHound evaluated against ExtractBench (370 documents). It observes the actual candidate pool and citations produced by the real current TonerHound production execution without modifying production behavior.

This methodology formalizes:
1. Exact definitions of Candidate Hit @ K, Recall@K, Oracle Grounding, and official evaluation metrics.
2. The mathematical resolution of the previous report's contradiction between reported Recall@5 (64.84%) and case-partition implied Top-5 presence (70.47%).
3. Explicit separation between Micro-aggregates (field-level) and Macro-aggregates (document-level).
4. Rigorous multi-evidence gold bounding box handling.

---

## 2. Core Definitions & Measurement Standards

### 2.1 Gradeable Benchmark Field
A benchmark field instance $f$ in a document $d$ is defined as **gradeable** for bounding-box grounding if and only if the ground truth test rule carries at least one physical bounding box annotation:
$$\text{Gradeable}(f) \iff \text{len}(\text{GoldBoxes}(f)) > 0$$
Fields carrying only value annotations or page annotations without bounding boxes are gradeable for Value F1 and Page Grounding, but strictly excluded from Word Grounding evaluation. Across the 370 ExtractBench documents, there are exactly:
- **Total evaluated documents**: 370
- **Grounded documents carrying GT bboxes**: 236
- **Ungrounded documents carrying value/page GT only**: 134
- **Total gradeable leaf field instances**: 445,950

### 2.2 Critical Definition: Candidate Hit
A candidate $c = (p_c, b_c)$ with page $p_c$ and normalized bounding box $b_c = (x, y, w, h)$ is a **geometrically correct candidate** for field $f$ with accepted gold evidence entries $G(f) = \{(p_g, b_g)_1, \dots, (p_g, b_g)_m\}$ if and only if:
$$\text{CandidateHit}(c, f) \iff \max_{(p_g, b_g) \in G(f), p_g = p_c} \text{IoU}(b_c, b_g) \ge 0.50$$
Where IoU between two bounding boxes $a$ and $b$ is computed as:
$$\text{IoU}(a, b) = \frac{\text{Area}(a \cap b)}{\text{Area}(a \cup b)}$$

### 2.3 Candidate Hit @ K
For field $f$, let $\mathcal{C}_K(f) = [c_1, c_2, \dots, c_K]$ denote the top $K$ candidates produced and ranked by the system. Then:
$$\text{CandidateHit@K}(f) \iff \exists c \in \mathcal{C}_K(f) \quad \text{such that} \quad \text{CandidateHit}(c, f) \text{ is True}$$
This definition is strictly uniform across all values of $K \in \{1, 3, 5, 10, 20\}$, across case partitions, and across oracle computations.

### 2.4 Multiple Accepted Gold Evidence Entries
ExtractBench ground truth often defines multiple accepted evidence locations (e.g. repeated table headers, summary rows, or multi-page citations).
Observer V2 **never arbitrarily selects a single gold box**. For every candidate $c$, the microscope computes:
$$\text{IoUByGold}[i] = \text{IoU}(b_c, b_{g,i}) \quad \text{if } p_c = p_{g,i} \text{ else } 0.0$$
$$\text{BestIoU}(c) = \max_i \text{IoUByGold}[i]$$
$$\text{BestGoldIndex}(c) = \arg\max_i \text{IoUByGold}[i]$$

---

## 3. Metric Formulations & Denominators

### 3.1 Official Word Grounding F1 (Macro-Average, $N=236$)
ExtractBench's official evaluator (`ExtractEvaluator` / `compute_unified_evidence_metrics`) computes Word Grounding F1 under Hungarian row alignment for array tables:
- **True Positives ($TP_w$)**: Aligned predicted cells that match value AND match page AND achieve $\text{IoU} \ge 0.50$ against gold evidence.
- **Precision Denominator ($g_{\text{claims}}$)**: Predicted citations emitted on cells aligned to a bbox-bearing gold cell.
- **Recall Denominator ($g_{\text{expected}}$)**: Total gold cells carrying a bounding box annotation.
- **Per-Document F1**:
  $$\text{Word F1}_d = \frac{2 \cdot TP_{w,d}}{g_{\text{claims},d} + g_{\text{expected},d}}$$
- **Official Dataset Word Grounding F1**: Unweighted macro-average across all $N=236$ grounded documents:
  $$\text{Macro Word Grounding F1} = \frac{1}{236} \sum_{d \in \text{GroundedDocs}} \text{Word F1}_d = 45.31\%$$

### 3.2 Official Page Grounding F1 (Macro-Average, $N=236$)
- **True Positives ($TP_p$)**: Aligned predicted cells that match value AND whose cited page is in the gold accepted page set.
- **Precision Denominator ($p_{\text{claims}}$)**: Predicted citations on cells aligned to page-annotated gold cells.
- **Recall Denominator ($p_{\text{expected}}$)**: Total gold cells carrying a page annotation.
- **Official Dataset Page Grounding F1**:
  $$\text{Macro Page Grounding F1} = \frac{1}{236} \sum_{d \in \text{GroundedDocs}} \text{Page F1}_d = 81.28\%$$

### 3.3 Candidate Recall@K
Candidate Recall@K is computed across two distinct, complementary aggregation modes:

1. **Micro Candidate Recall@K (Field-Level, Denominator = 445,950 fields)**:
   $$\text{Micro Recall@K} = \frac{\sum_{f \in \text{AllGradeableFields}} \mathbb{I}(\text{CandidateHit@K}(f))}{445,950}$$
2. **True Macro Candidate Recall@K (Document-Level, Denominator = 236 documents)**:
   For each grounded document $d$ with $M_d$ gradeable fields:
   $$\text{DocRecall@K}_d = \frac{1}{M_d} \sum_{f \in \text{DocGradeableFields}} \mathbb{I}(\text{CandidateHit@K}(f))$$
   $$\text{True Macro Recall@K} = \frac{1}{236} \sum_{d \in \text{GroundedDocs}} \text{DocRecall@K}_d$$

### 3.4 Top-K Oracle Grounding F1
For each $K \in \{1, 3, 5, 10, 20\}$, an oracle prediction set is constructed by substituting the system's selected candidate with the candidate from the top $K$ candidates having the highest IoU against accepted gold evidence.
The oracle prediction is then scored using the official ExtractBench evaluator (`compute_unified_evidence_metrics`), yielding the exact achievable Word Grounding F1 ceiling if candidate association within the top $K$ were solved.

---

## 4. Reconciliation of Previous Contradictions

The previous report presented:
- Reported Recall@5 = **64.84%**
- Case-partition implied Top-5 presence = **70.47%** (4.16% + 8.17% + 58.14%)

Observer V2 uncovers the exact mathematical causes of this discrepancy:
1. **Flaw 1 (Polluted Macro Average Across Ungrounded Documents)**:
   In previous instrumentation (`microscope_runner.py`), when a document had zero gold bounding boxes (`total_gradeable_leaves == 0`, 134 ungrounded docs), its `recall_at_k` was hardcoded to $1.0$ (100%).
   `run_full_observer.py` then averaged all 370 documents together:
   $$\frac{236 \times 44.88\% + 134 \times 100\%}{370} = 64.84\%$$
   The reported 64.84% was thus an artificial artifact of averaging 134 ungrounded documents hardcoded to 100%! The true macro Recall@5 across the 236 grounded documents is **44.88%**.
2. **Flaw 2 (Mixing Micro and Macro Aggregates)**:
   The case partition (29.53% + 4.16% + 8.17% + 58.14% = 100%) was computed as a **micro-average** across all 445,950 fields. Presenting the corrupted 370-doc macro recall (64.84%) alongside the micro case partition created an apples-to-oranges contradiction.
3. **Flaw 3 (Candidate Ranking Desynchronization)**:
   The previous observer re-scored candidates with a stripped context function that omitted row coordinates and structural reranking. In large array tables, this caused the top-1 candidate to desynchronize from TonerHound's actual output, misattributing passing fields to retrieval failures in candidate metrics while counting them as passes in case partitions.

Observer V2 eliminates all three flaws by freezing unified definitions, evaluating candidates directly from the production execution, and reporting micro and macro metrics with explicit numerators and denominators.
