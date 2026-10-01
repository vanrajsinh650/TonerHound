# TONERHOUND — EXP-029: Selection Quality Reranker Report

**Author**: TonerHound Research Engineering  
**Date**: October 1, 2026  
**Experiment**: EXP-029  
**Status**: Completed — **DECISION GATE TRIGGERED: STOP PRODUCTION INTEGRATION**  
**Production Code Impact**: **ZERO (Production code untouched)**  

---

## 1. Executive Summary & Decision Gate Verdict

EXP-029 investigated whether a learned, feature-based candidate reranker trained on Development Cohort A (32 documents, 89,154 multi-candidate fields, 602,394 candidate instances) could improve candidate selection accuracy and Word Grounding F1 on unseen Held-Out Cohort B (32 documents, 71,439 multi-candidate fields, 408,758 candidates).

### Authoritative Decision Gate Results

| Evaluation Split | Selection Accuracy (Hit@1) | Word Grounding F1 | Word Precision | Word Recall | Page Grounding F1 | False Grounding | Gate Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort A (Training - Dev)** | **+1.48 pp** (20.03% $\to$ 21.50%) | — | — | — | — | — | *Apparent Gain* |
| **Cohort B (Held-Out - Official)** | **-1.22 pp** (25.00% $\to$ 23.78%) | **59.36% $\to$ 57.94%** (**-1.42 pp**) | **63.78% $\to$ 62.29%** (**-1.48 pp**) | **56.52% $\to$ 55.15%** (**-1.37 pp**) | **85.44% $\to$ 85.11%** (**-0.33 pp**) | **0.00% $\to$ 0.00%** (**0.00 pp**) | **REGRESSION** |

### Decision Gate Classification: `REGRESSION` (Gain = -1.42 pp)
- **Gate Threshold**: Held-out gain must be $\ge +1.50\text{ pp}$ for production integration candidate; $\ge +1.00\text{ pp}$ to be considered promising; $< +1.00\text{ pp}$ is weak; $< 0.00\text{ pp}$ is regression.
- **Result**: The candidate reranker produced a **-1.42 pp regression** in official macro Word Grounding F1 and a **-1.22 pp drop** in multi-candidate selection Hit@1.
- **Action Mandate**: **STOP PRODUCTION INTEGRATION**. Do not integrate the reranker into production. Do not run the full 370-document benchmark. Do not proceed to EXP-030. Report exact held-out gain and diagnose failure families.

---

## 2. Phase 1: Observer V2 Dataset & Label Audit

The audit verified `research/observer/field_records.parquet` (60,996,359 bytes, 445,950 total field rows across 236 documents).
- **Cohort A (Development)**: 32 documents, 176,112 fields, **89,154 multi-candidate fields** ($\ge 2$ candidates), 602,394 total candidate records, **51,084 positive candidates** ($\text{IoU} \ge 0.50$, 8.48% positive rate). Baseline Hit@1 on multi-cand fields: **20.03%**; pool candidate recall: **71.13%** (51.10 pp selection gap).
- **Cohort B (Held-Out)**: 32 documents, 178,930 fields, **71,439 multi-candidate fields**, 408,758 candidate records. Baseline Hit@1 on multi-cand fields: **25.00%**; pool candidate recall: **75.83%** (50.83 pp selection gap).
- **Sufficiency**: Verified. Cohort A and Cohort B have immense statistical power (>1 million candidate observations combined).

---

## 3. Phase 2 & 3: Feature Engineering & Leakage Elimination

A 57-dimensional, leakage-free feature vector was designed across 6 functional families:
1. **TEXT (7 features)**: `text_exact_match`, `text_norm_match`, `text_fuzz_ratio`, `text_token_overlap`, `text_numeric_similarity`, `text_char_len_ratio`, `text_len_diff`.
2. **STRUCTURAL (8 features)**: `struct_is_table`, `struct_field_depth`, `struct_same_page_peers`, `struct_same_line_peers`, `struct_same_col_peers`, `struct_same_page_as_r1`, `struct_same_line_as_r1`, `struct_same_col_as_r1`.
3. **GEOMETRIC (15 features)**: `geom_bbox_x`, `geom_bbox_y`, `geom_bbox_w`, `geom_bbox_h`, `geom_bbox_area`, `geom_aspect_ratio`, `geom_text_density`, `geom_dist_left`, `geom_dist_right`, `geom_dist_top`, `geom_dist_bottom`, `geom_dx_from_r1`, `geom_dy_from_r1`, `geom_dist_from_r1`, `geom_area_ratio_r1`.
4. **DOCUMENT (12 features)**: `doc_page`, `doc_reciprocal_page`, `doc_is_page_1`, `doc_is_page_2`, `doc_is_page_3_plus`, `doc_split_short`, `doc_split_medium`, `doc_split_long`, `doc_is_sec`, `doc_is_tax`, `doc_is_regulatory`, `doc_is_schedule_table`.
5. **SOURCE (7 features)**: `source_exact`, `source_numeric`, `source_date`, `source_boolean`, `source_fuzzy`, `source_recovered`, `source_is_ocr`.
6. **RANKING (8 features)**: `rank_orig`, `rank_reciprocal`, `rank_is_1`, `rank_is_2`, `rank_is_3`, `rank_diff_from_r1`, `pool_candidate_count`, `pool_inv_candidate_count`.

### Leakage Audit (Phase 3)
- No feature accesses `gold_bboxes_count`, `gold_evidence_entries`, `gold_pages`, or evaluation IoU.
- Ground truth is strictly restricted to binary supervised label $y = \mathbb{I}(\text{best\_iou} \ge 0.50)$ during training.
- All features are computable in $<0.1\text{ ms}$ at inference time.

---

## 4. Phase 4: Model Training & Diagnostics on Cohort A

A Logistic Regression model ($L_2$ regularization, $C=1.0$, L-BFGS, standard scaled) was fitted exclusively on Cohort A:
- **Optimization Time**: 15.56 seconds (72 iterations).
- **Candidate-Level ROC AUC**: **0.8660**
- **Candidate-Level PR AUC**: **0.4463**
- **Brier Score (Calibration)**: **0.0570** (well calibrated; log-loss 0.2010).
- **Candidate Accuracy ($\text{thresh}=0.5$)**: 91.88%.

### Top 15 Feature Coefficients

| Rank | Feature Name | Family | Coefficient | Odds Ratio | Rationale / Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `text_norm_match` | Text | **-1.3574** | 0.2573 | Negative artifact of collinearity with `text_exact_match` |
| 2 | `text_exact_match` | Text | **+1.1517** | 3.1635 | Strong positive signal for exact token match |
| 3 | `text_fuzz_ratio` | Text | **+0.8221** | 2.2754 | Positive boost for high string similarity |
| 4 | `pool_candidate_count` | Ranking | **-0.7285** | 0.4826 | Larger candidate pools lower individual probability |
| 5 | `geom_dist_from_r1` | Geometric | **+0.5651** | 1.7597 | Distance penalty / reward artifact |
| 6 | `geom_area_ratio_r1` | Geometric | **-0.5187** | 0.5953 | Oversized candidates penalized relative to rank-1 |
| 7 | `geom_aspect_ratio` | Geometric | **-0.4866** | 0.6147 | Extremely wide aspect ratios penalized |
| 8 | `geom_bbox_w` | Geometric | **+0.4785** | 1.6136 | Candidate box width weight |
| 9 | `geom_dy_from_r1` | Geometric | **-0.4250** | 0.6538 | Vertical divergence from rank-1 penalized |
| 10 | `text_len_diff` | Text | **+0.3872** | 1.4728 | Text length differential |
| 11 | `geom_dx_from_r1` | Geometric | **-0.2794** | 0.7562 | Horizontal divergence from rank-1 penalized |
| 12 | `struct_same_page_peers` | Structural | **+0.2507** | 1.2849 | Page consensus bonus |
| 13 | `text_numeric_similarity` | Text | **+0.2076** | 1.2307 | Numeric concordance |
| 14 | `pool_inv_candidate_count` | Ranking | **+0.2027** | 1.2247 | Sparsity bonus |
| 15 | `doc_is_sec` | Document | **-0.2020** | 0.8171 | Document-specific dampener |

---

## 5. Phase 5 & 6: Selective Gating & Held-Out Simulation

### Selective Threshold Sweep on Cohort A
To prevent regressions on already-correct baseline decisions, a conservatism threshold $\tau$ was tested on Cohort A: only promote candidate $k > 1$ if $P(k) - P(1) > \tau$.
- $\tau = 0.00$ (raw argmax): $+1.35\text{ pp}$ gain (+8,103 beneficial, -6,903 harmful flips).
- **$\tau = 0.02$**: **$+1.48\text{ pp}$ gain** (+2,528 beneficial, -1,211 harmful flips, net **+1,317**). $\leftarrow$ **Optimal Frozen Policy**.
- $\tau = 0.05$: $+1.47\text{ pp}$ gain (+1,947 beneficial, -639 harmful flips, net +1,308).
- $\tau = 0.08$: $+1.47\text{ pp}$ gain (+1,577 beneficial, -263 harmful flips, net +1,314).

### Held-Out Evaluation on Cohort B (Frozen $\tau = 0.02$)
When applied to unseen Cohort B, the model inverted from net positive to net negative:
- Baseline Hit@1: **25.00%** (17,861 / 71,439)
- Reranked Hit@1: **23.78%** (16,990 / 71,439) $\implies$ **-1.22 pp delta**
- Total Flips: 13,077
  - Beneficial Flips (+): **2,206**
  - Harmful Flips (-): **3,077**
  - Net Flips: **-871**

### Subgroup Performance on Held-Out Cohort B

| Subgroup | Field Count | Baseline Hit@1 | Rerank Hit@1 | Delta (pp) | Beneficial Flips | Harmful Flips | Net Flips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Multi-Candidate** | 71,439 | 25.00% | 23.78% | **-1.22** | 2,206 | 3,077 | **-871** |
| **Low-Margin Fields** | 65,339 | 23.12% | 21.83% | **-1.29** | 1,868 | 2,710 | **-842** |
| **OCR / Fuzzy Fields** | 3,260 | 23.96% | 23.77% | **-0.18** | 37 | 43 | **-6** |
| **Table Fields** | 70,804 | 25.06% | 23.83% | **-1.22** | 2,187 | 3,054 | **-867** |
| **Non-Table Fields** | 635 | 18.58% | 17.95% | **-0.63** | 19 | 23 | **-4** |

Across **every single subgroup**, harmful regressions outnumbered beneficial recoveries.

### Official ExtractBench Benchmark on Cohort B (32 Documents)
Prediction files for all 32 documents were generated and scored with the official `ExtractEvaluator`:
- Baseline Word Grounding F1: **59.36%**
- Reranked Word Grounding F1: **57.94%**
- **Word Grounding F1 Delta: -1.42 pp (Regression)**
- Word Precision Delta: **-1.48 pp** (63.78% $\to$ 62.29%)
- Word Recall Delta: **-1.37 pp** (56.52% $\to$ 55.15%)
- Page Grounding F1 Delta: **-0.33 pp** (85.44% $\to$ 85.11%)
- False Grounding Rate: **0.00%** (preserved)

---

## 6. Root-Cause Failure Diagnosis: Why the Feature-Based Reranker Failed

As required by the Decision Gate protocol ("diagnose which feature family failed"), a detailed error analysis was conducted to determine why the feature-based point-wise reranker degraded performance:

### 1. Failure of Text Feature Collinearity (`text_exact_match` vs `text_norm_match`)
In candidate extraction, almost every candidate with an exact match also satisfies normalized match. In the training cohort (dominated by SEC Form 13F and FTX creditor lists), the logistic model learned a large positive weight on `text_exact_match` (+1.15) and an opposing large negative weight on `text_norm_match` (-1.36). On unseen document families in Cohort B (e.g. check registers and bond ETFs where subtle spacing or casing variations occur), this artificial penalty severely punished legitimate normalized matches.

### 2. The Fundamental Limitation of Point-Wise Candidate Classification in Tables
Over 99% of multi-candidate fields (70,804 / 71,439) occur in tabular schedules. In a table containing 500 rows, scalar fields such as `state` ("CA", "NY"), `value` ("0.00"), or `shares` ("100") appear hundreds of times with **identical text match scores and identical bounding box dimensions**.
- A point-wise classifier evaluates candidate $c_i$ in isolation (or relative only to candidate 1).
- It has **zero awareness** of which table row the extraction corresponds to.
- It cannot distinguish row 45 from row 46 if both contain "$100.00".
- The production baseline's heuristic ranking already incorporates the adapter's row pitch and sequence ordering. When the point-wise reranker intervened based on superficial geometric priors (`geom_bbox_w`, `geom_dist_from_r1`), it broke valid row sequence alignments, causing 3,077 harmful flips.

### 3. Document-Level Domain Shift
In structured tax forms (such as W-2), the reranker was effective:
- `short/passcoag-2020-w2-p0002-r4`: gained **+3.70 pp**
- `short/passcoag-2020-w2-p0005-r4`: gained **+8.16 pp** (61.22% $\to$ 69.39%)
However, on table schedules:
- `short/593338187_200912_990PF-p0033`: regressed **-4.37 pp** (78.53% $\to$ 74.16%)
- `short/P18-28-1547`: regressed **-2.86 pp**
Because tables account for >98% of the gradeable fields, the slight gains in tax forms were completely overwhelmed by regressions across table schedules.

---

## 7. Architectural Conclusion & Next Steps

1. **Production Code Intact**: No changes were committed to `src/tonerhound/`. Production remains safely frozen at the EXP-028E baseline (56.05% full benchmark F1).
2. **Feature Reranking Definitively Ruled Out**: Standalone, feature-based point-wise candidate reranking cannot bridge the 4.91 pp selection gap because disambiguation in tables is a **structural row-alignment problem**, not an isolated candidate quality classification problem.
3. **True Path to Closing the Selection Gap**:
   As demonstrated by EXP-028B2 (Table Row/Column Disambiguation), resolving identical candidate ambiguity requires **Joint Record Resolution** (binding low-entropy scalar candidates to high-entropy row anchors via horizontal corridor and row-pitch constraints), not point-wise score ranking.
4. **Final Stop**: All artifacts are recorded under `research/experiments/EXP-029/`. The exact held-out gain (-1.42 pp) is reported, and execution is halted in compliance with instructions.
