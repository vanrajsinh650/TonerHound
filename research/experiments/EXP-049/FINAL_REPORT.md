# TONERHOUND — EXP-049: IRS FORM LINE-ITEM GRID LOCALIZER
## FINAL REPORT

**Experiment ID:** EXP-049  
**Baseline:** EXP-048 (Word Grounding F1: 72.9918%, Page Grounding F1: 90.0613%, Passing: 352,570, Failing: 145,570)  
**Evaluated Documents:** 370 benchmark documents (236 evaluated, 134 unextractable/preserved)  
**Total Gradeable Fields:** 498,140  
**Passing Fields:** 352,687 (+117 net rescued)  
**Failing Fields:** 145,453 (-117 reduction)  
**Regressed Fields:** 0 (Strict Zero Regressions across all 370 benchmark documents)  
**Official Metrics:**
- **Word Grounding F1:** **73.1786%** (+0.1868 pp)
- **Page Grounding F1:** **90.1317%** (+0.0704 pp)
- **Word Precision:** **78.1118%** (+0.0383 pp)
- **Word Recall:** **69.7305%** (+0.4163 pp)

---

## Executive Summary

Following the diagnostic-first document repair paradigm from EXP-048, EXP-049 targeted the IRS Tax Return document family. The EXP-048 distribution audit identified 18 documents in the 40–50% Word Grounding F1 bucket, of which all tax filings (`becerra`, `bar-lev`, `cabrera`, `arif`) shared standardized IRS Form 1040 line numbers, column rails, and schedules.

### Diagnostic Discoveries
1. **Form Layout Regularity:** IRS Form 1040 partitions lines strictly across pages: Page 1 holds lines 1–15, Page 2 holds lines 16–38, and Schedule 1 spans Pages 3–4.
2. **Three Core Failure Modes:**
   - **`WRONG_ROW` (351 failures):** Core dollar totals cascade down the return (e.g., Line 16 Tax -> Line 18 -> Line 22 -> Line 24 Total Tax). When identical dollar amounts recur across multiple rows, unguided global token search collapsed citations onto the wrong row.
   - **`PAGE_DRIFT` (386 failures):** Identical values recur across supporting schedules (Schedule 1, Schedule D, Form 8995). The baseline drifted to later pages (e.g. Page 24) instead of grounding to the primary Form 1040 line item.
   - **`ZERO_CANDIDATE` (299 failures):** Unaligned candidates failed to locate numeric tokens within dense tabular OCR streams.
3. **Physical Token Sparsity:** Of 1,268 failing `line_` fields in the target documents, **708 fields are blank or unchecked checkboxes** (`[ ]`) having zero OCR tokens inside the gold region. Only ~530 failing fields contained physical OCR characters. Furthermore, tight single-digit tokens (e.g. `0`, `3`) required horizontal cell padding to satisfy ExtractBench's coarse entry cell IoU threshold (IoU ≥ 0.50).

### Experimental Outcomes
The developed **Line-Item Grid Localizer** (`irs_line_grid.py`) anchors schema fields to detected line tokens (`^\d{1,2}[a-z]?$`), restricts vertical candidate search to `y ∈ [anchor_y - 0.008, anchor_y + 0.008]`, bounds the value column rail (`x ∈ [0.78, 0.98]`), and pads cell widths.
- **Isolated Testing (Phase C):** Rescued **117 fields** across 13 target documents with **0 regressions** and 2.95s runtime overhead.
- **Held-Out Cohort Gate (Phase D):** Rescued **51 fields** across the 4 tax documents in the 32-document cohort with **0 regressions** and 16.43s latency. Marked as **FAIL** against the rigid `>= 300` threshold due to limited tax form representation in the frozen cohort (4 docs total, ~120 reachable OCR tokens).
- **Full Benchmark (Phase E):** Achieved **73.1786% Word Grounding F1** (+0.1868 pp) and **90.1317% Page Grounding F1** (+0.0704 pp), rescuing **117 net fields** with **strict zero regressions** across all 370 benchmark documents.

---

## Mandatory Section 8 Answers to Directive Questions

### 1. Baseline (EXP-048)?
- **Word Grounding F1:** 72.9918%
- **Page Grounding F1:** 90.0613%
- **Word Precision:** 78.0735%
- **Word Recall:** 69.3142%
- **Passing Fields:** 352,570 / 498,140
- **Failing Fields:** 145,570

### 2. How many target documents identified?
- **15 IRS tax documents total** across the benchmark, of which **6 documents** were in the [30%, 50%) Word F1 tail bucket (`medium/cabrera-2023`, `medium/becerra-2021`, `medium/cabrera-2022`, `medium/becerra-2024`, `medium/bar-lev-2024`, `medium/becerra-2022`). All 15 documents share the IRS Form 1040/8879 geometry.

### 3. How many distinct form types?
- **1 distinct form family:** IRS Form 1040 (U.S. Individual Income Tax Return) with supporting Schedules (Schedule 1, 2, 3, A, B, C, D, E, SE, 8812) and Form 8879 (IRS e-file Signature Authorization).

### 4. How many line anchors detected per form?
- **Average of 87.2 line anchors** detected per document across Pages 1–4 (range: 72 to 98 anchors per document).
  - Page 1 (Form 1040 Lines 1–15): average 18.3 anchors (range: 7–29)
  - Page 2 (Form 1040 Lines 16–38): average 34.1 anchors (range: 28–37)
  - Page 3 (Schedule 1 Part I): average 14.7 anchors (range: 9–23)
  - Page 4 (Schedule 1 Part II): average 20.3 anchors (range: 7–24)

### 5. How many schema fields successfully anchored?
- **14,354 total schema fields** across the 15 documents had field paths matching IRS line-number patterns (average of 956.9 line-numbered fields per document).
- In the failing subset analyzed during Phase A, **1,143 candidate fields** were evaluated against detected line anchors.

### 6. How many fields had no anchor (fallback)?
- **5,026 schema fields** across the 15 documents lacked line-number prefixes (e.g. `taxpayer_name`, `ssn`, `filing_status`, `signature_date`, `routing_number`). All safely triggered fallback preservation to prevent baseline degradation.

### 7. Fields rescued in isolation?
- **117 fields rescued** in isolation across 13 target documents (`research/experiments/EXP-049/results/fix_metrics.json`).

### 8. Fields regressed in isolation?
- **0 fields regressed** (verified in isolation testing).

### 9. Runtime overhead?
- **2.95 seconds** across all 15 target tax documents in isolation.
- **16.43 seconds** across the 32 documents in the held-out gate.

### 10. Held-out gate: PASS / FAIL?
- **FAIL** against the rigid `>= 300` rescues threshold.
  - Net Rescues: **+51**
  - Regressions: **0**
  - Latency: **16.43s** (threshold: ≤ 300s)
  - Cause of shortfall: The frozen 32-document held-out cohort contains only 4 tax documents (`bar-lev-2024`, `becerra-2021`, `becerra-2022`, `arif-2021`). These 4 documents possess only ~120 total failing fields containing physical OCR characters; the remaining ~430 failing fields are blank boxes or checkboxes without OCR text. Recovering 51 fields represents a 42.5% recovery rate of reachable text fields, but mathematically could not reach 300.

### 11. Full benchmark: Word F1 / Page F1 / passing / failing?
- **Word Grounding F1:** **73.1786%**
- **Page Grounding F1:** **90.1317%**
- **Word Precision:** **78.1118%**
- **Word Recall:** **69.7305%**
- **Passing Fields:** **352,687**
- **Failing Fields:** **145,453**

### 12. Delta vs EXP-048?
- **Word Grounding F1:** **+0.1868 pp** (72.9918% -> 73.1786%)
- **Page Grounding F1:** **+0.0704 pp** (90.0613% -> 90.1317%)
- **Net Fields Rescued:** **+117 fields** (352,570 -> 352,687)
- **Net Fields Regressed:** **0**

### 13. Top 5 documents improved?
1. `medium/bar-lev-2024`: **+13.7498 pp** (47.0842% -> 60.8340%, +33 rescued)
2. `medium/bar-lev-2023`: **+9.1585 pp** (54.6237% -> 63.7822%, +22 rescued)
3. `medium/bar-lev-2021`: **+4.3611 pp** (50.5747% -> 54.9358%, +12 rescued)
4. `medium/cabrera-2021`: **+3.6562 pp** (52.9262% -> 56.5824%, +8 rescued)
5. `medium/bar-lev-2022`: **+2.3648 pp** (54.2797% -> 56.6445%, +6 rescued)

*(Additional improvements: `short/arif-2024`: +1.9774 pp, `medium/becerra-2021`: +1.9514 pp, `medium/becerra-2023`: +1.8874 pp, `medium/becerra-2022`: +1.5354 pp, `medium/cabrera-2023`: +1.5290 pp).*

### 14. Any document regressed?
- **None.** Verified in `research/experiments/EXP-049/results/regression_analysis.json`: 0 documents and 0 fields regressed out of 370 benchmark documents.

### 15. Failure class migration?
- `WRONG_ROW`: 351 -> 299 (**-52 failures / -14.8%**)
- `PAGE_DRIFT`: 386 -> 346 (**-40 failures / -10.4%**)
- `ZERO_CANDIDATE`: 299 -> 276 (**-23 failures / -7.7%**)
- `INSUFFICIENT_IOU`: 102 -> 102 (unchanged)
- `MALFORMED_BBOX`: 5 -> 5 (unchanged)

### 16. Remaining gap to 75%?
- Baseline gap (EXP-048): 2.0082 pp
- Current Word Grounding F1: **73.1786%**
- Remaining gap to 75.0000%: **1.8214 pp** (-0.1868 pp reduction).

### 17. Remaining gap to 80%?
- Baseline gap (EXP-048): 7.0082 pp
- Remaining gap to 80.0000%: **6.8214 pp** (-0.1868 pp reduction).

### 18. Should this fix be integrated into production?
- **REFINE / HOLD BEFORE PRODUCTION MERGE.**
- While the fix is strictly safe and monotonic (+117 rescues, +0.1868 pp macro lift, 0 regressions across all 370 documents), it did not meet the gate threshold of ≥ 300 rescues on the held-out cohort because only 4 tax documents exist in the held-out split and >70% of tax form failures are empty/blank checkbox cells.
- The module should remain preserved in `research/experiments/EXP-049/fixes/` as an accepted research component and be merged into `src/tonerhound/` once blank cell bounding box synthesis is introduced, or when family-calibrated gate thresholds are adopted.

### 19. What other document families should be targeted next?
1. **Bankruptcy Creditor Matrices & Schedules (`real_ftx_full_corrupted`, `real_imedia_full_corrupted`):** 34,575 failing fields accounting for **4.32 pp of unharvested macro lift potential**.
2. **OFAC Sanctions Multi-Line Continuation Entities (`real_ofac_ssi_full`):** 7,934 failing fields where continuation blocks separated by semicolons are prematurely truncated.
3. **Fund Portfolio Holdings Schedules (`real_ishares_*`, `real_fidelity_*`):** Multi-column asset allocation tables with sub-column alignment challenges.

### 20. What is the next experiment?
- **EXP-050: Bankruptcy Creditor Schedule Multi-Column Block Localizer.** Target the massive failure cluster in `real_ftx_full` and `real_imedia_full` to capture the 4.32 pp macro lift required to surpass the 75% milestone.
