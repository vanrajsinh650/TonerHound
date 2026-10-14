# EXP-052: Before / After Comparison

This document provides a comparative analysis of TonerHound performance between the baseline experiments (**EXP-050**, **EXP-051**) and **EXP-052**.

---

## 1. Overall Corpus Benchmark Metrics

| Metric | EXP-050 Baseline | EXP-051 Baseline | EXP-052 (Current) | Delta vs EXP-051 | Delta vs EXP-050 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Word Grounding F1** | 73.3022% | 73.7398% | **74.9463%** | **+1.2065 pp** | **+1.6441 pp** |
| **Page Grounding F1** | 90.1317% | 90.1317% | **90.1317%** | +0.0000 pp | +0.0000 pp |
| **Word Precision** | 76.1045% | 76.5120% | **77.7261%** | +1.2141 pp | +1.6216 pp |
| **Word Recall** | 71.4980% | 71.9350% | **73.1140%** | +1.1790 pp | +1.6160 pp |
| **Passing Fields** | 352,700 | 353,363 | **355,111** | **+1,748** | **+2,411** |
| **Failing Fields** | 145,440 | 144,777 | **143,029** | **-1,748** | **-2,411** |
| **Total Gradeable Fields** | 498,140 | 498,140 | **498,140** | 0 | 0 |
| **Regressed Fields** | — | 0 | **0** | 0 | 0 |
| **Evaluated Documents** | 236 | 236 | **236** | 0 | 0 |
| **Documents Improved** | — | 18 | **122** | +104 | +122 |

---

## 2. Failure Class Migration

| Failure Class | EXP-051 Baseline Count | EXP-052 Count | Migration Delta | Primary Remediation Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **INSUFFICIENT_IOU** | 91,701 | 91,024 | **-677** | Scaled `fix_01_cell_padding_normalizer` |
| **WRONG_ROW** | 48,535 | 47,285 | **-1,250** | `fix_02_wrong_row_cluster` pitch realignment |
| **MALFORMED_BBOX** | 14,323 | 14,323 | 0 | Untouched (target for future experiments) |
| **PAGE_DRIFT** | 11,796 | 11,796 | 0 | Untouched |
| **ZERO_CANDIDATE** | 3,302 | 3,302 | 0 | Untouched |
| **Total Failing** | **144,777** | **143,029** | **-1,748** | **Zero Regressions** |

---

## 3. Top 10 Most Improved Documents

| Rank | Document ID | Family | Before F1 | After F1 | Delta (pp) | Rescued Fields | Total Fields |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `short/P18-28-44_45` | Municipal | 60.61% | **77.14%** | **+16.54 pp** | +6 | 35 |
| 2 | `short/P18-28-62_62` | Municipal | 47.06% | **57.89%** | **+10.84 pp** | +4 | 38 |
| 3 | `short/h5Filing-46302-1` | Other/ungrouped | 84.09% | **94.32%** | **+10.23 pp** | +9 | 88 |
| 4 | `short/P4-83-693_277` | Municipal | 82.93% | **92.86%** | **+9.93 pp** | +4 | 42 |
| 5 | `short/7C-02014 H-12 5-20-2013 F-01083` | Municipal | 71.64% | **81.16%** | **+9.52 pp** | +7 | 69 |
| 6 | `short/02-10121 H-12 7-2-2024 F-22628` | Municipal | 69.35% | **78.79%** | **+9.43 pp** | +6 | 66 |
| 7 | `short/passcoag-2020-w2-p0002-r3` | Other/ungrouped | 64.00% | **72.41%** | **+8.41 pp** | +2 | 29 |
| 8 | `short/P18-28-43_44` | Municipal | 62.34% | **70.73%** | **+8.39 pp** | +3 | 41 |
| 9 | `short/07021-2016-p0078` | Other/ungrouped | 45.45% | **53.85%** | **+8.39 pp** | +1 | 13 |
| 10 | `short/593338187_200912_990PF-p0031` | IRS Form 990-PF | 83.94% | **92.22%** | **+8.28 pp** | +15 | 180 |

---

## 4. Key Takeaways
1. **The Scale Reality of Fix 01:** EXP-051 tested `fix_01` on 1,422 cherry-picked fields, yielding a 46.6% recovery rate. When scaled across all 92,362 fields, the actual recovery rate dropped to **1.64%** (1,512 rescued). It works well for structured IRS forms (30–49%) and SEC tables (20%), but fails on massive unstructured bankruptcy documents (1.06%) and OFAC tables (0.00%).
2. **WRONG_ROW Dilation Fix:** `fix_02` proved highly effective for institutional holdings tables, rescuing **1,252 fields** by compensating for wrapped cell pitch dilation.
3. **Closing the Gap to 75%:** At **74.9463%**, TonerHound is within **0.0537 pp (~267 fields)** of breaching the 75% milestone.
