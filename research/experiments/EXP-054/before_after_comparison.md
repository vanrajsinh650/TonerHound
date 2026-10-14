# EXP-054: Before vs. After Comprehensive Comparison

**Experiment:** EXP-054  
**Date:** October 14, 2026  
**Baseline:** EXP-053  

---

## 1. High-Level Benchmark Metrics Comparison

| Metric | Baseline (EXP-053) | Post-Fix (EXP-054) | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1 (237 docs, with Oklahoma)** | 75.0241% | **75.2099%** | **+0.1858 pp** | **Gate Maintained (v0.4.0 Released)** |
| **Word Grounding F1 (236 docs, standard suite)** | 74.9594% | **75.1459%** | **+0.1865 pp** | Continuous Growth |
| **Word Grounding F1 (370 docs, all-inclusive)** | N/A (guard triggered) | **48.1750%** | Defined | First All-370 Evaluation |
| **Page Grounding F1** | 90.1317% | **90.1733%** | **+0.0416 pp** | Lifted |
| **Total Evaluated Fields** | 498,140 | 498,140 | 0 | 100% Accounted (Gate V8) |
| **Passing Fields** | 394,909 | **411,098** | **+16,189** | Rescued |
| **Failing Fields** | 103,231 | **87,042** | **-16,189** | 15.7% Failure Reduction |
| **Regressions** | 0 | **0** | **0** | Strict Zero Regressions |
| **Gap to 80.00% Milestone** | 4.9759 pp | **4.7901 pp** | **-0.1858 pp** | Closing |
| **Gap to 90.00% Milestone** | 14.9759 pp | **14.7901 pp** | **-0.1858 pp** | Headroom mapped |
| **Theoretical Ceiling F1** | 99.9956% | **99.9956%** | Defined | 498,118 gradeable fields |

---

## 2. Failure Class Migration

Through the combined application of Fix 05 (OFAC Row Tracker) and Fix 06 (Malformed Bounding Box Repair), failure counts migrated as follows:

| Failure Taxonomy Class | EXP-053 Failures | EXP-054 Failures | Net Rescued | Resolution Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **WRONG_ROW** | 42,025 | **27,991** | **-14,034** | Fix 05 Localized Entity Row Tracking on OFAC |
| **MALFORMED_BBOX** | 22,505 | **20,350** | **-2,155** | Fix 06 Degenerate Bbox Repair on Holdings/ETFs |
| **INSUFFICIENT_IOU** | 31,546 | **31,546** | 0 | Stable (Target for EXP-056) |
| **PAGE_DRIFT** | 6,271 | **6,271** | 0 | Target for EXP-055 Multi-Page Tracking |
| **ZERO_CANDIDATE** | 863 | **863** | 0 | Stable |
| **UNGRADEABLE_NO_VALUE** | 22 | **22** | 0 | Ground truth empty/whitespace values (Ungradeable) |
| **TOTAL** | **103,231** | **87,042** | **-16,189** | **Zero Regressions** |

---

## 3. Targeted Document Performance Deltas

### A. `long/real_ofac_ssi_full` (Fix 05: Localized Entity Row Tracking)
- **Problem:** Repeated schema tokens (`a.k.a.`, `Secondary sanctions risk:`, `Tax ID No.`, `Russia`) bound to the first occurrence on each of the 192 pages.
- **Intervention:** Built `fix_05_ofac_row_tracker.py` isolating search to each entity's vertical window $[y_i, y_{i+1}]$.
- **Results:**
  - Gradeable Fields: 36,303
  - Passing Fields: 10,841 -> **24,875**
  - Failing Fields: 25,462 -> **11,428**
  - Rescued Fields: **+14,034 fields**
  - Regressions: **0**
  - Document Word Grounding F1: **29.86% -> 63.14% (+33.28 pp)**
  - Runtime: **7.16 seconds**

### B. Institutional Holdings & Bond ETFs (Fix 06: Degenerate Bbox Repair)
- **Problem:** Numeric zero values (`0`), dashes, and sub-pixel single characters collapsed into zero-width or degenerate bboxes.
- **Intervention:** Built `fix_06_malformed_bbox_repair.py` expanding degenerate coordinates to minimum canonical line heights and token spans.
- **Results:**
  - `long/real_ishares_iboxx_bond_etfs`: +1,071 fields rescued (F1: 67.42% -> **82.42%**, +15.00 pp)
  - `long/real_sm0801_eco_full_corrupted`: +310 fields rescued (F1: 52.14% -> **56.79%**, +4.65 pp)
  - `long/real_sm0801_ae_full`: +99 fields rescued (F1: 61.20% -> **62.69%**, +1.49 pp)
  - Other institutional schedules and municipal documents: +675 fields rescued
  - Total non-Oklahoma rescues: **+2,155 fields**

---

## 4. Top 10 Improved Documents Across Entire Benchmark

| Document ID | Type | Baseline F1 | EXP-054 F1 | F1 Delta | Rescued Fields |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `long/real_ofac_ssi_full` | OFAC Sanctions Stream | 29.86% | **63.14%** | **+33.28 pp** | **+14,034** |
| `long/real_ishares_iboxx_bond_etfs` | Institutional Holdings | 67.42% | **82.42%** | **+15.00 pp** | **+1,071** |
| `long/real_sm0801_eco_full_corrupted` | Financial Statement | 52.14% | **56.79%** | **+4.65 pp** | +310 |
| `long/real_sm0801_ae_full` | Financial Statement | 61.20% | **62.69%** | **+1.49 pp** | +99 |
| `medium/real_fidelity_northstar` | Fund Holdings | 71.30% | **72.32%** | **+1.02 pp** | +68 |
| `long/real_imedia_full_corrupted` | Bankruptcy Matrix | 34.50% | **35.31%** | **+0.81 pp** | +54 |
| `medium/real_blackrock_schedule_full` | Asset Schedule | 68.90% | **69.46%** | **+0.56 pp** | +37 |
| `medium/sec_13f_0031_loomis_sayles` | SEC 13F Filing | 74.20% | **74.74%** | **+0.54 pp** | +36 |
| `short/W14-52342_W14` | Property Assessment | 58.10% | **58.61%** | **+0.51 pp** | +34 |
| `short/W2-27-217314_213200` | Property Assessment | 64.20% | **64.70%** | **+0.50 pp** | +33 |

---

## 5. Regression Audit Summary

Every evaluated document was tested against baseline predictions under strict equality checks:
$$\Delta \text{Passing}_d \ge 0 \quad \forall d \in \mathcal{D}$$
- **Total Documents Audited:** 370
- **Total Evaluated Documents:** 237
- **Regressing Documents:** 0
- **Regressing Fields:** 0
- **Regression Rate:** **0.0000%**
