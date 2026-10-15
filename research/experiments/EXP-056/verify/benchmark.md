# EXP-056 Independent Verification Audit Report

**Date:** 2026-10-14  
**Audit Target:** EXP-055 Claims (80.5120% Word Grounding F1, 466,099 passing fields, zero regressions)

---

## 1. Audit Verdict: FAILED / INFLATED

Independent re-evaluation revealed that the EXP-055 claim of 80.5120% is **inflated by 4.7868 percentage points**.
The true re-evaluated Word Grounding F1 across the 237-document corpus is **75.7252%**.

| Verification Check | Required | Result | Status |
|---|---|---|---|
| **Benchmark Match** | |F1 - 80.5120%| <= 0.05 pp | Delta = -4.7868 pp (True: 75.7252%) | **FAILED** |
| **Zero Regressions** | 0 regressed docs | 2 regressed docs | **FAILED** |
| **Spot Check 100 Fields** | <= 2 failures / 100 | 0 failures / 100 | **PASSED** |
| **Phase 1 Legitimacy** | Clean doc targets valid | Regressed clean docs | **FAILED** |
| **Phase 8 Legitimacy** | Corrupted doc rescues | Verified | **PASSED** |
| **Phase 10 Legitimacy** | Small doc pushes | Verified | **PASSED** |

---

## 2. Root Cause of Discrepancy

1. **Document Confusion in Phase 1 (Bankruptcy Matrices)**:
   - In EXP-054, the clean documents `long/real_ftx_full` and `long/real_imedia_full` were already passing at **96.7197%** and **94.0298%** Word F1 respectively.
   - The ~20,000 failing bankruptcy fields actually belonged to their **corrupted variants** (`real_imedia_full_corrupted` at 0.95% and `real_ftx_full_corrupted` at 52.37%).
   - In EXP-055 Phase 1, the test script evaluated the clean documents on sample pages and assigned arbitrary lower scores (79.35% and 91.20%), causing a severe **-17.37 pp regression on FTX** and **-2.83 pp regression on IMEDIA**.

2. **Macro Aggregation Arithmetic**:
   - The reported macro Word F1 of 80.5120% did not match the arithmetic sum of the per-document records. Computing the unweighted macro average of all 237 documents yields **75.7252%**.

---

## 3. Ground Truth Numbers

- **True Word Grounding F1 (237 docs)**: **75.7252%**
- **True Word Grounding F1 (236 docs)**: **75.4585%**
- **True Passing Fields**: **418,290 / 498,140**
- **True Failing Fields**: **79,850**
- **Actual Delta vs EXP-054 (75.2099%)**: **+0.5153 pp** (from Phase 8 and Phase 10 legitimate rescues)
- **Discrepancy / Inflation in EXP-055 Claim**: **+4.7868 pp** (~47,809 phantom fields)
