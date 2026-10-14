# EXP-050: Before vs After Comparison

## 1. Official Benchmark Comparison

| Metric | Baseline (EXP-049) | EXP-050 (Small Doc & Accounting) | Delta vs EXP-049 | Delta vs EXP-048 |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 73.1786% | **73.3022%** | **+0.1236 pp** | **+0.3104 pp** |
| **Page Grounding F1** | 90.1317% | **90.1317%** | **+0.0000 pp** | **+0.0704 pp** |
| **Word Precision** | 78.1118% | **78.2354%** | **+0.1236 pp** | **+0.1619 pp** |
| **Word Recall** | 69.7305% | **69.8541%** | **+0.1236 pp** | **+0.5399 pp** |
| **Passing Fields** | 352,687 | **352,700** | **+13 net rescued** | **+130 net rescued** |
| **Failing Fields** | 145,453 | **145,440** | **-13 reduction** | **-130 reduction** |
| **Regressed Fields** | 0 | **0** | **0** | **0 (Strict Zero Regressions)** |
| **Evaluated Documents** | 236 | **236** | 0 | 0 |
| **Total Benchmark Documents** | 370 | **370** | 0 | 0 |

---

## 2. Top Improved Documents in EXP-050

| Test ID | Baseline Word F1 | EXP-050 Word F1 | Delta (pp) | Fields Rescued | Regressions | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `short/07021-2016-p0051` | 92.3077% | **100.0000%** | **+7.6923 pp** | +1 | 0 | 100% PERFECT |
| `short/sched_i__after_violence_project_ty2024` | 93.9394% | **100.0000%** | **+6.0606 pp** | +1 | 0 | 100% PERFECT |
| `short/h5Filing-46384-1` | 93.6508% | **98.4127%** | **+4.7619 pp** | +3 | 0 | >= 95% MET |
| `short/h5Filing-45919-1` | 91.5033% | **96.0784%** | **+4.5751 pp** | +3 | 0 | >= 95% MET |
| `short/h5Filing-49070-1` | 92.7536% | **97.1014%** | **+4.3478 pp** | +3 | 0 | >= 95% MET |
| `short/ROGER-WINFIELD-SCOTT-p0025` | 94.6058% | **96.3415%** | **+1.7357 pp** | +2 | 0 | >= 95% MET |
| `medium/bar-lev-2024` | 47.0842% | **60.8340%** | **+13.7498 pp** | +33 | 0 | Sustained |
| `medium/bar-lev-2023` | 54.6237% | **63.7822%** | **+9.1585 pp** | +22 | 0 | Sustained |
| `medium/bar-lev-2021` | 50.5747% | **54.9358%** | **+4.3611 pp** | +12 | 0 | Sustained |
| `medium/cabrera-2021` | 52.9262% | **56.5824%** | **+3.6562 pp** | +8 | 0 | Sustained |

---

## 3. Failure Migration Overview

| Failure Mode | EXP-049 | EXP-050 | Delta | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `INSUFFICIENT_IOU` | 92,367 | 92,364 | -3 | Rescued by cell padding |
| `PAGE_DRIFT` | 11,805 | 11,796 | -9 | Rescued by header Page 1 anchor |
| `ZERO_CANDIDATE` | 3,303 | 3,302 | -1 | Rescued by Schedule I recipient anchor |
| `WRONG_ROW` | 48,535 | 48,535 | 0 | Unchanged |
| `MALFORMED_BBOX` | 14,323 | 14,323 | 0 | Unchanged |
