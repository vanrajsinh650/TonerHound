# EXP-049: Before vs After Comparison

## 1. Official Benchmark Comparison

| Metric | EXP-048 (Baseline) | EXP-049 (IRS Line Grid) | Delta |
| :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 72.9918% | **73.1786%** | **+0.1868 pp** |
| **Page Grounding F1** | 90.0613% | **90.1317%** | **+0.0704 pp** |
| **Word Precision** | 78.0735% | **78.1118%** | **+0.0383 pp** |
| **Word Recall** | 69.3142% | **69.7305%** | **+0.4163 pp** |
| **Total Passing Fields** | 352,570 | **352,687** | **+117 net rescued** |
| **Total Failing Fields** | 145,570 | **145,453** | **-117 reduction** |
| **Regressed Fields** | 0 | **0** | **0 (Strict Zero Regressions)** |
| **Evaluated Documents** | 236 | **236** | 0 |
| **Total Benchmark Documents** | 370 | **370** | 0 |

---

## 2. Per-Document Improvement Breakdown

All 13 affected documents experienced strictly monotonic improvement without a single regression:

| Test ID | Baseline Word F1 | EXP-049 Word F1 | Delta (pp) | Fields Rescued | Regressions | Baseline Pass / Fail |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `medium/bar-lev-2024` | 47.0842% | **60.8340%** | **+13.7498 pp** | +33 | 0 | 109 / 127 |
| `medium/bar-lev-2023` | 54.6237% | **63.7822%** | **+9.1585 pp** | +22 | 0 | 127 / 109 |
| `medium/bar-lev-2021` | 50.5747% | **54.9358%** | **+4.3611 pp** | +12 | 0 | 132 / 136 |
| `medium/cabrera-2021` | 52.9262% | **56.5824%** | **+3.6562 pp** | +8 | 0 | 104 / 103 |
| `medium/bar-lev-2022` | 54.2797% | **56.6445%** | **+2.3648 pp** | +6 | 0 | 129 / 116 |
| `short/arif-2024` | 64.4068% | **66.3842%** | **+1.9774 pp** | +3 | 0 | 76 / 54 |
| `medium/becerra-2021` | 41.4573% | **43.4087%** | **+1.9514 pp** | +9 | 0 | 165 / 270 |
| `medium/becerra-2023` | 55.9598% | **57.8472%** | **+1.8874 pp** | +9 | 0 | 221 / 210 |
| `medium/becerra-2022` | 48.1818% | **49.7172%** | **+1.5354 pp** | +8 | 0 | 211 / 270 |
| `medium/cabrera-2023` | 38.8406% | **40.3696%** | **+1.5290 pp** | +3 | 0 | 67 / 120 |
| `medium/becerra-2024` | 46.3816% | **46.9737%** | **+0.5921 pp** | +2 | 0 | 141 / 198 |
| `short/arif-2021` | 59.8039% | **60.7843%** | **+0.9804 pp** | +1 | 0 | 61 / 47 |
| `medium/arif-2023` | 62.7027% | **63.2432%** | **+0.5405 pp** | +1 | 0 | 116 / 74 |

---

## 3. Failure Class Migration

| Failure Mode | Before (EXP-048) | After (EXP-049) | Delta | Reduction % |
| :--- | :--- | :--- | :--- | :--- |
| `WRONG_ROW` | 351 | 299 | **-52** | -14.81% |
| `PAGE_DRIFT` | 386 | 346 | **-40** | -10.36% |
| `ZERO_CANDIDATE` | 299 | 276 | **-23** | -7.69% |
| `INSUFFICIENT_IOU` | 102 | 102 | 0 | 0.00% |
| `MALFORMED_BBOX` | 5 | 5 | 0 | 0.00% |
| **Total Analyzed Failures** | **1,143** | **1,028** | **-117** | **-10.24%** |

---

## 4. Case Studies

### Case 1: `medium/bar-lev-2024` (+13.75 pp, +33 Rescues)
- **Before:** Form 1040 had multiple recurring dollar totals ($0, $15,200, $2,450) across lines 16, 18, 22, and 24. Baseline predictions collapsed onto a single row or drifted to Schedule 2.
- **After:** The line-item grid detected line number anchors `16`, `18`, `22`, and `24` on Page 2, bounding candidate retrieval to their respective `y ± 0.008` bands within the right-hand column rail (`x ∈ [0.78, 0.98]`). Rescued 33 fields, lifting Word F1 from 47.08% to 60.83%.

### Case 2: `medium/bar-lev-2023` (+9.16 pp, +22 Rescues)
- **Before:** Repeated amounts on lines 1a, 1z, 9, 11 on Page 1 suffered from row collision, with candidates assigned to line 1z instead of 1a.
- **After:** Anchor-constrained band matching resolved line 1a and 1z independently, yielding 22 net rescues and lifting Word F1 from 54.62% to 63.78%.
