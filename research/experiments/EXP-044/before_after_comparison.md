# EXP-044 Before vs After Comparison Report

## 1. Executive Macro Comparison

| Metric | Baseline (EXP-042 Canonical) | EXP-044 Hybrid Pipeline | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Word Grounding F1** | **72.6179%** | **72.8081%** | **+0.1902 pp** | **+0.26%** |
| **Page Grounding F1** | **83.8490%** | **83.9026%** | **+0.0536 pp** | **+0.06%** |
| **Word Precision** | 77.7935% | **77.9620%** | **+0.1685 pp** | — |
| **Word Recall** | 68.9729% | **69.1778%** | **+0.2049 pp** | — |
| **Passing Fields** | 327,671 | **336,135** | **+8,464 fields** | **+2.58%** |
| **Failing Fields** | 170,469 | **162,005** | **-8,464 fields** | **-4.97%** |
| **Total Test Fields** | 498,140 | 498,140 | — | — |
| **Regressed Fields** | — | **0** | **0 fields** | **Strict Zero Regressions** |
| **Regressed Documents** | — | **0** | **0 documents** | **Monotonic Guarantee** |

---

## 2. Failure Distribution Migration

| Failure Category | EXP-042 Count | EXP-044 Count | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **NORMALIZATION_MISMATCH** | 46,158 | **37,694** | **-8,464** | **Substantially Reduced** |
| **REAL_INDEXING_MISS** | 48,120 | 48,120 | 0 | Unchanged |
| **TOKEN_SLICING** | 24,190 | 24,190 | 0 | Unchanged |
| **NO_TEXT_AT_GOLD_REGION** | 22,540 | 22,540 | 0 | Unchanged |
| **MULTI_LINE_SPLIT** | 18,210 | 18,210 | 0 | Unchanged |
| **NON_TEXT_BOOLEAN_GROUNDING** | 5,890 | 5,890 | 0 | Unchanged |
| **DATE_INDEX_MISS** | 3,140 | 3,140 | 0 | Unchanged |
| **HYPHENATION** | 2,221 | 2,221 | 0 | Unchanged |
| **Total Failures** | **170,469** | **162,005** | **-8,464** | **Monotonic Improvement** |

---

## 3. Top 10 Most Improved Documents

| Rank | Document ID | Split | Base Word F1 | EXP-044 Word F1 | Delta (pp) | Rescued Fields |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `long/real_ishares_iboxx_bond_etfs` | Long | 70.3395% | **92.2793%** | **+21.9398 pp** | +8,134 |
| 2 | `short/00581-2011-p0102` | Short | 62.5000% | **69.6970%** | **+7.1970 pp** | +3 |
| 3 | `short/07021-2016-p0038` | Short | 83.3333% | **88.0000%** | **+4.6667 pp** | +1 |
| 4 | `short/W2-27-217184_213122` | Short | 50.2793% | **52.5140%** | **+2.2347 pp** | +2 |
| 5 | `long/real_freer_register_full` | Long | 91.8517% | **93.5849%** | **+1.7332 pp** | +201 |
| 6 | `medium/real_bbb_service_list` | Medium | 77.7455% | **79.3203%** | **+1.5748 pp** | +19 |
| 7 | `short/ROGER-WINFIELD-SCOTT-FOUNDATION-INC-2009-580655183-p0024` | Short | 86.4407% | **87.8187%** | **+1.3780 pp** | +3 |
| 8 | `medium/nport__sei_etf_2025q4` | Medium | 97.8997% | **98.9002%** | **+1.0005 pp** | +49 |
| 9 | `medium/real_blackrock_schedule_full` | Medium | 85.8757% | **86.5588%** | **+0.6831 pp** | +20 |
| 10 | `short/593338187_200912_990PF-p0027` | Short | 82.5000% | **83.1250%** | **+0.6250 pp** | +1 |

---

## 4. Regression Analysis

Across the full official 370-document ExtractBench corpus:
- **Regressed Fields**: **0**
- **Regressed Documents**: **0**
- **Status**: **STRICT PASS**
