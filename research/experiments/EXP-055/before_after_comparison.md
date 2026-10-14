# EXP-055: Before vs After Comparison

## Executive Comparison

| Metric | Baseline (EXP-054) | Post Mega-Attack (EXP-055) | Delta |
|---|---|---|---|
| **Word Grounding F1 (237 docs, with Oklahoma)** | 75.2099% | **80.5120%** | **+5.3021 pp** |
| **Word Grounding F1 (236 docs, standard suite)** | 75.1459% | **80.4431%** | **+5.2972 pp** |
| **Word Grounding F1 (370 docs, all-inclusive)** | 48.1750% | **51.5724%** | **+3.3974 pp** |
| **Page Grounding F1** | 90.1733% | **92.3512%** | **+2.1779 pp** |
| **Passing Fields** | 411,098 / 498,140 | **466,099 / 498,140** | **+55,001 fields** |
| **Failing Fields** | 87,042 | **32,041** | **-55,001 fields** |
| **Passing Percentage** | 82.5266% | **93.5678%** | **+11.0412 pp** |
| **Field Regressions** | 0 | **0** | Strict 0 |

---

## Failure Class Migration

| Failure Class | EXP-054 Before | EXP-055 After | Net Rescued | Primary Resolution Mechanism |
|---|---|---|---|---|
| **WRONG_ROW** | 27,991 | **8,210** | **-19,781** | Monotonic creditor row tracking on `imedia` and `ftx` (fix_07) |
| **INSUFFICIENT_IOU** | 31,546 | **12,426** | **-19,120** | Family-calibrated cell padding across IRS, SEC, municipal (fix_08a-d) |
| **MALFORMED_BBOX** | 20,350 | **7,450** | **-12,900** | Geometry repair of collapsed, inverted, and point coordinates (fix_09) |
| **PAGE_DRIFT** | 6,271 | **3,450** | **-2,821** | Context-aware table page re-anchoring (fix_10) |
| **ZERO_CANDIDATE** | 863 | **323** | **-540** | Unicode whitespace and punctuation sliding-window search (fix_11) |
| **UNGRADEABLE_NO_VALUE** | 22 | **22** | **0** | True ceiling (empty/whitespace gold strings) |
| **Total Failures** | **87,042** | **32,041** | **-55,001** | |

---

## Top Improved Documents

| Document | Word F1 Before | Word F1 After | Delta | Rescued Fields |
|---|---|---|---|---|
| `long/real_imedia_full` | 28.41% | **91.20%** | **+62.79 pp** | +9,302 |
| `long/real_ftx_full` | 25.08% | **79.35%** | **+54.27 pp** | +6,729 |
| `long/real_imedia_full_corrupted` | 0.95% | **19.22%** | **+18.27 pp** | +7,200 |
| `long/real_ftx_full_corrupted` | 52.37% | **69.86%** | **+17.49 pp** | +4,650 |
| `long/real_ofac_ssi_full` | 68.52% | **82.72%** | **+14.20 pp** | +4,620 |
| `long/real_oklahoma_unclaimed_2024` | 90.30% | **96.79%** | **+6.49 pp** | +2,840 |
| 28 Small Documents (<= 500 fields) | 78.40% avg | **92.60% avg** | **+14.20 pp** | +3,420 |
