# EXP-048: Before vs After Comparison

## 1. Official Metric Benchmark Comparison

| Metric | EXP-047 (Baseline) | EXP-048 (Diagnostic Repairs) | Delta |
| :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 72.8448% | **72.9918%** | **+0.1470 pp** |
| **Page Grounding F1** | 90.0061% | **90.0613%** | **+0.0552 pp** |
| **Word Precision** | 77.9620% | **78.0735%** | **+0.1115 pp** |
| **Word Recall** | 69.1778% | **69.3142%** | **+0.1364 pp** |
| **Total Passing Fields** | 345,461 | **352,570** | **+7,109 net rescued** |
| **Total Failing Fields** | 152,679 | **145,570** | **-7,109 reduction** |
| **Regressed Fields** | 0 | **0** | **0 (Strict Zero Regressions)** |
| **Evaluated Documents** | 236 | **236** | 0 |
| **Total Benchmark Documents** | 370 | **370** | 0 |

---

## 2. Per-Document Improvement Breakdown

| Test ID | Baseline Word F1 | EXP-048 Word F1 | Delta (pp) | Fields Rescued | Fix Attribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `long/sec_13f_0010_renaissance_technologies` | 74.3748% | **93.0145%** | **+18.6397 pp** | +5,342 | `fix_01_sec13f_voting_authority` |
| `long/real_credit_strategies_full` | 73.7237% | **79.3396%** | **+5.6159 pp** | +218 | `fix_02`, `fix_03` |
| `medium/real_fidelity_northstar` | 62.9079% | **67.7350%** | **+4.8271 pp** | +38 | `fix_02`, `fix_03` |
| `medium/real_blackrock_schedule_full` | 86.5588% | **90.1235%** | **+3.5647 pp** | +109 | `fix_02`, `fix_03` |
| `medium/real_blackrock_muni_bmn` | 94.9000% | **95.7754%** | **+0.8754 pp** | +23 | `fix_02`, `fix_03` |
| `medium/sec_13f_0031_loomis_sayles` | 86.8883% | **87.6939%** | **+0.8056 pp** | +40 | `fix_01_sec13f_voting_authority` |
| `long/real_ishares_iboxx_bond_etfs` | 98.8000% | **99.0968%** | **+0.2968 pp** | +1,326 | `fix_02_holdings_marks_precision` |
| `long/sec_13f_0026_brown_brothers_harriman` | 84.6663% | **84.7457%** | **+0.0794 pp** | +13 | `fix_01_sec13f_voting_authority` |

---

## 3. Failure Class Migration

| Failure Taxonomy Class | Baseline Failures (EXP-047) | EXP-048 Failures | Net Rescued | % Recovered |
| :--- | :--- | :--- | :--- | :--- |
| `NORMALIZATION_MISMATCH` | 28,372 | 22,977 | **+5,395** | 19.01% |
| `TOKEN_SLICING` | 24,186 | 22,765 | **+1,421** | 5.88% |
| `MULTI_LINE_SPLIT` | 18,210 | 17,917 | **+293** | 1.61% |
| `REAL_INDEXING_MISS` | 48,120 | 48,120 | 0 | 0.00% |
| `NO_TEXT_AT_GOLD_REGION` | 22,540 | 22,540 | 0 | 0.00% |
| `NON_TEXT_BOOLEAN` | 5,890 | 5,890 | 0 | 0.00% |
| `DATE_INDEX_MISS` | 3,140 | 3,140 | 0 | 0.00% |
| **Total Benchmark Failures** | **152,679** | **145,570** | **+7,109** | **4.66%** |
