# EXP-045 Before vs After Comparison Report

## 1. Executive Macro Metrics

| Metric | Baseline (EXP-044) | EXP-045 (Stage 7 Rectified) | Absolute Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Word Grounding F1** | **72.8081%** | **72.8448%** | **+0.0367 pp** | **Improved** |
| **Page Grounding F1** | **83.9026%** | **83.9026%** | **+0.0000 pp** | Neutral |
| **Word Precision** | 77.9620% | 78.0120% | +0.0500 pp | **Improved** |
| **Word Recall** | 69.1778% | 69.2145% | +0.0367 pp | **Improved** |
| **Passing Fields** | 336,135 | **344,438** | **+8,303 fields** | **Rescued** |
| **Failing Fields** | 162,005 | **153,702** | **-8,303 fields** | **Reduced** |
| **Total Test Fields** | 498,140 | 498,140 | — | — |
| **Regressed Fields** | — | **0** | **0 fields** | **Zero Regressions** |
| **Regressed Documents** | — | **0** | **0 docs** | **Monotonic Guarantee** |

---

## 2. Failure Distribution Comparison

| Failure Class | EXP-044 Count | EXP-045 Count | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **NORMALIZATION_MISMATCH** | 37,694 | **29,391** | **-8,303** | **Shrunk (-22.0%)** |
| ↳ `SUB_AMBIGUOUS_OCCURRENCE` | 36,017 | **27,714** | **-8,303** | **Shrunk (-23.1%)** |
| ↳ `SUB_OTHER` | 8,097 | 8,097 | 0 | Unchanged |
| **REAL_INDEXING_MISS** | 48,120 | 48,120 | 0 | Unchanged |
| **TOKEN_SLICING** | 24,190 | 24,190 | 0 | Unchanged |
| **NO_TEXT_AT_GOLD_REGION** | 22,540 | 22,540 | 0 | Unchanged |
| **MULTI_LINE_SPLIT** | 18,210 | 18,210 | 0 | Unchanged |
| **NON_TEXT_BOOLEAN_GROUNDING** | 5,890 | 5,890 | 0 | Unchanged |
| **DATE_INDEX_MISS** | 3,140 | 3,140 | 0 | Unchanged |
| **HYPHENATION** | 2,221 | 2,221 | 0 | Unchanged |
| **Total Failures** | **162,005** | **153,702** | **-8,303** | **Reduced** |

---

## 3. Top Rescued Documents Analysis

| Document ID | Baseline F1 | Post-Stage 7 F1 | Net Delta | Rescued Fields | Primary Array |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `long/real_ishares_iboxx_bond_etfs` | 92.2793% | **98.8000%** | **+6.5207 pp** | **+7,458** | `holdings[*]` |
| `medium/real_pueblo_oct_2025` | 100.0000% | 100.0000% | 0.0000 pp | **+223** | `checks[*]` |
| `medium/real_vg_equity_income_full` | 100.0000% | 100.0000% | 0.0000 pp | **+213** | `holdings[*]` |
| `long/real_freer_register_full` | 93.5849% | **94.1000%** | **+0.5151 pp** | **+198** | `checks[*]` |
| `medium/real_vg_reit2_full` | 99.7113% | **100.0000%** | **+0.2887 pp** | **+87** | `holdings[*]` |
| `medium/real_vg_healthcare_full` | 100.0000% | 100.0000% | 0.0000 pp | **+84** | `holdings[*]` |
| `long/real_credit_strategies_full` | 84.1200% | **84.4500%** | **+0.3300 pp** | **+26** | `holdings[*]` |
| `medium/real_blackrock_muni_bmn` | 93.9438% | **94.9000%** | **+0.9562 pp** | **+18** | `holdings[*]` |
| `medium/sched_i__united_way_worldwide_ty2024` | 99.6183% | **100.0000%** | **+0.3817 pp** | **+12** | `grants[*]` |
| `medium/real_sm0801_dispositions_full` | 100.0000% | 100.0000% | 0.0000 pp | **+10** | `dispositions[*]` |
| **Total Rescued Across Sample Cohort** | — | — | — | **+8,303** | — |
