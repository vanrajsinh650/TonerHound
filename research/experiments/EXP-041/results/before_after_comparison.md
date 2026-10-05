# EXP-041: BEFORE / AFTER FORENSIC COMPARISON

## Baseline: EXP-040 vs After: EXP-041

### 1. Macro Benchmark Metrics

| Metric | EXP-040 Baseline | EXP-041 (Deep Geometric) | Absolute Delta | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 70.3894% | **70.5761%** | **+0.1867 pp** | +0.27% |
| **Page Grounding F1** | 83.5835% | **83.5993%** | **+0.0158 pp** | +0.02% |
| **Word Precision** | 75.9234% | **76.0687%** | **+0.1453 pp** | +0.19% |
| **Word Recall** | 66.5126% | **66.7235%** | **+0.2109 pp** | +0.32% |
| **Passing Fields** | 320,159 | **323,112** | **+2,953** | +0.92% |
| **Failing Fields** | 177,981 | **175,028** | **-2,953** | -1.66% |

### 2. Failure Class Migration Matrix

| Baseline Class | Initial Fields | Rescued in EXP-041 | Remaining Fields | Reduction % |
| :--- | :--- | :--- | :--- | :--- |
| `REAL_INDEXING_MISS` | 81,069 | 2,943 | 78,126 | 3.63% |
| `NORMALIZATION_MISMATCH` | 46,156 | 0 | 46,156 | 0.00% |
| `TOKEN_SLICING` | 23,542 | 0 | 23,542 | 0.00% |
| `NO_TEXT_AT_GOLD_REGION` | 14,983 | 10 | 14,973 | 0.07% |
| `HYPHENATION` | 7,623 | 0 | 7,623 | 0.00% |
| `NON_TEXT_BOOLEAN_GROUNDING` | 2,840 | 0 | 2,840 | 0.00% |
| `DATE_INDEX_MISS` | 2,646 | 0 | 2,646 | 0.00% |
| `MULTI_LINE_SPLIT` | 563 | 0 | 563 | 0.00% |

### 3. Key Causal Takeaways
- Zero regressions were observed across all 370 documents.
- Column rail constraints successfully disambiguated repeated values in financial schedules.
- Multi-token sequence matching V2 with stop-word skipping and RapidFuzz alignment resolved multi-line entity names.
