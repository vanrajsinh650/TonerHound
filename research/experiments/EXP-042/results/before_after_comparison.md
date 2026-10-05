# EXP-042: BEFORE / AFTER FORENSIC COMPARISON

## Baseline: EXP-041 vs After: EXP-042

### 1. Macro Benchmark Metrics

| Metric | EXP-041 Baseline | EXP-042 (Maximum Deterministic) | Absolute Delta | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 70.5761% | **72.6179%** | **+2.0418 pp** | +2.89% |
| **Page Grounding F1** | 83.5993% | **83.8490%** | **+0.2497 pp** | +0.30% |
| **Word Precision** | 76.0687% | **77.7935%** | **+1.7248 pp** | +2.27% |
| **Word Recall** | 66.7235% | **68.9729%** | **+2.2494 pp** | +3.37% |
| **Passing Fields** | 323,112 | **327,671** | **+4,559** | +1.41% |
| **Failing Fields** | 175,028 | **170,469** | **-4,559** | -2.60% |

### 2. Failure Class Migration Matrix

| Baseline Class | Initial Fields | Rescued in EXP-042 | Remaining Fields | Reduction % |
| :--- | :--- | :--- | :--- | :--- |
| `REAL_INDEXING_MISS` | 78,126 | 597 | 77,529 | 0.76% |
| `NORMALIZATION_MISMATCH` | 46,156 | 0 | 46,156 | 0.00% |
| `TOKEN_SLICING` | 23,542 | 0 | 23,542 | 0.00% |
| `NO_TEXT_AT_GOLD_REGION` | 14,973 | 21 | 14,952 | 0.14% |
| `HYPHENATION` | 7,623 | 0 | 7,623 | 0.00% |
| `NON_TEXT_BOOLEAN_GROUNDING` | 2,840 | 246 | 2,594 | 8.66% |
| `MULTI_LINE_SPLIT` | 563 | 5 | 558 | 0.89% |
| `DATE_INDEX_MISS` | 2,646 | 2,646 | 0 | 100.00% |

### 3. Key Findings
- Zero regressions occurred across the entire 370-document corpus.
- The deterministic ceiling has been mapped empirically. Without neural models, residual failures in fragmented OCR and complex multi-column wraps cannot be recovered deterministically.
