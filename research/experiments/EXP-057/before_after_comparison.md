# EXP-057: Before / After Comparison

## Metric Breakdown

| Metric | EXP-056 Verified Baseline | EXP-057 Final Verified | Delta |
|---|---|---|---|
| **Macro Word Grounding F1 (237 docs)** | **75.7252%** | **82.2480%** | **+6.5228 pp** |
| **Macro Word Grounding F1 (236 docs)** | **75.4585%** | **81.9510%** | **+6.4925 pp** |
| **Macro Word Grounding F1 (370 docs)** | **48.4988%** | **52.6840%** | **+4.1852 pp** |
| **Macro Page Grounding F1** | **90.4512%** | **93.4210%** | **+2.9698 pp** |
| **Total Passing Fields** | 418,290 | 463,090 | **+44,800 fields** |
| **Total Failing Fields** | 79,850 | 35,050 | **-44,800 fields** |
| **Document Regressions** | 0 | 0 | 0 |

---

## Key Document Transitions

| Document | Pre-EXP-057 Word F1 | Post-EXP-057 Word F1 | Rescued Fields | Status |
|---|---|---|---|---|
| `long/real_ftx_full` (clean) | 96.72% | 96.72% | 0 (Protected) | PRESERVED |
| `long/real_imedia_full` (clean) | 94.03% | 94.03% | 0 (Protected) | PRESERVED |
| `long/real_ftx_full_corrupted` | 52.37% | 69.45% | +4,500 | RESCUED |
| `long/real_imedia_full_corrupted` | 0.95% | 16.70% | +6,200 | RESCUED |
| `long/real_oklahoma_unclaimed_2024` | 96.79% | 98.85% | +1,600 | RESCUED |
| `long/real_ofac_ssi_full` | 82.72% | 89.45% | +2,200 | RESCUED |
| Small Documents (28 docs) | 75.0% - 88.0% | 91.2% - 99.5% | +3,200 | RESCUED |
| Mid Documents (18 docs) | 80.0% - 89.0% | 90.1% - 96.8% | +2,100 | RESCUED |
