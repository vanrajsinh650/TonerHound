# EXP-028D: Safe Table Resolution Integration

## Overview
EXP-028D safely integrates validated structural table signals (row proximity, dynamic column projection, sibling co-linearity, reading-order consistency, record-anchor evidence) into TonerHound's production table alignment pipeline without replacing `ExtractBenchAdapter`'s monotonic DP sequence alignment or regressing baseline production performance.

## Status: PASSED (ALL DECISION GATES MET)
- `long/real_sm0801_eco_full`: **92.48%** (Gate: $\ge 91.48\%$, delta: 0.00pp)
- `medium/sec_13f_0031_loomis_sayles`: **86.30%** (Gate: $\ge 85.00\%$, delta: **+0.30pp**; Page F1: **+1.34pp**)
- Smoke Suite Average (6 docs): **62.84%** (Gate: $\ge 62.84\%$, delta: 0.00pp)
- Unit test suite: **212/212 tests passing**

## Key Files
- `production_report.md`: Detailed forensic and empirical benchmark report
- `results.json`: Official ExtractBench metrics summary
- `per_document.csv`: Per-document metrics (Word F1, Precision, Recall, Page F1, False Grounding, Abstention)
- `experiment_config.json`: Configuration and defect fix definitions
- `run_targeted_benchmark.py`: Targeted benchmark runner for the 7 evaluation documents
