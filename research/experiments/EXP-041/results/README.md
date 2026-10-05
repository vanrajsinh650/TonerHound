# EXP-041: Deep Research-Backed Geometric Recovery

## Quick Reference
- **Status:** COMPLETED
- **Word Grounding F1:** **70.5761%** (+0.1867 pp vs EXP-040 baseline 70.3894%)
- **Page Grounding F1:** **83.5993%**
- **Passing Fields:** **323,112**
- **Rescued Fields:** **+2,953**
- **Regressions:** **0**
- **Decision:** **REJECT** (Score < 71.50% threshold)

## File Artifacts
- `forensic_audit.json`: Pre-experiment audit of baseline failure subclasses.
- `FINAL_REPORT.md`: Comprehensive experimental report.
- `before_after_comparison.md`: Detailed metric deltas and migration matrix.
- `decision.md`: Decision gate analysis.
- `implementation_notes.md`: Algorithmic design and implementation notes.
- `README.md`: Executive summary.
- `full_benchmark_results.json`: Complete 370-document official benchmark results.
- `fix_attribution.json`: Rescue counts attributed by phase.
- `failure_migration.json`: Migration matrix of failure classes.
- `regression_analysis.json`: Regression verification report.
- `failure_summary.json`: Failure Microscope V4 post-audit summary.
- `per_document.csv`: Per-document metrics across all 370 documents.
