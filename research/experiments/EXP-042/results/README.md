# EXP-042: Maximum Deterministic Recovery

## Quick Reference
- **Status:** COMPLETED
- **Word Grounding F1:** **72.6179%** (+2.0418 pp vs EXP-041 baseline 70.5761%)
- **Page Grounding F1:** **83.8490%**
- **Passing Fields:** **327,671**
- **Rescued Fields:** **+4,559**
- **Regressions:** **0**
- **Decision:** **INTEGRATE**
- **Deterministic Ceiling Reached:** **YES**

## File Artifacts
- `forensic_audit_v2.json`: Forensic audit across all 8 target classes.
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
