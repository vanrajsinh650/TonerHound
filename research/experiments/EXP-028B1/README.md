# EXP-028B1: True Text Path Productionization

## Objective
Determine the actual end-to-end production Word Grounding F1 after integrating the 10 information-loss fixes discovered in EXP-028B0.

## Headline Results
- **EXP-026 Production Baseline:** 55.98% Word Grounding F1
- **EXP-028B0 True Text Oracle:** 75.12% Word Grounding F1 (Upper Bound)
- **EXP-028B1 Production:** **56.05%** Word Grounding F1

## Artifacts
- `results.json`: Full metrics and cohort breakdowns.
- `production_report.md`: Authoritative technical report.
- `per_document.csv`: Per-document metrics.
- `field_failure_analysis.parquet`: Parquet trace of field-level failure classification.
