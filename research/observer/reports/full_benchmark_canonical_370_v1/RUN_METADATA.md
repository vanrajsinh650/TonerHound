# Run Metadata — canonical_370_v1

- **Run ID**: `canonical_370_v1`
- **Timestamp (UTC)**: `2026-10-03T09:25:11.330342+00:00`
- **Git Commit**: `178f81c`
- **Git Branch**: `exp-036c-deterministic-grounding`
- **Working Tree State**:
  - `src/tonerhound/` unmodified by new interventions; preserves existing unstaged flags (`ENABLE_EXP033_CANDIDATE_EXPANSION = False`).
- **Benchmark Corpus**: ExtractBench Official 370-Document Full Corpus (`research/data/full`)
- **Evaluator**: `ExtractBench ExtractEvaluator` & `EvaluationRunner._aggregate_metrics`
- **Python Environment**: Python 3.12.14, Linux x86_64
- **Execution Command**: `./.venv/bin/python research/observer/run_full_benchmark_audit.py`
- **Evaluated Documents**: 370 total documents (237 with grounded field rules)
- **Evaluated Fields**: 498140 total gradeable fields
- **Total Runtime**: 145.86 seconds
