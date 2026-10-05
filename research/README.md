# TonerHound Research Directory

This directory contains all research artifacts, experiments, and diagnostic tools.

## Structure

- `experiments/` — Full experimental history
  - `EXP-027/` through `EXP-043/` — Final deterministic attack experiments
  - `archive/` — Older experiments (EXP-005 through EXP-018)
- `observer/` — Failure Microscope diagnostic engine
  - Microscope V1–V4
  - Golden sets and canonical ground-truth audits
  - `validation_report_v4.md` — V1–V8 gate verification
- `official_eval/` — Prediction caches and evaluation JSONs (regenerable, not source of truth)
- `data/` — Benchmark dataset (PDFs)
- `legacy_analysis/` — Archived scripts and experiments no longer in use

## Key Artifacts

- **Microscope validation**: `observer/validation_report_v4.md`
- **Latest experiment**: `experiments/EXP-043/`
- **Held-out cohort**: `benchmarks/held_out_manifest.json`
- **Canonical baseline**: `observer/reports/canonical_baseline_v2/`

## Running Research Experiments

All experiments use the official ExtractBench evaluator:

```bash
python research/experiments/EXP-043/run_full_benchmark.py
```

## Proprietary Content

This directory contains TonerHound's proprietary research methodology.
If you are forking the public repo, you can safely exclude this directory.
