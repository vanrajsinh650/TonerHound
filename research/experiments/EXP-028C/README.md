# EXP-028C: Fast Safe Production Benchmark — JointRecordResolver

**Status:** STOPPED AT PHASE 1 DECISION GATE  
**Date:** 2026-09-30  
**Harness:** Official ExtractBench `ExtractEvaluator`  
**Primary Metric:** Word Grounding F1 (IoU >= 0.50)  

---

## Overview

EXP-028C measured the production impact of the **EXP-028B2 `JointRecordResolver`** on ExtractBench against the **EXP-028B1** baseline (56.05% production Word F1).

### Key Results & Decision Gate Verdict

- **Phase 1 Smoke Benchmark (6 docs):** 59.07% Word F1 vs 62.84% in EXP-028B1 (**-3.77pp delta**).
- **Clinical Table (`real_sm0801_eco_full`):** 75.02% vs 92.48% in B1 (**-17.46pp regression**).
- **Financial Table (`sec_13f_0031_loomis_sayles`):** 69.77% vs 86.00% in B1 (**-16.23pp regression**).
- **Decision Gate Action:** Execution of the full 370-document benchmark was halted pursuant to the protocol rule (*"If there is an obvious regression, STOP and diagnose only the regression"*).

## Artifacts in this Directory

- `production_report.md`: Authoritative report containing executive summary, baseline comparisons, root-cause diagnosis, and recommendations for EXP-028D.
- `results.json`: Complete machine-readable metrics and diagnostic payload.
- `per_document.csv`: Per-document metrics across evaluated smoke and table comparison documents.
- `experiment_config.json`: Experiment configuration and Decision Gate specification.
- `run_phase1_smoke.py`: Reproducible execution script for the Phase 1 smoke benchmark.
- `smoke_predictions/`: Complete prediction JSON files.
- `smoke_eval_cache/`: Official ExtractEvaluator metric outputs.
