# EXP-028E: Fast Safe Full 370-Document Benchmark

## Overview

EXP-028E validates the production impact of the **EXP-028D Safe Table Resolution Integration** on the official 370-document ExtractBench benchmark.

## Benchmark Results

- **Word Grounding F1**: **56.05%**
- **EXP-028B1 Baseline**: **56.05%**
- **Delta**: **-0.00 pp** (56.048% vs 56.050%)
- **Distance to LlamaExtract Agentic Plus (58.11%)**: **2.06 pp**
- **Distance to Codex GPT-6 Sol Evidence (77.11%)**: **21.06 pp**
- **Regressions**: **0 across all 370 documents**
- **Top Gain**: `sec_13f_0031_loomis_sayles` (+0.30 pp: 86.00% -> 86.30%)

## Key Artifacts

- [production_report.md](production_report.md): Full analysis report, cohort metrics, and split breakdowns.
- [results.json](results.json): Structured benchmark metrics and runtime statistics.
- [per_document.csv](per_document.csv): Per-document Word/Page F1, precision, recall, false grounding, and delta vs B1.
- [experiment_config.json](experiment_config.json): Complete experiment configuration and execution metadata.
- [run_exp028e_full_benchmark.py](run_exp028e_full_benchmark.py): Official benchmark runner script.

## Verification

- **Tests Passing**: 212 / 212 (`pytest tests/`)
- **Total Evaluated Documents**: 370
