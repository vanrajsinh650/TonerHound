# EXP-005 Non-FTX Control Document Regression Audit

**Date**: 2026-09-16  
**Auditor**: Agent D (Regression Auditor) — EXP-005 Part 4 Pass 4 G2 Near-Miss Pass  
**Target Candidate**: ExtractBenchAdapter with FTX 52.29% Word F1  
**Runner Script**: [`scripts/verify_control_regressions.py`](file:///home/vanrajsinh/Projects/TonerHound/scripts/verify_control_regressions.py)  
**JSON Artifact**: [`experiments/EXP-005-control-regressions.json`](file:///home/vanrajsinh/Projects/TonerHound/experiments/EXP-005-control-regressions.json)  

---

## 1. Executive Summary

As part of EXP-005 Part 4 Pass 4 G2 near-miss pass, this regression audit evaluated all 5 non-FTX control documents against their established baseline checkpoints.

### Key Finding:
**ZERO REGRESSIONS VERIFIED ACROSS ALL 5 NON-FTX CONTROL DOCUMENTS.**  
Every control document produced identical performance (within floating-point precision $\Delta < 10^{-8}$ pp) to its established baseline checkpoint. The current adapter changes (which elevated FTX from 49.87% to 52.29% Word F1) are strictly isolated to FTX creditor table handling and have zero negative side-effects on other benchmark documents.

---

## 2. Control Document Benchmark Summary

| Control Document | Domain | Pages | Citations | Baseline Word F1 | Current Word F1 | Delta Word F1 | Baseline Page F1 | Current Page F1 | Delta Page F1 | Status | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `short/13f__sl_advisors_llc` | 13F Holdings | 2 | 507 | **99.80%** | **99.80%** | $+0.00$ pp | **100.00%** | **100.00%** | $+0.00$ pp | **PASS** | 0.22s |
| `short/nport__bullfinch_fund_inc` | Mutual Fund Schedule | 3 | 549 | **91.44%** | **91.44%** | $+0.00$ pp | **100.00%** | **100.00%** | $+0.00$ pp | **PASS** | 0.17s |
| `medium/cabrera-2023` | IRS Form 1040 Tax | 28 | 158 | **26.67%** | **26.67%** | $+0.00$ pp | **68.99%** | **68.99%** | $+0.00$ pp | **PASS** | 1.37s |
| `long/real_credit_strategies_full` | Investment Schedule | 59 | 4687 | **19.29%** | **19.29%** | $+0.00$ pp | **77.57%** | **77.57%** | $+0.00$ pp | **PASS** | 16.56s |
| `short/real_clinton_property_25_11073_corrupted` | Deed & Conveyance (OCR) | 9 | 178 | **47.83%** | **47.83%** | $+0.00$ pp | **94.12%** | **94.12%** | $+0.00$ pp | **PASS** | 0.44s |

- **Total Suite Runtime**: 18.76s across 5 documents (101 total pages, 6,079 citations).
- **Macro Regression Rate**: **0.00% (0 / 5 regressed)**.

---

## 3. Exact Precision and Recall Metrics

### 1. `short/13f__sl_advisors_llc`
- **Domain**: Form 13F Holdings Report
- **Word Grounding**:
  - Exact Word F1: **0.99802372** (99.80%)
  - Word Precision: **0.99802372** (99.80%)
  - Word Recall: **0.99802372** (99.80%)
- **Page Grounding**:
  - Exact Page F1: **1.00000000** (100.00%)
  - Page Precision: **1.00000000** (100.00%)
  - Page Recall: **1.00000000** (100.00%)
- **Latency**: 0.22s (Indexing: 0.00s, Grounding: 0.19s, Eval: 0.01s)

### 2. `short/nport__bullfinch_fund_inc`
- **Domain**: Form N-PORT Mutual Fund Holdings Schedule
- **Word Grounding**:
  - Exact Word F1: **0.91438980** (91.44%)
  - Word Precision: **0.91438980** (91.44%)
  - Word Recall: **0.91438980** (91.44%)
- **Page Grounding**:
  - Exact Page F1: **1.00000000** (100.00%)
  - Page Precision: **1.00000000** (100.00%)
  - Page Recall: **1.00000000** (100.00%)
- **Latency**: 0.17s (Indexing: 0.01s, Grounding: 0.14s, Eval: 0.01s)

### 3. `medium/cabrera-2023`
- **Domain**: IRS Form 1040 Tax Return (Multi-year, structural tax grounding)
- **Word Grounding**:
  - Exact Word F1: **0.26666667** (26.67%)
  - Word Precision: **0.29113924** (29.11%)
  - Word Recall: **0.24598930** (24.60%)
- **Page Grounding**:
  - Exact Page F1: **0.68985507** (68.99%)
  - Page Precision: **0.75316456** (75.32%)
  - Page Recall: **0.63636364** (63.64%)
- **Latency**: 1.37s (Indexing: 0.13s, Grounding: 1.18s, Eval: 0.03s)

### 4. `long/real_credit_strategies_full`
- **Domain**: Long Schedule of Investments (59 pages, dense multi-page tabular)
- **Word Grounding**:
  - Exact Word F1: **0.19292833** (19.29%)
  - Word Precision: **0.21743782** (21.74%)
  - Word Recall: **0.17338452** (17.34%)
- **Page Grounding**:
  - Exact Page F1: **0.77571845** (77.57%)
  - Page Precision: **0.86087358** (86.09%)
  - Page Recall: **0.70589350** (70.59%)
- **Latency**: 16.56s (Indexing: 0.46s, Grounding: 15.47s, Eval: 0.42s)

### 5. `short/real_clinton_property_25_11073_corrupted`
- **Domain**: Deed & Conveyance (Scanned OCR with severe character substitutions and noise)
- **Word Grounding**:
  - Exact Word F1: **0.47826087** (47.83%)
  - Word Precision: **0.49624060** (49.62%)
  - Word Recall: **0.46153846** (46.15%)
- **Page Grounding**:
  - Exact Page F1: **0.94117647** (94.12%)
  - Page Precision: **0.97297297** (97.30%)
  - Page Recall: **0.91139241** (91.14%)
- **Latency**: 0.44s (Indexing: 0.04s, Grounding: 0.39s, Eval: 0.01s)

---

## 4. Automation and Execution

The verification runner can be executed at any time via:
```bash
python scripts/verify_control_regressions.py
```
Options supported:
- `--doc <DOC_ID>`: Run verification on an individual document
- `--tolerance <FLOAT>`: Set maximum allowable regression threshold (default: `1e-4`)
- `--json-output <PATH>`: Destination path for JSON report
- `-v` / `--verbose`: Detailed execution trace

All 100 existing unit tests in `tests/` pass concurrently.
