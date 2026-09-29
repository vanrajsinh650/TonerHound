# EXP-028B2: Table Row/Column Disambiguation Report

**Experiment ID:** EXP-028B2  
**Date:** 2026-09-29  
**Status:** COMPLETE  
**Primary Metric:** Word Grounding F1 (IoU >= 0.50)  
**Harness:** Official ExtractBench `ExtractEvaluator`  

---

## 1. Executive Summary

In **EXP-028B1**, 98.14% of remaining Oracle-to-Production gap failures were classified as table row/column assignment failures. This experiment (**EXP-028B2**) systematically verified this diagnosis, developed structural context signals, implemented a **Joint Row/Record Resolution** engine, and executed a 6-stage controlled ablation suite (Modes A through F).

### Key Empirical Findings:
1. **Phase 1 Diagnosis Confirmed:** Table fields account for **99.43%** of all real Oracle-to-Production gap fields across the benchmark population.
2. **Controlled Ablation Demonstrates Decisive Gains:**
   - **Mode A (Baseline):** 45.31% Word F1 (39.46% on major table case `loomis_sayles`, 59.83% false grounding).
   - **Mode B (Row-Anchor Only):** 57.19% Word F1 (+11.88pp suite gain, 63.53% on `loomis_sayles`).
   - **Mode C (Column Only):** 45.43% Word F1.
   - **Mode D (Row + Column Context):** 57.30% Word F1 (+11.99pp suite gain, 64.22% on `loomis_sayles`).
   - **Mode E (Joint Multi-Field Record Resolution):** **58.23% Word F1** (+12.92pp suite gain; **69.78%** on `loomis_sayles`, **+30.32pp gain** over baseline, cutting false grounding from 59.83% down to 29.75%).
   - **Mode F (Ambiguity/Abstention Gating):** **58.10% Word F1**, Precision **61.10%** (up from 47.86% in baseline), while reducing false grounding to **14.69%** (a 56% relative reduction).
3. **Strict Regression Controls Preserved:**
   - `real_wyo_Goshen_2024`: **99.49%** Word F1 preserved.
   - `real_pueblo_oct_2025`: **99.39%** Word F1 preserved.
   - `bianco-2024` (Tax Form 1040): **30.34%** Word F1 preserved without regression.
   - `W14-Atascosa SWD Well No. 4` (Regulatory Form): **51.09%** Word F1 preserved without regression.
   - Non-table decks and scalar forms unaffected.

---

## 2. Phase 1: Verification of Diagnosis

To ensure engineering effort directly addressed the true bottleneck rather than classification artifacts, Phase 1 audited `field_failure_analysis.parquet` (498,140 rows):

### A. Sample Audit (N = 200 Category D Fields):
- **False Alarms (Passed in B1 with IoU >= 0.50):** 121 (60.5%)
- **Oracle Also Failed (Beyond text reachability):** 31 (15.5%)
- **Genuine Oracle-to-Production Gap Failures:** 48 (24.0%)

### B. Breakdown of Genuine Category D Failures:
| Failure Mechanism | Share | Root Cause |
| :--- | :---: | :--- |
| `different_cell_or_table` | 39.6% | Candidate selected from adjacent table or wrong cell |
| `omitted_unresolved` | 22.9% | Unanchored rows dropped when evidence was uncertain |
| `wrong_column_same_row` | 16.7% | Ambiguous values placed into adjacent column |
| `wrong_page_assignment` | 10.4% | Identical numbers/codes assigned to incorrect page |
| `wrong_row_same_column` | 10.4% | Repeated values (e.g. 0, "SOLE") assigned to row above/below |

### C. Population-Level Audit (50-Document Cross-Section):
- Total gradeable fields: 441,457
- Real Oracle-to-Production Gap Fields: 99,165
- **Table field share (`[` or `table`): 99.43%**
- Scalar field share: 0.57%
- **Phase 1 Decision Gate Verdict: CONFIRMED.**

---

## 3. Phase 2 & 3: Joint Row/Record Resolution Engine

The `JointRecordResolver` replaces naive isolated string-matching with structural record context:

$$\text{Score}(c) = S_{\text{page}}(c) + S_{\text{row}}(c) + S_{\text{column}}(c) + S_{\text{sibling}}(c) + S_{\text{reading\_order}}(c) + 2.0 \cdot \text{Sim}(c)$$

1. **Record Anchor Discovery:** Automatically selects distinctive high-entropy tokens (e.g. 9-character CUSIP codes, unique issuer names) to establish physical page and vertical center line ($y_{\text{center}}$).
2. **Vertical Row Confinement:** Evaluates vertical deviation $\Delta y = |y_c - y_{\text{row\_center}}|$, applying Gaussian reward within row bounds and steep penalties outside the corridor.
3. **Horizontal Column Rails:** Aligns numeric fields to right column rails and text fields to left column rails, penalizing candidates that stray into neighboring column corridors.
4. **Sibling Co-Linearity Bonus:** Rewards candidates that share an exact baseline with already-resolved sibling fields within the same record.
5. **Horizontal Reading-Order Check:** Enforces column sequence consistency (e.g. Description $\rightarrow$ Quantity $\rightarrow$ Price).
6. **Ambiguity Gating (Mode F):** Abstains from silent guesses when candidate score margin is within $\epsilon_{\text{ambiguity}}$.

---

## 4. Phase 4: Controlled Experiment Matrix

Comprehensive evaluation across the 6 controlled ablation modes:

| Mode | Configuration | Word Grounding F1 | Precision | Recall | Page Grounding F1 | False Grounding Rate | Abstention Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **BASELINE** (Pure text match) | 45.31% | 47.86% | 43.57% | 71.53% | 33.51% | 0.00% |
| **B** | **ROW_ANCHOR_ONLY** (Row + Page) | 57.19% | 59.81% | 55.39% | 71.67% | 21.80% | 0.00% |
| **C** | **COLUMN_ONLY** (Column Rails) | 45.43% | 47.98% | 43.69% | 71.67% | 33.53% | 0.00% |
| **D** | **ROW_AND_COLUMN** (Row + Column) | 57.30% | 59.92% | 55.50% | 71.67% | 21.68% | 0.00% |
| **E** | **JOINT_RECORD** (Full Structural Context) | **58.23%** | **60.85%** | **56.43%** | **71.67%** | **16.99%** | 0.00% |
| **F** | **AMBIGUITY_GATED** (Mode E + Abstention) | 58.10% | **61.10%** | 55.96% | 70.39% | **14.69%** | 21.09% |

### Major Table Document Deep-Dive (`medium/sec_13f_0031_loomis_sayles`):
| Mode | Word Grounding F1 | Precision | Recall | False Grounding Rate | Delta vs Baseline |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Baseline)** | 39.46% | 39.52% | 39.40% | 59.83% | — |
| **B (Row Only)** | 63.53% | 63.62% | 63.43% | 35.99% | +24.07pp |
| **C (Column Only)** | 40.18% | 40.24% | 40.12% | 59.30% | +0.72pp |
| **D (Row + Column)** | 64.22% | 64.31% | 64.12% | 35.31% | +24.76pp |
| **E (Joint Record)** | **69.78%** | **69.88%** | **69.67%** | **29.75%** | **+30.32pp** |
| **F (Ambiguity Gated)** | 68.34% | **69.75%** | 66.99% | **28.78%** | +28.88pp |

---

## 5. Phase 5: Regression Controls

All standard regression controls were continuously monitored:
1. **6-Document Smoke Set:** Maintained high fidelity with 0 regressions on table, form, and non-table slides.
2. **Form 1040 Tax Documents:** `short/bianco-2024` achieved 30.34% F1 (slight gain due to improved attachment record alignment).
3. **Scanned Regulatory Forms:** `short/W14-Atascosa SWD Well No. 4` held steady at 51.09% F1.
4. **Non-Table Presentation Decks:** `medium/veralto_earnings_deck_q4fy25` remained 100% stable without unintended interference.
5. **Unit Test Suite:** All 206 unit tests passed in 21.19s.

---

## 6. Artifact Inventory

- `controlled_experiments_metrics.json`: Complete metrics payload across all documents and modes.
- `experiment_config.json`: Precise experiment parameters, tolerances, and ablation configurations.
- `representative_examples.json`: Detailed case studies showing before/after bounding box coordinates.
- `failure_samples.json`: Catalog of residual failures and physical OCR boundary limits.
- `phase1_diagnosis.py`: Verification script auditing category distributions and genuine gap fields.
- `run_controlled_experiments.py`: Modular, memory-safe execution suite for ablation Modes A through F.
- `src/tonerhound/resolution/joint_record_resolver.py`: Core production implementation of `JointRecordResolver`.
- `tests/test_joint_record_resolver.py`: Comprehensive test suite verifying all 11 required structural scenarios.
