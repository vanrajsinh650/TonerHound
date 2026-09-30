# EXP-028C: Production Measurement Report — JointRecordResolver

**Experiment ID:** EXP-028C  
**Date:** 2026-09-30  
**Harness:** Official ExtractBench `ExtractEvaluator`  
**Primary Metric:** Word Grounding F1 (IoU >= 0.50)  
**Status:** STOPPED AT PHASE 1 DECISION GATE (Regression Detected)  

---

## 1. Executive Summary

The goal of **EXP-028C** was to measure the real-world production impact of the **EXP-028B2 `JointRecordResolver`** against the official **EXP-028B1** production baseline across the ExtractBench benchmark.

In **EXP-028B2**, a controlled 6-stage ablation (Modes A through F) reported that **Mode E (Joint Multi-Field Record Resolution)** achieved **58.23% Word F1** on a 6-document evaluation suite and **69.78% Word F1** on the major SEC 13F table case `sec_13f_0031_loomis_sayles`, which was interpreted as a **+12.92pp suite gain** and **+30.32pp table gain** over the baseline.

In accordance with the experimental protocol, **Phase 1 (Quick Sanity Check)** was executed on the official 6-document smoke benchmark. Direct measurement against **EXP-028B1** revealed that **the EXP-028B2 `JointRecordResolver` causes severe regressions across all structured table documents**:

- **Official 6-Document Smoke Benchmark:** Regressed from **62.84%** down to **59.07%** (**-3.77pp delta**).
- **Major Clinical Table (`long/real_sm0801_eco_full`):** Regressed from **92.48%** down to **75.02%** (**-17.46pp regression**; 1,279 fields dropped).
- **Major Financial Table (`medium/sec_13f_0031_loomis_sayles`):** Regressed from **86.00%** down to **69.77%** (**-16.23pp regression**; 2,538 fields dropped).
- **Regulatory Scanned Form (`short/W14-Atascosa`):** Regressed from **54.01%** down to **51.09%** (**-2.92pp regression**).
- **Form 1040 Tax Return (`short/bianco-2024`):** Regressed from **31.44%** down to **29.43%** (**-2.01pp regression**).
- **County Record (`medium/real_pueblo_oct_2025`):** Regressed from **99.60%** down to **99.39%** (**-0.21pp regression**).
- **Control Documents (`Goshen`, `veralto`):** Remained flat (99.49% and 0.00%).

### Decision Gate Verdict: STOP TRIGGERED
> **Protocol Rule:** *"If there is an obvious regression, STOP and diagnose only the regression. If healthy, immediately continue to the full benchmark."*  
> **Verdict:** **STOPPED AT PHASE 1.** Execution of the full 370-document benchmark was halted pursuant to the protocol rule. A full forensic diagnosis was conducted to establish the exact root causes of why the controlled EXP-028B2 gains failed to transfer to production.

---

## 2. Direct Comparison against Baselines

### A. Smoke Benchmark Suite Performance

| Document ID | EXP-026 Production | EXP-028B1 Production | EXP-028C (Smoke) | Delta vs B1 | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `long/real_sm0801_eco_full` | 92.46% | **92.48%** | 75.02% | **-17.46pp** | ❌ Severe Regression |
| `medium/sec_13f_0031_loomis_sayles` | 86.00% | **86.00%** | 69.77% | **-16.23pp** | ❌ Severe Regression |
| `short/W14-Atascosa SWD Well No. 4` | 54.01% | **54.01%** | 51.09% | **-2.92pp** | ❌ Regression |
| `short/bianco-2024` | 31.44% | **31.44%** | 29.43% | **-2.01pp** | ❌ Regression |
| `medium/real_pueblo_oct_2025` | 99.60% | **99.60%** | 99.39% | **-0.21pp** | ❌ Minor Regression |
| `short/real_wyo_Goshen_2024` | 99.49% | **99.49%** | 99.49% | 0.00pp | ⏸️ Stable |
| `medium/veralto_earnings_deck_q4fy25`| 0.00% | **0.00%** | 0.00% | 0.00pp | ⏸️ Stable |
| **Official 6-Document Smoke Macro F1**| 62.83% | **62.84%** | **59.07%** | **-3.77pp** | ❌ Decision Gate Stop |
| **B2 6-Doc Suite Macro F1 (with Loomis)**| 61.76% | **61.76%** | **58.20%** | **-3.56pp** | ❌ Decision Gate Stop |

### B. Why Did EXP-028B2 Report a "+12.92pp Gain"? (The Baseline Mirage)

In EXP-028B2, the developer constructed **Mode A (Baseline)** by running unconstrained exact text matching across all table fields in isolation, which threw away all existing production table geometry and alignment logic:
- In Mode A, `loomis_sayles` scored **39.46%** Word F1.
- Mode E then added anchor discovery and row/column rails to reach **69.78%** Word F1 (+30.32pp relative to Mode A).
- Across the 6-document ablation set, Mode A scored **45.31%** Word F1, while Mode E scored **58.23%** Word F1 (+12.92pp relative to Mode A).

**However, the production baseline (EXP-028B1 / EXP-026) was NOT 45.31%.**  
Production `ExtractBenchAdapter` already possessed a dynamic programming monotonic row alignment engine and adaptive character-span geometry, which achieved **61.76%** on that suite and **86.00%** on `loomis_sayles`.  

When Mode E is compared to the actual production baseline, **58.23% is 3.53pp LOWER than production (61.76%)**, and **69.78% on Loomis Sayles is 16.22pp LOWER than production (86.00%)**. The "+12.92pp gain" was an artifact of measuring against a degraded synthetic baseline.

---

## 3. Root Cause Forensic Diagnosis

A field-level diff was performed comparing `predictions` from EXP-028B1 and EXP-028C across the regressed documents (2,538 regressed fields on `loomis_sayles`, 1,279 on `sm0801`). Four distinct failure mechanisms were identified:

### Root Cause 1: Anchor Collapse on Multi-Row Code Tokens (Primary Driver)
In `src/tonerhound/resolution/joint_record_resolver.py` (`identify_record_anchor`):
```python
cand_pages = {c.page for c in cands}
# An anchor must be unambiguous (exactly 1 candidate or clearly dominant)
if len(cands) > 1 and not (is_code and len(cand_pages) == 1):
    continue
...
best_anchor_cand = cands[0]  # Line 342
```
- **The Defect:** When a distinctive alphanumeric code (such as a 9-digit CUSIP `000361105`) appears across multiple rows on the same page, the condition `(is_code and len(cand_pages) == 1)` evaluates to `True`. The resolver allows multiple candidates, but then **unconditionally selects `cands[0]`** for every row that contains that code.
- **The Failure:** Row 1 (`holdings[1]`) selects `cands[0]` at $y = 0.347$. Row 2 (`holdings[2]`, also AAR CORP with CUSIP `000361105`) ALSO selects `cands[0]` at $y = 0.347$. As a result, Row 2 is collapsed onto Row 1's vertical position! All fields belonging to Row 2 (`name_of_issuer`, `shares`, `value`, `voting_authority`) search within Row 1's corridor and fail the ground-truth IoU.
- **Production Difference:** `ExtractBenchAdapter` uses a global dynamic programming alignment step that enforces monotonic vertical progression ($y_0 < y_1 < y_2 < \dots$), preventing duplicate token snapping.

### Root Cause 2: Hardcoded Column Corridor Boundaries
In `STANDARD_TABLE_COLUMNS["holdings"]`:
```python
"voting_authority.none": (0.82, 0.92)
```
- **The Defect:** In `sec_13f_0031_loomis_sayles`, the actual `voting_authority.none` column is positioned at $x \in [0.940, 0.955]$.
- **The Failure:** Because the hardcoded corridor stops at $0.92$, `JointRecordResolver.score_candidate` assesses a $-10.0$ penalty against candidates at $x = 0.9441$. It instead forces selection of a "0" token located inside $[0.82, 0.92]$ (which belongs to `voting_authority.shared`). Consequently, hundreds of voting authority fields received $0.00$ IoU.

### Root Cause 3: Loss of Character-Span Geometry Enhancements
- `run_controlled_experiments.py` directly assigned `new_citations[fpath] = outcome.candidate.bbox.to_coco()`.
- **The Defect:** Raw token boxes omit `reconstruct_safe_character_span` (EXP-015) and `extend_same_line_tokens` (EXP-017). When a ground-truth evidence string is a sub-span of an OCR word or a concatenated line, the un-enhanced candidate box fails the strict $0.50$ IoU threshold against ground truth.

### Root Cause 4: Lack of Cross-Record Sequence Constraints
- `JointRecordResolver` evaluates each `StructuralRecord` as an isolated island without awareness of preceding or succeeding records.
- In documents with dense, homogeneous rows containing repeated values (e.g. `real_sm0801_eco_full`, where timestamps and measurement values like `0` and `Week` appear 50+ times per page), isolated records lose their sequence context and cluster toward the top of the page.

---

## 4. Phase 1 Metrics Detail

| Metric | EXP-028B1 Baseline | EXP-028C Sanity Check | Delta |
| :--- | :---: | :---: | :---: |
| **Word Grounding F1** | **62.84%** | **59.07%** | **-3.77pp** |
| **Word Precision** | 66.80% | 61.64% | -5.16pp |
| **Word Recall** | 59.32% | 57.32% | -2.00pp |
| **Page Grounding F1** | 71.36% | 71.67% | +0.31pp |
| **False Grounding Rate** | 12.44% | 16.18% | +3.74pp |
| **Abstention Rate** | 0.00% | 0.00% | 0.00pp |
| **Runtime (7 docs)** | ~8.2s | 5.8s | -2.4s |

---

## 5. Architectural Recommendations for EXP-028D

To successfully productionize table resolution without regressions:

1. **Do Not Discard Monotonic DP Alignment:**  
   `JointRecordResolver`'s spatial scoring ($S_{\text{row}}, S_{\text{col}}, S_{\text{sib}}$) should be embedded **inside** `ExtractBenchAdapter`'s dynamic programming table aligner as an enhanced cost function, rather than replacing the DP pass with unsequenced per-row calls.
2. **Dynamic Column Detection over Static Constants:**  
   Replace `STANDARD_TABLE_COLUMNS` with dynamic column corridor estimation derived from header text geometry and line x-projections.
3. **Multi-Occurrence Code Disambiguation:**  
   In `identify_record_anchor`, when multiple occurrences of a code exist on a page, use the monotonic sequence index ($i$-th record matches $i$-th code) rather than defaulting to `cands[0]`.
4. **Mandatory Geometry Enhancement Pipeline:**  
   All candidates produced by record resolution must pass through `_apply_geometry_enhancements` (`reconstruct_safe_character_span`).

---

## 6. Artifact Inventory

- `experiment_config.json`: Configuration, baseline definitions, and decision gate specification.
- `per_document.csv`: Detailed evaluation metrics for all smoke and comparison documents.
- `results.json`: Full machine-readable metrics payload.
- `smoke_predictions/`: Complete ExtractBench `InferenceResult` files for the smoke cohort.
- `smoke_eval_cache/`: Official evaluation JSONs generated by `ExtractEvaluator`.
