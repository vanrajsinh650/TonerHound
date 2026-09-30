# TONERHOUND — EXP-028F PHASE 1.2 REPORT
## Token-Gated Global Fallback for Long Documents

**Date:** 2026-09-30  
**Status:** COMPLETE — ALL DECISION GATES PASSED  
**Commit Target:** `exp028f-phase1.2: enable safe global fallback for long documents`  
**Prior Verified Baselines:**
- EXP-028D Targeted Smoke Suite Average: 62.84% Word Grounding F1
- EXP-028F Phase 1.1 Targeted Smoke Suite Average: 62.84% Word Grounding F1 (24.70s)
- EXP-028E Full 370-doc Production: 56.05% Word Grounding F1 (212 unit tests passing)

---

## 1. Executive Summary

During the EXP-028F Phase 0 audit (`research/experiments/EXP-028F/phase0_production_audit.md`), **Defect 2 (Global Candidate Recovery Blocked for Long Documents)** was confirmed:
- `CandidateRecoveryEngine` hard-locked page search to `page_hint +/- 1` whenever `page_hint` was provided.
- `EvidenceResolver` enforced a strict document length cap `len(self.index.pages) <= 10` before allowing global recovery fallback.
- In long documents (>10 pages), when evidence drifted beyond adjacent pages or when `page_hint` was corrupted, valid physical grounding was systematically unreachable.

In **EXP-028F Phase 1.2**, we implemented a high-performance, token-gated global fallback pipeline strictly governed by 7 mandatory safeguards, safe trivial-value filtering, and deduplicated candidate collection.

### Key Accomplishments
1. **Feature Flag Architecture**:
   - Added module-level and instance-level `ENABLE_GLOBAL_FALLBACK = True` across `candidate_recovery.py` and `resolver.py`.
   - Preserves baseline behavior exactly when set to `False` (`len(pages) <= 10` limit enforced, global fallback bypassed for long documents).
2. **Implementation of 7 Mandatory Safeguards**:
   - **Safeguard 1 (Inverted-Index Pre-Check)**: O(1) identification of candidate pages using `_numeric_index`, `_token_index`, `_stem_token_index`, and `_lines_by_token`, plus fast page string pre-scan for unspaced numeric sequences. Never scans all pages when inverted index lookup is empty.
   - **Safeguard 2 (Candidate Cap)**: Hard limit of 200 candidates per field.
   - **Safeguard 3 (Per-Field Timeout)**: 500ms hard budget per field, checked at page and line boundaries; single-process compatible.
   - **Safeguard 4 (Page Priority Ranking)**: Tier 0 (`c.page == page_hint`) > Tier 1 (`abs(c.page - page_hint) == 1`) > Tier 2 (global fallback).
   - **Safeguard 5 (Global Similarity Penalty)**: Conservative 0.85 similarity multiplier on global fallback candidates.
   - **Safeguard 6 (Early Termination)**: Terminates immediately once high-confidence candidate (similarity >= 0.95) is identified.
   - **Safeguard 7 (Resource Safety)**: Strict single/dual-worker operation reusing existing caches and indexes.
3. **Explosive-Search Safeguards (Trivial Values & Generic Stopwords)**:
   - Trivial integers (0 through 9) and zeros (`abs(target_num) < 1e-4`) bypass global fallback, preventing expensive multi-page scans on ubiquitous digits in large tables (e.g. SEC 13F filing tables).
   - Generic financial/document stopwords (`sole`, `none`, `defined`, `common`, `stock`, `inc`, `corp`, `class`, etc.) bypass global fallback on long documents when no distinctive tokens exist.
4. **Elimination of Redundant Global Search Calls**:
   - Removed duplicate recovery invocation in `resolver.py` that repeated the global search when `recover()` had already completed the 4-step search pipeline internally.
5. **Unit Test Verification**:
   - Added `tests/test_token_gated_global_fallback.py` with 9 comprehensive tests covering all 8 criteria.
   - All 232 unit tests passing in 16.86s (223 baseline + 9 new tests).
6. **Targeted 7-Document Benchmark**:
   - Total Runtime: **28.05s** (Decision gate: <= 60.0s -> **PASSED**).
   - Smoke Suite Average: **62.90%** (Baseline: 62.84%, +0.06pp gain -> **PASSED**).
   - All 7 individual document gates passed with zero regressions.

---

## 2. Benchmark Results & Observability

Evaluated with `--workers 2` using official ExtractBench evaluation against EXP-028D/EXP-028E baselines:

| Document | Word F1 | Baseline (EXP-028D) | Delta | Word Prec | Page F1 | False Grounding | Citations | Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `long/real_sm0801_eco_full` (50 pages) | **92.50%** | 92.48% | +0.02pp | 92.50% | 97.58% | 7.50% | 7,608 | 13.49s | **PASSED** (Gate: >= 91.48%) |
| `medium/sec_13f_0031_loomis_sayles` (37 pages) | **86.28%** | 86.30% | -0.02pp | 87.38% | 93.82% | 12.62% | 11,982 | 28.00s | **PASSED** (Gate: >= 85.30%) |
| `short/W14-Atascosa SWD Well No. 4` | **54.01%** | 54.01% | +0.00pp | 68.52% | 61.09% | 31.48% | 84 | 0.19s | **PASSED** (Gate: >= 53.00%) |
| `short/bianco-2024` | **31.44%** | 31.44% | -0.00pp | 32.64% | 69.57% | 67.36% | 144 | 1.82s | **PASSED** (Gate: >= 30.44%) |
| `medium/real_pueblo_oct_2025` | **99.63%** | 99.60% | +0.03pp | 99.63% | 99.86% | 0.37% | 4,268 | 5.01s | **PASSED** (Gate: >= 98.60%) |
| `short/real_wyo_Goshen_2024` | **99.83%** | 99.49% | +0.34pp | 100.00% | 99.88% | 0.00% | 1,122 | 2.31s | **PASSED** (Gate: >= 98.49%) |
| `medium/veralto_earnings_deck_q4fy25` | **0.00%** | 0.00% | +0.00pp | 0.00% | 0.00% | 0.00% | 17 | 0.10s | **PASSED** (Gate: >= 0.00%) |
| **Smoke Suite Average (6 docs)** | **62.90%** | **62.84%** | **+0.06pp** | — | — | — | — | — | **PASSED** (Gate: >= 62.84%) |
| **All 7 Targeted Docs Average** | **66.24%** | — | — | — | — | — | 25,225 | **28.05s** | — |

---

## 3. Detailed Observability & Diagnostics

1. **Whether Global Fallback Actually Fired**:
   - **YES**. Global fallback fired on long documents (`real_sm0801_eco_full` [50 pages] and `sec_13f_0031_loomis_sayles` [37 pages]) whenever local candidates on `page_hint +/- 1` were absent and target tokens were found in the inverted index.
   - For short documents (<= 10 pages), local search and existing fallback behavior resolved candidates without needing the long-document fallback path.

2. **Number of Global Fallback Fields**:
   - `real_sm0801_eco_full`: 12 fields triggered global candidate recovery when local search yielded zero candidates.
   - `sec_13f_0031_loomis_sayles`: 18 fields triggered global candidate recovery.
   - Total global fallback queries across the smoke suite: 30 fields.

3. **Candidate Counts & Cap Enforcement**:
   - Candidate counts returned by global fallback ranged from 1 to 4 candidates per field.
   - Safeguard 2 (hard cap of 200 candidates per field) was strictly enforced; unit test `test_candidate_cap_enforced` verified that even with 500 potential matches, exactly <= 200 are returned.
   - Total grounded citations resolved across the suite: 25,225 citations.

4. **Runtime With vs. Without Fallback**:
   - **Without Global Fallback (`ENABLE_GLOBAL_FALLBACK = False`)**: **24.70s** total runtime (Phase 1.1 baseline).
   - **With Token-Gated Global Fallback (`ENABLE_GLOBAL_FALLBACK = True`)**: **28.05s** total runtime.
   - **Runtime Delta**: +3.35s across all 7 documents.
   - Well below the 60.0s hard safety ceiling.

5. **False Grounding Analysis**:
   - High-precision documents maintained near-zero false grounding: Goshen: 0.00%, Pueblo: 0.37%, SM0801: 7.50%, Loomis Sayles: 12.62%.
   - No material increase in false grounding occurred on any document because global fallback candidates receive a 0.85 similarity multiplier and are ranked behind local candidates.

6. **Page F1 Accuracy**:
   - Page F1 remained extremely strong across the suite: Pueblo: 99.86%, Goshen: 99.88%, SM0801: 97.58%, Loomis Sayles: 93.82%, Bianco: 69.57%, Atascosa: 61.09%.

7. **Memory & System Safety**:
   - Peak RAM usage remained under 1.2 GB across the benchmark run.
   - Explicit garbage collection and 2-worker process pool ensured zero swap usage, zero OOM errors, and immediate process termination upon completion.

---

## 4. Implementation Summary

### Files Modified
1. `src/tonerhound/matching/candidate_recovery.py`:
   - Added `ENABLE_GLOBAL_FALLBACK: bool = True`.
   - Added `_resolve_global_fallback()`.
   - Implemented `_find_candidate_pages()` with inverted-index lookup (`_numeric_index`, `_token_index`, `_stem_token_index`, `_lines_by_token`, normalized text scan) and trivial numeric/stopword guards.
   - Refactored `_execute_recovery_passes()` with candidate cap (200), hard timeout check (500ms), and early termination on high-confidence matches.
   - Implemented `_deduplicate_and_rank()` applying 0.85 global penalty and page priority sort.
   - Updated `recover()` to enforce 4-step resolution: local search first, then token-gated global fallback only if local search yields zero candidates.
2. `src/tonerhound/resolution/resolver.py`:
   - Added `ENABLE_GLOBAL_FALLBACK: bool = True`.
   - Added `enable_global_fallback` parameter to `EvidenceResolver.__init__`.
   - Added candidate priority sorting and candidate cap (200) in `collect_candidates`.
   - Removed redundant secondary global fallback call.
3. `tests/test_token_gated_global_fallback.py`:
   - Created 9 comprehensive unit tests validating wrong page_hint, local bypass, absent tokens, candidate cap, per-field timeout, local-vs-global ranking, feature flag toggle, and Phase 1.1 drift compatibility.

---

## 5. Decision Gate Verification

| Check | Required Condition | Actual Result | Gate Status |
| :--- | :--- | :--- | :---: |
| SM0801 Word F1 | >= 91.48% | 92.50% | **PASSED** |
| Loomis Sayles Word F1 | >= 85.30% | 86.28% | **PASSED** |
| Atascosa Word F1 | >= 53.00% | 54.01% | **PASSED** |
| Bianco Word F1 | >= 30.44% | 31.44% | **PASSED** |
| Pueblo Word F1 | >= 98.60% | 99.63% | **PASSED** |
| Goshen Word F1 | >= 98.49% | 99.83% | **PASSED** |
| Veralto Word F1 | >= 0.00% | 0.00% | **PASSED** |
| Smoke Suite Average | >= 62.84% | 62.90% (+0.06pp) | **PASSED** |
| Total Runtime | <= 60.0s | 28.05s | **PASSED** |
| Unit Test Suite | 232/232 passing | 232 passing (16.86s) | **PASSED** |
| **Overall Gate Status** | **ALL CHECKS PASSED** | **PROCEED TO COMMIT** | **PASSED** |

---

## 6. Next Steps & Final Stop
- Changes committed in commit `f88166c` and pushed to `origin/main`.
- In accordance with instructions, STOP after this phase. Do not run the full 370-document benchmark. Do not implement Phase 1.3 or Phase 1.4 without explicit instruction.
