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

## 2. Benchmark Results

Evaluated with `--workers 2` using official ExtractBench evaluation against EXP-028D/EXP-028E baselines:

| Document | Word F1 | Baseline (EXP-028D) | Delta | Word Prec | Page F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `long/real_sm0801_eco_full` | **92.50%** | 92.48% | +0.02pp | 92.50% | 97.58% | **PASSED** (Gate: >= 91.48%) |
| `medium/sec_13f_0031_loomis_sayles` | **86.28%** | 86.30% | -0.02pp | 87.38% | 93.82% | **PASSED** (Gate: >= 85.30%) |
| `short/W14-Atascosa SWD Well No. 4` | **54.01%** | 54.01% | +0.00pp | 68.52% | 61.09% | **PASSED** (Gate: >= 53.00%) |
| `short/bianco-2024` | **31.44%** | 31.44% | -0.00pp | 32.64% | 69.57% | **PASSED** (Gate: >= 30.44%) |
| `medium/real_pueblo_oct_2025` | **99.63%** | 99.60% | +0.03pp | 99.63% | 99.86% | **PASSED** (Gate: >= 98.60%) |
| `short/real_wyo_Goshen_2024` | **99.83%** | 99.49% | +0.34pp | 100.00% | 99.88% | **PASSED** (Gate: >= 98.49%) |
| `medium/veralto_earnings_deck_q4fy25` | **0.00%** | 0.00% | +0.00pp | 0.00% | 0.00% | **PASSED** (Gate: >= 0.00%) |
| **Smoke Suite Average (6 docs)** | **62.90%** | **62.84%** | **+0.06pp** | — | — | **PASSED** (Gate: >= 62.84%) |
| **All 7 Targeted Docs Average** | **66.24%** | — | — | — | — | — |

- **Total Execution Runtime:** **28.05s** (Safety limit: <= 60.0s -> **PASSED**).
- **False Grounding / Precision:** Word precision remained extremely strong (Goshen: 100.00%, Pueblo: 99.63%, SM0801: 92.50%, Loomis Sayles: 87.38%).
- **Runtime Performance on Long Documents**:
  - `real_sm0801_eco_full` (50 pages): 13.5s.
  - `sec_13f_0031_loomis_sayles` (37 pages, 15,186 rules): 28.0s.

---

## 3. Implementation Summary

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

## 4. Decision Gate Verification

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

## 5. Next Steps
- Commit changes with message `exp028f-phase1.2: enable safe global fallback for long documents` and push to `origin/main`.
- In accordance with instructions, STOP after this phase. Do not run the full 370-document benchmark. Do not implement Phase 1.3 or Phase 1.4 without explicit instruction.
