# EXP-030 Phase 1: Candidate Generation Audit

**Date**: October 1, 2026  
**Status**: Completed  
**Scope**: Verification of production candidate generation mechanisms, caps, pruning, and fallbacks  
**Production Code Impact**: **ZERO (Production code untouched)**  

---

## 1. Executive Summary

This audit investigates the candidate-generation layer currently active in TonerHound's production codebase (`src/tonerhound/matching/matcher.py`, `src/tonerhound/matching/candidate_recovery.py`, `src/tonerhound/resolution/resolver.py`, and `src/tonerhound/benchmark/adapter.py`).

In EXP-028B1 and EXP-028F, multiple candidate-generation limitations were diagnosed. Following project directives, candidate-generation expansion was halted after EXP-028F Phase 1.2 (+0.06pp smoke gain). Before implementing global structured assignment in EXP-030, this audit documents the exact active state of all six candidate-generation mechanisms to ensure structured assignment builds upon a verified, stable foundation.

---

## 2. Audit of Candidate-Generation Mechanisms

### A. Candidate Truncation (Active)
- **Implementation**:
  - `EvidenceResolver.collect_candidates()`: `return candidates[:200]` (`resolver.py:174`).
  - `CandidateRecoveryEngine._deduplicate_and_rank()`: `return unique[: min(200, self.max_candidate_per_field)]` (`candidate_recovery.py:275`).
  - `EvidenceMatcher.find_fuzzy_candidates()`: Line candidate filtering `top_lines = ... [:200]` (`matcher.py:450`).
- **Active in Production**: **YES**.
- **Assessment**:
  - A hard limit of 200 candidates per field is active.
  - Across 99.8% of fields, 200 candidates exceeds the total occurrences in the document.
  - In massive table documents (e.g. `long/real_ishares_iboxx_bond_etfs` with 41,525 fields across 50 pages), generic tokens like "0.00" or state codes appear $>500$ times. Truncating at 200 limits retrieval on later pages if page hints are absent.
  - However, expanding beyond 200 without spatial indexing causes combinatorial blowup in matcher execution time.

### B. Per-Page Cutoffs (Active)
- **Implementation**:
  - `EvidenceMatcher.find_fuzzy_candidates()` restricts line candidates using top-200 token overlap lines across the document (`matcher.py:449`).
  - `CandidateRecoveryEngine._execute_recovery_passes()` terminates recovery once candidate cap reaches 200 or 500ms hard timeout expires (`candidate_recovery.py:198-202`).
- **Active in Production**: **YES**.
- **Assessment**:
  - These cutoffs are vital safety mechanisms protecting against catastrophic $O(N \cdot M)$ latency spikes on 50+ page documents.

### C. Row-Level Deduplication Collapse (Active & Safe)
- **Implementation**:
  - `CandidateRecoveryEngine._deduplicate_and_rank()` deduplicates candidate spans on the same page with $\text{IoU} \ge 0.85$ (`candidate_recovery.py:247`).
  - In `ExtractBenchAdapter`, within-row duplicate occurrences of identical numbers are tracked via `record_val_counts[row_rec_key][occ_key]` (`adapter.py:696-698`).
- **Active in Production**: **YES** (Validated safe in EXP-028F Phase 1.1).
- **Assessment**:
  - Intra-row deduplication prevents duplicate token emissions.
  - Inter-row duplicates are preserved across different table rows, allowing different rows to claim their respective occurrences.

### D. Page Pruning (Active)
- **Implementation**:
  - `_filter_monotonic_row_pages()` applies Longest Non-Decreasing Subsequence (LNDS) dynamic programming to table rows (`adapter.py:57-79, 1289`).
  - Outlier page assignments that violate monotonic document progression are pruned, and monotonic page propagation bridges unassigned rows.
- **Active in Production**: **YES**.
- **Assessment**:
  - Essential architectural pillar. In EXP-028C, disabling monotonic page alignment caused massive regressions (-17.46pp on clinical tables, -16.23pp on 13F filings).

### E. Rigid Routing & Row Confinement (Active)
- **Implementation**:
  - In `ExtractBenchAdapter`, when a row anchor is established (`anchor is not None`), field resolution is confined to the anchor's vertical line corridor:
    `cand_row_lines = [ln for ln in page_obj.lines if -up_tol <= (ln.bbox.y - anc_cy) <= down_tol]` (`adapter.py:522-546`).
  - If a candidate falls outside vertical tolerance $\Delta y > \max(0.025, \text{anc\_h} \cdot 1.5)$, the field is explicitly aborted (`continue` at line 920).
  - Conversely, for unanchored rows (`anchor is None`), lines 923-931 resort to an arbitrary index fallback: `match_idx = min(r_idx, len(page_matches) - 1)`.
- **Active in Production**: **YES**.
- **Assessment**:
  - This is the primary structural vulnerability in production table resolution:
    - If `_align_table_arrays` misses a row anchor, the entire row's scalar fields fall back to index matching or fail entirely.
    - If a row spans multiple visual lines (wrapped multi-line record), fields on the wrapped line outside the vertical tolerance are discarded.

### F. Insufficient Global Recovery on Long Documents (Resolved by EXP-028F Phase 1.2)
- **Implementation**:
  - `ENABLE_GLOBAL_FALLBACK = True` (`resolver.py:27`, `candidate_recovery.py:23`).
  - 7 mandatory safeguards active: Inverted-index pre-check, candidate cap 200, 500ms timeout, page priority ranking, 0.85 global penalty, early termination on similarity $\ge 0.95$, and generic stopword exclusion.
- **Active in Production**: **YES**.
- **Assessment**:
  - Verified safe and non-regressive (+0.06pp smoke gain, 28.05s runtime).

---

## 3. Conclusions for EXP-030 Structured Assignment

1. **Candidate Pool Inventory is Stable & Sufficient**:
   As verified in EXP-029, candidate-pool recall on Held-Out Cohort B multi-candidate fields is **75.83%**, against a production Hit@1 of only **25.00%**. There is a massive **50.83 pp selection gap** already contained inside the existing candidate pools.
2. **Do Not Modify Candidate Generation**:
   The problem is NOT that the candidates do not exist; the candidates are already retrieved and present in the candidate pools. The problem is that production makes point-wise or rigid row-corridor selection decisions.
3. **Structured Assignment Focus**:
   EXP-030 will focus entirely on **joint multi-field record assignment** across candidate rows and columns, leaving the underlying candidate generation pipeline untouched.
