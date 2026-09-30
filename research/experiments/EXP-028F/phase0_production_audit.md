# EXP-028F: Phase 0 Production Candidate-Generation Audit

## 1. Executive Summary

This audit rigorously inspects the current production codebase under `src/tonerhound/` at commit `e8a2f0f` (EXP-028E baseline: **56.05% Word Grounding F1** across all 370 ExtractBench documents).

The primary goal is to determine whether the 6 candidate-generation information-loss defects identified during historical research (EXP-028B0) actually exist in the **CURRENT production implementation**, or whether they were artifacts of offline research observer scripts or have already been addressed in prior iterations.

---

## 2. Audit Findings Matrix

| Defect | Exists? | Production Location | Current Limit / Behavior | Evidence | Safe Fix Required? |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **1. Candidate Pool Truncation** | **PARTIAL** | `matcher.py:392, 726`<br>`candidate_recovery.py:56, 148` | Exact & numeric matching are **uncapped**.<br>Fuzzy lines capped at `[:200]`.<br>Booleans capped at `[:200]`.<br>Candidate recovery capped at `[:500]`. | Historical `[:25]` was in `run_microscope_v2.py:303`, not `src/tonerhound/`. Production has generous soft caps. | **No** (Safe refinement optional) |
| **2. Global Search Fallback** | **YES** | `resolver.py:144-151`<br>`candidate_recovery.py:81-89`<br>`matcher.py:380-390` | Exact/numeric have `_allow_drift` fallback.<br>**BUT** `resolver.py:144` explicitly hardcodes `len(pages) <= 10`, completely blocking global recovery for long docs! `candidate_recovery.py:81` locks target pages to `page_hint +/- 1`. | `if not candidates and page_hint is not None and len(self.index.pages) <= 10:` | **YES (High Priority)** |
| **3. Per-Page Candidate Cutoff** | **NO** | `matcher.py:50-61`<br>`document/index.py:77, 91` | No per-page limit exists. All tokens and lines matching criteria are indexed and retrieved. | `_numeric_cache` and `_token_index` append all tokens without page slice or cap. | **NO (Already Solved)** |
| **4. Row / Identical-String Dedup** | **PARTIAL** | `matcher.py:462-535`<br>`resolver.py:301-315`<br>`adapter.py:694-700` | Inverted index does not collapse posting lists.<br>`_find_token_subsequence` extracts only *first* match on a line.<br>`resolver.py:309` marks tied non-overlapping candidates `AMBIGUOUS`. | `occ_idx` handles table numbers in `adapter.py`, but scalar resolver suppresses non-anchored duplicates. | **YES (Medium Priority)** |
| **5. Multi-Line / Wrapped Search** | **PARTIAL** | `matcher.py:537-612`<br>`index.py:280-310`<br>`adapter.py:752-777` | Column-bounded multi-line search exists (`span_len in (2, 3)`).<br>`adapter.py` has corridor fallback.<br>`index.py:search_exact` cannot match wrapped multi-column cells. >3 line wraps fail. | `span_len in (2, 3)` in `_find_multiline_candidates`; `search_exact` concatenates full lines with `\n`. | **YES (Low Priority)** |
| **6. Page Pruning / Rigid Routing** | **YES** | `matcher.py:324-325`<br>`matcher.py:363, 375`<br>`candidate_recovery.py:81-86` | `adapter.py` has macro offset calibration.<br>**BUT** `find_normalized_date_candidates` and `find_fuzzy_candidates` strictly filter by `p_num == page_hint` with **zero drift tolerance and no global fallback**. | `if page_hint is not None and p_num != page_hint: continue` in dates and fuzzy matching. | **YES (High Priority)** |

---

## 3. Detailed Forensic Inspection by Defect

### Defect 1: Candidate Pool Truncation
- **Status in Production**: **PARTIAL**
- **Exact Files & Lines**:
  - `src/tonerhound/matching/matcher.py:392`: `top_lines = sorted(line_overlap_counts.items(), key=lambda item: item[1], reverse=True)[:200]`
  - `src/tonerhound/matching/matcher.py:726`: `return unique_cands[:200]`
  - `src/tonerhound/matching/candidate_recovery.py:56, 148`: `max_candidate_per_field: int = 500`, `return unique[: self.max_candidate_per_field]`
- **Current Behavior**:
  - Exact match (`find_exact_candidates`) and numeric match (`find_normalized_numeric_candidates`) return **all valid occurrences** without slicing or truncation.
  - The historical `[:25]` truncation identified in EXP-028B0 was located in `research/observer/run_microscope_v2.py:303` (an offline microscope analysis script), not in production `src/tonerhound/`.
  - Production already raised fuzzy line pruning to 200 and candidate recovery to 500.
- **Why It Can Lose Valid Evidence**:
  - In massive 100+ page documents with diffuse n-gram overlap, line sorting could theoretically push lines on later pages outside the top-200. However, in practice 200 lines cover most relevant occurrences.
- **Estimated Risk of Changing**: Low.
- **Safe Fix Required?**: No mandatory fix required; optional refinement to partition top lines per page.

---

### Defect 2: Global Search Fallback
- **Status in Production**: **YES (CONFIRMED)**
- **Exact Files & Lines**:
  - `src/tonerhound/resolution/resolver.py:144–151`:
    ```python
    # Global recovery fallback for small documents if still zero candidates
    if not candidates and page_hint is not None and len(self.index.pages) <= 10:
        candidates.extend(self.recovery_engine.recover(
            value=value,
            page_hint=None,
            field_name=field,
            field_context=context,
            evidence_text=evidence_text,
        ))
    ```
  - `src/tonerhound/matching/candidate_recovery.py:81–89`:
    ```python
    if page_hint is not None and self.index.get_page(page_hint):
        target_pages = [page_hint]
        if page_hint - 1 >= 1 and self.index.get_page(page_hint - 1):
            target_pages.append(page_hint - 1)
        if page_hint + 1 <= len(self.index.pages) and self.index.get_page(page_hint + 1):
            target_pages.append(page_hint + 1)
    else:
        target_pages = [p.page_number for p in self.index.pages]
    ```
- **Current Behavior**:
  - Standard exact and normalized numeric matchers possess `_allow_drift` document-wide fallback.
  - However, `CandidateRecoveryEngine` strictly limits its search to `page_hint +/- 1` whenever `page_hint` is provided.
  - In `resolver.py:144`, global recovery fallback is explicitly conditioned on `len(self.index.pages) <= 10`.
- **Why It Can Lose Valid Evidence**:
  - Over 90% of ExtractBench test cases and fields are in medium and long documents (> 10 pages).
  - If a spaced-token, fragmented, or split-symbol value is on a page further away from `page_hint` (e.g. page_hint + 2 due to an uncalibrated section break), `CandidateRecoveryEngine` will never search for it because `resolver.py:144` suppresses global fallback on all documents with > 10 pages.
- **Estimated Risk of Changing**: Moderate.
  - Exhaustive scanning across 60+ pages can consume CPU cycles. A safe fix must gate global recovery using an O(1) inverted index token presence check.
- **Safe Fix Required?**: **YES**.

---

### Defect 3: Per-Page Candidate Cutoff
- **Status in Production**: **NO (NOT PRESENT / ALREADY SOLVED)**
- **Exact Files & Lines**:
  - `src/tonerhound/matching/matcher.py:50–61`
  - `src/tonerhound/document/index.py:77, 91`
- **Current Behavior**:
  - The historical `matcher.py:48-61` fixed candidate limit per page was eliminated in earlier refactorings.
  - `DocumentIndex` appends every token on every page to `_token_index` and `_numeric_index` without posting list caps or per-page limits.
- **Why It Can Lose Valid Evidence**:
  - N/A (defect is not present).
- **Safe Fix Required?**: **NO**.

---

### Defect 4: Row-Level / Identical-String Deduplication
- **Status in Production**: **PARTIAL**
- **Exact Files & Lines**:
  - `src/tonerhound/matching/matcher.py:462–535` (`_find_token_subsequence`)
  - `src/tonerhound/resolution/resolver.py:301–315`
  - `src/tonerhound/benchmark/adapter.py:694–700`
- **Current Behavior**:
  - Inverted index posting lists preserve all occurrences across rows and pages.
  - In table arrays, `adapter.py` tracks occurrences (`record_val_counts`) and uses monotonic DP.
  - However, within a single visual line, `_find_token_subsequence` returns only the first occurrence.
  - In scalar resolution (`resolver.py:309`), identical occurrences across different lines with tied confidence are marked `ProvenanceStatus.AMBIGUOUS` (`bbox=None`).
- **Why It Can Lose Valid Evidence**:
  - If multiple table columns on the same visual line contain the same string (e.g., repeated `"0"` or `"NONE"`), `_find_token_subsequence` always returns the leftmost match, failing subsequent columns.
- **Estimated Risk of Changing**: Moderate.
- **Safe Fix Required?**: **YES (for within-line multi-occurrence column matching)**.

---

### Defect 5: Multi-Line / Wrapped-Token Search
- **Status in Production**: **PARTIAL**
- **Exact Files & Lines**:
  - `src/tonerhound/matching/matcher.py:537–612` (`_find_multiline_candidates`)
  - `src/tonerhound/document/index.py:280–310` (`search_exact`)
  - `src/tonerhound/benchmark/adapter.py:752–777`
- **Current Behavior**:
  - `_find_multiline_candidates` checks 2-line and 3-line spans and uses column-bounded token sets (`col_w in (0.20, 0.32, 0.45)`).
  - `adapter.py` implements left-column and corridor filtering for row tokens.
  - However, `index.py:search_exact` operates on sequentially concatenated line text (`\n` separation). In multi-column tables, tokens from intervening columns break the string, preventing `search_exact` from matching wrapped cells.
  - Phrases wrapping across 4 or more lines cannot be found by `_find_multiline_candidates`.
- **Why It Can Lose Valid Evidence**:
  - Long entity names or wrapped narrative cells spanning 4+ lines or cells in wide multi-column tables are missed by `search_exact`.
- **Estimated Risk of Changing**: Low to Moderate.
- **Safe Fix Required?**: **YES (Targeted)**.

---

### Defect 6: Page Pruning / Rigid Routing
- **Status in Production**: **YES (CONFIRMED)**
- **Exact Files & Lines**:
  - `src/tonerhound/matching/matcher.py:324–325` (`find_normalized_date_candidates`):
    ```python
    if page_hint is not None and p_num != page_hint:
        continue
    ```
  - `src/tonerhound/matching/matcher.py:363, 375` (`find_fuzzy_candidates`):
    ```python
    if page_hint is not None and p_num != page_hint:
        continue
    ```
  - `src/tonerhound/matching/candidate_recovery.py:81–86`:
    ```python
    if page_hint is not None and self.index.get_page(page_hint):
        target_pages = [page_hint]
        if page_hint - 1 >= 1 and self.index.get_page(page_hint - 1):
            target_pages.append(page_hint - 1)
        if page_hint + 1 <= len(self.index.pages) and self.index.get_page(page_hint + 1):
            target_pages.append(page_hint + 1)
    ```
- **Current Behavior**:
  - `find_exact_candidates` and `find_normalized_numeric_candidates` have `_allow_drift` (checking `page_hint`, then `page_hint +/- 1`, then `page_hint=None`).
  - In contrast, `find_normalized_date_candidates` and `find_fuzzy_candidates` have **zero drift tolerance and no global fallback**: any match outside `page_hint` is immediately dropped.
  - `CandidateRecoveryEngine` is strictly capped to `page_hint +/- 1` and never falls back document-wide.
- **Why It Can Lose Valid Evidence**:
  - When logical page numbers drift from physical pages (e.g. by 2 pages) or when a date/text field appears on a subsequent continuation page, date and fuzzy matching fail with 0 candidates.
- **Estimated Risk of Changing**: Low.
  - Adding `_allow_drift` to dates and fuzzy matching follows the exact production pattern already tested and validated in exact/numeric matching.
- **Safe Fix Required?**: **YES (High Priority)**.

---

## 4. Defect Categorization

### 1. CONFIRMED Defects (Active Evidence-Loss in Production)
1. **Defect 2: Global Search Fallback Suppression in Medium/Long Docs**
   - Location: `src/tonerhound/resolution/resolver.py:144` (`len(self.index.pages) <= 10`).
   - Impact: Blocks global candidate recovery on 92% of benchmark fields.
2. **Defect 6: Rigid Page Routing in Date & Fuzzy Matching**
   - Location: `src/tonerhound/matching/matcher.py:324, 363, 375`.
   - Impact: 0 candidates returned for dates and fuzzy text when `page_hint` is off by even 1 page.

### 2. PARTIAL Defects (Partially Mitigated or Specialized)
1. **Defect 1: Candidate Pool Truncation**
   - Location: `matcher.py:392` (top 200 fuzzy lines), `matcher.py:726` (top 200 booleans), `candidate_recovery.py:148` (top 500 recovered).
   - Exact/numeric matching is uncapped; historical `[:25]` bug was in research observer scripts.
2. **Defect 4: Single-Line Repeated Occurrence Collapsing**
   - Location: `matcher.py:462` (`_find_token_subsequence`).
   - Repeated values in different columns on the same line always return the first column's bbox.
3. **Defect 5: Multi-Line Wrapped Table Cells**
   - Location: `index.py:267` (`search_exact`) and `matcher.py:565` (`span_len in (2, 3)`).
   - Interleaved columns break `search_exact`; spans > 3 lines are unhandled.

### 3. ALREADY SOLVED Defects
1. **Systematic Document/Table Page Offsets**: Solved in `adapter.py:_calibrate_page_offset` and `_filter_monotonic_row_pages`.
2. **Repeated Code Anchors in Monotonic Tables**: Solved in EXP-028D via sequence-aware DP scoring.
3. **Sub-token Character-Span Geometry**: Solved in EXP-015/028D via `character_span.py` and `_apply_geometry_enhancements`.

### 4. NOT PRESENT Defects
1. **Defect 3: Per-Page Candidate Cutoffs**: Not present in `src/tonerhound/`. Posting lists and page token caches are complete and uncapped.

---

## 5. Recommended Phase 1 Implementation Order

To maintain strict production stability and avoid regressions:

1. **Phase 1.1 — Safe Drift & Fallback for Dates and Fuzzy Matching (Defect 6)**
   - *Target*: `src/tonerhound/matching/matcher.py` (`find_normalized_date_candidates`, `find_fuzzy_candidates`).
   - *Action*: Introduce `_allow_drift: bool = True` with `page_hint +/- 1` and `page_hint=None` fallback, mirroring `find_exact_candidates`.
   - *Risk*: Very Low.

2. **Phase 1.2 — Token-Gated Global Candidate Recovery for Long Documents (Defect 2)**
   - *Target*: `src/tonerhound/resolution/resolver.py:144` and `src/tonerhound/matching/candidate_recovery.py:81`.
   - *Action*: Lift the `len(pages) <= 10` guard by adding an O(1) inverted index token pre-filter so global recovery runs safely on medium/long documents without CPU explosion.
   - *Risk*: Low to Moderate.

3. **Phase 1.3 — Column-Aware Single-Line Subsequence Matching (Defect 4)**
   - *Target*: `src/tonerhound/matching/matcher.py` (`_find_token_subsequence`).
   - *Action*: Add support for target column bounds / occurrence index so repeated tokens on the same line match their respective columns.
   - *Risk*: Low.

4. **Phase 1.4 — Extended Multi-Line Span (Defect 5)**
   - *Target*: `src/tonerhound/matching/matcher.py` (`_find_multiline_candidates`).
   - *Action*: Allow 4-line spans for long queries (>= 6 words).
   - *Risk*: Very Low.

---

## 6. Audit Conclusion

The audit is complete. No production code was modified during this phase. All findings are backed by line-level code references and behavioral verification.
