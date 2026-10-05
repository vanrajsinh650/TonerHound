# EXP-033: Candidate Generation Funnel Analysis

**Experiment**: EXP-033 (Candidate Generation Reconciliation Audit + Productionization)  
**Date**: October 2026  
**Scope**: Full 370-document ExtractBench corpus (445,950 total gradeable fields)

---

## 1. Executive Summary

This document traces the complete candidate-generation funnel in TonerHound, mapping how 445,950 ground-truth fields progress through text extraction, page routing, tokenization, candidate generation, candidate expansion, and post-filtering.

Of the **48,744** failed fields that terminate with literally zero candidates (`RETRIEVAL_NO_CANDIDATE`):
- **76.5% (~37,289 fields)** are trapped in OCR transcription noise across two massive corrupted creditor matrices (`real_imedia_full_corrupted` and `real_ftx_full_corrupted`). In document macro-averaged Word F1, these two documents account for only **0.84%** total weight.
- **23.5% (~11,455 fields)** occur across clean and moderately complex documents due to algorithmic gaps in the candidate generation funnel:
  1. Token punctuation & symbol clipping (`5482'`, currency/percent symbols): ~5,361 fields (11.0%)
  2. Page hint drift & global search restriction on documents >10 pages: ~2,193 fields (4.5%)
  3. Table cell alignment & dense token clipping: ~1,706 fields (3.5%)
  4. Boolean / Checkbox keyword filtering omissions: ~974 fields (2.0%)
  5. Multi-line whitespace/newline breaks: ~487 fields (1.0%)
  6. OCR / PDF text layer omission: ~731 fields (1.5%)

---

## 2. The Complete Candidate Generation Funnel

```text
                  445,950 Gold Gradeable Fields
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    378,988 Initial Hits             66,962 No Initial Candidate
    (Adapter / Table DP)                       │
                               ┌───────────────┴───────────────┐
                               ▼                               ▼
                     18,218 SUCCESS                  48,744 FAILED
                   (Resolved by Adapter)       (RETRIEVAL_NO_CANDIDATE)
                                                               │
        ┌──────────────────────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                                      ▼                                        ▼
37,289 OCR Noise Trap                                10,481 Text-Recoverable                   974 Checkbox Fields
(2 long corrupted matrices)                                    │                                        │
- real_imedia_full_corrupted (~22k)            ┌───────────────┴───────────────┐                 (W-14, 1040, W-2, H-9)
- real_ftx_full_corrupted (~15k)               ▼                               ▼                 - Name missing in
                                     5,361 Punctuation/Token         2,193 Page Routing            hardcoded list
                                     - trailing quotes (5482')       - >10 page block          - Evaluated as string
                                     - currency / % signs            - drift > +/- 1             "False" instead of
                                     - alphanumeric boundary         - boilerplate across        box geometry
                                       slicing rejections              distant pages
                                               │                               │
                                               ▼                               ▼
                                     1,706 Table Geometry            487 Multiline Spans
                                     - dense repeated rows           - \n inside queries
                                     - column boundary trims         - line-break splits
```

---

## 3. Funnel Stages & Detailed Dropout Root Causes

### Stage 1: Document / Page Routing
* **Mechanism**: Fields in structured extraction carry an optional `page_hint` indicating the predicted or expected page.
* **Dropout Mechanism**:
  - In `resolver.py` line 130:
    ```python
    if not candidates and page_hint is not None:
        # Tries exact, boolean, numeric on page_hint=None
    ```
  - However, in `matcher.py` lines 126–140:
    ```python
    if q_words:
        postings = [self.index._lines_by_token.get(w, []) for w in q_words]
        ...
    elif len(self.index.pages) <= 10:
        # Only falls back to checking all lines if <= 10 pages!
    ```
    If `len(self.index.pages) > 10` and `q_words` is empty or common stopwords, candidate generation completely drops the query and checks 0 lines!
  - Furthermore, boilerplate notices (e.g. OFAC directives comments on `real_ofac_ssi_full`) carry a `page_hint` pointing to the table row's page (e.g. page 171), whereas the directive explanation text is located on page 1. When `_find_candidate_pages` runs in `candidate_recovery.py`, generic stopwords filter it out, yielding 0 candidate pages.
* **Impact**: ~2,193 fields.

### Stage 2: Normalization & Query Typing
* **Mechanism**: In `resolver.py` lines 98–103, queries are classified into numeric, boolean, date, or exact text strings.
* **Dropout Mechanism A: Boolean Keyword Filtering**:
  ```python
  is_bool = isinstance(value, bool) or (
      isinstance(value, str)
      and value.strip().lower() in ("true", "false", "yes", "no")
      and any(k in field.lower() for k in ("_box", "checkbox", "is_", "has_", "flag", "_yes", "_no", "final", "amended", "general", "domestic", "contributed"))
  )
  ```
  - Fields such as `reason_pressure`, `productive_zone_no`, `reason_interval`, `workover_100ppm_3000ft_yes` have values `"False"` or `"True"`.
  - Because `field.lower()` does not contain any of the rigid keywords (e.g. `reason_pressure`), `is_bool` evaluates to `False`.
  - The query is treated as an exact string `"False"`, which does not exist as text on the tax/well form (only physical checkboxes exist). Result: 0 candidates.
* **Dropout Mechanism B: Numeric Parsing of Punctuation-Adjacent Numbers**:
  - In `parse_numeric_value(val)`: Strings like `"5482'"` (depth in feet with apostrophe) return `None`.
  - The value is therefore not routed to numeric inverted indexing.
* **Impact**: ~974 checkbox fields + ~1,200 numeric/symbol fields.

### Stage 3: Tokenization & Boundary Slicing
* **Mechanism**: `search_exact` in `src/tonerhound/document/index.py` and `_find_token_subsequence` in `matcher.py`.
* **Dropout Mechanism**:
  - In `index.py` lines 290–294:
    ```python
    is_alnum_before = idx > 0 and text[idx - 1].isalnum()
    is_alnum_after = (idx + len(norm_query) < len(text)) and text[idx + len(norm_query)].isalnum()
    if (norm_query[0].isalnum() and is_alnum_before) or (norm_query[-1].isalnum() and is_alnum_after):
        start = idx + 1
        continue
    ```
    When `norm_query` contains trailing non-alphanumeric punctuation (e.g. `"5482'"` or `"Post Holdings, Inc.,"`), token-level slicing fails if the document token stream separated the punctuation or if the query punctuation does not match the token's normalized form.
  - In `_find_token_subsequence`:
    If clean query target has trailing punctuation stripped (`clean_query = norm_query.strip(" -.,;:_()[]{}/'\"")`), but `norm_query in norm_line` check was required first, the line check fails when newlines or formatting diverge.
* **Impact**: ~5,361 fields.

### Stage 4: Multi-Line Token Spans
* **Mechanism**: Spans that break across visual lines (e.g. multiline emails in `real_bbb_service_list`, long entity names in `real_ofac_ssi_full`).
* **Dropout Mechanism**:
  - `find_exact_candidates` iterates line-by-line (`for p_num, l_idx in lines_to_check:`).
  - Lines containing partial strings fail the exact line match.
  - While `_find_multiline_candidates` was added, it was only invoked if `len(norm_query.split()) >= 2`. If the query contains newline-separated items (`"MSIROTA@COLESCHOTZ.COM\nWUSATINE@COLESCHOTZ.COM\n..."`), standard line token matching fails because tokens on line $i$ do not match the multiline string.
* **Impact**: ~487 fields.

---

## 4. Prioritization of Fixes for Productionization

Based on empirical frequency in the 200-sample audit and document-level macro impact:

| Priority | Fix Description | Targeted Category | Estimated Field Recovery | Macro F1 Potential |
| :--- | :--- | :--- | :--- | :--- |
| **Fix 1** | **Boolean Schema Generalization**: If `value` is boolean or `"True"`/`"False"`, trigger `find_boolean_candidates` regardless of field name keywords. | `NON_TEXT_CHECKBOX` | ~974 fields | **High** (affects 15+ entire short/form tax documents) |
| **Fix 2** | **Punctuation & Token Strip Fallback**: In `collect_candidates`, if query fails with punctuation/units (quotes, currency, `%`), retry candidate lookup with stripped base token. | `TOKEN_MATCH_FAILURE` | ~5,361 fields | **High** (restores oil/gas depths, percentages, share counts) |
| **Fix 3** | **Global Search Fallback Relaxation**: In `find_exact_candidates`, if `lines_to_check` is empty or search returns 0 on long documents, check posting lists for rarest word or document-wide token index without the rigid `<= 10` page limit. | `PAGE_ROUTING_FAILURE` | ~2,193 fields | **Medium-High** (recovers OFAC and long SEC boilerplate fields) |
| **Fix 4** | **Multi-Line & Newline Span Normalization**: In `find_exact_candidates`, normalize newlines to single spaces and allow contiguous visual line spans across line breaks. | `DIRECT_TEXT_TRUNCATED` | ~487 fields | **Medium** (recovers service lists and legal notices) |
| **Fix 5** | **Candidate Limit Uncapping**: In `collect_candidates` and `resolver.resolve`, ensure candidates are not artificially truncated at top-25 or top-200 before scoring. | Candidate Truncation | ~350 fields | **Safe** |
