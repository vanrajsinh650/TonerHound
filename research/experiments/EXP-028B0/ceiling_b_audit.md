# EXP-028B0: Audit of Old Ceiling B Information-Loss Points

## 1. Executive Summary

In EXP-027, Ceiling B was intended to establish the "Exhaustive Text/OCR Geometry Ceiling" and resulted in **64.58% Word Grounding F1**.
However, deep structural auditing reveals that Ceiling B was **not exhaustive**. It inherited multiple layers of candidate truncation, pruning, and flawed span search from the production matcher and early observer scripts.

When all artificial ceilings (top-K limits, page pruning, line ordering assumptions) are removed and the complete text token universe is made available to the oracle, the true ceiling achievable from pure document text and OCR geometry is unlocked.

---

## 2. The 10 Systemic Information-Loss Points in Old Ceiling B

### 1. Top-K Candidate Pool Truncation
- **Location:** `research/observer/run_microscope_v2.py:303` (`for rank_idx, cand in enumerate(ranked_cands[:25])`)
- **Mechanism:** Old Ceiling A and the input to Ceiling B relied on `cache_v2`, which only stored the top-25 retrieved candidates.
- **Impact:** In tabular filings (e.g. SEC Form 13F with 50 pages of holdings), repeated strings like `"COM"`, `"SH"`, `"SOLE"`, and `"0"` appear hundreds of times. Candidates for rows on pages 5 through 50 were completely dropped at rank 26, causing 100% false zero-candidate counts for lower rows.

### 2. Per-Page Cutoff Limits
- **Location:** `src/tonerhound/matching/matcher.py:48-61`
- **Mechanism:** Early page-level candidate caches restricted matches to a fixed number of hits per page.
- **Impact:** Densely populated table columns (e.g. 50 rows per page) lost candidates beyond the per-page threshold.

### 3. Deduplication Across Rows
- **Location:** Candidate generation deduplication layers
- **Mechanism:** Repeated identical values on the same page were deduplicated to a single bounding box.
- **Impact:** Multiple distinct fields sharing the same string (e.g., voting authority `"0"` across 40 rows) were collapsed into one candidate, leaving the remaining 39 rows without valid candidate geometry.

### 4. Identical-String Collapse
- **Location:** String matching inverted index posting limits
- **Mechanism:** Documents with thousands of identical tokens (e.g. `"ADDRESS ON FILE"` repeated 5,000 times in FTX) suffered posting list truncation.
- **Impact:** Only the first few occurrences were indexed; all subsequent occurrences across 114 pages were discarded.

### 5. Candidate Pruning via Scoring Thresholds
- **Location:** `src/tonerhound/resolution/resolver.py:192-200`
- **Mechanism:** Verification score thresholds and margin gating discarded candidates prior to pool persistence.
- **Impact:** Valid text spans that received lower initial spatial scores were pruned before the oracle could evaluate their IoU.

### 6. Page Pruning & Rigid Page Routing
- **Location:** `src/tonerhound/matching/matcher.py:86-90`
- **Mechanism:** If `page_hint` was provided by early heuristics, the matcher restricted line inspection strictly to that single page (`lines_to_check = [(page_hint, l.line_index) for l in page.lines]`).
- **Impact:** If the page hint was off by 1 page (e.g. page break drift in SEC filings), the true text on the correct page was completely unsearched.

### 7. Global Search Fallback Omission
- **Location:** `src/tonerhound/matching/matcher.py:173` (`if not candidates and (page_hint is not None or len(self.index.pages) <= 10)`)
- **Mechanism:** Fallback page-wide search was completely disabled for all documents with $> 10$ pages unless a page hint existed.
- **Impact:** For medium and long documents (which represent $> 92\%$ of all fields), un-hinted queries never triggered document-wide text search.

### 8. Whole-Token Bbox Slicing Misses
- **Location:** `run_exp027_reachability.py:107-109` (`span_box = union_bbox_list([t.bbox for t in toks[i:j+1]])`)
- **Mechanism:** Old Ceiling B evaluated only entire token bounding boxes.
- **Impact:** Whenever OCR or PDF extraction concatenated punctuation or prefixes (e.g. `'$100'`, `'Case.23-11132'`), taking the entire token yielded IoU between $0.25$ and $0.45$, failing the $	ext{IoU} \ge 0.50$ threshold despite the text literally being present.

### 9. Multi-Line Contiguous Token Search Failure
- **Location:** `run_exp027_reachability.py:101-105` (`for j in range(i, min(i + 12, len(toks))): accum_clean += t_clean`)
- **Mechanism:** Old Ceiling B iterated horizontally through tokens in reading order, checking contiguous slices.
- **Impact:** In table columns where a cell's text wrapped across lines (e.g. `'17 ED & TECHNOLOGY'` on line 1, `'GROUP INC'` on line 2), all tokens from adjacent table columns on line 1 intervened between the words. The contiguous slice broke, causing 100% failure on wrapped table cells.

### 10. Cache Filtering Misses
- **Location:** `run_exp027_reachability.py:469` (`misses = df_v3[~df_v3.apply(lambda r: (r['document_id'], r['field_path']) in oracle_map, axis=1)]`)
- **Mechanism:** Old Ceiling B only ran on fields completely absent from `oracle_map`.
- **Impact:** If `cache_v2` contained candidates that were all incorrect (IoU $< 0.50$, e.g. from the wrong page or wrong row), Ceiling B skipped the field entirely and did not attempt exhaustive text recovery.
