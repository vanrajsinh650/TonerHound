# EXP-005: Performance & Algorithmic Profiling Report

**Project**: TonerHound  
**Document ID**: `EXP-005-profile`  
**Date**: 2026-09-14  
**Profiler**: Python `cProfile` + high-resolution `time.perf_counter` stage timers  
**Target Benchmark**: Stratified Local Benchmark (EXP-005 Representative Cross-Section)  
**Profiled Documents**:
- `short/07021-2016-p0014` (1 page, Form 1065 K-1, 52 fields)
- `short/real_clinton_property_25_11073_corrupted` (9 pages, Scanned Property Deed, 18 fields)
- `medium/cabrera-2023` (28 pages, Auto Insurance Valuation, 126 fields)
- `medium/13f__leonteq_securities_2025q4` (31 pages, Institutional 13F, 1,248 fields)
- `long/sec_13f_0010_renaissance_technologies` (81 pages, Renaissance 13F, 4,892 fields)

---

## 1. Executive Summary & Headline Profiling Findings

Across the 5-document profile run (150 total pages, 6,336 extraction queries):
- **Total Suite Execution Time**: **1,029.99 seconds (17.17 minutes)**
- **PDF Indexing & Text Extraction Time**: **28.63 seconds (2.78%)**
- **Official Evaluation Metric Time**: **3.55 seconds (0.34%)**
- **Evidence Grounding Time (`ground_extracted_data`)**: **997.81 seconds (96.88%)**

### The Smoking Gun: Redundant String Normalization Hotspot
A staggering **68.5% of total runtime (689.72 seconds)** was consumed by a single function: `normalize_unicode_and_case` called **23,927,861 times**!

```
Total function calls: 3,150,678,365 (3.15 BILLION calls)
normalize_unicode_and_case: 23,927,861 calls -> 386.05s self time -> 689.72s cumtime
```

Two algorithmic pathologies cause this explosion:
1. **Redundant Visual Line Re-Normalization in Context Scoring**: In `src/tonerhound/resolution/resolver.py:192` (`_score_candidates_with_context`), line text is re-normalized in a nested loop `for line in p.lines: line_norm = normalize_unicode_and_case(line.text).text` instead of reading the pre-cached `line.norm_text` property. This runs for every field query on every candidate page!
2. **Quadratic Window Normalization in Subsequence Search**: In `src/tonerhound/matching/matcher.py:332` (`_find_token_subsequence`), a sliding window $O(W^2)$ re-joins token strings and calls `normalize_unicode_and_case(" ".join(...))` **2,607,280 times (280.86s cumtime)**.

---

## 2. Stage Breakdown & Latency by Document

| Document Test ID | Pages | PDF Index Time | Grounding Time | Eval Time | Total Time | Word Grounding F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `short/07021-2016-p0014` | 1 | 0.237s | 0.410s | 0.007s | **0.654s** | 27.8% | Fast, but F1 degraded by checkbox/booleans |
| `short/real_clinton_property_25_11073_corrupted` | 9 | 0.011s | 0.181s | 0.015s | **0.207s** | 0.0% | Fast, but F1 0.0% due to OCR corruption drop |
| `medium/cabrera-2023` | 28 | 0.009s | 0.142s | 0.098s | **0.249s** | 0.0% | Fast because >10 pages truncated candidate search! |
| `medium/13f__leonteq_securities_2025q4` | 31 | 6.962s | 184.770s | 0.846s | **192.578s** | 66.7% | 3.2 minutes for 1,248 fields (dense tabular) |
| `long/sec_13f_0010_renaissance_technologies` | 81 | 21.407s | 812.308s | 2.584s | **836.299s** | 62.8% | 13.9 minutes for 4,892 fields (linear scan explosion) |
| **TOTALS (5 Documents)** | **150** | **28.63s (2.8%)** | **997.81s (96.9%)** | **3.55s (0.3%)** | **1029.99s (100%)** | — | — |

---

## 3. Top Call-Stack Bottlenecks (from `cProfile`)

### Top 10 by Self-Time (`tottime`)
| Rank | Function | Calls | Self Time (s) | Cumulative Time (s) | Call Site / Culprit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `normalize_unicode_and_case` | 23,927,861 | **386.05s** | 689.72s | Re-normalizing lines and windows in loops |
| **2** | `list.append` | 901,903,563 | **101.94s** | 101.94s | Character map list construction in normalizer |
| **3** | `_score_candidates_with_context` | 10,341 | **74.11s** | 585.39s | Nested line scanning across pages for context |
| **4** | `dict.get` | 454,495,120 | **68.97s** | 68.97s | Glyph replacement table lookups |
| **5** | `str.casefold` | 445,820,554 | **56.98s** | 56.98s | Repeated character folding |
| **6** | `unicodedata.normalize` | 446,105,816 | **54.97s** | 54.97s | Repeated NFKC normalization |
| **7** | `find_exact_candidates` | 55,885 | **39.49s** | 391.40s | Linear line scanning for exact queries |
| **8** | `str.join` | 49,559,893 | **30.31s** | 48.53s | Window string recreation |
| **9** | `_find_token_subsequence` | 2,607,280 | **24.89s** | 280.86s | Quadratic sliding window over visual lines |
| **10** | `min` (built-in) | 94,079,024 | **21.70s** | 30.13s | Coordinate bounding box math |

### Top 5 by Cumulative Time (`cumtime`)
1. `ground_extracted_data`: **997.80s** (Adapter orchestration)
2. `resolve`: **897.49s** (Field resolver)
3. `normalize_unicode_and_case`: **689.72s** (Unicode normalization workhorse)
4. `_score_candidates_with_context`: **585.39s** (Context proximity scoring)
5. `find_exact_candidates`: **391.40s** (Candidate matching)

---

## 4. Architectural Weaknesses & Optimization Targets

### Bottleneck A: The O(N_fields × N_lines) Redundant Context Normalization
- **Diagnosis**: Lines 191–193 in `src/tonerhound/resolution/resolver.py`:
  ```python
  for line in p.lines:
      line_norm = normalize_unicode_and_case(line.text).text  # BUG: recomputed 24 million times!
      if any(w in line_norm for w in context_words):
          boxes.append(line.bbox)
  ```
- **Remedy**: Replace with cached property `line.norm_text` which is computed once per line during indexing.
- **Expected Speedup**: Removes ~580 seconds of redundant CPU time; **70% immediate speedup** on long documents.

### Bottleneck B: Quadratic String Normalization in `_find_token_subsequence`
- **Diagnosis**: Lines 337–342 in `src/tonerhound/matching/matcher.py`:
  ```python
  for window_size in range(1, len(tokens) + 1):
      for start_i in range(len(tokens) - window_size + 1):
          window = tokens[start_i : start_i + window_size]
          text = normalize_unicode_and_case(" ".join(t.text for t in window)).text.strip()
  ```
- **Remedy**: Pre-tokenize or compare normalized token sequences directly using string/token matching without re-normalizing the joined string.
- **Expected Speedup**: Removes ~280 seconds of CPU time; **80% faster line candidate search**.

### Bottleneck C: Absence of Inverted Token Index for Exact & Numeric Lookups
- **Diagnosis**: For every single query, `find_exact_candidates` iterates over all visual lines across all pages (`for line in page.lines`). On an 81-page document with 4,892 fields, this performs $4,892 \times 81 \times \text{lines\_per\_page} \approx 20,000,000$ line string checks!
- **Remedy**: Query the inverted index `_token_index[norm_token]` in $O(1)$ amortized time.
- **Expected Speedup**: Linear scans replaced with instant posting list lookups; **10x to 50x faster candidate generation**.

---

## 5. Algorithmic Roadmap for EXP-005 Implementations

1. **Sprint 1 (Immediate Hotspot Elimination)**:
   - Fix line 192 in `resolver.py` to use `line.norm_text`.
   - Vectorize / simplify `_find_token_subsequence` to avoid re-normalizing strings in inner loops.
2. **Sprint 2 (Inverted Multi-Token & Value Index)**:
   - Build a document-level inverted index for single tokens, token bigrams, and canonical numbers.
   - Remove the artificial 10-page truncation in `find_normalized_numeric_candidates` and `find_fuzzy_candidates` safely.
3. **Sprint 3 (Structural & Table Grounding)**:
   - Implement row-level anchoring and monotonic sequence constraints to resolve dense tabular records.
4. **Sprint 4 (Verification & Calibrated Abstention)**:
   - Gate uncertain candidates with calibrated score margins to maintain zero silent false groundings.
