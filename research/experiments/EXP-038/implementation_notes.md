# EXP-038 Implementation Notes

## Architecture Overview

EXP-038 implemented a multi-problem deterministic resolution harness (`UnifiedEXP038Resolver`) orchestrating 8 targeted fixes designed to address the specific root causes mapped by Failure Microscope V4.

```
Incoming Ungrounded Field / Baseline Failure
                      │
                      ├──► [Fix 2: Visual Checkbox Provider] (Strict Boolean Gating: bool / checkbox in field)
                      │       └── Morphological wireframe + Hough diagonal line detection (Page-level cached)
                      │
                      ├──► [Fix 4: Date Normalizer] (Target is string)
                      │       └── Extended regex parser for YYYY-MMM-DD, DD-MMM-YYYY, DD-MM-YYYY, MM-DD-YY
                      │
                      ├──► [Fix 6: Normalization Expansion] (Non-boolean Target)
                      │       └── Accounting negative (xxx) -> -xxx & currency symbol stripping
                      │
                      ├──► [Fix 5: Multi-Line Assembly] (Target contains whitespace)
                      │       └── Bounded multi-line bounding box union assembler
                      │
                      ├──► [Fix 8: Hyphenation Joiner] (Target is string)
                      │       └── Terminal line hyphen detector and joiner
                      │
                      ├──► [Fix 1: OCR-Noise Index] (Target is string, on OCR pages)
                      │       └── Bounded character 3-gram index with edit distance <= 1
                      │
                      └──► [Standard Resolver Fallback]
                              └── TonerHound ExtractBenchAdapter with geometry enhancement & token slicing
```

---

## Component Details

### 1. Accounting Negative Normalizer (`normalization_v2.py`)
- **Problem**: Real-content financial reports (`real_vg_divappr_full`, `real_penn_hills_pa`, `bar-lev`, Schedule C/SE) format negative balances using parentheses: `(1,234.56)` or `($1,234.56)`. Gold labels frequently record these as `-1234.56`. Standard numeric indexing failed exact string and float match.
- **Solution**: `generate_numeric_query_variants()` detects leading/trailing parentheses, parses commas and currency symbols (`$`, `€`, `£`, `USD`), and generates both parenthesized and signed numeric string variants.
- **Performance**: Contributed **494 rescued fields** (+0.59 pp Word F1) across 41 documents with sub-millisecond overhead.

### 2. Standard Resolver Fallback
- **Problem**: Baseline predictions frequently had empty citations for valid targets due to conservative thresholding or early abortion on massive tabular schemas.
- **Solution**: For any ungrounded field that had no baseline citation (or failed baseline), the full `ExtractBenchAdapter` with geometry enhancements was invoked.
- **Performance**: Contributed **723 rescued fields** (58.2% of total rescues).

### 3. OCR Noise-Tolerant Inverted Index (`ocr_noise_index.py`)
- **Problem**: Tesseract OCR introduces single-character substitution errors on scanned pages (`0` vs `O`, `1` vs `l`, `S` vs `5`, `8` vs `B`), causing exact-token inverted index queries to fail.
- **Solution**: A secondary character 3-gram index with bounded edit distance ($\le 1$) was constructed exclusively on pages where `page_mode == 'ocr'`. Digital pages were completely untouched.
- **Performance**: Contributed **16 rescued fields** on degraded scans (`passcoag-2020-w2`, `07021-2016`) with zero false-positive contamination.

### 4. Visual Checkbox Provider V2 (`visual_provider_v2.py`)
- **Problem**: Checkbox detection was previously slow when called redundantly per field on the same page.
- **Solution**: Added page-level memoization (`_checkbox_cache[(pdf_path, page_num)]`) so that wireframe rendering, thresholding, and contour extraction execute at most once per page. Hough line detection was added to detect diagonal cross lines (`X`) alongside core occupancy (`✓`).
- **Performance**: Contributed **5 rescued fields** across complex forms while speeding up execution by $>10\times$.

### 5. Date Normalizer (`date_normalizer.py`)
- **Problem**: Date formats like `05-OCT-2021` or `2021/10/05` failed standard ISO 8601 parsers.
- **Solution**: Implemented regex patterns to capture 3-letter month abbreviations and slashes, normalizing into standard `YYYY-MM-DD` queries.
- **Performance**: Contributed **4 rescued fields**.

---

## Memory & Performance Engineering

1. **Explicit Garbage Collection**:
   - `HybridDocumentIndex.from_pdf()` and PyMuPDF document instances consume significant memory when rendering at 300 DPI.
   - Explicit `del doc_idx, resolver, tc, rules` followed by `gc.collect()` after each document kept process RSS memory strictly below 800 MB throughout the entire 370-document benchmark.
2. **Single-Thread Math Enforcement**:
   - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` ensured zero CPU contention or thermal throttling on laptop environments.
3. **Execution Time**:
   - 370-document prediction generation + official ExtractBench evaluation took **186.2 seconds** (3.10 minutes).
