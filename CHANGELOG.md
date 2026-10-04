# Changelog

All notable changes to TonerHound will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-04

### Added
- **Hungarian Table Assigner** (`src/tonerhound/resolution/table_assigner.py`): Global bipartite matching via `scipy.optimize.linear_sum_assignment` resolving repeated values in dense tabular arrays (EXP-039/040: +5,845 fields).
- **Visual Checkbox & Signature Provider** (`src/tonerhound/vision/checkbox.py`): Computer vision detection of checkboxes embedded in grid lines using morphological line removal, plus stroke-variance signature detection (EXP-039/042: +1,882 fields).
- **Table Cell Extractor** (`src/tonerhound/document/table_cells.py`): Table cell bounding and column rail clamping via PyMuPDF native table layout (EXP-039: +112 fields).
- **OCR Noise-Tolerant Index** (`src/tonerhound/document/ocr_noise_index.py`): Character 3-gram inverted index with bounded Levenshtein distance for degraded/scanned documents (EXP-039: +16 fields).
- **Multi-Token Sequence Matcher** (`src/tonerhound/matching/multi_token.py`): Contiguous and bounded-gap multi-word token matching (EXP-040: +1,196 fields).
- **Multi-Region Assembler** (`src/tonerhound/geometry/multi_region.py`): Spatial proximity clustering for cross-column and disconnected tokens (EXP-040: +689 fields).
- **Hyphen Joiner** (`src/tonerhound/geometry/hyphen_joiner.py`): Line-wrap hyphenation pair identification and bounding box unioning (EXP-041).
- **Date Literal Variants** (`src/tonerhound/matching/date_variants.py`): Exact literal string variant matching across 18 canonical date formats without lossy semantic drift (EXP-042: +3,690 fields).
- **Multiline Assembler** (`src/tonerhound/geometry/multiline.py`): Sequential reading-order token assembly across consecutive lines (EXP-042: +5 fields).
- **Deterministic Resolution Pipeline** (`src/tonerhound/resolution/pipeline.py`): Strict priority-order orchestration with unconditional baseline preservation contract (zero regressions).

### Benchmark Results
- **Word Grounding F1:** **72.6179%** (+27.31 pp over initial baseline)
- **Page Grounding F1:** **83.8490%**
- **Word Precision:** **77.7935%**
- **Word Recall:** **68.9729%**
- **Passing Fields:** 327,671 / 498,140
- **Regressions:** 0 across all 498,140 fields

## [0.2.0] - 2026-09-30
- Initial experimental candidates and structural reranking.
