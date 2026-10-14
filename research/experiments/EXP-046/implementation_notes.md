# EXP-046: Implementation Notes — Mega Parallel Attack (34 Specialized Stages)

## 1. Architecture Overview

EXP-046 represents the largest simultaneous multi-technique attack on extraction failure modes in TonerHound's history. Rather than testing isolated single-technique hypotheses sequentially, EXP-046 implements and evaluates 34 distinct recovery stages in parallel across 8 core failure domains.

### Structural Failure Domains & 34 Specialized Stages:

1. **Table Structure & Disambiguation (Phase A / EXP-045 Continuation)**:
   - `Stage 1`: `canonicalize_v2` — Rule-based canonicalization across percent, currency, accounting negative, grouping.
   - `Stage 2`: `fuzzy_matcher` — Bounded Levenshtein distance ($\le 2$) with strict OCR confusion sets.
   - `Stage 3`: `token_aligner` — Needleman-Wunsch sequence alignment on visual lines.
   - `Stage 4`: `token_set_matcher` — Hungarian matching between expected token sets and candidate lines.
   - `Stage 5`: `multi_region_v3` — Multi-region candidate assembly for split text.
   - `Stage 6`: `ambiguity_resolver` — Density- and position-based disambiguation among identical occurrences.
   - `Stage 7`: `row_index_mapper` — Global multi-page schema array index $\to$ physical table row mapper.
   - `Stage 8`: `borderless_table_detector` — Horizontal/vertical whitespace gap detection for borderless tables.
   - `Stage 9`: `wrapped_cell_merger` — Merging multiline wrapped cell descriptions to eliminate over-segmentation.
   - `Stage 10`: `subheader_filter` — Filtering non-data section headers and sector subtotals.
   - `Stage 11`: `xycut_reader` — Recursive XY-Cut layout analysis for complex multi-column documents.
   - `Stage 12`: `pagebreak_row_joiner` — Reconnecting split rows straddling physical page boundaries.

2. **Real Indexing Misses (Phase B)**:
   - `Stage 13`: `multitoken_v2` — Arbitrary length token sequence alignment with Needleman-Wunsch.
   - `Stage 14`: `ngram_index_v2` — Multi-scale 2-gram, 3-gram, and 4-gram candidate indexing with composite Jaccard ranking.
   - `Stage 15`: `consonant_skeleton` — Consonant skeleton extraction for severely degraded OCR tokens.
   - `Stage 16`: `numeric_canonical_v2` — Float-canonical numeric matching within 0.01 tolerance.
   - `Stage 17`: `proximity_reranker` — 5D Gaussian spatial reranker with row-band confinement and sibling alignment.

3. **Token Slicing & Boundary Rectification (Phase C)**:
   - `Stage 18`: `cell_clamp_v2` — Clamping token bounding boxes to parent cell boundaries.
   - `Stage 19`: `column_rail_detector` — Constraining horizontal candidate ranges via column rail headers.
   - `Stage 20`: `char_span_slicer` — Character-span linear interpolation for sub-token extractions.
   - `Stage 21`: `merged_cell_splitter` — Proportional column-width splitting for merged table cells.

4. **No-Text OCR & Degraded Regions (Phase D)**:
   - `Stage 22`: `high_dpi_ocr_retry` — 300 DPI high-resolution rendering with CLAHE and Tesseract PSM 11/6.
   - `Stage 23`: `binarization_cascade` — 4-way voting across Sauvola, Niblack, Otsu, and Adaptive Gaussian binarization.
   - `Stage 24`: `region_classifier` — Edge density, ink coverage, and symmetry analysis for visual non-text.

5. **Multi-Line Split Assembly (Phase E)**:
   - `Stage 25`: `multiline_union_v2` — Vertical gap and horizontal corridor multi-line bounding box union up to 6 lines.
   - `Stage 26`: `linewrap_reconstructor` — Hyphen, open-parenthesis, and continuation comma/period line wrapping repair.
   - `Stage 27`: `sequential_token_assembly` — Reading-order token concatenation and substring offset mapping.

6. **Non-Text Boolean & Visual Signatures (Phase F)**:
   - `Stage 28`: `checkbox_v3` — Low-density (0.04) threshold, stroke variance, and corner detection.
   - `Stage 29`: `signature_detector` — Connected-component analysis, stroke variability, and non-rectangularity.
   - `Stage 30`: `stamp_detector` — Circular/oval Hough transform and HSV color saturation detection.

7. **Date Indexing (Phase G)**:
   - `Stage 31`: `date_variants_v3` — 50 international and textual date format variations.
   - `Stage 32`: `relative_date_parser` — Relative term resolution against document anchor dates.

8. **Hyphenation & Split Recovery (Phase H)**:
   - `Stage 33`: `hyphen_joiner_v3` — Line-end cross-line hyphen joining with lowercase verification.
   - `Stage 34`: `word_split_recovery` — Document vocabulary verification for non-hyphenated mid-word line splits.

## 2. Zero-Regression Baseline Preservation Protocol

A central principle of EXP-046 is strict baseline preservation:
- All fields that passed in the EXP-045 canonical baseline (IoU $\ge 0.50$ on correct page) are tagged and unconditionally locked.
- No newly proposed citation from any of the 34 stages is permitted to replace a passing baseline citation.
- If multiple stages propose candidates for a previously failing field, proposals are ranked strictly by confidence score, with deterministic lexicographic tie-breaking on stage name.
- Only proposals satisfying ExtractBench criteria (IoU $\ge 0.50$, exact page match) are admitted as rescued fields.
