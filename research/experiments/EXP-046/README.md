# TONERHOUND EXP-046: MEGA PARALLEL ATTACK (100+ TECHNIQUES AT ONCE)

## Experiment Summary

EXP-046 is the largest coordinated algorithmic research cycle in TonerHound's history. It executes a comprehensive, multi-technique assault across all 8 remaining extraction failure domains simultaneously.

### Hard Constraints Strictly Enforced:
- NO Large Language Models (LLMs)
- NO Vision-Language Models (VLMs)
- NO Neural Networks
- NO Cloud AI APIs
- NO Embeddings
- NO Generative AI
- Production code (`src/tonerhound/`) UNTOUCHED until integration gating.
- Unconditional baseline preservation: strict zero regressions.

---

## The 34 Algorithmic Stages

EXP-046 orchestrates 34 specialized stages:

| Stage # | Stage Name | Target Failure Class | Core Mechanism |
|---|---|---|---|
| 1 | `canonicalize_v2` | Normalization Mismatch | Rule-based numeric, currency, and percent canonical representations |
| 2 | `fuzzy_matcher` | OCR Noise / Normalization | Bounded Levenshtein ($\le 2$) with strict typography confusion sets |
| 3 | `token_aligner` | Token Segmentation | Needleman-Wunsch sequence alignment across visual line boundaries |
| 4 | `token_set_matcher` | Permuted / Partial Sets | Hungarian matching on token distance matrices |
| 5 | `multi_region_v3` | Disjoint Evidence | Multi-region candidate clustering and bounding box unions |
| 6 | `ambiguity_resolver` | Ambiguous Occurrence | Density and contextual proximity disambiguation |
| 7 | `row_index_mapper` | Table Row Alignment | Global multi-page schema array index $\to$ table row mapper |
| 8 | `borderless_table_detector` | Borderless Tables | Horizontal & vertical whitespace gap projection detection |
| 9 | `wrapped_cell_merger` | Over-segmentation | Multi-line wrapped cell merging into unified logical rows |
| 10 | `subheader_filter` | Intermediate Subheaders | Section sub-header and subtotals classification and filtering |
| 11 | `xycut_reader` | Complex Multi-column | Recursive XY-Cut reading order layout analysis |
| 12 | `pagebreak_row_joiner` | Pagebreak Splits | Continuation marker and column alignment row bridging |
| 13 | `multitoken_v2` | Real Indexing Miss | Arbitrary-length token sequence alignment |
| 14 | `ngram_index_v2` | OCR Degradation | Multi-scale 2-gram, 3-gram, 4-gram composite Jaccard indexing |
| 15 | `consonant_skeleton` | Severely Corrupted OCR | Consonant skeleton extraction and fuzzy ratio matching |
| 16 | `numeric_canonical_v2` | Number Format Variation | Float-canonical numeric matching within 0.01 tolerance |
| 17 | `proximity_reranker` | Candidate Disambiguation | 5D Gaussian spatial reranking with row context |
| 18 | `cell_clamp_v2` | Token Slicing | Bounding box clamping to parent cell boundaries |
| 19 | `column_rail_detector` | Horizontal Drift | Column boundary constraints via header rails |
| 20 | `char_span_slicer` | Substring Extraction | Proportional character-span linear interpolation |
| 21 | `merged_cell_splitter` | Merged Cells | Header-proportional column splitting of merged table cells |
| 22 | `high_dpi_ocr_retry` | Missing OCR Text | 300 DPI high-resolution rendering with CLAHE and Tesseract PSM 11/6 |
| 23 | `binarization_cascade` | Degraded Region OCR | 4-way voting across Sauvola, Niblack, Otsu, Gaussian thresholds |
| 24 | `region_classifier` | Visual Non-Text | Edge density, ink coverage, and symmetry feature classification |
| 25 | `multiline_union_v2` | Multi-line Split | Multi-line bounding box union up to 6 lines |
| 26 | `linewrap_reconstructor` | Line Breaks | Hyphen, comma, and parenthesis continuation wrap recovery |
| 27 | `sequential_token_assembly` | Dispersed Tokens | Visual reading order concatenation and substring span mapping |
| 28 | `checkbox_v3` | Non-Text Boolean | Low-density (0.04) threshold, stroke variance, corner detection |
| 29 | `signature_detector` | Non-Text Signatures | Connected component, stroke variability, cursive contour analysis |
| 30 | `stamp_detector` | Official Stamps / Seals | Hough circle detection and HSV color saturation verification |
| 31 | `date_variants_v3` | Date Indexing | 50 international and textual date format variations |
| 32 | `relative_date_parser` | Relative Terms | Term parsing and anchor date offset computation |
| 33 | `hyphen_joiner_v3` | Line-end Hyphenation | Cross-line hyphen joining with lowercase verification |
| 34 | `word_split_recovery` | Mid-word Splitting | Document internal vocabulary verification for line break joins |

---

## Verification & Gating

- **Held-Out Cohort Gate**: Evaluated on 32 frozen held-out benchmark documents.
- **Success Criteria**: Net rescues $\ge 3,000$, 0 regressions, latency $\le 5\times$.
- **Full Benchmark**: Evaluated across all 370 benchmark documents.
