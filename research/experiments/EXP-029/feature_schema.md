# EXP-029 Phase 2 & Phase 3: Feature Schema & Leakage Audit

**Date**: October 1, 2026  
**Status**: Verified & Audited  
**Feature Dimension**: 57 features across 6 functional families  

---

## 1. Feature Catalog

All features are extracted strictly from information available during production inference prior to observing ground-truth citations.

### Family 1: Text Match & Alignment Features (7 features)
Features measuring syntactic, lexical, phonetic, and semantic concordance between candidate text and target field query value:
- `text_exact_match`: Binary indicator $\mathbb{I}(\text{cand\_text} == \text{query\_val})$.
- `text_norm_match`: Binary indicator $\mathbb{I}(\text{cand\_norm} == \text{query\_norm})$.
- `text_fuzz_ratio`: Normalized Levenshtein ratio $\in [0, 1]$ computed via RapidFuzz.
- `text_token_overlap`: Token Jaccard index $|T_c \cap T_q| / \max(1, |T_c \cup T_q|)$.
- `text_numeric_similarity`: Inverse relative numeric error $1.0 / (1.0 + |v_c - v_q|)$ for numerical values; fallback to exact match.
- `text_char_len_ratio`: $\min(|c|, |q|) / \max(1, |c|, |q|)$.
- `text_len_diff`: Absolute character length discrepancy $||c| - |q||$.

### Family 2: Structural & Contextual Features (8 features)
Features capturing layout hierarchies and spatial consensus within candidate clusters:
- `struct_is_table`: Binary indicator $\mathbb{I}(\text{field\_path contains } [ \dots ])$.
- `struct_field_depth`: Count of hierarchy levels (`.` and `[` delimiters).
- `struct_same_page_peers`: Ratio of pool candidates co-occurring on candidate's page.
- `struct_same_line_peers`: Fraction of pool peers co-linear within $|\Delta y| < 0.012$.
- `struct_same_col_peers`: Fraction of pool peers co-column within $|\Delta x| < 0.025$.
- `struct_same_page_as_r1`: Binary flag whether candidate shares the same page as baseline rank-1.
- `struct_same_line_as_r1`: Binary flag whether candidate shares the same horizontal line band as baseline rank-1 ($|\Delta y| < 0.012$).
- `struct_same_col_as_r1`: Binary flag whether candidate shares the same vertical column band as baseline rank-1 ($|\Delta x| < 0.025$).

### Family 3: Geometric & Layout Features (15 features)
Features capturing 2D coordinates, bounding box envelope, and relative spatial proximity:
- `geom_bbox_x`: Left coordinate $x \in [0, 1]$.
- `geom_bbox_y`: Top coordinate $y \in [0, 1]$.
- `geom_bbox_w`: Bounding box width $w \in [0, 1]$.
- `geom_bbox_h`: Bounding box height $h \in [0, 1]$.
- `geom_bbox_area`: Bounding box surface area $w \times h$.
- `geom_aspect_ratio`: Aspect ratio $w / \max(10^{-4}, h)$.
- `geom_text_density`: Character count per unit area $|c| / \max(10^{-4}, \text{area})$.
- `geom_dist_left`: Distance from left page margin ($x$).
- `geom_dist_right`: Distance from right page margin ($1 - (x + w)$).
- `geom_dist_top`: Distance from top page margin ($y$).
- `geom_dist_bottom`: Distance from bottom page margin ($1 - (y + h)$).
- `geom_dx_from_r1`: Horizontal displacement from baseline rank-1 candidate $|x - x_{r1}|$.
- `geom_dy_from_r1`: Vertical displacement from baseline rank-1 candidate $|y - y_{r1}|$.
- `geom_dist_from_r1`: Euclidean distance from baseline rank-1 candidate.
- `geom_area_ratio_r1`: Surface area ratio relative to baseline rank-1 candidate.

### Family 4: Document & Macro Context Features (12 features)
Features capturing document length, page priors, and document taxonomy:
- `doc_page`: 1-indexed document page number.
- `doc_reciprocal_page`: Inverse page index $1.0 / \text{page}$.
- `doc_is_page_1`, `doc_is_page_2`, `doc_is_page_3_plus`: Page positional priors.
- `doc_split_short`, `doc_split_medium`, `doc_split_long`: Document length indicators.
- `doc_is_sec`, `doc_is_tax`, `doc_is_regulatory`, `doc_is_schedule_table`: Document category indicators.

### Family 5: Source Match Tier Features (7 features)
Binary flags indicating the retrieval channel that produced the candidate:
- `source_exact`: Native exact token match.
- `source_numeric`: Normalized numeric match.
- `source_date`: Normalized date match.
- `source_boolean`: Checkbox / boolean match.
- `source_fuzzy`: Fuzzy alignment fallback.
- `source_recovered`: Modular recovery engine (fragmented/interleaved/spaced/symbol).
- `source_is_ocr`: OCR text-layer provenance.

### Family 6: Baseline Ranking Prior Features (8 features)
Features capturing production baseline resolver priority and pool competition:
- `rank_orig`: Baseline rank integer (1, 2, 3...).
- `rank_reciprocal`: Reciprocal rank $1.0 / \text{rank}$.
- `rank_is_1`, `rank_is_2`, `rank_is_3`: One-hot indicators for top baseline ranks.
- `rank_diff_from_r1`: $\text{rank} - 1$.
- `pool_candidate_count`: Total cardinality of candidates in pool.
- `pool_inv_candidate_count`: Inverse candidate count $1.0 / |\text{pool}|$.

---

## 2. Mandatory Leakage Audit (Phase 3)

A systematic verification was conducted across all 57 features against potential information leakage vectors:

| Leakage Vector | Checked | Result | Verification Rationale |
| :--- | :---: | :---: | :--- |
| **Gold Bounding Box** | Yes | **PASS** | No feature references `gold_bboxes_count` or `gold_evidence_entries`. Coordinates are derived strictly from candidate's own bounding box $c.bbox$. |
| **Gold Ground-Truth Page** | Yes | **PASS** | `doc_page` is the candidate's own page $c.page$; no reference to `gold_pages`. |
| **Gold Ground-Truth IoU** | Yes | **PASS** | `best_iou` and `iou_by_gold` are excluded from all input vectors; only used to define the supervised training label $y \in \{0, 1\}$. |
| **Ground-Truth Row / Occurrence** | Yes | **PASS** | No ground truth row or occurrence index is accessed. Peer features use only candidate pool coordinates. |
| **Post-Evaluation Outcomes** | Yes | **PASS** | Columns `value_correct`, `page_correct`, `grounded_correct`, `failure_class`, and `selected_candidate_iou` are strictly prohibited and absent from the feature pipeline. |
| **Inference Time Validity** | Yes | **PASS** | Every feature can be computed in $<0.1\text{ms}$ at runtime using only the extraction request and candidate objects generated by `EvidenceResolver.collect_candidates()`. |
