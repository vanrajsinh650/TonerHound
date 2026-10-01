# EXP-031A: Simulated Native Integration of Global Structured Assignment

**Date**: October 1, 2026  
**Status**: In Progress  
**Objective**: Determine whether the +7.73 pp selection accuracy gain from EXP-030 translates into official ExtractBench Word Grounding F1 when selected candidates pass through TonerHound's complete native production geometry pipeline.  
**Production Code Impact**: **ZERO (Production code untouched, simulation only)**  

---

## 1. Context & Starting State

- **Target Benchmark**: Official ExtractBench leaderboard target: **58.11% Overall Word Grounding F1** (LlamaExtract Agentic Plus). Minimum passing target: $\ge 58.12\%$.
- **Starting Baseline (EXP-028E full 370-document benchmark)**: **56.05%** Word Grounding F1.
- **Held-Out Cohort B Baseline (32 documents)**:
  - Multi-Candidate Hit@1: **25.00%** (17,861 / 71,439 fields).
  - Candidate Pool Recall: **75.83%** (54,171 / 71,439 fields).
  - Selection Gap: **50.82 pp** (36,310 fields).
  - Word Grounding F1: **59.36%**.
  - Word Precision: **63.78%**.
  - Word Recall: **56.52%**.
  - Page Grounding F1: **85.44%**.
- **EXP-030 Findings**:
  - Global Structured Assignment achieved **32.73% Hit@1** (+7.73 pp gain, 6,203 beneficial flips vs 684 harmful flips, +5,519 net flips).
  - However, post-hoc substitution of raw candidate bounding boxes from `field_records.parquet` yielded **58.89% Word F1** (-0.47 pp delta) due to geometry pipeline decoupling.

---

## 2. Production Geometry Pipeline Audit (Phase 0)

In TonerHound's production execution path:
```text
Candidate Selection (EvidenceResolver / Structured Assigner)
        ↓
Candidate Object (MatchCandidate with raw token BBox & matched_text)
        ↓
Line-Height Snapping (box.align_to_line_height(target_height))
        ↓
Sibling Boundary Clamping (max_right_boundary calculation)
        ↓
_apply_geometry_enhancements:
  ├─ 1. reconstruct_safe_character_span (EXP-015: sub-token character-span slicing)
  ├─ 2. extend_same_line_tokens (EXP-017 / EXP-017R: multi-word same-line expansion)
  └─ 3. trim_dot_leaders (EXP-018: tabular leader and dot padding removal)
        ↓
Final FieldCitation (page, bbox.to_coco(), reference_text, confidence)
        ↓
ExtractBench Official ExtractEvaluator (Hungarian bipartite matching, IoU >= 0.50)
```

### Exact Production Functions & Locations
1. **`ExtractBenchAdapter`**: Located in `src/tonerhound/benchmark/adapter.py`.
2. **`_apply_geometry_enhancements`**: Defined at `src/tonerhound/benchmark/adapter.py:129-198`.
3. **`reconstruct_safe_character_span`**: Defined at `src/tonerhound/geometry/character_span.py:14-73`.
   - Computes proportional character offsets within the raw token box based on substring offsets of the extracted target value.
   - Prevents IoU threshold failures ($\text{IoU} < 0.50$) caused by extraneous currency symbols, commas, or parentheses in raw token bboxes.
4. **`extend_same_line_tokens`**: Defined at `src/tonerhound/geometry/same_line_recovery.py:46-170`.
   - Extends bounding boxes across contiguous tokens on the same visual line, bounded by `max_right_boundary`.
5. **`trim_dot_leaders`**: Defined at `src/tonerhound/geometry/dot_leader_trimming.py:60-170`.
   - Crops trailing dot leaders and whitespace padding commonly found in tabular line items.
6. **`align_to_line_height`**: Defined at `src/tonerhound/geometry/coordinates.py:171-179`.
   - Centers and expands tight font glyph boxes to standard text-line height ($0.016-0.018$).

---

## 3. The Core Experimental Question

When EXP-030's 6,203 beneficial flips are routed through this complete geometry pipeline, does the raw token bbox IoU increase from sub-threshold ($0.44 - 0.48$) to passing ($\ge 0.50$), and does Cohort B Word Grounding F1 exceed the baseline of 59.36%?
