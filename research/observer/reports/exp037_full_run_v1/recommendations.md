# Failure Microscope Recommendations: exp037_full_run_v1

## Executive Synthesis

The EXP-037 Deterministic Page-Level OCR Routing run established a measured, empirical reduction in text-layer failure modes across the official 370-document benchmark.

### 1. Macro-Weighted Failure Rankings (Post-OCR)

1. **NON_TEXT_BOOLEAN_GROUNDING**: +16.5056 pp macro-weighted opportunity (2,845 fields across 160 documents, 67.80% breadth).
   - *Observation*: Unresolved visual checkbox targets remain the single highest-leverage macro bottleneck across the document corpus. EXP-036D demonstrated viability on a 71-document subset; complete coverage across all 160 documents is required.
2. **REAL_INDEXING_MISS**: +11.5335 pp macro-weighted opportunity (90,255 fields across 229 documents, 97.03% breadth).
   - *Observation*: After introducing OCR tokens into the index, 39,037 fields migrated from `NO_TEXT_AT_GOLD_REGION` to `REAL_INDEXING_MISS`. The text was successfully extracted by OCR, but exact string queries fail due to OCR character noise (e.g. "Comsolidated" vs "Consolidated", "Dog 574" vs "Doc 574").
3. **NO_TEXT_AT_GOLD_REGION**: +7.8108 pp macro-weighted opportunity (16,886 fields across 122 documents, 51.69% breadth).
   - *Observation*: Reduced from 58,778 fields down to 16,886 fields (-71.27%). Remaining cases correspond to degraded stamp regions, un-OCR'd graphical headers, or low-resolution image borders.
4. **NORMALIZATION_MISMATCH**: +5.4780 pp macro-weighted opportunity (46,158 fields across 204 documents, 86.44% breadth).
   - *Observation*: Numeric formatting, currency symbol stripping, decimal precision, and date format variations.
5. **TOKEN_SLICING**: +4.5411 pp macro-weighted opportunity (23,646 fields across 156 documents, 66.10% breadth).
   - *Observation*: Multi-word table cell tokens where the gold bounding box spans only a sub-slice of an OCR word cluster.

### 2. Next Research Priority Recommendations

- **Do NOT blindly reopen naive n-gram indexing**: The EXP-035 post-mortem demonstrated that unconstrained n-gram indexing causes false-positive candidate explosion.
- **Priority 1: Robust Fuzzy OCR Token Matching (OCR Noise Tolerance)**: Address the 90,255 `REAL_INDEXING_MISS` fields by leveraging Levenshtein/consonant-skeleton alignment (already present in `tonerhound.ocr.normalizer`) during candidate retrieval on OCR-tagged pages.
- **Priority 2: Unified EXP-036D Boolean Grounding Integration**: Integrate the deterministic visual provider for boolean/checkbox fields to unlock the +16.51 pp opportunity.
