# EXP-035: Implementation Notes & Gate Decision Record

## 1. Decision Gate Evaluation

According to the **EXP-035 Decision Gate** specified in the research directive:

```text
GATE A — THE CLAIM IS INVALID
If the alleged 571 cases substantially disappear after reclassification:
  INDEXING_MISS was overstated
Do not implement the proposed indexing fix.
Report:
- original count
- corrected count
- why the original classification was wrong
- replacement failure classes
- new macro ranking
Then stop EXP-035 implementation.
```

### Gate Outcome: **TRIGGERED GATE A (THE CLAIM IS INVALID)**

The forensic audit traced all 572 alleged `INDEXING_MISS` fields through the real production pipeline (`DocumentIndex`, `EvidenceMatcher`, `CandidateRecoveryEngine`, `CandidateVerifier`, and `EvidenceResolver`).

The findings definitively falsified the premise that indexing failure is the #1 candidate-generation bottleneck:
- **Alleged Count:** 571 fields (actual extracted: 572 fields across 137 documents), claiming `+3.4991 pp` macro Word Grounding F1.
- **Genuine `REAL_INDEXING_MISS` Count:** **25 fields** across **19 documents** (a **95.6% reduction**).
- **Corrected Macro Headroom:** Only **+0.2293 pp** (vs. +3.4991 pp claimed, a 93.4% reduction).
- **Held-Out Cohort B Headroom:** Only **3 fields** across **2 documents** out of 32 held-out documents.

## 2. Why the Original Classification Was Flawed

The alleged 571 cases in EXP-034R originated from a heuristic classifier in `research/experiments/EXP-034R/gap_characterization.py` (lines 51–62):

```python
# Heuristic in EXP-034R gap_characterization.py:
if not cur_cit or not cur_cit.get("bbox") or cur_cit.get("page") is None:
    if "\n" in str(val):
        return "MULTI_LINE_SPLIT"
    if isinstance(val, (int, float)) or (isinstance(val, str) and re.search(r"\d", val)):
        if is_tbl:
            return "TOP_K_TRUNCATION"
        return "INDEXING_MISS"
    if is_tbl:
        return "DEDUPLICATION_COLLAPSE"
    return "GLOBAL_ROUTING"
```

This classification rule suffered from three fatal diagnostic defects:

1. **Python Boolean Inheritance Bug:**
   In Python, `bool` is a subclass of `int` (`issubclass(bool, int) is True`). Therefore, `isinstance(val, (int, float))` evaluated to `True` for boolean fields (`val = False` or `val = True`).
   Every scalar boolean checkbox field (such as tax form checkboxes `part2_box_e: False` or `taxpayer_pin_self_entered_box: False`) with no baseline citation was automatically labeled `INDEXING_MISS`. This single bug accounted for **286 fields (50.0%)** across 71 documents.

2. **Conflation with OCR-Deficient Documents:**
   The heuristic checked `if "corrupted" in doc_id` for OCR, but failed to check whether scanned PDFs without `"corrupted"` in the test ID had native text tokens. For **148 fields (25.87%)** across 68 documents, the PDF text layer had zero tokens inside the gold bounding box region. These are OCR transcription failures, not indexing failures.

3. **Ignoring Existing Production Retrieval:**
   The heuristic did not trace whether `EvidenceMatcher` or `CandidateRecoveryEngine` actually located the evidence.
   - For **15 fields (2.62%)**, the current production pipeline ALREADY successfully resolves the field with `IoU >= 0.50` (the baseline prediction had abstained or was omitted from the old oracle map).
   - For **16 fields (2.80%)**, candidate recovery generated a candidate with `IoU >= 0.50`, but the resolver/verifier rejected it during verification.

## 3. Production Code Modification Policy

Per Section 11 of the directive:
> *"Only if genuine indexing failure is confirmed... Design the smallest deterministic change... Do not automatically implement n-gram length 2–8, new global index, new retrieval architecture, semantic embeddings, LLM retrieval, VLM retrieval, vector database unless the evidence specifically requires it."*

Because:
1. Only 25 fields across the entire 370-document corpus are genuine indexing misses,
2. In Held-Out Cohort B, only 3 fields exist across 2 documents,
3. The total macro potential of genuine indexing misses is just **+0.2293 pp**,
4. Implementing a speculative character n-gram index or complex token resegmentation would risk false-positive collisions, candidate explosion, latency degradation, and regressions on the 236 passing regression tests,

**No production modifications to `src/tonerhound/` are permitted or justified under Gate A.**

## 4. Replacement Failure Mechanisms & Real Headroom

The actual failure distribution of the alleged 572 cases is:
1. `WRONG_CLASSIFICATION` (Boolean Checkboxes): 286 fields / 71 docs (+1.2162 pp)
2. `OCR_CORRUPTION` (Zero tokens in text layer): 148 fields / 68 docs (+0.9649 pp)
3. `DATE_INDEX_MISS` (Non-standard date formats): 38 fields / 33 docs (+0.4499 pp)
4. `REAL_INDEXING_MISS` (Genuine retrieval failure): 25 fields / 19 docs (+0.2293 pp)
5. `VERIFICATION_REJECTION` (Valid candidate rejected by verifier): 16 fields / 16 docs (+0.1842 pp)
6. `NORMALIZATION_MISMATCH` (Parenthesized negatives, currency codes): 22 fields / 12 docs (+0.1428 pp)
7. `ALREADY_RESOLVED` (Production already resolves with IoU >= 0.50): 15 fields / 13 docs (+0.1401 pp)
8. `BBOX_RECONSTRUCTION` (Geometry / padding / span bounds): 12 fields / 8 docs (+0.0913 pp)
9. `HYPHENATION` (Hyphenated tokens across line ends): 5 fields / 4 docs (+0.0410 pp)
10. `PAGE_ROUTING` (Page hint restricted search to wrong page): 5 fields / 2 docs (+0.0395 pp)
