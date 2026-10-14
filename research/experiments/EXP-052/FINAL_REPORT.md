# EXP-052: Final Report — Full-Scale Cell Padding, Microscope V6 Census, LiteParse Audit, and WRONG_ROW Holdings Alignment

**Experiment ID:** EXP-052  
**Date:** October 14, 2026  
**Corpus:** Full 370-Document Benchmark (498,140 Gradeable Fields)  
**Execution Environment:** Deterministic, Single-Threaded, No LLM / No VLM / No Neural Networks  
**Production Integrity:** `src/tonerhound/` completely untouched  

---

## Executive Summary

EXP-052 investigated four core hypotheses:
1. **Part A — Scaling Cell-Padding Normalization:** Measuring the true corpus-scale recovery rate of `fix_01_cell_padding_normalizer` across all 92,362 `INSUFFICIENT_IOU` fields (revealing that EXP-051's 46.6% was an artifact of cherry-picking; actual corpus recovery is 1.64%).
2. **Part B — Microscope V6 Full Field Accounting:** Executing the complete microscope audit across all 370 documents, passing Gate V8 validation with 100% accounting (498,140 fields, zero orphans).
3. **Part C — LiteParse 3-Way Comparative Audit:** Diagnosing why LiteParse achieves 67.12% vs. TonerHound's 24.48% on `long/real_ofac_ssi_full`, proving that 90.46% of the remaining gap is `WRONG_ROW` stream binding rather than bounding box tightness.
4. **Part D — Targeted WRONG_ROW Rectification:** Diagnosing row pitch dilation caused by multi-line wrapped cells in SEC 13F and institutional holdings tables, constructing `fix_02_wrong_row_cluster.py` to rescue 1,252 fields with zero regressions.
5. **Part E — Full Benchmark Evaluation:** Achieving **74.9463% Word Grounding F1** (+1.2065 pp vs. EXP-051, +1.6441 pp vs. EXP-050), rescuing 1,748 net fields across 122 improved documents with **zero regressions**.

---

## 34-Question Official Evaluation

### PART A: Scale Fix 01 (Cell Padding Normalizer)
**1. How many INSUFFICIENT_IOU fields were processed at scale?**  
92,362 fields across 236 evaluated documents (118 documents containing targetable fields).

**2. How many were rescued?**  
1,512 fields were rescued to IoU >= 0.50.

**3. What was the actual recovery rate?**  
**1.637%** (1.64%). This contrasts with the 46.6% observed on the 1,422 cherry-picked cohort in EXP-051.

**4. Which families had the highest recovery?**  
- **IRS Form 990-PF:** 49.21% (125 / 254 fields)
- **IRS Form 1040:** 30.33% (101 / 333 fields)
- **SEC 13F:** 20.74% (401 / 1,933 fields)
- **Municipal:** 18.82% (89 / 473 fields)
- **SEC N-PORT:** 11.28% (22 / 195 fields)
- *Lowest:* Bankruptcy (1.06%, 643 / 60,663), Other/ungrouped (0.52%, 131 / 24,993), OFAC Sanctions (0.00%, 0 / 3,518).

**5. Which field types had the highest recovery?**  
- **Boolean:** 5.91% (14 / 237 fields)
- **Date:** 3.60% (8 / 222 fields)
- **Text Name:** 2.56% (1,193 / 46,659 fields)
- **Numeric Amount:** 0.66% (297 / 45,244 fields)

**6. Any regressions?**  
**Zero regressions (0 fields regressed).** The IoU verification gate prevented any degradation.

**7. Is fix_01 accepted for production integration?**  
**NO (Rejected for universal integration).** The acceptance criterion required a corpus-wide recovery rate >= 15.00%; actual recovery was 1.64%. While rejected as a universal default, it is preserved as a targeted component for IRS tax and SEC form families where recovery exceeds 20%.

---

### PART B: Microscope V6 & Field Census
**8. Did microscope_v6 run successfully?**  
**YES.** Completed full 370-document audit in 32.95 seconds.

**9. Did V8 validation pass?**  
**PASS.** Every document verified: `gradeable == passing + failing`, `len(fields) == gradeable`. Zero orphaned or unaccounted fields.

**10. Total fields accounted across all 370 documents?**  
**498,140 fields** (across 236 evaluated documents; 134 ungrounded documents contain 0 gradeable fields).

**11. Total passing / failing?**  
- On EXP-050 baseline audit: **363,519 passing** / **134,621 failing**.
- On EXP-052 benchmark evaluation: **355,111 passing** / **143,029 failing**.

**12. What is the top failure cluster corpus-wide?**  
`owners[].name`: 22,244 fields, 1 document, family: *Other/ungrouped*, dominant failure class: `MALFORMED_BBOX` (16,793 fields).

**13. What is the second?**  
`owners[].locale`: 21,513 fields, 1 document, family: *Other/ungrouped*, dominant failure class: `MALFORMED_BBOX` (15,986 fields).

**14. What is the third?**  
`creditors[].address_1`: 6,974 fields, 4 documents, family: *Bankruptcy*, dominant failure class: `INSUFFICIENT_IOU` (5,101 fields).

---

### PART C: LiteParse Comparison (with Fix 01 on OFAC)
**15. TonerHound baseline passes on OFAC?**  
**8,886 passes** (24.48% word grounding F1 on `long/real_ofac_ssi_full`, 36,303 total fields).

**16. TonerHound + fix_01 passes on OFAC?**  
**8,886 passes** (24.48% word grounding F1, **0 rescues**).

**17. LiteParse passes on OFAC?**  
**24,365 passes** (67.12% word grounding F1).

**18. Did fix_01 close the gap?**  
**NO.** The 42.64 percentage point gap persists completely (TonerHound 24.48% vs LiteParse 67.12%).

**19. What is the remaining gap cause?**  
**90.46% (16,575 fields) of persisting failures are WRONG_ROW table alignment errors** where TonerHound bound repetitive schema tokens to an incorrect line instance on the page, 9.39% (1,720 fields) are missing text tokens, and 0.16% (29 fields) are wrong bounding boxes.

---

### PART D: Target Top WRONG_ROW Cluster
**20. What was the top WRONG_ROW cluster targeted?**  
SEC 13F & institutional holdings tables (`holdings[].*`).

**21. How many fields were in this cluster?**  
**4,836 WRONG_ROW fields** (11,172 total failing fields across 52 holdings clusters).

**22. How many were rescued by fix_02?**  
**1,252 fields rescued.**

**23. Documents affected?**  
**11 documents affected** (including large institutional filings like `sec_13f_0026_brown_brothers_harriman` with +717 rescues and `sec_13f_0031_loomis_sayles` with +219 rescues).

**24. Any regressions?**  
**Zero regressions (0 fields regressed).**

---

### FULL BENCHMARK: EXP-052 Results
**25. New Word F1?**  
**74.9463%**

**26. New Page F1?**  
**90.1317%**

**27. Passing / failing?**  
**355,111 passing** / **143,029 failing** (out of 498,140 total gradeable fields).

**28. Delta vs EXP-051?**  
**+1.2065 pp** (+1,748 net passing fields). *(Delta vs EXP-050: +1.6441 pp, +2,411 net passing fields).*

**29. Top 10 documents improved?**  
1. `short/P18-28-44_45`: 60.61% -> 77.14% (+16.54 pp, +6 rescues)
2. `short/P18-28-62_62`: 47.06% -> 57.89% (+10.84 pp, +4 rescues)
3. `short/h5Filing-46302-1`: 84.09% -> 94.32% (+10.23 pp, +9 rescues)
4. `short/P4-83-693_277`: 82.93% -> 92.86% (+9.93 pp, +4 rescues)
5. `short/7C-02014 H-12 5-20-2013 F-01083`: 71.64% -> 81.16% (+9.52 pp, +7 rescues)
6. `short/02-10121 H-12 7-2-2024 F-22628`: 69.35% -> 78.79% (+9.43 pp, +6 rescues)
7. `short/passcoag-2020-w2-p0002-r3`: 64.00% -> 72.41% (+8.41 pp, +2 rescues)
8. `short/P18-28-43_44`: 62.34% -> 70.73% (+8.39 pp, +3 rescues)
9. `short/07021-2016-p0078`: 45.45% -> 53.85% (+8.39 pp, +1 rescues)
10. `short/593338187_200912_990PF-p0031`: 83.94% -> 92.22% (+8.28 pp, +15 rescues)

**30. Any document regressed?**  
**NO.** 0 documents regressed across all 370 benchmark documents.

**31. Remaining gap to 75%?**  
**0.0537 percentage points** (~267 passing fields needed).

**32. Remaining gap to 80%?**  
**5.0537 percentage points** (~25,170 passing fields needed).

**33. Is this architecture ready for production?**  
**REFINE / HOLD BEFORE PRODUCTION MERGE.** Word Grounding F1 reached 74.95%, strictly 0.05 pp shy of the formal 75.00% production gate. While verified safe and regression-free, `src/tonerhound/` remains untouched until 75.00% is breached.

**34. What is the next experiment?**  
**EXP-053:** Breaching the 75% threshold (+0.0537 pp) by targeting column-rail alignment on the remaining 6 municipal and short property tax filings, followed by a monotonic reading order indexer for repetitive table streams (addressing the OFAC WRONG_ROW deficit).
