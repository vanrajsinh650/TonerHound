# EXP-047: Promotion Decision & Architectural Verdict

## Verdict: **REFINE (Selective Staged Integration)**

### Recommendation Summary
- **Promote to Core:** `CharIndex` (Phase A) and the Hungarian block optimizer (`GlobalAssigner`, Phase D).
- **Refine before Full Production Deployment:** `TableLocalizer` (Phase C) requires integration with visual column rails to prevent horizontal column cross-assignment.
- **Do NOT integrate blindly into standard single-field lookup:** Global assignment should be executed strictly as a document-level pass on unassigned fields after initial exact/normalized parsing.

---

## Decision Rationale

### 1. Evidence of Paradigm Shift
- In EXP-046, expanding from 7 to 34 field-level heuristic stages yielded a meager +32 rescues across the entire 370-document corpus.
- In EXP-047, document-level global constraint assignment rescued **+991 fields** (a 31x acceleration) while preserving **zero regressions**.
- Suffix array character search demonstrated an overwhelming advantage: in ablation testing, removing `char_index` caused rescues to collapse from 729 down to 219 (-70%).

### 2. Why Word F1 Remained Structurally Flat (+0.0000 pp)
ExtractBench reports a **macro unweighted average** over 370 documents:
- Rescuing 660 fields in `long/real_ofac_ssi_full` (a document with 45,440 fields) improves that document's Word F1 from 18.10% to 19.55% (+1.45 pp).
- When averaged across 370 documents:
  $$\Delta_{\text{macro}} = \frac{1.45 \text{ pp}}{370} \approx +0.0039 \text{ pp}$$
- Across the 28 improved documents, the total sum of per-document percentage improvements was ~1.5 pp, resulting in an unrounded macro delta of $\approx +0.004$ pp, which rounds to **+0.0000 pp** at 4 decimal places.

### 3. Failure Mode: The Horizontal Column Boundary Problem
Forensic inspection of unresolved fields revealed the key bottleneck of the current global assigner:
- Vertical ordering (row assignment) was solved accurately via monotonicity enforcement and table localization priors.
- However, within a detected row, multiple numeric fields often have identical or indistinguishable string representations (e.g., Voting Authority "Sole" vs. "Shared" vs. "None", or "Shares" vs. "Par Value").
- Because `TableLocalizer` estimated column horizontal centers coarsely from first-page text blocks, the Hungarian solver frequently assigned the correct row but the wrong column within that row (resulting in $IoU = 0.0$ against gold).

---

## Action Plan for EXP-048

To convert the +991 field rescue momentum into a macro Word F1 breakthrough:
1. **EXP-048: Column-Rail Spatial Constrained Global Assignment:**
   - Integrate physical line separator detection (from EXP-046 Stage 19 `column_rail_detector`) into `table_localizer.py`.
   - Restrict Hungarian candidate search for each schema subfield strictly to its detected physical column corridor.
   - Enforce both vertical monotonicity AND horizontal corridor membership simultaneously.
