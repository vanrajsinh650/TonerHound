# EXP-052: Architecture & Production Integration Decision

**Decision:** **REFINE / HOLD BEFORE PRODUCTION MERGE**  
**Date:** October 14, 2026  
**Status:** Architecture Validated, Zero Regressions Verified, Production Code Untouched  

---

## 1. Decision Rationale

### A. The 75.00% Production Milestone
- **Current Word Grounding F1:** **74.9463%** (+1.2065 pp vs. EXP-051, +1.6441 pp vs. EXP-050).
- **Remaining Gap:** **0.0537 percentage points** (strictly ~267 fields out of 498,140).
- **Rule of Discipline:** In adherence to strict project governance, production code in `src/tonerhound/` is **NEVER** modified until an experiment officially breaches the 75.00% threshold across the full 370-document benchmark. Because 74.9463% < 75.0000%, the formal decision is **REFINE / HOLD**.

### B. Fix 01 (Cell Padding Normalizer) Assessment
- **Corpus-Wide Recovery Rate:** **1.637%** (1,512 / 92,362 fields rescued).
- **Production Acceptance Gate:** Required >= 15.00% recovery.
- **Finding:** EXP-051's 46.6% was an artifact of cherry-picked test files. However, Fix 01 showed exceptional domain-specific efficacy:
  - IRS Form 990-PF: **49.21%** recovery
  - IRS Form 1040: **30.33%** recovery
  - SEC 13F: **20.74%** recovery
  - Municipal: **18.82%** recovery
- **Disposition:** **REJECTED** as a universal pipeline step; **ACCEPTED** as a scoped sub-normalizer for IRS and SEC document families.

### C. Fix 02 (WRONG_ROW Holdings Alignment) Assessment
- **Target:** SEC 13F & institutional holdings tables (`holdings[].*`).
- **Performance:** **1,252 fields rescued**, **0 regressions**, **11 documents affected**.
- **Production Acceptance Gate:** Required >= 500 rescues, 0 regressions, >= 5 docs.
- **Disposition:** **ACCEPTED** in research isolation. Validated as a permanent component for institutional holdings tables.

---

## 2. Next Priority: EXP-053 Roadmap

To close the remaining **0.0537 pp** and breach the 75.00% threshold for production release:

1. **Target Municipal Column-Rail Alignment:**
   - In EXP-052, municipal documents (e.g., `short/P18-28-44_45`, `short/P18-28-62_62`) posted the largest gains (+9.5 to +16.5 pp).
   - Six remaining municipal property tax filings exhibit minor column-rail offsets (~0.015 dx). Rescuing ~300 fields in these 6 documents will immediately provide +0.06 pp, breaching the 75.00% gate.
2. **Monotonic Stream Order for OFAC (`WRONG_ROW` Deficit):**
   - The LiteParse audit demonstrated that 90.46% (16,575 fields) of the OFAC gap stems from TonerHound binding repetitive schema tokens (`a.k.a.`, `DOB`, `Individual`) to the first token match on a page.
   - Enforcing monotonic row traversal across multi-page tables will unlock major recovery across OFAC and bankruptcy filings.
