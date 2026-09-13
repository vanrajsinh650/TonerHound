# EXP-002 Baseline Failure Analysis & Candidate Recall Report

**Date**: 2026-09-13 08:35:24 UTC  
**Scope**: Target failure cases (`W14-Atascosa`, `bianco-2024`, `real_pueblo_oct_2025`, `real_sm0801_eco_full`)  
**Evaluator**: ExtractBench Unified Evidence Grounding  

---

## 1. Executive Failure Taxonomy Summary

Total Ground-Truth BBox Rules Analyzed: **10652**  
Total Successfully Grounded (IoU $\ge$ 0.5): **5180** (48.63%)  
Total Failed / Missed Groundings: **5472** (51.37%)  

| Failure Category | Total Occurrences | Share of Failures | Primary Root Cause |
| :--- | :---: | :---: | :--- |
| `table/structure failure` | 3859 | 70.5% | See per-document details below |
| `ambiguity` | 911 | 16.6% | See per-document details below |
| `wrong region` | 350 | 6.4% | See per-document details below |
| `OCR absence` | 238 | 4.3% | See per-document details below |
| `wrong page` | 113 | 2.1% | See per-document details below |
| `wrong occurrence` | 1 | 0.0% | See per-document details below |

---

## 2. Candidate Pool Recall Diagnostic (Recall@K)

Mandatory architectural gate: Does the correct physical bounding box even exist in the candidate pool prior to resolution ranking?

| Document Case | GT BBoxes | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Candidate Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025)` | 83 | 3.61% | 4.82% | 4.82% | 4.82% | Zero text layer (OCR absence) |
| `short/bianco-2024` | 155 | 0.00% | 0.00% | 0.00% | 0.00% | Zero text layer (OCR absence) |
| `medium/real_pueblo_oct_2025` | 3757 | 53.37% | 59.81% | 64.49% | 72.88% | Candidate generation needs improvement |
| `long/real_sm0801_eco_full` | 6657 | 55.43% | 84.83% | 90.88% | 95.54% | Candidate pool healthy; ranking/disambiguation is bottleneck |

---

## 3. Per-Document Deep Dive

### Document: `short/W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025)`
- **PDF File**: `W14-Atascosa SWD Well No. 4 - W-14 (Updated 01.22.2025).pdf`
- **Pages**: 1 | **Extracted Tokens**: 13
- **Official Metrics**: Word F1 = **6.45%** | Page F1 = **9.95%**
- **Candidate Recall**: R@1 = 3.61% | R@5 = 4.82% | R@10 = 4.82% | R@20 = 4.82%
- **Failure Taxonomy Breakdown**:
  - `OCR absence`: 83

### Document: `short/bianco-2024`
- **PDF File**: `bianco-2024.pdf`
- **Pages**: 10 | **Extracted Tokens**: 0
- **Official Metrics**: Word F1 = **0.00%** | Page F1 = **0.00%**
- **Candidate Recall**: R@1 = 0.00% | R@5 = 0.00% | R@10 = 0.00% | R@20 = 0.00%
- **Failure Taxonomy Breakdown**:
  - `OCR absence`: 155

### Document: `medium/real_pueblo_oct_2025`
- **PDF File**: `real_pueblo_oct_2025.pdf`
- **Pages**: 11 | **Extracted Tokens**: 5329
- **Official Metrics**: Word F1 = **53.81%** | Page F1 = **96.27%**
- **Candidate Recall**: R@1 = 53.37% | R@5 = 59.81% | R@10 = 64.49% | R@20 = 72.88%
- **Failure Taxonomy Breakdown**:
  - `table/structure failure`: 1513
  - `ambiguity`: 274
  - `wrong page`: 17
  - `wrong region`: 5

### Document: `long/real_sm0801_eco_full`
- **PDF File**: `real_sm0801_eco_full.pdf`
- **Pages**: 66 | **Extracted Tokens**: 9206
- **Official Metrics**: Word F1 = **51.01%** | Page F1 = **94.20%**
- **Candidate Recall**: R@1 = 55.43% | R@5 = 84.83% | R@10 = 90.88% | R@20 = 95.54%
- **Failure Taxonomy Breakdown**:
  - `table/structure failure`: 2346
  - `ambiguity`: 637
  - `wrong region`: 345
  - `wrong page`: 96
  - `wrong occurrence`: 1
