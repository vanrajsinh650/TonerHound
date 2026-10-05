# EXP-037: Deterministic Page-Level OCR Routing

## Mission & Executive Objective
Determine, with actual measured evidence, how much deterministic page-level OCR can recover from the TonerHound failure population identified by the Failure Microscope (`canonical_370_v1`).

- **Do NOT assume the answer.**
- **Do NOT assume OCR will help.**
- **Do NOT assume 90% is reachable or unreachable.**
- **Measure it.**

---

## Experiment Summary

| Parameter | Specification |
| :--- | :--- |
| **Run ID (Targeted)** | `exp037_targeted_run_v1` |
| **Run ID (Full 370)** | `exp037_full_run_v1` |
| **Baseline Run ID** | `canonical_370_v1` (Git commit `178f81c`) |
| **Evaluator** | Official ExtractBench `ExtractEvaluator` (`ExtractAssociationF1Metric`) |
| **Routing Principle** | IF page has usable native digital text ($\ge 25$ tokens): **DO NOT OCR**<br>IF page has negligible native text ($< 25$ tokens): **Invoke deterministic OCR** |
| **Deterministic Stack** | PyMuPDF (PDFium) bitmap rendering at scale 2.78 + Tesseract OCR + `repair_ocr_text` + PSM 11 sparse fallback. Zero AI/LLM/VLM/Neural Networks. |

---

## Headline Measured Results

| Metric | Canonical Baseline (`canonical_370_v1`) | EXP-037 After (`exp037_full_run_v1`) | Measured Delta |
| :--- | :---: | :---: | :---: |
| **Word Grounding F1** | **56.0477%** | **56.6707%** | **+0.6230 pp** |
| **Page Grounding F1** | **81.6639%** | **81.9786%** | **+0.3147 pp** |
| **Word Grounding Precision** | **61.7275%** | **62.4511%** | **+0.7236 pp** |
| **Word Grounding Recall** | **52.5686%** | **53.1233%** | **+0.5547 pp** |
| **Total Gradeable Fields** | 498,140 | 498,140 | 0 |
| **Passing Fields ($IoU \ge 0.50$)** | 307,373 | 307,514 | **+141 fields** |
| **Failing Fields** | 190,767 | 190,626 | **-141 fields** |
| **Fields Rescued** | 0 | 141 | **+141 fields** |
| **Fields Regressed** | 0 | 0 | **0 fields** |
| **Documents with Word F1 Improved** | — | 45 | **45 docs** |
| **Documents with Word F1 Regressed** | — | 3 | **3 docs** |
| **Documents Unchanged** | — | 82 | **82 docs** |

---

## Arithmetic Distance to 90.00% Word F1

$$56.0477\% \text{ (Baseline)} + 0.6230\% \text{ (OCR Gain)} = 56.6707\% \text{ (New Measured Score)}$$

$$\text{Remaining Gap to } 90.00\% = 90.0000\% - 56.6707\% = \mathbf{33.3293\text{ percentage points}}$$

---

## Directory Contents

- [before_after_comparison.md](file:///home/vanrajsinh/Projects/TonerHound/research/experiments/EXP-037/before_after_comparison.md): Comprehensive before vs after metric tables, document gains, and latency audit.
- [implementation_notes.md](file:///home/vanrajsinh/Projects/TonerHound/research/experiments/EXP-037/implementation_notes.md): Technical routing design, OCR extraction mechanics, and index ingestion.
- [decision.md](file:///home/vanrajsinh/Projects/TonerHound/research/experiments/EXP-037/decision.md): Final engineering and research decision.
- [FINAL_REPORT.md](file:///home/vanrajsinh/Projects/TonerHound/research/experiments/EXP-037/FINAL_REPORT.md): Complete 17-section master report.
- `ocr_router.py`: Deterministic page-selective OCR routing implementation.
- `targeted_harness.py`: Phase C targeted feasibility harness on the 130 target documents.
- `run_exp037_full_benchmark.py`: Phase D & E full 370-document official benchmark runner and Failure Microscope analyzer.
- `ocr_runtime.json`: Section 7 required instrumentation metrics.
- `targeted_results.json`: Raw results from Phase C targeted run.
- `targeted_failure_analysis.json`: Causal migration analysis from Phase C.
- `full_benchmark_results.json`: Official evaluation outputs across all 370 documents.
- `failure_migration.json`: Complete failure class migration matrix.
- `regression_analysis.json`: Audit of zero regressions.
- `success_analysis.json`: Characterization of rescued fields across document families.
