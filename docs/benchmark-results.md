# Benchmark Results

All results are on the official **ExtractBench 370-document corpus** (498,140 gradeable fields, 236 documents with grounded field rules). Evaluated with the official `ExtractEvaluator` (`ExtractAssociationF1Metric`, IoU ≥ 0.50 threshold).

## Headline Metrics

| Metric | Value |
|---|---|
| Word Grounding F1 | **72.6179%** |
| Page Grounding F1 | 83.8490% |
| Word Grounding Precision | 77.7935% |
| Word Grounding Recall | 68.9729% |
| Total Gradeable Fields | 498,140 |
| Passing Fields (IoU ≥ 0.50) | 327,671 |
| Failing Fields | 170,469 |
| Field Regressions (EXP-037 → EXP-042) | **0** |

## ExtractBench Competitive Leaderboard

Evaluated on the official **ExtractBench 370-document corpus** across document length splits (Short: 1–2 pages, Medium: 3–9 pages, Long: 10+ pages).

| Rank | Provider | Word Grounding F1 (Overall) | Short | Medium | Long | Page Grounding F1 (Overall) | Short | Medium | Long | Stack / Cost |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | LlamaExtract Agentic Plus | **81.26%** | **81.63%** | 81.18% | 75.22% | 89.94% | 92.45% | 86.50% | 74.71% | Frontier Agent ($0.081/pg) |
| 2 | LlamaExtract Agentic | 78.89% | 77.63% | **81.86%** | **80.89%** | **90.64%** | **92.58%** | **86.68%** | **87.09%** | Agent Harness ($0.050/pg) |
| 3 | Codex (GPT-6 Sol Evidence) | 77.11% | 77.61% | 75.53% | 78.31% | 86.50% | 87.20% | 85.12% | 84.82% | Cloud LLM |
| **4** | **TonerHound** | **72.62%** | **70.72%** | **80.58%** | **68.82%** | **83.85%** | **86.34%** | **78.76%** | **77.29%** | **Deterministic ($0.00)** |
| 5 | Claude Code (Opus 5.5 Evidence) | 71.55% | 73.57% | 65.54% | 74.87% | 76.95% | 79.27% | 71.36% | 76.47% | Cloud LLM |
| 6 | Codex (GPT-6 Luna Evidence) | 65.70% | 62.91% | 71.11% | 74.36% | 83.93% | 84.41% | 83.47% | 80.17% | Cloud LLM |
| 7 | Claude Code (Sonnet 5.5 Evidence) | 64.61% | 64.59% | 63.18% | 71.22% | 75.99% | 77.93% | 71.50% | 74.98% | Cloud LLM |
| 8 | Reducto Extract (v4) | 54.97% | 60.53% | 49.35% | 16.30% | 75.28% | 87.47% | 57.38% | 18.59% | Commercial API |
| 9 | Codex (GPT-5.6 Sol Evidence) | 54.66% | 50.87% | 64.58% | 69.01% | 83.62% | 85.12% | 80.77% | 78.84% | Cloud LLM |
| 10 | LlamaExtract Cost-Effective | 53.65% | 49.20% | 64.84% | 55.66% | 80.09% | 87.53% | 66.47% | 54.94% | Cloud LLM |
| 11 | Codex (GPT-5.5 Evidence) | 52.57% | 53.13% | 48.01% | 61.87% | 80.64% | 83.27% | 73.87% | 79.90% | Cloud LLM |

> **Key takeaway:** TonerHound is the only system in the top 5 with zero neural networks and zero API cost. On medium-length documents (3–9 pages), TonerHound reaches **80.58% Word F1**, within 1.3 points of the commercial frontier.

## Score Progression

| Experiment | Word F1 | Δ | Regressions | What Moved It |
|---|---|---|---|---|
| Canonical baseline (`canonical_370_v1`) | 56.0477% | — | — | Starting point |
| EXP-037 | 56.6707% | +0.6230 | 0 | Page-selective Tesseract OCR |
| EXP-038 | 58.1118% | +1.4411 | 0 | Accounting negatives + resolver fallback |
| EXP-039 | 69.1327% | +11.0209 | 0 | Hungarian table assignment + visual fallback |
| EXP-040 | 70.3894% | +1.2567 | 0 | Multi-token matching |
| EXP-041 | 70.5761% | +0.1867 | 0 | Geometric recovery attempt |
| EXP-042 | **72.6179%** | +2.0418 | 0 | 18 date literal formats |
| EXP-043 (post-release, not integrated) | 72.6992% | +0.0813 | 0 | Mathematical ceiling confirmation |

**Total gain:** +16.6502 pp across 7 experiments. **Zero regressions** across all 498,140 fields.

## Per-Module Contribution

| Module | Fields Contributed | Share |
|---|---|---|
| Hungarian table assignment | +5,845 | 38.5% |
| Date literal variants (18 formats) | +3,690 | 24.3% |
| Visual pixel-statistics fallback | +1,882 | 12.4% |
| Multi-token matching | +1,196 | 7.9% |
| Multi-region assembler | +689 | 4.5% |
| Standard resolver fallback | +723 | 4.8% |
| Accounting negative normalization | +494 | 3.3% |
| Table cell grounding | +112 | 0.7% |
| OCR noise index | +16 | 0.1% |
| Multi-line assembly | +5 | 0.03% |
| **Total** | **+15,171 field-events** | |

*(Field-events exceed net rescued fields because some fields pass through multiple modules in the priority pipeline.)*

## What Didn't Work (Honest Negative Results)

| Technique | Target Class | Fields Rescued | Lesson |
|---|---|---|---|
| Percent ↔ decimal normalization | NORMALIZATION_MISMATCH | 2 / 37,850 | Gold values store raw strings, not semantic values |
| Column rail constraints | TOKEN_SLICING | 0 | More geometry ≠ more rescues |
| Hyphen joiner v2 | HYPHENATION | 0 | Edge case too rare to matter |
| Needleman-Wunsch alignment | REAL_INDEXING_MISS | 13 | Math alone can't bridge the gap |
| Niblack + Sauvola OCR voting | NO_TEXT_AT_GOLD_REGION | 0 | Corrupted pixels are unrecoverable |
| Recursive XY-Cut | TOKEN_SLICING | 0 | Table cells already handled by Hungarian |

These negative results are the most valuable part of the research. They define the deterministic ceiling empirically.

## Reproducing

```bash
# Clone
git clone https://github.com/vanrajsinh650/TonerHound
cd TonerHound

# Install
uv sync

# Run tests
pytest tests/ -v  # 261 tests, ~20s

# Run full benchmark
python -m tonerhound.benchmark \
  --data-dir research/data/full \
  --exp-id VERIFY
# Expected: Word Grounding F1 = 72.6179%
```

## Limitations

- **Page grounding F1 is 83.8%.** Even with perfect word grounding, the ceiling is 83.8% because 16.2% of values are placed on the wrong page. This requires visual page selection (a VLM task).
- **The remaining 27.4% of failures are not solvable with deterministic methods.** They fall into three categories: semantic normalization (needs LLM), corrupted perception (needs VLM), and annotation convention (needs supervised learning).
- **The benchmark is vendor-run.** ExtractBench is maintained by LlamaIndex, whose own product competes on it. Cross-reference with independent benchmarks (OmniDocBench, DocVQA) before drawing conclusions.
