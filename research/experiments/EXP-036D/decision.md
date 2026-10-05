# EXP-036D Held-Out Decision Gate

## Status: POSITIVE — AUTHORIZED FOR FULL BENCHMARK

### 1. Executive Summary
The deterministic visual provider was evaluated across all four mandatory conflict-resolution policies (A, B, C, D) on the frozen held-out cohort (25 target fields across 32 documents from `benchmarks/held_out_manifest.json`).

All acceptance criteria for a **POSITIVE** gate under Section 32 of the Directive were satisfied:
- **Meaningful Net Improvement**: Rescued 20 of 25 failing fields (+20 net change, **80.0% recovery**).
- **Regressions Controlled**: **0** regressions across all evaluated fields.
- **False-Positive Explosion**: **0** false-positive regressions.
- **Acceptable Latency**: Average page processing time is ~530ms; total held-out evaluation took 16.09s.

---

### 2. Policy Comparison

| Metric | Policy A (Visual only when no prod) | Policy B (Threshold >= 0.80) | Policy C (Explicit scoring) | Policy D (Strict boolean only) |
|:---|:---:|:---:|:---:|:---:|
| **Targets Evaluated** | 25 | 25 | 25 | 25 |
| **Rescued Fields** | **20 (80.0%)** | 19 (76.0%) | **20 (80.0%)** | **20 (80.0%)** |
| **Regressed Fields** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** | **0 (0.0%)** |
| **Net Field Change** | **+20** | +19 | **+20** | **+20** |
| **Mean IoU** | 0.5516 | 0.5301 | 0.5516 | 0.5516 |
| **Median IoU** | 0.6384 | 0.6384 | 0.6384 | 0.6384 |
| **State Accuracy** | 96.0% (24/25) | 96.0% (24/25) | 96.0% (24/25) | 96.0% (24/25) |
| **Evaluation Runtime** | 16.09s | 16.20s | 16.15s | 16.05s |

### 3. Policy Selection Rationale
- **Policy A & Policy D** both achieved maximum recovery (+20 net, 80.0%).
- Policy B missed 1 field where visual detector confidence was 0.76 (below the 0.80 cutoff).
- Policy D provides architectural safety by strictly enforcing boolean type gating (distinguishing `bool` from `int`/`float`), ensuring zero risk to numerical or text fields.
- **Selected Canonical Configuration for Full Benchmark**: Combined Policy A with Policy D gating (Policy A/D: visual candidates supplied only when production has no candidate, strictly restricted to boolean/checkbox fields).

---

### 4. Gate Decision
Under Section 32:
**POSITIVE** gate triggered. Proceed immediately to **Phase G — Full Benchmark Evaluation**.
