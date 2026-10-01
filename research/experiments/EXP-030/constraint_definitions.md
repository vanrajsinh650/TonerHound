# EXP-030 Phase 5: Structured Assignment Constraint Definitions

**Date**: October 1, 2026  
**Status**: Formalized & Verified  
**Scope**: Mathematical and algorithmic definitions of Constraints A through F for Table-Level Global Structured Assignment.

---

## 1. Mathematical Formulation

Let a document contain a set of tabular records $\mathcal{R} = \{ R_1, R_2, \dots, R_M \}$, where each record $R_i$ represents a schema row (e.g. `holdings[i]` or `creditors[i]`).
Each record $R_i$ comprises a set of schema fields:
$$\mathcal{F}_i = \{ f_{i, 1}, f_{i, 2}, \dots, f_{i, K} \}$$

For each field $f_{i, k}$, the candidate retrieval engine produces a set of candidate occurrences:
$$\mathcal{C}(f_{i, k}) = \{ c_1, c_2, \dots, c_{N_{i, k}} \}$$
where each candidate $c$ is characterized by page $p_c \in \mathbb{N}$, bounding box $[x_c, y_c, w_c, h_c] \in [0, 1]^4$, vertical center $y_c^{\text{mid}} = y_c + h_c / 2$, horizontal center $x_c^{\text{mid}} = x_c + w_c / 2$, and match similarity $\operatorname{Sim}(c) \in [0, 1]$.

The goal is to select an assignment mapping:
$$\pi: (i, k) \mapsto c_{i, k}^* \in \mathcal{C}(f_{i, k})$$
maximizing global structural coherence across all records and fields.

---

## 2. Constraint Definitions & Objectives

### Constraint A — Row Coherence (Intra-Record Vertical Binding)
**Hypothesis**: All fields belonging to the same physical record $R_i$ reside within a unified horizontal row corridor on the same document page.
- **Anchor Discovery**: For record $R_i$, select the most distinctive field $f_{i, \text{anc}} \in \mathcal{F}_i$ possessing the lowest candidate entropy ($1 \le |\mathcal{C}(f)| \le 5$, non-boilerplate). The anchor's baseline candidate defines the physical record anchor:
  $$\mathbf{a}_i = (p_{\text{anc}}, y_{\text{anc}})$$
- **Corridor Penalty**: For every sibling field $f_{i, k} \neq f_{i, \text{anc}}$, penalize candidate vertical divergence:
  $$S_{\text{row}}(c \mid \mathbf{a}_i) = \begin{cases} - \lambda_{\text{row}} \cdot |y_c^{\text{mid}} - y_{\text{anc}}| & \text{if } p_c = p_{\text{anc}} \\ -\infty & \text{if } p_c \neq p_{\text{anc}} \end{cases}$$
  where $\lambda_{\text{row}} = 100.0$.
- **Corridor Tolerance**: Candidates with $|y_c^{\text{mid}} - y_{\text{anc}}| > \delta_y$ (where $\delta_y = 0.025$) are rejected from row binding.

### Constraint B — Row Monotonicity (Inter-Record Vertical Ordering)
**Hypothesis**: Extracted table records $R_1, R_2, \dots, R_M$ follow a top-to-bottom reading progression across pages.
- For consecutive records $R_i, R_{i+1}$ residing on the same page $p$:
  $$y(\mathbf{a}_{i+1}) \ge y(\mathbf{a}_i) - \epsilon_{\text{slack}}$$
  where $\epsilon_{\text{slack}} = 0.008$ accounts for minor visual line tilt.
- Anchors violating monotonic progression are pruned or re-anchored using Longest Non-Decreasing Subsequence (LNDS) dynamic programming.

### Constraint C — Column Rail Consistency (Horizontal Spatial Alignment)
**Hypothesis**: Within a table, instances of the same schema column subfield $f$ (e.g. `cusip`, `value`, `shares`) align to a consistent horizontal column corridor.
- **Corridor Estimation**: The median horizontal coordinate for subfield $f$ across all records in table $T$ is computed:
  $$\bar{x}_{T, f} = \operatorname{median}_{i} \left( x_{c_{i, f, 1}} \right)$$
- **Column Penalty**:
  $$S_{\text{col}}(c \mid \bar{x}_{T, f}) = - \lambda_{\text{col}} \cdot |x_c - \bar{x}_{T, f}|$$
  where $\lambda_{\text{col}} = 20.0$.

### Constraint D — Sibling Proximity (Co-Linearity & Horizontal Order)
**Hypothesis**: Candidates within the same record must satisfy natural left-to-right reading order and vertical overlap:
$$S_{\text{sib}}(c) = \sum_{j \neq k} \mathbb{I}\left( |y_c^{\text{mid}} - y_{c_{i, j}}^{\text{mid}}| \le 0.012 \right) \cdot \omega_{\text{colinear}}$$

### Constraint E — Table Membership (Page & Bounding Envelope)
**Hypothesis**: Table records do not interleave arbitrarily across disjoint document sections.
- Candidates must fall within the page range $[\min(p_T), \max(p_T)]$ established by table anchor calibration.

### Constraint F — Ambiguity & Abstention Gating
**Hypothesis**: When structural row context cannot be established (e.g. all fields in record have high candidate entropy $>10$) or candidate scores are tied within margin $\epsilon_{\text{margin}}$, the system falls back to the conservative baseline or abstains rather than making an ungrounded guess:
$$\text{If } \max_{c} S(c) < \theta_{\text{abstain}}, \quad \text{retain baseline rank 1}.$$

---

## 3. Empirical Ablation Results on Development Cohort A

| Mode | Active Constraints | Hit@1 Count | Hit@1 Rate | Gain over Baseline | Net Flips |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Baseline (Production) | 17,744 | 20.04% | — | — |
| **1** | Constraint A (Row Coherence) | 27,899 | 31.51% | **+11.47 pp** | +10,155 |
| **2** | Constraints A + C (Row + Column) | **29,056** | **32.82%** | **+12.78 pp** | **+11,312** |
| **3** | Constraints A + B + C (Row + Monotonicity + Column) | 28,045 | 31.68% | **+11.64 pp** | +10,301 |

**Selected Primary Production Candidate**: **Mode 2 (A + C: Row Coherence + Column Rail Consistency)** achieves the highest net gain (+12.78pp on Cohort A) while remaining fast and mathematically clean.
