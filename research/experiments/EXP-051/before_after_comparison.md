# EXP-051: Before vs After Benchmark Comparison

## Macro Metrics Comparison

| Metric | EXP-050 Baseline | EXP-051 Final | Absolute Delta | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Word Grounding F1** | 73.3022% | **73.7398%** | **+0.4376 pp** | +0.60% |
| **Page Grounding F1** | 90.1317% | **90.1317%** | +0.0000 pp | +0.00% |
| **Passing Fields** | 352,700 | **353,363** | **+663 fields** | +0.19% |
| **Failing Fields** | 145,440 | **144,777** | **-663 fields** | -0.46% |
| **Total Gradeable Fields** | 498,140 | 498,140 | 0 | - |
| **Evaluated Documents** | 236 | 236 | 0 | - |
| **Total Documents** | 370 | 370 | 0 | - |
| **Regressed Documents** | 0 | **0** | **0 (Zero)** | - |
| **Gap to 75.0% Target** | 1.6978 pp | **1.2602 pp** | **-0.4376 pp** | -25.77% gap reduction |
| **Gap to 80.0% Milestone** | 6.6978 pp | **6.2602 pp** | **-0.4376 pp** | -6.53% gap reduction |

---

## Top 15 Improved Documents

| Document ID | Family | Before Word F1 | After Word F1 | Delta (pp) | Rescued Fields |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `short/h5Filing-46302-1` | Miscellaneous Corporate Filings | 84.09% | **92.86%** | **+8.77 pp** | +9 |
| `short/593338187_200912_990PF-p0030` | IRS Form 990-PF | 78.81% | **86.86%** | **+8.05 pp** | +14 |
| `short/593338187_200912_990PF-p0031` | IRS Form 990-PF | 83.94% | **91.80%** | **+7.86 pp** | +14 |
| `short/593338187_200912_990PF-p0033` | IRS Form 990-PF | 82.58% | **90.16%** | **+7.58 pp** | +14 |
| `short/593338187_200912_990PF-p0032` | IRS Form 990-PF | 79.75% | **87.27%** | **+7.53 pp** | +12 |
| `short/593338187_200912_990PF-p0036` | IRS Form 990-PF | 84.36% | **91.26%** | **+6.90 pp** | +13 |
| `short/593338187_200912_990PF-p0038` | IRS Form 990-PF | 88.83% | **94.54%** | **+5.71 pp** | +10 |
| `short/593338187_200912_990PF-p0037` | IRS Form 990-PF | 88.96% | **94.25%** | **+5.30 pp** | +9 |
| `short/593338187_200912_990PF-p0035` | IRS Form 990-PF | 91.36% | **95.63%** | **+4.26 pp** | +8 |
| `short/593338187_200912_990PF-p0027` | IRS Form 990-PF | 83.12% | **87.27%** | **+4.15 pp** | +7 |
| `short/h5Filing-45919-1` | Miscellaneous Corporate Filings | 83.33% | **86.84%** | **+3.51 pp** | +3 |
| `short/h5Filing-49070-1` | Miscellaneous Corporate Filings | 83.33% | **86.84%** | **+3.51 pp** | +3 |
| `medium/bar-lev-2024` | IRS Form 1040 | 47.96% | **51.34%** | **+3.38 pp** | +19 |
| `medium/bar-lev-2021` | IRS Form 1040 | 47.51% | **50.36%** | **+2.85 pp** | +16 |
| `medium/becerra-2021` | IRS Form 1040 | 49.33% | **52.12%** | **+2.79 pp** | +14 |

---

## Major Form & SEC Table Impacts
- **IRS Form 1040 Cohort**: 14 documents improved (becerra: +29 rescues; bar-lev: +49 rescues; cabrera: +12 rescues; arif: +12 rescues).
- **SEC 13F Cohort**: 5 large tabular filings recovered 403 fields (Renaissance Technologies: +155; Loomis Sayles: +153; Brown Brothers Harriman: +88; Nalls Sherbakoff: +6; Leonteq Securities: +1).
- **SEC Form N-PORT**: 2 filings recovered 23 fields (SEI ETF: +15; Dunham Funds: +8).
- **IRS Form 990-PF**: 9 filings recovered 97 fields.
- **Municipal Assessments**: 11 filings recovered 20 fields.
