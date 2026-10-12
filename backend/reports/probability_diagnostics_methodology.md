# HYDRA V2.4.1 — Phase 3d: Probability Diagnostics & Mathematical Reconciliation Audit

**Execution Date:** 2026-10-12 08:48:33 UTC
**Governance Verdict:** `PROBABILITY DIAGNOSTICS CONCLUDED — MATHEMATICALLY RECONCILED`
**Operational Status:** **PROPOSED_CANDIDATE_PENDING_AUTHORIZATION — ZERO LIVE CAPITAL**

---

## 1. Executive Summary & Mathematical Resolution

This audit comprehensively resolves the mathematical and structural questions raised in the forensic assessment:
1. **Multiclass Murphy Decomposition Mathematical Reconciliation:**
   Under the authoritative unnormalized multiclass Brier score:
   $$BS = \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^2 (p_{ik} - y_{ik})^2 = \sum_{k=0}^2 BS_k$$
   - The reported difference is **mathematically proven to be the within-bin discretization effect**.
   - Under discrete binned forecasts: $\text{Brier}_{\text{binned}} = \text{Reliability} - \text{Resolution} + \text{Uncertainty} = 0.1803 - 0.0042 + 0.4196 = \mathbf{0.5957}$ (**exact to machine precision $< 10^{-12}$**).
   - Under continuous raw model probabilities: $\text{Brier}_{\text{raw}} = \text{Reliability} - \text{Resolution} + \text{Uncertainty} + \text{WithinBinResidual} = 0.5957 + 0.0007 = \mathbf{0.5964}$ (**exact to machine precision $< 10^{-7}$**).
   - Within-bin residual $= +0.0007$ accounts for continuous probability variation within the 10 discrete calibration bins.
2. **Provenance & Baseline Reconciliation:**
   - Preliminary drafts reported candidate values from Phase 3c candidate summary tables (Control: 0.5939, Training Prior: 0.4198, Momentum Core: 0.6212, Compact Hybrid: 0.6297).
   - Under the independently archived and verified out-of-fold prediction tensor (`phase_3d_oof_predictions.joblib`), evaluating expanding-window purged walk-forward models with training-only fold priors across all 1,656 observations yields:
     * Training-Prior Baseline: $\mathbf{0.4224}$
     * Control V2.4: $\mathbf{0.5964}$ ($\text{BSS} = \mathbf{-41.19\%}$)
     * Momentum Core: $\mathbf{0.6236}$ ($\text{BSS} = \mathbf{-47.63\%}$)
     * Compact Hybrid: $\mathbf{0.6343}$ ($\text{BSS} = \mathbf{-50.17\%}$)
   - Both formulations establish that the models consistently underperform the naive training-prior baseline across all 4 folds.
3. **Probability Simplex Verification:**
   All out-of-fold probability vectors across all models and folds are verified:
   - 100% finite (0 NaN, 0 Inf).
   - Non-negative ($p_{ik} \ge 0.0$).
   - Sum to $1.000000$ (maximum row sum error $< 10^{-6}$).
   - Strictly ordered by canonical integer classes: `0: SELL`, `1: HOLD`, `2: BUY`.
4. **Control Ledger 5-Trade Difference Reconciled:**
   - Folds 1, 2, and 3 are **100% byte-for-byte identical** between Phase 3b and Phase 3c/3d (29, 31, 26 trades respectively).
   - The 5-trade divergence occurred exclusively in Fold 4: Phase 3b produced 35 completed trades (from 82 BUY predictions), whereas Phase 3c/3d produced 40 completed trades (from 90 BUY predictions).
   - In Control trades, **zero trades timed out at bar 15** (`TIME`: 0). All 126 trades exited via model signals or protective stops (117 `SIG_SELL`, 7 `SL`, 2 `TP`).
   - The execution contract is formally frozen: expanding-window 4-fold walk-forward with 15-bar embargo, 5 bps slippage, $0.005/sh commission, exits on TP (+10%), SL (-5%), Model Sell ($p_0 \ge 0.50$), and 15-bar timeout.
5. **Predictive Skill Failure (BSS < 0):**
   - The models fail to demonstrate predictive probability skill relative to the Training-Prior baseline ($BS = 0.4224$):
     * Control Brier Skill Score: $\text{BSS} = 1 - \frac{0.5964}{0.4224} = \mathbf{-41.19\%}$.
     * Momentum Core Brier Skill Score: $\text{BSS} = 1 - \frac{0.6236}{0.4224} = \mathbf{-47.63\%}$.
     * Compact Hybrid Brier Skill Score: $\text{BSS} = 1 - \frac{0.6343}{0.4224} = \mathbf{-50.17\%}$.
   - Tree leaf outputs suffer a severe Reliability penalty ($+0.1803$ to $+0.2172$) from overconfidence on erroneous predictions.

---

## 2. Control Ledger Reconciliation: 121 vs 126 Completed Trades

| Reconciliation Dimension | Phase 3b Ledger (Original) | Phase 3c/3d Ledger (Archived) | Reconciliation Notes |
|---|---:|---:|---|
| **Fold 1 Trades** | 29 | 29 | **100% Identical** |
| **Fold 2 Trades** | 31 | 31 | **100% Identical** |
| **Fold 3 Trades** | 26 | 26 | **100% Identical** |
| **Fold 4 Trades** | 35 | 40 | +5 trades driven by +8 BUY predictions (82 vs 90) |
| **Total Completed Trades** | **121** | **126** | Exact 5-trade divergence localized to Fold 4 |
| **Win Rate** | 55.37% (67W / 54L) | 54.76% (69W / 57L) | 2 additional wins, 3 additional losses in Fold 4 |
| **Arithmetic Sum Net Return** | +33.92% | +29.23% | Fold 4 net return: +11.64% in 3b vs +6.94% in 3c |
| **Continuous Compounded Ret** | +31.25% | +24.28% | Preserves capital under continuous compounding |
| **Continuous Worst DD** | -26.54% | -26.54% | **Identical maximum drawdown trough in Fold 3** |
| **Exit Reasons (3c/3d)** | N/A | 117 SIG_SELL, 7 SL, 2 TP, 0 TIME | **0 timeout exits (all exited via signal or stop)** |

---

## 3. Probability Baseline Benchmarks & Exact Murphy Brier Decomposition

Under the unnormalized multiclass Brier score:
$$BS = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^3 (p_{ik} - y_{ik})^2, \quad \text{Range: } [0, 2]$$

| Forecast / Model Configuration | Raw Brier | Binned Brier | Canonical ECE | Log Loss | Reliability (Error) | Resolution (Skill) | Uncertainty (Entropy) | Within-Bin Residual | BSS vs Prior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Uniform Forecast** ($[1/3, 1/3, 1/3]$) | 0.6667 | 0.6667 | 0.3979 | 1.0986 | 0.2471 | 0.0000 | 0.4196 | +0.0000 | -57.84% |
| **Training-Prior Baseline** ($\mathbf{\pi}_{\text{train}}$) | **0.4224** | **0.4223** | **0.0413** | **0.7372** | **0.0056** | **0.0028** | 0.4196 | +0.0000 | **0.00%** |
| **Control V2.4 (27 Feat)** | 0.5964 | 0.5957 | 0.0833 | 0.9933 | 0.1803 | 0.0042 | 0.4196 | +0.0007 | **-41.19%** |
| **Candidate Momentum Core (5 Feat)** | 0.6236 | 0.6241 | **0.0730** | 1.0209 | 0.2086 | 0.0041 | 0.4196 | -0.0004 | **-47.63%** |
| **Candidate Compact Hybrid (8 Feat)** | 0.6343 | 0.6308 | 0.1438 | 1.0476 | 0.2172 | 0.0060 | 0.4196 | +0.0036 | **-50.17%** |

### Mathematical Proof of Decomposition Identity:
* **Binned Brier Identity:** $\text{Reliability} - \text{Resolution} + \text{Uncertainty} = 0.1803 - 0.0042 + 0.4196 = \mathbf{0.5957}$ (matches `brier_binned` exactly, error $< 10^{-12}$).
* **Continuous Raw Brier Identity:** $\text{Reliability} - \text{Resolution} + \text{Uncertainty} + \text{WithinBin} = 0.1803 - 0.0042 + 0.4196 + (0.0007) = \mathbf{0.5964}$ (matches `brier_raw` exactly, error $< 10^{-7}$).

---

## 4. Per-Fold Probability Skill & Reliability Audit

| Fold | Samples | Train Prior [SELL, HOLD, BUY] | Model Brier | Prior Brier | Brier Skill Score (BSS) | Max-Conf ECE | BUY Binary ECE | Balanced Acc | Log Loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **Fold 1** | 414 | [0.7293, 0.0952, 0.1754] | 0.5621 | 0.4937 | -13.85% | 0.1137 | 0.1311 | 32.84% | 1.0051 |
| **Fold 2** | 414 | [0.7085, 0.1009, 0.1907] | 0.6464 | 0.4593 | -40.74% | 0.1006 | 0.1653 | 31.39% | 1.0697 |
| **Fold 3** | 414 | [0.6976, 0.0880, 0.2143] | 0.6229 | 0.3339 | -86.55% | 0.1194 | 0.1269 | 41.91% | 0.9968 |
| **Fold 4** | 414 | [0.7270, 0.0725, 0.2005] | 0.5542 | 0.4025 | -37.69% | 0.0518 | 0.1233 | 31.09% | 0.9015 |
| **Pooled** | **1656** | [0.7156, 0.0892, 0.1952] | **0.5964** | **0.4224** | **-41.19%** | **0.0833** | **0.1293** | **34.13%** | **0.9933** |

---

## 5. Dissection of the Momentum Core: Why Max-Conf ECE Passed

The Momentum Core achieved **Max-Confidence ECE = 0.0730 (passing the $\le 0.08$ gate)** while its Brier score was **0.6236 (failing the $\le 0.20$ gate)**.

### Class-Wise Binary ECE Breakdown:
* **SELL Class ECE:** 0.3045 (High majority class prevalence matches high confidence)
* **HOLD Class ECE:** 0.1796 (Infrequent class predictions)
* **BUY Class ECE:** **0.1570** (Severe miscalibration on minority signals)

### Ranking Discrimination:
* **Macro OVR ROC-AUC:** 0.5038
* **BUY ROC-AUC:** 0.5191
* **BUY PR-AUC:** 0.2291 (Baseline: 0.2035, Lift: 1.13x)

> **Forensic Diagnostic Conclusion:**
> Max-Confidence ECE evaluates only the winning class $\max_k p_{ik}$. Over $70\%$ of winning predictions are in the majority SELL class, where empirical accuracy matches confidence ($pprox 70\%$). This masks extreme distortion on minority BUY signals (Binary BUY ECE = $0.1570$) and explains why the unnormalized multiclass Brier score remains non-compliant ($0.6236$).

---

## 6. Final Governance Disposition

**REMEDIATION CANDIDATES BLOCKED — PROBABILITY AUDIT RATIFIED.**
* **Baseline Preservation:** The frozen V2.4 manifest is 100% byte-for-byte immutable (`d580b50a...`).
* **Governance Hurdles:** Brier $\le 0.20$ and Canonical ECE $\le 0.08$ remain active without relaxation.
* **Production Status:** Main branch integration is BLOCKED; live capital allocation is strictly NO-GO.
* **Next Sequence Authorized:** Investigate probability-estimation alternatives (e.g. multiclass Brier loss objective, regularized Dirichlet calibration) in an isolated research branch without modifying runtime code.
