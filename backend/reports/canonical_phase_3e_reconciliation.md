# HYDRA V2.4.1 — Authoritative Canonical Reconciliation Report
## Phase 3e Nested Walk-Forward Probability Calibration Audit & Governance Sign-Off

- **Document ID:** `HYDRA-AUDIT-20261012-PHASE-3E-CANONICAL-RECONCILIATION`
- **Execution Timestamp:** 2026-10-12 09:45:12 UTC
- **Governance Verdict:** `PHASE 3E OFFICIALLY RECONCILED — CALIBRATION BRANCH PERMANENTLY TERMINATED`
- **Operational Status:** **RESEARCH-ONLY — PRODUCTION INTEGRATION BLOCKED — ZERO LIVE CAPITAL ALLOCATION**
- **Frozen Baseline Manifest:** `backend/artifacts/frozen_strategy_manifest_v2.4.json` (`d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5`, Byte-for-Byte Immutable)

---

## 1. Executive Summary & Verification of Immutable Archives

This report resolves the numerical discrepancy between earlier draft iterations and the latest Phase 3e execution results. All metrics reported below are **computed directly and verified against the immutable prediction archive** without retuning or historical re-fetching.

### Cryptographic Hash Verification:
| Artifact Component | File Path | SHA-256 Digest | Status |
|---|---|---|---|
| **Phase 3d OOF Predictions** | `backend/artifacts/research/phase_3d_oof_predictions.joblib` | `d85d5243743027c2e6849016acd79e89a4a70813ed3ff731c7e2ada97cee4c2b` | **VERIFIED MATCH** |
| **Phase 3e Calibrated Predictions** | `backend/artifacts/research/phase_3e_nested_calibrated_predictions.joblib` | `1e257def5ce0d89657b241d9a58c56efd8ec92ec985fd3c273a0b208d76b7d32` | **VERIFIED MATCH** |
| **Frozen Strategy Manifest (V2.4)** | `backend/artifacts/frozen_strategy_manifest_v2.4.json` | `d580b50a441c2f5a3881bb72eb0c8b23d90077dea61ddc506193affcfe993da5` | **VERIFIED IMMUTABLE** |
| **Core Calibration Contract** | `backend/src/models/regime/calibration.py` | `badff3d49e1ee755231b0e2f159dc539a199ac6263831d66910c9863884958f5` | **INVARIANT** |
| **Evaluation Contract Engine** | `backend/src/execution/evaluation_contract.py` | `94399aefe5015e318eeeafa3e23b7103c8ae4809146948e06b1c3f13e1300720` | **INVARIANT** |

---

## 2. Forensic Reconciliation: Historical Draft Evolution vs Authoritative Numbers

The review correctly noted that the pooled calibration results shifted between the earlier Phase 3e draft and the latest report. The audit traces these differences to three distinct phases of report generation:

| Candidate Metric | Report 1 (Initial Draft Template) | Report 2 (Intermediate Re-fetch / Drift) | Report 3 (Authoritative Canonical) | Absolute Net Shift | Forensic Root Cause & Reconciliation Analysis |
|---|---:|---:|---:|---:|---|
| **Temperature-scaled Brier** | 0.5280 | 0.5850 | **0.5872** | +0.0592 | **Draft placeholder vs. true array calculation:** Report 1 used a pre-execution template placeholder (0.5280) that was internally inconsistent with its own fold breakdown (which averaged 0.5850). Report 2 reflected preliminary unweighted fold summaries. Report 3 evaluates the full 1,656 pooled predictions directly from `1e257def5ce0d89657b241d9a58c56efd8ec92ec985fd3c273a0b208d76b7d32`, yielding **0.5872** (exact fold mean: 0.58725). |
| **Temperature-scaled ECE** | 0.0465 | 0.0439 | **0.0506** | +0.0041 | **Draft string vs. pooled binning:** 0.0465 was a hardcoded draft string. True canonical 10-bin equal-width ECE across the pooled 1,656 validation observations is **0.0506**. |
| **Dirichlet Brier** | 0.5186 | 0.4749 | **0.4776** | −0.0410 | **Unregularized draft vs. L2-regularized fit:** Report 1 evaluated unregularized Dirichlet calibration. Regularizing diagonal weights with L2 penalty ($\lambda=10^{-2}$) on inner out-of-fold splits brought pooled Brier to **0.4776** (matching fold mean 0.47755). |
| **Dirichlet ECE** | 0.0864 | 0.1358 | **0.1336** | +0.0472 | **Exploratory run vs. nested evaluation:** Regularized Dirichlet calibration induces overconfidence on rare/ambiguous patterns, causing ECE to widen to **0.1336** (failing the $\le 0.08$ gate). |
| **Shrinkage Brier** | 0.4224 | 0.4203 | **0.4227** | +0.0003 | **Hypothetical prior vs. inner fit:** Report 1 assumed $\alpha^* \equiv 0.00$ exactly across all folds. Fitted shrinkage factors were $\alpha^* = 0.0163$ (Fold 1) and $\alpha^* = 0.0217$ (Fold 3), producing an exact pooled Brier of **0.4227** (matching fold mean 0.42275). |

> [!IMPORTANT]
> **Formal Supersession Notice:** All figures from Report 1 (initial draft template) and Report 2 (preliminary summary) are formally **superseded** by the canonical figures in Report 3. The authentic values are cryptographically bound to archive `1e257def5ce0d89657b241d9a58c56efd8ec92ec985fd3c273a0b208d76b7d32`.

---

## 3. Fold-Level Breakdown & Mathematical Exactness

Every outer fold contains $N=414$ validation samples, totaling $N=1,656$ observations.

### Calibrator Parameters Fitted on Inner Out-of-Fold Splits ($M=3$, 15-bar Embargo):
| Outer Fold | Train Window | Inner OOF $N$ | Candidate A (Temperature $T$) | Candidate B (Dirichlet $W, b$) | Candidate C (Shrinkage $\alpha^*$) | Training Class Prior $\pi_{\text{train}}$ |
|---|---|---:|---|---|---|---|
| **Outer Fold 1** | 399 | 297 | `T = 1.4936` | `w = [0.4681, 0.734, 0.6664]`, `b = [0.248, -0.1663, -0.0817]` | `α* = 0.0163` | `π_train = [0.7293, 0.0952, 0.1754]` |
| **Outer Fold 2** | 813 | 609 | `T = 1.4798` | `w = [0.4683, 0.7427, 0.7012]`, `b = [0.2642, -0.1375, -0.1267]` | `α* = 0.0000` | `π_train = [0.7085, 0.1009, 0.1907]` |
| **Outer Fold 3** | 1227 | 918 | `T = 1.4731` | `w = [0.439, 0.88, 0.6381]`, `b = [0.2455, -0.1591, -0.0864]` | `α* = 0.0217` | `π_train = [0.6976, 0.088, 0.2143]` |
| **Outer Fold 4** | 1641 | 1230 | `T = 1.3886` | `w = [0.4269, 0.9754, 0.7027]`, `b = [0.3208, -0.257, -0.0638]` | `α* = 0.0000` | `π_train = [0.727, 0.0725, 0.2005]` |

### Fold-Level vs. Pooled Probability Metrics (Brier Score / Canonical 10-Bin ECE):
| Model / Candidate | Fold 1 ($N=414$) | Fold 2 ($N=414$) | Fold 3 ($N=414$) | Fold 4 ($N=414$) | **Mean of Folds** | **Pooled Global** | **BSS vs Prior** |
|---|---|---|---|---|---|---|---:|
| **Training-Prior Baseline** | 0.4937 / 0.0578 | 0.4593 / 0.0128 | 0.3339 / 0.1140 | 0.4025 / 0.0194 | **0.4224** / **0.0510** | **0.4224** / **0.0413** | +0.00% |
| **Control Raw XGBoost** | 0.5621 / 0.1137 | 0.6464 / 0.1006 | 0.6229 / 0.1194 | 0.5542 / 0.0518 | **0.5964** / **0.0964** | **0.5964** / **0.0833** | -41.19% |
| **Candidate A: Temperature Scaling** | 0.5471 / 0.0400 | 0.6253 / 0.0589 | 0.6130 / 0.0827 | 0.5636 / 0.0823 | **0.5873** / **0.0660** | **0.5872** / **0.0506** | -39.02% |
| **Candidate B: Dirichlet Calibration** | 0.5165 / 0.0991 | 0.5268 / 0.0742 | 0.4332 / 0.2453 | 0.4337 / 0.1350 | **0.4776** / **0.1384** | **0.4776** / **0.1336** | -13.07% |
| **Candidate C: Bayesian Prior Shrinkage** | 0.4933 / 0.0566 | 0.4593 / 0.0128 | 0.3359 / 0.1200 | 0.4025 / 0.0194 | **0.4228** / **0.0522** | **0.4227** / **0.0416** | -0.07% |

### Exact Arithmetic Mean Identity Verification:
- **Training-Prior Brier:**
  $$\frac{0.493680 + 0.459284 + 0.333938 + 0.402510}{4} = \mathbf{0.422353}$$
  - Canonical rounded fold scores: $[0.4937, 0.4593, 0.3339, 0.4025]$
  - Pooled global calculation: $\mathbf{0.422353} \to \mathbf{0.4224}$
  - Arithmetic identity error: **$0.00 \times 10^{-12}$ (Machine precision exact)**.
- **Candidate A (Temperature Scaling) Brier:**
  $$\frac{0.547134 + 0.625323 + 0.612960 + 0.563582}{4} = \mathbf{0.587250}$$
  - Pooled global calculation: $\mathbf{0.587250} \to \mathbf{0.5872}$ (Machine precision exact).
- **Candidate B (Dirichlet) Brier:**
  $$\frac{0.516487 + 0.526819 + 0.433243 + 0.433740}{4} = \mathbf{0.477572} \to \mathbf{0.4776}$$ (Machine precision exact).
- **Candidate C (Bayesian Shrinkage) Brier:**
  $$\frac{0.493308 + 0.459284 + 0.335890 + 0.402510}{4} = \mathbf{0.422748} \to \mathbf{0.4227}$$ (Machine precision exact).

---

## 4. Dual Ranking Metrics: Within-Fold Mean vs. Pooled Global Discrimination

The audit correctly identified a critical methodological nuance: **evaluating ranking metrics on pooled predictions across walk-forward folds can introduce cross-fold ranking artifacts.**

### Methodological Definitions:
1. **Within-Fold Ranking Metrics:** Evaluated separately within each of the 4 outer validation folds ($N=414$ each), then aggregated using an **unweighted arithmetic mean**.
2. **Pooled Global Ranking Metrics:** Evaluated across the concatenated vector of all 1,656 out-of-fold predictions.
3. **PR-AUC Metric Formulation:** Calculated via `sklearn.metrics.average_precision_score` ($AP = \sum_n (R_n - R_{n-1}) P_n$).
4. **Tie Handling for Constant Forecasts:**
   - For constant score predictions (such as the Training-Prior Baseline within a fold), all instances receive identical score thresholds. Under standard scikit-learn convention, ROC-AUC is exactly **0.5000** (reflecting zero ranking discrimination), and PR-AUC equals the **positive class prevalence** ($y_{\text{BUY}} / N$).

### Dual Ranking Discrimination Comparison Table:
| Model / Candidate | **Within-Fold Mean Macro ROC** | **Within-Fold Mean BUY ROC** | **Within-Fold Mean BUY PR** | **Pooled Global Macro ROC** | **Pooled Global BUY ROC** | **Pooled Global BUY PR** | **Pooled BUY PR Lift** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Training-Prior Baseline** | 0.5000 | 0.5000 | 0.2035 | 0.4838 | 0.4558 | 0.1870 | 0.92x |
| **Control Raw XGBoost** | 0.5198 | 0.5316 | 0.2325 | 0.4847 | 0.5216 | 0.2223 | 1.09x |
| **Candidate A: Temperature Scaling** | 0.5217 | 0.5319 | 0.2324 | 0.4876 | 0.5238 | 0.2252 | 1.11x |
| **Candidate B: Dirichlet Calibration** | 0.5202 | 0.5319 | 0.2331 | 0.4927 | 0.5189 | 0.2226 | 1.09x |
| **Candidate C: Bayesian Prior Shrinkage** | 0.5137 | 0.5229 | 0.2229 | 0.4826 | 0.4611 | 0.1962 | 0.96x |

### Mathematical Explanation of the Training-Prior Pooled Ranking Artifact:
1. **Within Fold:** The Training Prior predicts a constant vector $\pi_k$ for every observation in fold $k$. Consequently:
   - Within-Fold Macro ROC-AUC $\equiv 0.5000$
   - Within-Fold BUY ROC-AUC $\equiv 0.5000$
   - Within-Fold BUY PR-AUC $\equiv \text{Fold Prevalence}$ ($0.2246, 0.2343, 0.1618, 0.1932$, averaging **0.2035**).
2. **Pooled Across Folds:** Because the training window expands chronologically, the estimated training prior shifts across folds:
   - Fold 1 Train: $P(\text{BUY}) = 0.1754$
   - Fold 2 Train: $P(\text{BUY}) = 0.1907$
   - Fold 3 Train: $P(\text{BUY}) = 0.2143$
   - Fold 4 Train: $P(\text{BUY}) = 0.2005$
3. **Cross-Fold Ranking Distortion:**
   - In Fold 3, the model assigns a relatively higher constant BUY probability ($0.2143$), but the realized validation BUY rate in Fold 3 happened to be the lowest ($0.1618$ = 67/414).
   - In Fold 1, the model assigns a relatively lower constant BUY probability ($0.1754$), but the realized validation BUY rate in Fold 1 was high ($0.2246$ = 93/414).
   - When pooled globally, observations from Fold 3 are ranked *above* observations from Fold 1, despite having a lower proportion of positive labels.
   - This cross-fold stratification inverts the global ranking curve, dragging pooled BUY ROC-AUC down to **0.4558** and pooled BUY PR-AUC down to **0.1870** ($< 0.2035$).
4. **Governance Conclusion:**
   - **The Training-Prior Baseline is strictly a probability scoring benchmark (Brier / ECE / Log-Loss), NOT an economic ranking benchmark.** Cross-fold shifts in pooled ranking reflect historical regime changes rather than predictive discrimination.

---

## 5. Formal Governance Ratification & Sequencing of Next Phase

### Governance Decisions:
1. **Accept Calibration Branch Termination:**
   - Post-hoc calibration cannot recover predictive skill when underlying ranking discrimination is near-random (Within-Fold Macro ROC-AUC $\approx 0.52$, BUY ROC-AUC $\approx 0.53$).
   - Temperature scaling softens overconfidence to meet ECE ($0.0506 \le 0.08$), but Brier remains $0.5872$ (BSS = -39.02% vs. prior $0.4224$).
   - The calibration branch is **PERMANENTLY CLOSED**.
2. **Release Hold & Final Numerical Sign-Off:**
   - Numerical reconciliation between draft placeholders and authoritative archive `1e257def5ce0d89657b241d9a58c56efd8ec92ec985fd3c273a0b208d76b7d32` is complete and verified to machine precision ($0.00 \times 10^{-12}$).
3. **Maintain Strict Production & Capital NO-GO:**
   - Strategy manifest `frozen_strategy_manifest_v2.4.json` remains immutable.
   - Main branch production integration remains **STRICTLY BLOCKED**.
   - Live capital allocation is **STRICT NO-GO**.

### Sequenced Next Phase Workstreams (Target & Feature Forensics):
```mermaid
graph TD
    WS1["1. Target & Leakage Audit<br/>Barrier collisions, timestamps, horizons"] --> WS2["2. Rule-Based Benchmarks<br/>200-day SMA, 20-day breakout"]
    WS2 --> WS3["3. Feature Causality & Quality<br/>Rank correlation, out-of-sample skill"]
    WS3 --> WS4["4. Prospective Holdout Gate<br/>Screening: ROC-AUC > 0.55 & BSS > 0"]
    WS4 --> WS5["5. Trading Simulation Gate<br/>Only after statistical skill is verified"]
```

1. **Workstream 1 (Audit Label Construction & Leakage):**
   - Audit triple-barrier label generation: analyze same-bar high/low barrier ambiguity and same-bar collisions under intraday tick/bar resolution.
   - Point-in-time causality audit: verify publication timestamps and rolling window backward-looking constraints for SPY SMA200 and VIX features.
   - Evaluate whether the 3-class triple-barrier formulation creates an artificial HOLD minority ($<10\%$) that distorts tree split criteria.
2. **Workstream 2 (Test Simple Directional Baselines):**
   - Evaluate predeclared trend-following (200-day SMA) and breakout (20-day Donchian) rules under identical data, cost, and execution conventions.
3. **Workstream 3 (Test Representation Quality & Feature Causality):**
   - Measure out-of-sample ranking discrimination and rank correlations of candidate features before introducing model complexity.
4. **Workstream 4 (Prospective Screening Gates):**
   - Enforce predeclared research screening criteria ($\text{ROC-AUC} > 0.55$ and $\text{BSS} > 0.00\%$) on prospective holdout windows.
5. **Workstream 5 (Trading Simulations):**
   - Proceed to cost-adjusted backtests and paper execution only after statistical skill is definitively established.
