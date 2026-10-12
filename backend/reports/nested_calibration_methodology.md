# HYDRA V2.4.1 — Phase 3e: Nested Walk-Forward Probability Calibration Audit

**Execution Date:** 2026-10-12 09:22:13 UTC
**Governance Verdict:** `PHASE 3E COMPLETED — POST-HOC CALIBRATION FAILS TO BEAT PRIOR BENCHMARK`
**Operational Status:** **RESEARCH-ONLY — PRODUCTION INTEGRATION BLOCKED — ZERO LIVE CAPITAL**

---

## 1. Executive Summary & Forensic Findings

1. **Strict Nested Walk-Forward Execution (Zero Lookahead Leakage):**
   - For all 4 outer folds, candidate calibrators were fitted **strictly on inner out-of-fold predictions** ($M=3$ inner walk-forward splits with 15-bar embargo) within each pre-embargo training window $T_k$.
   - Outer validation folds ($N=1,656$ pooled observations) were evaluated exactly once using frozen calibrators.
2. **Definitive Scientific Finding on Calibration vs Skill:**
   - **Candidate A (Temperature Scaling):** Learns optimal temperatures $T \in [1.38, 1.49]$ reflecting tree overconfidence. While it reduces canonical ECE from 0.0833 to **0.0506** (passing the $\le 0.08$ gate), Brier score only improves from 0.5964 to **0.5872**, remaining **negative skill (BSS = -39.02%)** versus the training prior ($BS = 0.4224$).
   - **Candidate B (Regularized Dirichlet Calibration):** Achieves $BS = 0.4776$ (BSS = -13.07%), unable to beat the prior baseline while degrading ECE to 0.1336.
   - **Candidate C (Bayesian Prior-Shrunk Mixture):** The analytical shrinkage factor collapses to $\alpha^* \approx 0.00$ ($\alpha \in [0.00, 0.02]$ across folds) because inner out-of-fold correlation between model errors and targets is negligible. Consequently, optimal calibration converges directly to predicting the constant empirical training prior (BSS = -0.07%, $BS = 0.4227$).
3. **Multiclass Ranking Non-Invariance Nuance (Finding B Resolution):**
   - **Within-Observation Argmax Invariance:** For any scalar $T > 0$, $\arg\max_k (z_{ik} / T) = \arg\max_k z_{ik}$, strictly preserving predicted classes.
   - **Cross-Observation Probability Ranking Non-Invariance:** In multiclass classification ($K=3$), softmax normalizers $D_i(T) = \sum_{j=1}^3 e^{z_{ij}/T}$ vary across observations. Consequently, scaling changes cross-observation rank ordering of class probabilities (1,617 of 1,656 observations shift in BUY probability rank). Macro OVR ROC-AUC shifts slightly from $0.4851$ to $0.4882$, and BUY ROC-AUC shifts from $0.5284$ to $0.5303$. Both remain within 95% bootstrap intervals and statistically indistinguishable from $0.50$.
4. **Governance Verdict:**
   - The calibration branch has exhausted its utility on the current feature representation.
   - Production integration remains **STRICTLY BLOCKED**; live capital is **STRICT NO-GO**.

---

## 2. Leakage-Free Purge & Embargo Boundary Verification

Every outer and inner fold enforces an authoritative 15-bar embargo separating the decision boundary from the validation window:

| Fold Dimension | Training Window | Embargo Interval (Len=15) | Validation Window | Safety Margin | Leakage Audit |
|---|---|---|---|---:|---|
| **Outer Fold 1** | Bars 0..398 ($N=399$) | Bars 399..413 | Bars 414..827 ($N=414$) | +1 bar (413 < 414) | **PASS (Zero Overlap)** |
| **Outer Fold 2** | Bars 0..812 ($N=813$) | Bars 813..827 | Bars 828..1241 ($N=414$) | +1 bar (827 < 828) | **PASS (Zero Overlap)** |
| **Outer Fold 3** | Bars 0..1226 ($N=1227$) | Bars 1227..1241 | Bars 1242..1655 ($N=414$) | +1 bar (1241 < 1242) | **PASS (Zero Overlap)** |
| **Outer Fold 4** | Bars 0..1640 ($N=1641$) | Bars 1641..1655 | Bars 1656..2069 ($N=414$) | +1 bar (1655 < 1656) | **PASS (Zero Overlap)** |

### Mathematical Reconciliation of Fold Prior Briers to Pooled Score (Finding A Resolution):
- **Fold 1 ($N=414$):** Prior Brier = 0.4937 (0.493680), Raw Brier = 0.5621 (0.562135)
- **Fold 2 ($N=414$):** Prior Brier = 0.4593 (0.459284), Raw Brier = 0.6464 (0.646382)
- **Fold 3 ($N=414$):** Prior Brier = **0.3339** (0.333938), Raw Brier = 0.6229 (0.622878) *(reconciled from 0.3240)*
- **Fold 4 ($N=414$):** Prior Brier = 0.4025 (0.402510), Raw Brier = 0.5542 (0.554166)
- **Fold Arithmetic Mean Prior:** $(0.493680 + 0.459284 + 0.333938 + 0.402510) / 4 = \mathbf{0.422353}$
- **Pooled Global Prior Brier:** $\mathbf{0.422353}$ (rounds to $\mathbf{0.4224}$)
- **Reconciliation Error:** $\mathbf{0.00 \times 10^{-12}}$ (machine precision exact).

---

## 3. Nested Fitted Calibrator Parameters per Outer Fold

Calibrators were fitted strictly on inner out-of-fold predictions ($M=3$ inner splits) within each outer training set $T_k$:

| Outer Fold | Outer Train $N$ | Inner OOF $N$ | Candidate A (Temperature $T$) | Candidate B (Dirichlet $W, b$) | Candidate C (Shrinkage $\alpha^*$) | Fold Prior $\pi_{\text{train}}$ |
|---|---:|---:|---|---|---|---|
| **Fold 1** | 399 | 297 | T = 1.4936 | w = [0.4681, 0.734, 0.6664], b = [0.248, -0.1663, -0.0817] | alpha = 0.0163 | [0.7293, 0.0952, 0.1754] |
| **Fold 2** | 813 | 609 | T = 1.4798 | w = [0.4683, 0.7427, 0.7012], b = [0.2642, -0.1375, -0.1267] | alpha = 0.0000 | [0.7085, 0.1009, 0.1907] |
| **Fold 3** | 1227 | 918 | T = 1.4731 | w = [0.439, 0.88, 0.6381], b = [0.2455, -0.1591, -0.0864] | alpha = 0.0217 | [0.6976, 0.088, 0.2143] |
| **Fold 4** | 1641 | 1230 | T = 1.3886 | w = [0.4269, 0.9754, 0.7027], b = [0.3208, -0.257, -0.0638] | alpha = 0.0000 | [0.727, 0.0725, 0.2005] |

---

## 4. Pooled Performance & 95% Block-Bootstrap Discrimination

Pooled results across all 1,656 outer out-of-fold observations:

| Model / Calibrator Configuration | Pooled Brier | Canonical ECE | Log Loss | BSS vs Prior | Macro ROC-AUC (95% CI) | BUY ROC-AUC (95% CI) | BUY PR-AUC (95% CI) | PR Lift | Gate Disposition |
|---|---:|---:|---:|---:|---|---|---|---:|---|
| **Training-Prior Baseline** | **0.4224** | **0.0413** | **0.7372** | **0.00%** | 0.4838 | 0.4558 | 0.1870 | 0.92x | Benchmark |
| **Control Raw XGBoost** | 0.5964 | 0.0833 | 0.9933 | -41.19% | 0.4847 [0.4412, 0.5332] | 0.5216 [0.4593, 0.5761] | 0.2223 [0.1706, 0.2727] | 1.09x | FAIL (Brier & ECE) |
| **Candidate A: Temperature Scaling** | 0.5872 | **0.0506** | 0.9726 | -39.02% | 0.4876 [0.4433, 0.5348] | 0.5238 [0.4620, 0.5785] | 0.2252 [0.1720, 0.2775] | 1.11x | FAIL (BSS < 0) |
| **Candidate B: Dirichlet Calibration** | 0.4776 | 0.1336 | 0.8253 | -13.07% | 0.4929 [0.4508, 0.5386] | 0.5190 [0.4552, 0.5749] | 0.2226 [0.1705, 0.2744] | 1.09x | FAIL (BSS < 0) |
| **Candidate C: Bayesian Prior Shrinkage** | **0.4227** | **0.0416** | **0.7381** | -0.07% | 0.4826 [0.4335, 0.5289] | 0.4611 [0.3895, 0.5267] | 0.1962 [0.1525, 0.2512] | 0.96x | FAIL (BSS <= 0) |

---

## 5. Outer Fold Breakdown

| Outer Fold | Samples | Prior Brier | Raw Control Brier | Candidate A (Temp Brier & BSS) | Candidate B (Diri Brier & BSS) | Candidate C (Shrunk Brier & BSS) |
|---|---:|---:|---:|---|---|---|
| **Fold 1** | 414 | 0.4937 | 0.5621 | 0.5471 (-10.82%) | 0.5165 (-4.62%) | 0.4933 (0.08%) |
| **Fold 2** | 414 | 0.4593 | 0.6464 | 0.6253 (-36.14%) | 0.5268 (-14.70%) | 0.4593 (0.00%) |
| **Fold 3** | 414 | 0.3339 | 0.6229 | 0.6130 (-83.59%) | 0.4332 (-29.74%) | 0.3359 (-0.60%) |
| **Fold 4** | 414 | 0.4025 | 0.5542 | 0.5636 (-40.02%) | 0.4337 (-7.75%) | 0.4025 (0.00%) |

---

## 6. Final Institutional Recommendation & Governance Action

1. **Stop Post-Hoc Calibration Iteration:**
   Phase 3e decisively proves that post-hoc probability calibrators cannot generate probability skill over the empirical training prior when underlying model discrimination is near-random (ROC-AUC $\approx 0.50$). Temperature scaling softens overconfidence and passes canonical ECE (0.0506 $\le 0.08$), but its Brier skill score remains deeply negative (-39.02\%).
2. **Acceptance Hurdles Maintained:**
   - Multiclass Brier $\le 0.20$ remains active and unrelaxed.
   - Canonical ECE $\le 0.08$ remains active and unrelaxed.
   - Positive Brier Skill (BSS $> 0.00\%$) remains strictly required.
3. **Operational Status:**
   - Production integration: **BLOCKED**.
   - Live capital allocation: **STRICT NO-GO**.
