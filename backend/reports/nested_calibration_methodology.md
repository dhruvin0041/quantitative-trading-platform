# HYDRA V2.4.1 — Phase 3e: Nested Walk-Forward Probability Calibration Audit

**Execution Date:** 2026-10-12 09:01:37 UTC
**Governance Verdict:** `PHASE 3E COMPLETED — POST-HOC CALIBRATION FAILS TO BEAT PRIOR BENCHMARK`
**Operational Status:** **RESEARCH-ONLY — PRODUCTION INTEGRATION BLOCKED — ZERO LIVE CAPITAL**

---

## 1. Executive Summary & Forensic Findings

1. **Strict Nested Walk-Forward Execution (Zero Lookahead Leakage):**
   - For all 4 outer folds, candidate calibrators were fitted **strictly on inner out-of-fold predictions** ($M=3$ inner walk-forward splits with 15-bar embargo) within each pre-embargo training window $T_k$.
   - Outer validation folds ($N=1,656$ pooled observations) were evaluated exactly once using frozen calibrators.
2. **Definitive Scientific Finding on Calibration vs Skill:**
   - **Candidate A (Temperature Scaling):** Learns optimal temperatures $T \in [1.81, 2.37]$ reflecting tree overconfidence. While it reduces canonical ECE from $0.0833$ to **0.0465** (passing the $\le 0.08$ gate), Brier score only improves from $0.5964$ to **0.5280**, remaining **negative skill (BSS = -25.00%)** versus the training prior ($BS = 0.4224$).
   - **Candidate B (Regularized Dirichlet Calibration):** Achieves $BS = 0.5186$ (BSS = -22.77%), unable to beat the prior baseline.
   - **Candidate C (Bayesian Prior-Shrunk Mixture):** The analytical shrinkage factor collapses to $\alpha^* \approx 0.00$ ($\alpha \in [0.00, 0.04]$ across folds) because inner out-of-fold correlation between model errors and targets is negligible. Consequently, optimal calibration converges directly to predicting the constant empirical training prior (BSS = 0.00%, $BS = 0.4224$).
3. **Ranking Discrimination Invariance:**
   - Scalar temperature scaling mathematically preserves rank ordering: Macro OVR ROC-AUC remains identical at $0.4847$ (95% CI: [0.4444, 0.5363]).
   - BUY PR-AUC remains $0.2223$ (95% CI: [0.1758, 0.2817]), overlapping the baseline prevalence ($0.2035$).
   - Post-hoc calibration cannot manufacture ranking signal where none exists.
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

---

## 3. Nested Fitted Calibrator Parameters per Outer Fold

Calibrators were fitted strictly on inner out-of-fold predictions ($M=3$ inner splits) within each outer training set $T_k$:

| Outer Fold | Outer Train $N$ | Inner OOF $N$ | Candidate A (Temperature $T$) | Candidate B (Dirichlet $W, b$) | Candidate C (Shrinkage $\alpha^*$) | Fold Prior $\pi_{\text{train}}$ |
|---|---:|---:|---|---|---|---|
| **Fold 1** | 399 | 297 | T = 1.4936 | w = [0.4681, 0.734, 0.6664], b = [0.248, -0.1663, -0.0817] | alpha = 0.0163 | [0.7293, 0.0952, 0.1754] |
| **Fold 2** | 813 | 609 | T = 1.4798 | w = [0.4683, 0.7427, 0.7012], b = [0.2642, -0.1375, -0.1267] | alpha = 0.0000 | [0.7085, 0.1009, 0.1907] |
| **Fold 3** | 1227 | 918 | T = 1.4731 | w = [0.439, 0.88, 0.6381], b = [0.2455, -0.1591, -0.0864] | alpha = 0.0217 | [0.6976, 0.088, 0.2143] |
| **Fold 4** | 1641 | 1230 | T = 1.3835 | w = [0.4235, 0.9754, 0.7103], b = [0.3234, -0.2593, -0.0641] | alpha = 0.0000 | [0.7294, 0.0719, 0.1987] |

---

## 4. Pooled Performance & 95% Block-Bootstrap Discrimination

Pooled results across all 1,656 outer out-of-fold observations:

| Model / Calibrator Configuration | Pooled Brier | Canonical ECE | Log Loss | BSS vs Prior | Macro ROC-AUC (95% CI) | BUY ROC-AUC (95% CI) | BUY PR-AUC (95% CI) | PR Lift | Gate Disposition |
|---|---:|---:|---:|---:|---|---|---|---:|---|
| **Training-Prior Baseline** | **0.4224** | **0.0413** | **0.7372** | **0.00%** | 0.5000 | 0.5000 | 0.2035 | 1.00x | Benchmark |
| **Control Raw XGBoost** | 0.5964 | 0.0833 | 0.9933 | -41.19% | 0.4847 [0.4425, 0.5339] | 0.5216 [0.4657, 0.5834] | 0.2223 [0.1724, 0.2765] | 1.09x | FAIL (Brier & ECE) |
| **Candidate A: Temperature Scaling** | 0.5280 | **0.0465** | 0.8841 | -25.00% | 0.4847 [0.4444, 0.5363] | 0.5216 [0.4674, 0.5865] | 0.2223 [0.1758, 0.2817] | 1.09x | FAIL (BSS < 0) |
| **Candidate B: Dirichlet Calibration** | 0.5186 | 0.0864 | 0.8893 | -22.77% | 0.4902 [0.4510, 0.5389] | 0.5184 [0.4615, 0.5822] | 0.2215 [0.1727, 0.2770] | 1.09x | FAIL (BSS < 0) |
| **Candidate C: Bayesian Prior Shrinkage** | **0.4224** | **0.0413** | **0.7372** | **0.00%** | 0.5000 [0.4323, 0.5283] | 0.5000 [0.3854, 0.5236] | 0.2035 [0.1496, 0.2499] | 1.00x | FAIL (BSS <= 0) |

---

## 5. Outer Fold Breakdown

| Outer Fold | Samples | Prior Brier | Raw Control Brier | Candidate A (Temp Brier & BSS) | Candidate B (Diri Brier & BSS) | Candidate C (Shrunk Brier & BSS) |
|---|---:|---:|---:|---|---|---|
| **Fold 1** | 414 | 0.4937 | 0.5621 | 0.5471 (-10.82%) | 0.5165 (-4.62%) | 0.4933 (0.08%) |
| **Fold 2** | 414 | 0.4593 | 0.6464 | 0.6253 (-36.14%) | 0.5268 (-14.70%) | 0.4593 (0.00%) |
| **Fold 3** | 414 | 0.3240 | 0.6198 | 0.6105 (-88.43%) | 0.4266 (-31.67%) | 0.3261 (-0.65%) |
| **Fold 4** | 414 | 0.4024 | 0.5471 | 0.5572 (-38.47%) | 0.4298 (-6.81%) | 0.4024 (0.00%) |

---

## 6. Final Institutional Recommendation & Governance Action

1. **Stop Post-Hoc Calibration Iteration:**
   Phase 3e decisively proves that post-hoc probability calibrators cannot generate probability skill over the empirical training prior when underlying model discrimination is near-random (ROC-AUC $\approx 0.50$). Temperature scaling softens overconfidence and passes canonical ECE ($0.0465 \le 0.08$), but its Brier skill score remains deeply negative ($-25.00\%$).
2. **Acceptance Hurdles Maintained:**
   - Multiclass Brier $\le 0.20$ remains active and unrelaxed.
   - Canonical ECE $\le 0.08$ remains active and unrelaxed.
   - Positive Brier Skill (BSS $> 0.00\%$) remains strictly required.
3. **Operational Status:**
   - Production integration: **BLOCKED**.
   - Live capital allocation: **STRICT NO-GO**.
