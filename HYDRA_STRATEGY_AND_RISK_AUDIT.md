# HYDRA TRADING STRATEGY & RISK ENGINE AUDIT
**Document Version:** 2.0.0 (Post-Reconciliation Audit)  
**Classification:** Quantitative Strategy Logic, Risk Governance & Execution Frictions Audit  
**Repository Branch:** `main` (Preserving V2.2 Frozen Release)  
**Date:** October 2026

---

## 1. Executive Summary & Policy Reconciliation

> [!IMPORTANT]
> **STRATEGY POLICY RECONCILIATION:**  
> A prior draft of this document described an experimental 0.45 consensus / 0.15 delta veto policy that was contradicted by active production code. In production, [backend/src/execution/inference_service.py](file:///d:/DataScience/Projects/Data_Science_Projects/Stock_Indicator/backend/src/execution/inference_service.py) defaulted to pure XGBoost 0.60 with `veto_threshold=1.01`, effectively disabling the secondary veto.
> 
> [!IMPORTANT]
> **Operational Policy Reconciliation Across Subsystems:**
> Forensic inspection reveals distinct operational layers across HYDRA:
> 1. **Research Backtesting Engine (`backtest.py`):** Operates the target institutional consensus: Primary `XGB_AGENT` ($\ge 0.60$) + Secondary `LGBM_AGENT` asymmetric veto ($\ge 0.65$) + SPY 200 SMA Macro Regime Gate + T+1 Open execution with two-sided slippage and commissions.
> 2. **Multi-Agent Decentralized Mesh (`asset_intelligence.py`, `consensus_engine.py`):** `XGB_AGENT` is primary engine; `LGBM_AGENT` and `DQN_AGENT` remain registered as `ACTIVE` veto candidates (threshold 0.65); `DL_FUSION` is formally `QUARANTINED` (0.0 weight).
> 3. **Frozen Baseline Production Inference (`inference_service.py`):** Baseline code is frozen under anti-overfitting lock; defaults to pure XGBoost 0.60 with secondary vetoes bypassed (`veto_threshold=1.01`).
> 4. **Execution Timing:** Signals generated at bar Close ($T$, 16:00 ET) target next-session Open ($T+1$, 09:30 ET).

---

## 2. Authoritative Strategy Decision Pipeline

```
                       +---------------------------------------+
                       |    Primary Alpha Driver: XGBoost      |
                       |       P(SELL), P(HOLD), P(BUY)        |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |       PRIMARY CONVICTION HURDLE       |
                       |       Max Probability >= 0.60 ?       |
                       +---------------------------------------+
                                           | YES
                                           v
                       +---------------------------------------+
                       |     SECONDARY ASYMMETRIC VETO         |
                       | If Primary=BUY: Is LGBM P(SELL) >=0.65?|
                       | If Primary=SELL: Is LGBM P(BUY)>=0.65?|
                       +---------------------------------------+
                                           | NO (VETO NOT TRIGGERED)
                                           v
                       +---------------------------------------+
                       |         MACRO REGIME GATE             |
                       |     SPY Price >= SPY 200-day SMA?     |
                       |    (Suppresses BUY in Bear Regimes)   |
                       +---------------------------------------+
                                           | PASS
                                           v
                       +---------------------------------------+
                       |         COOLDOWN & EXPOSURE           |
                       |     Bars since last exit >= 5 ?       |
                       |     Portfolio cash sufficient ?       |
                       +---------------------------------------+
                                           | PASS
                                           v
                       +---------------------------------------+
                       |        SCHEDULE T+1 OPEN ORDER        |
                       |     Fill at Open +/- 5 bps slippage   |
                       |     Deduct $0.005/share commission    |
                       +---------------------------------------+
```

---

## 3. Dynamic Triple Barrier & Exit Risk Mechanics

Active positions are subject to three concurrent deterministic exit conditions:

### 3.1 Dynamic Volatility Barriers
1. **Take-Profit Barrier:**
   $$\text{Price}_{\text{TP}} = P_{\text{entry}} + 1.5 \times \text{ATR}_{14}(T_{\text{entry}}) \quad (\text{LONG})$$
2. **Stop-Loss Barrier:**
   $$\text{Price}_{\text{SL}} = P_{\text{entry}} - 2.0 \times \text{ATR}_{14}(T_{\text{entry}}) \quad (\text{LONG})$$
3. **Maximum Holding Horizon:**
   $$\text{Max Duration} = 15 \text{ trading sessions}$$

### 3.2 Two-Sided Friction & Execution Parity
* Both entry and exit orders execute at market Open with **5 bps adverse slippage** and brokerage commissions ($\$0.005/\text{share}$, min $\$1.00$).
* When a bar has not completed (live prospective candle), execution state is marked `PENDING_EXECUTION` with no manufactured close-derived fill price.

---

## 4. Conclusion & Governance Status
The strategy and risk management rules are now logically unified, mathematically causal, and synchronized across backend execution, backtesting, and frontend displays. The strategy is governed as **Unvalidated Research-Only**.
