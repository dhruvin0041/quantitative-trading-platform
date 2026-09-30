"use client";

import React from 'react';
import { SignalAccuracyCenter } from './SignalAccuracyCenter';
import { ModelReliabilityDashboard } from './ModelReliabilityDashboard';
import { SignalHistoryExplorer } from './SignalHistoryExplorer';
import { UserTradeJournal } from './UserTradeJournal';
import { UserVsHydraAnalytics } from './UserVsHydraAnalytics';
import { ScenarioAnalysis } from './ScenarioAnalysis';

interface PaperTradingPerformanceProps {
  currency?: string;
}

export function PaperTradingPerformance({ currency = '$' }: PaperTradingPerformanceProps) {
  return (
    <div className="flex flex-col gap-8">
      {/* MANDATORY EVIDENCE CLASSIFICATION BANNER */}
      <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 flex flex-col sm:flex-row items-start sm:items-center gap-3">
        <span className="px-2.5 py-1 rounded text-[11px] font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/30 shrink-0">
          PRELIMINARY HISTORICAL EVIDENCE
        </span>
        <p className="text-[12px] text-muted-foreground leading-relaxed">
          The backtest metrics below (2024–2026) represent <strong className="text-foreground">Preliminary Historical Evidence</strong>. Because hyperparameters, probability thresholds, and cooldown rules were tuned using this historical period, these results must not be conflated with forward holdouts. Pure untouched forward tracking is monitored under <strong className="text-emerald-400">PROSPECTIVE PAPER TRADING</strong>.
        </p>
      </div>

      {/* PHASE 1: Signal Accuracy Intelligence */}
      <section>
        <div className="mb-4">
          <h2 className="text-[16px] font-black uppercase tracking-tight text-foreground">Signal Accuracy Center</h2>
          <p className="text-[12px] text-muted-foreground mt-1">Validation of historical signal performance across various regimes and lookback windows.</p>
        </div>
        <SignalAccuracyCenter currency={currency} />
      </section>

      {/* PHASE 2: Model Reliability Engine */}
      <section className="pt-6 border-t border-border/50">
        <div className="mb-4">
          <h2 className="text-[16px] font-black uppercase tracking-tight text-foreground">Model Reliability Engine</h2>
          <p className="text-[12px] text-muted-foreground mt-1">Real-time performance drift detection and model decay tracking.</p>
        </div>
        <ModelReliabilityDashboard currency={currency} />
      </section>

      {/* PHASE 9: Generative Scenario Analysis */}
      <section className="pt-6 border-t border-border/50">
        <div className="mb-4">
          <h2 className="text-[16px] font-black uppercase tracking-tight text-foreground">Scenario Analysis & Stress Testing</h2>
          <p className="text-[12px] text-muted-foreground mt-1">Generative GAN stress testing for non-historical black swan events and regime shifts.</p>
        </div>
        <ScenarioAnalysis data={null} />
      </section>

      {/* PHASE 6: Signal History Explorer */}
      <section className="pt-6 border-t border-border/50">
        <div className="mb-4">
          <h2 className="text-[16px] font-black uppercase tracking-tight text-foreground">Signal History Explorer</h2>
          <p className="text-[12px] text-muted-foreground mt-1">Immutable ledger of all historical signals, confidence levels, and final outcomes.</p>
        </div>
        <SignalHistoryExplorer currency={currency} />
      </section>

      {/* PHASE 8: User vs Hydra Analytics */}
      <section className="pt-6 border-t border-border/50">
        <div className="mb-4">
          <h2 className="text-[16px] font-black uppercase tracking-tight text-foreground">User vs Hydra Analytics</h2>
          <p className="text-[12px] text-muted-foreground mt-1">Benchmark manual trading performance against the baseline Hydra model signals.</p>
        </div>
        <UserVsHydraAnalytics />
      </section>

      {/* PHASE 7: User Trade Journal */}
      <section className="pt-6 border-t border-border/50">
        <div className="mb-4">
          <h2 className="text-[16px] font-black uppercase tracking-tight text-foreground">User Trade Journal</h2>
          <p className="text-[12px] text-muted-foreground mt-1">Track manual executions and monitor alignment with institutional system recommendations.</p>
        </div>
        <UserTradeJournal />
      </section>
    </div>
  );
}
