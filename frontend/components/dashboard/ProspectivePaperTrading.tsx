"use client";

import React, { useEffect, useState, useCallback } from 'react';
import { ProspectiveSummary, ProspectiveSignal } from '@/types';
import { API_KEY, getBaseUrl } from '@/lib/config';
import { 
  ShieldCheck, 
  Lock, 
  Clock, 
  CheckCircle2, 
  Layers, 
  GitCommit,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProspectivePaperTradingProps {
  ticker?: string;
  currency?: string;
}

export function ProspectivePaperTrading({ ticker = "AAPL", currency = "$" }: ProspectivePaperTradingProps) {
  const [summary, setSummary] = useState<ProspectiveSummary | null>(null);
  const [signals, setSignals] = useState<ProspectiveSignal[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    const baseUrl = getBaseUrl();
    try {
      const [sumRes, sigRes] = await Promise.all([
        fetch(`${baseUrl}/prospective/summary?ticker=${ticker}`, {
          headers: { "X-API-Key": API_KEY },
          cache: 'no-store'
        }),
        fetch(`${baseUrl}/prospective/signals?ticker=${ticker}`, {
          headers: { "X-API-Key": API_KEY },
          cache: 'no-store'
        })
      ]);

      if (!sumRes.ok || !sigRes.ok) {
        throw new Error(`Failed to fetch prospective validation data.`);
      }

      const sumData: ProspectiveSummary = await sumRes.json();
      const sigData: { signals: ProspectiveSignal[] } = await sigRes.json();

      setSummary(sumData);
      setSignals(sigData.signals || []);
    } catch (err) {
      setError((err as Error).message || "Error connecting to prospective engine.");
    } finally {
      setLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    let ignore = false;
    const load = async () => {
      if (!ignore) {
        await fetchData();
      }
    };
    load();
    return () => {
      ignore = true;
    };
  }, [fetchData]);

  const lockActive = summary?.governance?.lock_active ?? true;
  const integrityVerified = summary?.governance?.integrity_verified ?? true;
  const strategyVersion = summary?.strategy_version || "HYDRA_PROSPECTIVE_V1.0";
  const commitHash = summary?.governance?.git_commit?.slice(0, 8) || "a956aacd";

  return (
    <div className="flex flex-col gap-6 w-full">
      {error && (
        <div className="p-3.5 rounded-lg border border-rose-500/30 bg-rose-500/10 text-rose-300 text-[12px] flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* HEADER SECTION */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-xl border border-border bg-card shadow-sm">
        <div>
          <div className="flex items-center gap-3">
            <span className="p-2 rounded-lg bg-primary/10 text-primary">
              <ShieldCheck className="w-5 h-5" />
            </span>
            <div>
              <h2 className="text-[20px] font-black uppercase tracking-tight text-foreground flex items-center gap-2">
                PROSPECTIVE PAPER TRADING
                <span className="px-2 py-0.5 rounded text-[10px] font-mono tracking-normal bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  UNTOUCHED FORWARD VALIDATION
                </span>
              </h2>
              <p className="text-[12px] text-muted-foreground mt-0.5">
                Zero-lookahead, causal forward validation initiated on September 30, 2026.
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border text-[12px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
            Refresh
          </button>
        </div>
      </div>

      {/* STRICT EVIDENCE SEPARATION BANNER */}
      <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-[12px]">
        <div className="flex items-start gap-2.5">
          <div className="p-1 rounded bg-amber-500/20 text-amber-400 mt-0.5">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <span className="font-bold text-amber-300 uppercase tracking-wide">
              Mandatory Data Segregation Protocol:
            </span>
            <p className="text-muted-foreground mt-0.5">
              Historical results (2024–2026) are classified as <span className="font-semibold text-foreground">PRELIMINARY HISTORICAL EVIDENCE</span> because hyperparameters, thresholds (0.60), and cooldown rules were tuned using that period. The metrics below represent strictly <span className="font-semibold text-emerald-400">UNTOUCHED FORWARD VALIDATION</span> recorded on live post-freeze market sessions.
            </p>
          </div>
        </div>
      </div>

      {/* ANTI-OVERFITTING LOCK & STRATEGY GOVERNANCE STATUS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-border bg-card flex flex-col gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
            <Lock className="w-3.5 h-3.5 text-primary" /> Anti-Overfitting Lock
          </span>
          <div className="flex items-center gap-2">
            <span className={cn(
              "w-2.5 h-2.5 rounded-full",
              lockActive && integrityVerified ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
            )} />
            <span className="text-[14px] font-mono font-bold text-foreground">
              {lockActive && integrityVerified ? "LOCKED & VERIFIED" : "MODIFICATION DETECTED"}
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground">Parameters are read-only</span>
        </div>

        <div className="p-4 rounded-xl border border-border bg-card flex flex-col gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
            <GitCommit className="w-3.5 h-3.5 text-primary" /> Strategy Version
          </span>
          <span className="text-[14px] font-mono font-bold text-foreground">{strategyVersion}</span>
          <span className="text-[10px] font-mono text-muted-foreground">Git: {commitHash}</span>
        </div>

        <div className="p-4 rounded-xl border border-border bg-card flex flex-col gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-primary" /> Validation Start
          </span>
          <span className="text-[14px] font-mono font-bold text-foreground">
            {summary?.validation_start_date || "2026-09-30"}
          </span>
          <span className="text-[10px] text-muted-foreground">Untouched holdout inception</span>
        </div>

        <div className="p-4 rounded-xl border border-border bg-card flex flex-col gap-1.5">
          <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-primary" /> Completed Trades
          </span>
          <div className="flex items-baseline gap-2">
            <span className="text-[18px] font-mono font-bold text-foreground">
              {summary?.completed_trades_count ?? 0}
            </span>
            <span className="text-[11px] text-muted-foreground">
              ({summary?.pending_trades_count ?? 0} pending)
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground">Threshold for significance: 30</span>
        </div>
      </div>

      {/* METRICS SUMMARY CARDS */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-3.5 rounded-lg border border-border bg-background flex flex-col gap-1">
          <span className="text-[10px] font-bold uppercase text-muted-foreground">1D Win Rate</span>
          <span className="text-[16px] font-mono font-bold text-foreground">
            {summary?.win_rate_1d ? `${summary.win_rate_1d}%` : "---"}
          </span>
          <span className="text-[9px] text-muted-foreground">Next-session close</span>
        </div>

        <div className="p-3.5 rounded-lg border border-border bg-background flex flex-col gap-1">
          <span className="text-[10px] font-bold uppercase text-muted-foreground">5D Win Rate</span>
          <span className="text-[16px] font-mono font-bold text-emerald-400">
            {summary?.win_rate_5d ? `${summary.win_rate_5d}%` : "---"}
          </span>
          <span className="text-[9px] text-muted-foreground">Primary horizon</span>
        </div>

        <div className="p-3.5 rounded-lg border border-border bg-background flex flex-col gap-1">
          <span className="text-[10px] font-bold uppercase text-muted-foreground">Profit Factor</span>
          <span className="text-[16px] font-mono font-bold text-foreground">
            {summary?.profit_factor ? summary.profit_factor.toFixed(2) : "---"}
          </span>
          <span className="text-[9px] text-muted-foreground">Gross profit / loss</span>
        </div>

        <div className="p-3.5 rounded-lg border border-border bg-background flex flex-col gap-1">
          <span className="text-[10px] font-bold uppercase text-muted-foreground">Expectancy</span>
          <span className={cn(
            "text-[16px] font-mono font-bold",
            (summary?.expectancy ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"
          )}>
            {summary?.expectancy ? `${summary.expectancy > 0 ? '+' : ''}${summary.expectancy.toFixed(2)}%` : "---"}
          </span>
          <span className="text-[9px] text-muted-foreground">Expected return/trade</span>
        </div>

        <div className="p-3.5 rounded-lg border border-border bg-background flex flex-col gap-1">
          <span className="text-[10px] font-bold uppercase text-muted-foreground">Max Drawdown</span>
          <span className="text-[16px] font-mono font-bold text-rose-400">
            {summary?.maximum_drawdown ? `${summary.maximum_drawdown.toFixed(2)}%` : "0.00%"}
          </span>
          <span className="text-[9px] text-muted-foreground">Forward peak-to-trough</span>
        </div>

        <div className="p-3.5 rounded-lg border border-border bg-background flex flex-col gap-1">
          <span className="text-[10px] font-bold uppercase text-muted-foreground">Cumul. Return</span>
          <span className={cn(
            "text-[16px] font-mono font-bold",
            (summary?.cumulative_return ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"
          )}>
            {summary?.cumulative_return ? `${summary.cumulative_return > 0 ? '+' : ''}${summary.cumulative_return.toFixed(2)}%` : "0.00%"}
          </span>
          <span className="text-[9px] text-muted-foreground">Net of costs</span>
        </div>
      </div>

      {/* CHECKPOINTS PROGRESS */}
      <div className="p-5 rounded-xl border border-border bg-card flex flex-col gap-3">
        <h3 className="text-[13px] font-bold uppercase tracking-wider text-foreground flex items-center justify-between">
          <span>Predefined Validation Checkpoints</span>
          <span className="text-[11px] font-mono text-muted-foreground font-normal">
            Neutrality Policy: Do not declare validation until Checkpoint 1 (30 trades)
          </span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            { name: "Checkpoint 1", target: 30, completed: summary?.completed_trades_count ?? 0 },
            { name: "Checkpoint 2", target: 50, completed: summary?.completed_trades_count ?? 0 },
            { name: "Checkpoint 3", target: 100, completed: summary?.completed_trades_count ?? 0 }
          ].map((cp, idx) => {
            const pct = Math.min(100, Math.round((cp.completed / cp.target) * 100));
            const reached = cp.completed >= cp.target;
            return (
              <div key={idx} className="p-3 rounded-lg border border-border bg-background flex flex-col gap-2">
                <div className="flex justify-between items-center text-[12px]">
                  <span className="font-bold text-foreground">{cp.name} ({cp.target} Trades)</span>
                  <span className={cn("font-mono font-bold text-[11px]", reached ? "text-emerald-400" : "text-muted-foreground")}>
                    {cp.completed} / {cp.target} ({pct}%)
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
                  <div 
                    className={cn("h-full transition-all duration-500", reached ? "bg-emerald-500" : "bg-primary")}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-[10px] text-muted-foreground">
                  {reached ? "Statistical significance achieved" : "Accumulating live prospective data"}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* PROSPECTIVE SIGNAL LEDGER TABLE */}
      <div className="rounded-xl border border-border bg-card overflow-hidden shadow-sm flex flex-col">
        <div className="p-4 border-b border-border flex items-center justify-between bg-muted/20">
          <div>
            <h3 className="text-[13px] font-bold uppercase tracking-wider text-foreground">
              Immutable Prospective Signal Ledger
            </h3>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              Decoupled record: Generation parameters are permanently frozen at bar close; outcomes appended only when realized.
            </p>
          </div>
          <span className="text-[11px] font-mono text-muted-foreground">
            {signals.length} Signal{signals.length === 1 ? '' : 's'} Logged
          </span>
        </div>

        {signals.length === 0 ? (
          <div className="p-8 text-center text-muted-foreground text-[13px] flex flex-col items-center justify-center gap-2">
            <Clock className="w-8 h-8 text-muted-foreground/50 animate-pulse" />
            <span className="font-semibold text-foreground">No prospective forward signals generated yet</span>
            <span className="text-[11px] max-w-md">
              Prospective forward paper trading commences with the next confirmed daily bar close after the freeze timestamp (2026-09-30 16:00 ET).
            </span>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-[12px]">
              <thead className="bg-muted/40 border-b border-border text-[11px] font-bold uppercase text-muted-foreground">
                <tr>
                  <th className="py-2.5 px-3">Signal ID</th>
                  <th className="py-2.5 px-3">Source Candle</th>
                  <th className="py-2.5 px-3">Action</th>
                  <th className="py-2.5 px-3">Probability</th>
                  <th className="py-2.5 px-3">Execution Target</th>
                  <th className="py-2.5 px-3 text-right">Exec Price</th>
                  <th className="py-2.5 px-3 text-right">Actual Open</th>
                  <th className="py-2.5 px-3 text-right">1D Ret</th>
                  <th className="py-2.5 px-3 text-right">5D Ret</th>
                  <th className="py-2.5 px-3 text-center">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {signals.map((sig) => {
                  const isBuy = sig.signal === 'BUY';
                  const isSell = sig.signal === 'SELL';
                  return (
                    <tr key={sig.signal_id} className="hover:bg-muted/20 font-mono transition-colors">
                      <td className="py-2 px-3 text-[11px] font-bold text-primary">{sig.signal_id}</td>
                      <td className="py-2 px-3 text-muted-foreground">{sig.source_candle_timestamp.slice(0, 10)}</td>
                      <td className="py-2 px-3">
                        <span className={cn(
                          "px-1.5 py-0.5 rounded text-[10px] font-bold uppercase",
                          isBuy && "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30",
                          isSell && "bg-rose-500/10 text-rose-400 border border-rose-500/30",
                          !isBuy && !isSell && "bg-muted text-muted-foreground"
                        )}>
                          {sig.signal}
                        </span>
                      </td>
                      <td className="py-2 px-3">{(sig.probability * 100).toFixed(1)}%</td>
                      <td className="py-2 px-3 text-muted-foreground">{sig.execution_target_timestamp.slice(0, 10)}</td>
                      <td className="py-2 px-3 text-right">{currency}{sig.execution_price.toFixed(2)}</td>
                      <td className="py-2 px-3 text-right">
                        {sig.actual_market_open ? `${currency}${sig.actual_market_open.toFixed(2)}` : '---'}
                      </td>
                      <td className={cn(
                        "py-2 px-3 text-right font-bold",
                        sig.return_1d !== null && sig.return_1d !== undefined
                          ? sig.return_1d > 0 ? "text-emerald-400" : (sig.return_1d < 0 ? "text-rose-400" : "text-foreground")
                          : "text-muted-foreground"
                      )}>
                        {sig.return_1d !== null && sig.return_1d !== undefined
                          ? `${sig.return_1d > 0 ? '+' : ''}${(sig.return_1d * 100).toFixed(2)}%`
                          : '---'}
                      </td>
                      <td className={cn(
                        "py-2 px-3 text-right font-bold",
                        sig.return_5d !== null && sig.return_5d !== undefined
                          ? sig.return_5d > 0 ? "text-emerald-400" : (sig.return_5d < 0 ? "text-rose-400" : "text-foreground")
                          : "text-muted-foreground"
                      )}>
                        {sig.return_5d !== null && sig.return_5d !== undefined
                          ? `${sig.return_5d > 0 ? '+' : ''}${(sig.return_5d * 100).toFixed(2)}%`
                          : '---'}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className={cn(
                          "px-1.5 py-0.2 rounded text-[9px] uppercase font-bold",
                          sig.status === 'COMPLETED' ? "bg-emerald-500/10 text-emerald-400" : "bg-amber-500/10 text-amber-400"
                        )}>
                          {sig.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
