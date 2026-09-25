"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { ChartData, PortfolioStatusResponse, ReadOnlyPosition, ReadOnlyTrailingStop } from '@/types';
import {
  Briefcase,
  Wallet,
  Activity,
  Layers,
  RefreshCw,
  Shield,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Gauge,
  Sliders,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';
import { API_KEY, getBaseUrl } from '@/lib/config';

interface PortfolioAnalyticsProps {
  data: ChartData | null;
  currency?: string;
}

export function PortfolioAnalytics({ data, currency = '$' }: PortfolioAnalyticsProps) {
  const [statusData, setStatusData] = useState<PortfolioStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<string | null>(null);

  const fetchStatus = useCallback(async (isManualRefresh = false) => {
    if (isManualRefresh) {
      setRefreshing(true);
    }

    const baseUrl = getBaseUrl();
    try {
      const res = await fetch(`${baseUrl}/api/v1/portfolio/status`, {
        headers: {
          "X-API-Key": API_KEY,
          "Accept": "application/json",
        },
        cache: 'no-store',
      });

      if (!res.ok) {
        throw new Error(`Failed to load portfolio status: HTTP ${res.status}`);
      }

      const json: PortfolioStatusResponse = await res.json();
      setStatusData(json);
      setError(null);
      setLastRefreshed(new Date().toLocaleTimeString());
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error fetching portfolio status';
      setError(msg);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    const run = async () => {
      const baseUrl = getBaseUrl();
      try {
        const res = await fetch(`${baseUrl}/api/v1/portfolio/status`, {
          headers: {
            "X-API-Key": API_KEY,
            "Accept": "application/json",
          },
          cache: 'no-store',
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json: PortfolioStatusResponse = await res.json();
        if (!ignore) {
          setStatusData(json);
          setError(null);
          setLastRefreshed(new Date().toLocaleTimeString());
        }
      } catch (err) {
        if (!ignore) {
          const msg = err instanceof Error ? err.message : 'Failed to fetch status';
          setError(msg);
        }
      } finally {
        if (!ignore) setLoading(false);
      }
    };
    run();
    return () => {
      ignore = true;
    };
  }, []);

  // If neither live statusData nor fallback data.portfolio exists
  if (!statusData && (!data || !data.portfolio)) {
    return (
      <div className="flex flex-col items-center justify-center h-56 text-muted-foreground font-mono text-[13px] uppercase tracking-widest border border-border rounded-xl bg-card gap-3">
        {loading ? (
          <div className="flex items-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin text-primary" />
            <span>Connecting to Institutional Portfolio Database...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <AlertTriangle className="w-6 h-6 text-amber-500" />
            <span>Awaiting Portfolio Telemetry</span>
            {error && <span className="text-[11px] text-destructive lowercase font-normal">{error}</span>}
            <button
              onClick={() => fetchStatus(true)}
              className="mt-2 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider bg-primary text-primary-foreground rounded hover:opacity-90 transition-opacity"
            >
              Retry Connection
            </button>
          </div>
        )}
      </div>
    );
  }

  // Determine active metrics from statusData or fallback
  const account = statusData?.account;
  const legacyPortfolio = data?.portfolio;

  const totalEquity = account?.equity ?? legacyPortfolio?.equity ?? 100000;
  const cash = account?.cash ?? legacyPortfolio?.cash ?? 100000;
  const invested = account?.invested_capital ?? (totalEquity - cash);
  const investedPct = account?.allocation_pct ?? (totalEquity > 0 ? (invested / totalEquity) * 100 : 0);
  const cashPct = account?.cash_pct ?? (totalEquity > 0 ? (cash / totalEquity) * 100 : 100);
  const buyingPower = account?.buying_power ?? cash;
  const unrealizedPnl = account?.unrealized_pnl ?? legacyPortfolio?.unrealized_pnl ?? 0;
  const unrealizedPnlPct = account?.unrealized_pnl_pct ?? 0;
  const totalPnl = account?.total_pnl ?? legacyPortfolio?.today_pnl ?? 0;
  const totalPnlPct = account?.total_pnl_pct ?? legacyPortfolio?.return_pct ?? 0;

  // Normalized positions list
  const positions: ReadOnlyPosition[] = statusData?.positions ?? (
    legacyPortfolio?.positions
      ? Object.entries(legacyPortfolio.positions)
          .filter(([, pos]) => pos.shares > 0)
          .map(([sym, pos]) => {
            const currentMark = data?.current_price && sym === data.ticker ? data.current_price : pos.avg_price;
            const posPnl = (currentMark - pos.avg_price) * pos.shares;
            const posPnlPct = pos.avg_price > 0 ? ((currentMark - pos.avg_price) / pos.avg_price) * 100 : 0;
            return {
              symbol: sym,
              qty: pos.shares,
              side: 'LONG' as const,
              avg_entry_price: pos.avg_price,
              current_price: currentMark,
              market_value: currentMark * pos.shares,
              unrealized_pnl: posPnl,
              unrealized_pnl_pct: posPnlPct,
              stop_loss: null,
              take_profit: null,
            };
          })
      : []
  );

  const trailingStops: ReadOnlyTrailingStop[] = statusData?.trailing_stops ?? [];

  return (
    <div className="flex flex-col gap-6">
      {/* HEADER / TELEMETRY STATUS BAR */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-card border border-border shadow-sm">
        <div className="flex items-center gap-3">
          <div className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[13px] font-black uppercase tracking-wider text-foreground">
                Institutional Paper Portfolio
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-primary/10 text-primary border border-primary/20">
                READ-ONLY LEDGER
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              Live SQLite synchronized execution state (zero table locks / zero neural inference overhead)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {lastRefreshed && (
            <span className="text-[11px] font-mono text-muted-foreground">
              Synced: <span className="text-foreground">{lastRefreshed}</span>
            </span>
          )}
          <button
            onClick={() => fetchStatus(true)}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-background hover:bg-muted text-[11px] font-mono font-bold uppercase tracking-wider text-foreground transition-all disabled:opacity-50"
            title="Poll real-time portfolio status from paper_trading.db"
          >
            <RefreshCw className={cn("w-3.5 h-3.5", refreshing && "animate-spin text-primary")} />
            <span>{refreshing ? 'Refreshing...' : 'Refresh Status'}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-500 text-[12px] font-mono flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>Warning: Using cached telemetry ({error})</span>
          </div>
          <button
            onClick={() => fetchStatus(true)}
            className="underline hover:text-foreground text-[11px] uppercase tracking-wider font-bold"
          >
            Retry
          </button>
        </div>
      )}

      {/* SECTION: High Level Balance Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1 shadow-sm">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <Briefcase className="w-3.5 h-3.5 text-primary" /> Total Equity
          </span>
          <span className="text-[20px] font-mono font-black text-foreground">
            {currency}{totalEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="text-[10px] text-muted-foreground font-mono">
            Initial: {currency}{(account?.initial_capital ?? 100000).toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1 shadow-sm">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <Wallet className="w-3.5 h-3.5 text-primary" /> Available Cash
          </span>
          <span className="text-[20px] font-mono font-black text-foreground">
            {currency}{cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="text-[10px] text-muted-foreground font-mono">
            BP: {currency}{buyingPower.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1 shadow-sm">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <Activity className="w-3.5 h-3.5 text-primary" /> Unrealized PnL
          </span>
          <div className="flex items-center gap-1.5">
            {unrealizedPnl >= 0 ? (
              <TrendingUp className="w-4 h-4 text-emerald-500" />
            ) : (
              <TrendingDown className="w-4 h-4 text-rose-500" />
            )}
            <span
              className={cn(
                "text-[20px] font-mono font-black",
                unrealizedPnl >= 0 ? "text-emerald-500" : "text-rose-500"
              )}
            >
              {unrealizedPnl >= 0 ? '+' : ''}{currency}
              {Math.abs(unrealizedPnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
          <span
            className={cn(
              "text-[10px] font-mono font-bold",
              unrealizedPnlPct >= 0 ? "text-emerald-500" : "text-rose-500"
            )}
          >
            {unrealizedPnlPct >= 0 ? '+' : ''}{unrealizedPnlPct.toFixed(2)}% on capital
          </span>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1 shadow-sm">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <CheckCircle2 className="w-3.5 h-3.5 text-primary" /> Net Return
          </span>
          <div className="flex items-center gap-1.5">
            {totalPnl >= 0 ? (
              <TrendingUp className="w-4 h-4 text-emerald-500" />
            ) : (
              <TrendingDown className="w-4 h-4 text-rose-500" />
            )}
            <span
              className={cn(
                "text-[20px] font-mono font-black",
                totalPnl >= 0 ? "text-emerald-500" : "text-rose-500"
              )}
            >
              {totalPnl >= 0 ? '+' : ''}{totalPnlPct.toFixed(2)}%
            </span>
          </div>
          <span
            className={cn(
              "text-[10px] font-mono",
              totalPnl >= 0 ? "text-emerald-500" : "text-rose-500"
            )}
          >
            {totalPnl >= 0 ? '+' : ''}{currency}{totalPnl.toFixed(2)} net PnL
          </span>
        </div>
      </div>

      {/* SECTION: Allocation & Capital Deployment */}
      <div className="p-5 rounded-xl bg-card border border-border shadow-sm flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-border pb-2">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary" />
            Capital Deployment & Allocation
          </h3>
          <span className="text-[11px] font-mono text-muted-foreground">
            Target Cap: 80% Max
          </span>
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-end text-[12px] font-mono">
            <span className="font-bold text-foreground">
              Invested: {currency}{invested.toLocaleString(undefined, { maximumFractionDigits: 0 })} ({investedPct.toFixed(1)}%)
            </span>
            <span className="text-muted-foreground">
              Cash: {currency}{cash.toLocaleString(undefined, { maximumFractionDigits: 0 })} ({cashPct.toFixed(1)}%)
            </span>
          </div>
          <div className="h-4 w-full bg-muted rounded-full overflow-hidden flex shadow-inner border border-border/60">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(Math.max(investedPct, 0), 100)}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className={cn(
                "h-full transition-colors",
                investedPct > 80 ? "bg-amber-500" : "bg-primary"
              )}
            />
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(Math.max(cashPct, 0), 100)}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="h-full bg-muted/80"
            />
          </div>
        </div>
      </div>

      {/* SECTION: Volatility-Adaptive Trailing Stops & Distance Gauges */}
      <div className="p-5 rounded-xl bg-card border border-border shadow-sm flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between border-b border-border pb-2 gap-2">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
            <Shield className="w-4 h-4 text-primary" />
            Volatility-Adaptive Trailing Stops & Distance Gauges
          </h3>
          <span className="text-[11px] font-mono text-muted-foreground">
            {trailingStops.length} Active Stop{trailingStops.length === 1 ? '' : 's'}
          </span>
        </div>

        {trailingStops.length === 0 ? (
          <div className="p-6 rounded-lg bg-background/50 border border-border/70 text-center flex flex-col items-center gap-2">
            <Shield className="w-8 h-8 text-muted-foreground/50" />
            <span className="text-[13px] font-mono text-muted-foreground">
              No active trailing stops currently registered.
            </span>
            <span className="text-[11px] text-muted-foreground max-w-md">
              Trailing stops are dynamically computed via ATR volatility targeting upon order fill and ratchet higher as the underlying price advances.
            </span>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {trailingStops.map((ts) => {
              const absDist = Math.abs(ts.distance_to_stop_pct);
              const isLong = ts.side.toUpperCase() === 'LONG';

              // Buffer health level
              let bufferStatus: 'HEALTHY' | 'CAUTION' | 'CRITICAL';
              let gaugeColor: string;
              let textColor: string;
              let badgeBg: string;

              if (absDist >= 5.0) {
                bufferStatus = 'HEALTHY';
                gaugeColor = 'bg-emerald-500';
                textColor = 'text-emerald-500';
                badgeBg = 'bg-emerald-500/10 text-emerald-500 border-emerald-500/30';
              } else if (absDist >= 2.5) {
                bufferStatus = 'CAUTION';
                gaugeColor = 'bg-amber-500';
                textColor = 'text-amber-500';
                badgeBg = 'bg-amber-500/10 text-amber-500 border-amber-500/30';
              } else {
                bufferStatus = 'CRITICAL';
                gaugeColor = 'bg-rose-500 animate-pulse';
                textColor = 'text-rose-500';
                badgeBg = 'bg-rose-500/10 text-rose-500 border-rose-500/30';
              }

              // Visual gauge: normalized 0 to 12% distance scale
              const gaugeFillPct = Math.min((absDist / 12.0) * 100, 100);

              // Ratchet gain from entry
              const ratchetGain = isLong
                ? ((ts.peak_trough_price - ts.entry_price) / ts.entry_price) * 100
                : ((ts.entry_price - ts.peak_trough_price) / ts.entry_price) * 100;

              return (
                <div
                  key={ts.symbol}
                  className="p-4 rounded-lg bg-background border border-border flex flex-col gap-3.5 hover:border-primary/40 transition-colors shadow-sm"
                >
                  {/* Top line: Symbol, Side, and Buffer Badge */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[16px] font-black font-mono text-foreground uppercase">
                        {ts.symbol}
                      </span>
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase",
                          isLong
                            ? "bg-emerald-500/15 text-emerald-500 border border-emerald-500/30"
                            : "bg-amber-500/15 text-amber-500 border border-amber-500/30"
                        )}
                      >
                        {ts.side}
                      </span>
                    </div>

                    <div className={cn("px-2 py-0.5 rounded text-[10px] font-mono font-bold border", badgeBg)}>
                      {bufferStatus} BUFFER
                    </div>
                  </div>

                  {/* Distance to Stop Gauge */}
                  <div className="flex flex-col gap-1.5 p-3 rounded-md bg-card/60 border border-border/50">
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-bold uppercase text-muted-foreground flex items-center gap-1.5">
                        <Gauge className="w-3.5 h-3.5 text-primary" /> Stop Distance Cushion
                      </span>
                      <span className={cn("text-[14px] font-mono font-black", textColor)}>
                        {ts.distance_to_stop_pct.toFixed(2)}%
                      </span>
                    </div>

                    {/* Progress Bar Gauge */}
                    <div className="h-2.5 w-full bg-muted rounded-full overflow-hidden p-0.5">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${gaugeFillPct}%` }}
                        transition={{ duration: 0.5, ease: "easeOut" }}
                        className={cn("h-full rounded-full transition-all", gaugeColor)}
                      />
                    </div>

                    <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground mt-0.5">
                      <span>Threshold: {isLong ? 'Below Mark' : 'Above Mark'}</span>
                      <span>Stop: {currency}{ts.stop_price.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* Multi-parameter Grid */}
                  <div className="grid grid-cols-3 gap-2 text-center pt-1 border-t border-border/40">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-muted-foreground uppercase">Entry Price</span>
                      <span className="text-[12px] font-mono font-bold text-foreground">
                        {currency}{ts.entry_price.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-muted-foreground uppercase">Peak Ratchet</span>
                      <span className="text-[12px] font-mono font-bold text-foreground">
                        {currency}{ts.peak_trough_price.toFixed(2)}
                        {ratchetGain > 0 && (
                          <span className="text-[10px] text-emerald-500 ml-1">+{ratchetGain.toFixed(1)}%</span>
                        )}
                      </span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[10px] font-bold text-muted-foreground uppercase">Multiplier</span>
                      <span className="text-[12px] font-mono font-bold text-foreground flex items-center justify-center gap-1">
                        <Sliders className="w-3 h-3 text-primary" />
                        {ts.multiplier.toFixed(2)}x ATR
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground border-t border-border/30 pt-1.5">
                    <span>ATR: {currency}{ts.atr.toFixed(2)}</span>
                    <span>Updated: {new Date(ts.updated_at).toLocaleTimeString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* SECTION: Active Open Positions Table */}
      <div className="p-5 rounded-xl bg-card border border-border shadow-sm flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-border pb-2">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2">
            <Activity className="w-4 h-4 text-primary" />
            Active Open Positions ({positions.length})
          </h3>
          <span className="text-[11px] font-mono text-muted-foreground">
            Max: 2 Concurrent
          </span>
        </div>

        {positions.length === 0 ? (
          <div className="p-6 rounded-lg bg-background/50 border border-border/70 text-center flex flex-col items-center gap-2">
            <span className="text-[13px] font-mono text-muted-foreground">
              Portfolio 100% in cash (no active open positions).
            </span>
            <span className="text-[11px] text-muted-foreground max-w-md">
              Run <code className="px-1 py-0.5 rounded bg-muted font-mono text-foreground">paper_runner.py</code> to execute the daily EOD inference cycle and dispatch systematic orders.
            </span>
          </div>
        ) : (
          <div className="border border-border rounded-lg overflow-x-auto bg-background">
            <table className="w-full text-left border-collapse min-w-[600px]">
              <thead className="bg-muted/40 border-b border-border">
                <tr>
                  <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Symbol</th>
                  <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-center">Side</th>
                  <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Shares</th>
                  <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Avg Entry</th>
                  <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Current Mark</th>
                  <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Market Value</th>
                  <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Unrealized PnL</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {positions.map((pos) => {
                  const isPos = pos.unrealized_pnl >= 0;
                  const isLong = pos.side.toUpperCase() === 'LONG';
                  const qtyStr = Number.isInteger(pos.qty) ? pos.qty.toString() : pos.qty.toFixed(2);

                  return (
                    <tr key={pos.symbol} className="hover:bg-muted/40 transition-colors font-mono">
                      <td className="px-4 py-3 text-[13px] font-bold text-foreground uppercase">
                        {pos.symbol}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span
                          className={cn(
                            "px-2 py-0.5 rounded text-[10px] font-bold uppercase",
                            isLong
                              ? "bg-emerald-500/15 text-emerald-500 border border-emerald-500/30"
                              : "bg-amber-500/15 text-amber-500 border border-amber-500/30"
                          )}
                        >
                          {pos.side}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-[13px] text-foreground text-right">
                        {qtyStr}
                      </td>
                      <td className="px-4 py-3 text-[13px] text-foreground text-right">
                        {currency}{pos.avg_entry_price.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-[13px] text-foreground text-right font-bold">
                        {currency}{pos.current_price.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-[13px] text-foreground text-right">
                        {currency}{Math.abs(pos.market_value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td
                        className={cn(
                          "px-4 py-3 text-[13px] font-bold text-right",
                          isPos ? "text-emerald-500" : "text-rose-500"
                        )}
                      >
                        {isPos ? '+' : ''}{currency}{pos.unrealized_pnl.toFixed(2)} ({pos.unrealized_pnl_pct >= 0 ? '+' : ''}{pos.unrealized_pnl_pct.toFixed(2)}%)
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
