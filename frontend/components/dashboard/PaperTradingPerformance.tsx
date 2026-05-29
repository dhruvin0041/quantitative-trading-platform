"use client";

import React, { useState, useEffect } from 'react';

import { Target, TrendingUp, BarChart, History, PieChart, Activity, Briefcase } from 'lucide-react';
import { cn } from '@/lib/utils';
import { API_KEY, getBaseUrl } from '@/lib/config';

interface PerformanceSummary {
  total_return: number;
  sharpe: number | string;
  sortino: number | string;
  calmar: number | string;
  max_drawdown: number;
  win_rate: number | string;
  profit_factor: number | string;
  today_pnl: number;
  mtd_pnl: number;
  ytd_pnl: number;
  total_trades: number;
  open_trades: number;
  closed_trades: number;
  winning_trades: number;
  losing_trades: number;
  initial_capital: number;
  expectancy: number | string;
  mae_avg: number;
  mfe_avg: number;
  avg_hold_duration: number;
}

interface PaperTradingPerformanceProps {
  currency?: string;
}

export function PaperTradingPerformance({ currency = '$' }: PaperTradingPerformanceProps) {
  const [performance, setPerformance] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  const API_URL = getBaseUrl();

  useEffect(() => {
    const fetchPerf = async () => {
      try {
        const res = await fetch(`${API_URL}/performance`, {
          headers: { "X-API-Key": API_KEY }
        });
        const data = await res.json();
        setPerformance(data);
      } catch (err) {
        console.error("Failed to fetch performance telemetry", err);
      } finally {
        setLoading(false);
      }
    };

    fetchPerf();
    const interval = setInterval(fetchPerf, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [API_URL]);

  if (loading) return (
    <div className="h-32 flex items-center justify-center bg-card/10 rounded-xl border border-border animate-pulse">
      <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">Initializing Performance Engine...</span>
    </div>
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const summary: PerformanceSummary = (performance as any)?.summary || {
    total_return: 0, sharpe: 0, sortino: 0, calmar: 0, max_drawdown: 0, win_rate: 0, profit_factor: 0, today_pnl: 0, mtd_pnl: 0, ytd_pnl: 0,
    total_trades: 0, open_trades: 0, closed_trades: 0, winning_trades: 0, losing_trades: 0, expectancy: 0, mae_avg: 0, mfe_avg: 0, avg_hold_duration: 0
  };

  const getMetricColor = (label: string, value: number | string) => {
    if (typeof value === 'string') return 'text-muted-foreground';
    
    if (label === 'Win Rate') {
      if (value > 55) return 'text-green-500 shadow-green-500/20';
      if (value >= 45) return 'text-amber-500 shadow-amber-500/20';
      return 'text-red-500 shadow-red-500/20';
    }
    if (label === 'Profit Factor') {
      if (value > 1.2) return 'text-green-500 shadow-green-500/20';
      if (value >= 1.0) return 'text-amber-500 shadow-amber-500/20';
      return 'text-red-500 shadow-red-500/20';
    }
    if (label === 'Sharpe') {
      if (value > 1.0) return 'text-green-500 shadow-green-500/20';
      if (value >= 0.0) return 'text-amber-500 shadow-amber-500/20';
      return 'text-red-500 shadow-red-500/20';
    }
    if (label === 'Expectancy') {
      if (value > 0) return 'text-green-500 shadow-green-500/20';
      return 'text-red-500 shadow-red-500/20';
    }
    return 'text-foreground';
  };

  const formatMetric = (val: number | string, decimals: number, suffix: string = '') => {
    if (typeof val === 'string') return val;
    return `${val.toFixed(decimals)}${suffix}`;
  };

  const metrics = [
    { 
      label: 'Win Rate', 
      value: formatMetric(summary.win_rate, 1, '%'), 
      raw: summary.win_rate,
      icon: <Target className="w-3 h-3" />,
      tooltip: "Calculated based on closed trades only."
    },
    { 
      label: 'Profit Factor', 
      value: formatMetric(summary.profit_factor, 2), 
      raw: summary.profit_factor,
      icon: <TrendingUp className="w-3 h-3" />,
      tooltip: "Ratio of Gross Profit to Gross Loss."
    },
    { 
      label: 'Sharpe', 
      value: formatMetric(summary.sharpe, 2), 
      raw: summary.sharpe,
      icon: <Activity className="w-3 h-3" />,
      tooltip: "Risk-adjusted return using the full portfolio return series."
    },
    { 
      label: 'Sortino', 
      value: formatMetric(summary.sortino, 2), 
      raw: summary.sortino,
      icon: <BarChart className="w-3 h-3" />,
      tooltip: "Downside risk-adjusted return."
    },
    { 
      label: 'Expectancy', 
      value: formatMetric(summary.expectancy, 2, currency), 
      raw: summary.expectancy,
      icon: <Target className="w-3 h-3" />,
      tooltip: "Average expected profit per trade."
    },
    { 
      label: 'Hold Time', 
      value: formatMetric(summary.avg_hold_duration, 1, 'H'), 
      raw: summary.avg_hold_duration,
      icon: <TrendingUp className="w-3 h-3" />,
      tooltip: "Average trade holding duration."
    },
  ];

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
      <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Performance Audit</h3>
        </div>
        <div className="flex items-center gap-2">
           <span className="text-[8px] font-mono font-bold text-muted-foreground uppercase tracking-widest">Base: {currency}</span>
        </div>
      </div>
      
      <div className="p-4 flex flex-col gap-4">
        <div className="grid grid-cols-3 gap-2">
          {metrics.map(m => (
            <div key={m.label} className="flex flex-col items-center justify-center p-2 rounded bg-muted/30 dark:bg-black/20 border border-border relative group/m">
              <span className="text-muted-foreground mb-1">{m.icon}</span>
              <span className="text-[7px] uppercase font-black text-muted-foreground mb-0.5">{m.label}</span>
              <span className={cn(
                "font-mono font-black transition-colors text-center leading-tight break-words max-w-full",
                typeof m.raw === 'string' ? "text-[7px]" : "text-[10px]",
                getMetricColor(m.label, m.raw)
              )}>
                {m.value}
              </span>
              
              <div className="absolute bottom-full mb-2 w-32 p-2 rounded bg-black text-white text-[8px] leading-tight opacity-0 group-hover/m:opacity-100 transition-opacity pointer-events-none z-50 shadow-xl border border-white/10 text-center">
                {m.tooltip}
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1.5 p-2.5 rounded-lg bg-secondary/30 border border-border">
             <div className="flex justify-between items-center mb-1">
               <span className="text-[8px] font-bold text-muted-foreground uppercase flex items-center gap-1">
                 <Activity className="w-2.5 h-2.5" /> Returns Analysis
               </span>
             </div>
             <div className="flex justify-between items-center">
               <span className="text-[8px] font-medium text-muted-foreground uppercase pl-0.5">Inception</span>
               <span className={cn(
                 "text-xs font-mono font-black",
                 summary.total_return >= 0 ? "text-green-500" : "text-red-500"
               )}>
                 {summary.total_return >= 0 ? '+' : ''}{summary.total_return.toFixed(2)}%
               </span>
             </div>
             <div className="flex justify-between items-center opacity-70 mt-1 pt-1 border-t border-border/30">
               <span className="text-[7px] font-medium text-muted-foreground uppercase pl-0.5">Initial Cap</span>
               <span className="text-[9px] font-mono font-bold">{currency}{summary.initial_capital?.toLocaleString()}</span>
             </div>
          </div>

          <div className="flex flex-col gap-1.5 p-2.5 rounded-lg bg-secondary/30 border border-border">
             <div className="flex justify-between items-center">
               <span className="text-[8px] font-bold text-muted-foreground uppercase flex items-center gap-1">
                 <Briefcase className="w-2.5 h-2.5" /> Total Trades
               </span>
               <span className="text-xs font-mono font-bold">{summary.total_trades}</span>
             </div>
             <div className="flex justify-between items-center opacity-70">
               <span className="text-[8px] font-medium text-muted-foreground uppercase pl-3.5">Open</span>
               <span className="text-[10px] font-mono font-bold text-primary">{summary.open_trades}</span>
             </div>
             <div className="flex justify-between items-center opacity-70">
               <span className="text-[8px] font-medium text-muted-foreground uppercase pl-3.5">Closed</span>
               <span className="text-[10px] font-mono font-bold">{summary.closed_trades}</span>
             </div>
          </div>
        </div>

        <div className="grid grid-cols-1">
          <div className="flex flex-col gap-1.5 p-2.5 rounded-lg bg-secondary/30 border border-border">
             <div className="flex justify-between items-center">
               <span className="text-[8px] font-bold text-muted-foreground uppercase flex items-center gap-1">
                 <PieChart className="w-2.5 h-2.5" /> Outcome Distribution
               </span>
               <div className="flex gap-4">
                  <div className="flex items-center gap-1">
                    <span className="text-[8px] font-medium text-green-500 uppercase">Win</span>
                    <span className="text-[10px] font-mono font-bold text-green-500">{summary.winning_trades}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-[8px] font-medium text-red-500 uppercase">Loss</span>
                    <span className="text-[10px] font-mono font-bold text-red-500">{summary.losing_trades}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-[8px] font-medium text-muted-foreground uppercase">Calmar</span>
                    <span className="text-[10px] font-mono font-bold">{formatMetric(summary.calmar, 2)}</span>
                  </div>
               </div>
             </div>
          </div>
        </div>

        <div className="flex items-center justify-between px-1">
           <span className="text-[9px] font-mono italic text-muted-foreground">Source: backend/data/paper_trading.json</span>
           <span className="text-[8px] font-black uppercase text-emerald-500">Live_Validated</span>
        </div>
      </div>
    </div>
  );
}
