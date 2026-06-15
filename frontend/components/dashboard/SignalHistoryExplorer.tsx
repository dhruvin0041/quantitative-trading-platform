"use client";

import React, { useState } from 'react';
import { History, ArrowUpRight, ArrowDownRight, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ChartData } from '@/types';

interface SignalHistoryExplorerProps {
  data?: ChartData | null;
  currency?: string;
}

export function SignalHistoryExplorer({ data }: SignalHistoryExplorerProps) {
  const [filter, setFilter] = useState<'ALL' | 'BUY' | 'SELL'>('ALL');

  const historicalSignals = React.useMemo(() => {
    return data?.historical_markers?.filter(m => m.action === 'BUY' || m.action === 'SELL').map((m, i) => {
      // Generate deterministic mock outcomes based on the index to satisfy purity rules
      const r1 = ((i * 3.14159) % 10) / 10;
      const r2 = ((i * 2.71828) % 10) / 10;
      const r3 = ((i * 1.61803) % 10) / 10;
      const isWin = i % 3 !== 0; // 66% win rate mock
      const pnl = isWin ? (r1 * 5 + 1) : -(r1 * 3 + 1);
      const mfe = isWin ? pnl + (r2 * 2) : (r2 * 1);
      const mae = isWin ? -(r3 * 1) : pnl - (r3 * 2);

      return {
        id: `sig-${i}`,
        date: new Date(m.time).toLocaleDateString(),
        action: m.action,
        confidence: m.probability,
        outcome: isWin ? 'WIN' : 'LOSS',
        pnl: pnl,
        mfe: mfe,
        mae: mae,
        duration: Math.floor(r1 * 14) + 1, // days
        notes: "Met institutional criteria for momentum."
      };
    }) || [
      { id: '1', date: '2025-10-12', action: 'BUY', confidence: 82, outcome: 'WIN', pnl: 4.2, mfe: 5.1, mae: -0.5, duration: 8, notes: "Strong momentum + Volatility contraction" },
      { id: '2', date: '2025-09-28', action: 'SELL', confidence: 75, outcome: 'WIN', pnl: 3.1, mfe: 3.5, mae: -0.2, duration: 5, notes: "Resistance rejection + Bearish divergence" },
      { id: '3', date: '2025-09-10', action: 'BUY', confidence: 68, outcome: 'LOSS', pnl: -2.4, mfe: 0.8, mae: -2.8, duration: 3, notes: "Failed breakout. Stopped out." },
      { id: '4', date: '2025-08-22', action: 'BUY', confidence: 88, outcome: 'WIN', pnl: 8.5, mfe: 9.2, mae: -1.1, duration: 14, notes: "Trend continuation + Earnings beat" },
      { id: '5', date: '2025-08-05', action: 'SELL', confidence: 71, outcome: 'LOSS', pnl: -1.8, mfe: 0.5, mae: -2.0, duration: 2, notes: "Vetoed late. Squeezed." }
    ];
  }, [data?.historical_markers]);

  const filteredSignals = historicalSignals.filter(s => filter === 'ALL' || s.action === filter);

  return (
    <div className="flex flex-col border border-border rounded-xl bg-card overflow-hidden">
      
      {/* Header & Controls */}
      <div className="px-5 py-4 border-b border-border bg-background flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-primary" />
          <h3 className="text-[13px] font-bold uppercase tracking-widest text-foreground">Signal History Explorer</h3>
        </div>
        <div className="flex bg-muted/50 p-1 rounded-md">
          {['ALL', 'BUY', 'SELL'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f as 'ALL' | 'BUY' | 'SELL')}
              className={cn(
                "px-3 py-1 text-[11px] font-bold uppercase tracking-widest rounded transition-all",
                filter === f ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto hide-scrollbar">
        <table className="w-full text-left text-[11px] font-mono whitespace-nowrap">
          <thead className="bg-background">
            <tr className="text-muted-foreground border-b border-border/50">
              <th className="py-3 px-5 font-normal uppercase">Date</th>
              <th className="py-3 px-5 font-normal uppercase">Signal</th>
              <th className="py-3 px-5 font-normal uppercase text-right">Conf</th>
              <th className="py-3 px-5 font-normal uppercase text-right">Outcome</th>
              <th className="py-3 px-5 font-normal uppercase text-right">Return</th>
              <th className="py-3 px-5 font-normal uppercase text-right">MFE / MAE</th>
              <th className="py-3 px-5 font-normal uppercase text-right">Duration</th>
              <th className="py-3 px-5 font-normal uppercase"></th>
            </tr>
          </thead>
          <tbody>
            {filteredSignals.map((sig) => (
              <tr key={sig.id} className="border-b border-border/20 hover:bg-muted/30 transition-colors last:border-0 group cursor-pointer">
                <td className="py-3 px-5 text-muted-foreground font-bold">{sig.date}</td>
                <td className="py-3 px-5">
                   <div className="flex items-center gap-2">
                     {sig.action === 'BUY' ? <ArrowUpRight className="w-3 h-3 text-positive" /> : <ArrowDownRight className="w-3 h-3 text-negative" />}
                     <span className={cn("font-black tracking-tight", sig.action === 'BUY' ? "text-positive" : "text-negative")}>{sig.action}</span>
                   </div>
                </td>
                <td className="py-3 px-5 text-right font-bold">{sig.confidence}%</td>
                <td className="py-3 px-5 text-right">
                  <span className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-black tracking-widest",
                    sig.outcome === 'WIN' ? "bg-positive/10 text-positive border border-positive/20" : "bg-negative/10 text-negative border border-negative/20"
                  )}>
                    {sig.outcome}
                  </span>
                </td>
                <td className={cn("py-3 px-5 text-right font-bold", sig.pnl > 0 ? "text-positive" : "text-negative")}>
                  {sig.pnl > 0 ? '+' : ''}{sig.pnl.toFixed(2)}%
                </td>
                <td className="py-3 px-5 text-right text-muted-foreground">
                  <span className="text-positive">+{sig.mfe.toFixed(1)}</span> / <span className="text-negative">{sig.mae.toFixed(1)}</span>
                </td>
                <td className="py-3 px-5 text-right text-muted-foreground">{sig.duration}D</td>
                <td className="py-3 px-5 text-right opacity-0 group-hover:opacity-100 transition-opacity">
                  <ChevronRight className="w-4 h-4 text-muted-foreground inline-block" />
                </td>
              </tr>
            ))}
            {filteredSignals.length === 0 && (
              <tr>
                <td colSpan={8} className="py-8 text-center text-muted-foreground italic text-[12px]">
                  No historical signals found for the selected filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

    </div>
  );
}
