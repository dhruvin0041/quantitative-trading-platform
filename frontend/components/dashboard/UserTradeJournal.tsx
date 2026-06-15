"use client";

import React, { useState } from 'react';
import { BookOpen, Check, X, User, BrainCircuit } from 'lucide-react';
import { cn } from '@/lib/utils';
export function UserTradeJournal() {
  const [filter, setFilter] = useState<'ALL' | 'AGREEMENT' | 'DIVERGENCE'>('ALL');

  // Mocking user trades vs Hydra signals
  const journalEntries = [
    { id: 't1', date: '2025-10-12', ticker: 'NVDA', userAction: 'BUY', hydraAction: 'BUY', userPnl: 4.2, hydraPnl: 4.2, matched: true, notes: "Followed Hydra signal. Clean breakout." },
    { id: 't2', date: '2025-09-28', ticker: 'TSLA', userAction: 'BUY', hydraAction: 'SELL', userPnl: -2.1, hydraPnl: 3.1, matched: false, notes: "Ignored Hydra sell signal. Caught in pullback." },
    { id: 't3', date: '2025-09-10', ticker: 'AAPL', userAction: 'HOLD', hydraAction: 'BUY', userPnl: 0, hydraPnl: -2.4, matched: false, notes: "Missed entry, but Hydra signal was a loss anyway." },
    { id: 't4', date: '2025-08-22', ticker: 'AMD', userAction: 'BUY', hydraAction: 'BUY', userPnl: 6.5, hydraPnl: 8.5, matched: true, notes: "Took profit early. Hydra held for full target." },
    { id: 't5', date: '2025-08-05', ticker: 'MSFT', userAction: 'SELL', hydraAction: 'HOLD', userPnl: 1.2, hydraPnl: 0, matched: false, notes: "Felt top heavy. Scalped a quick short." }
  ];

  const filteredEntries = journalEntries.filter(e => {
    if (filter === 'AGREEMENT') return e.matched;
    if (filter === 'DIVERGENCE') return !e.matched;
    return true;
  });

  return (
    <div className="flex flex-col border border-border rounded-xl bg-card overflow-hidden">
      
      <div className="px-5 py-4 border-b border-border bg-background flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-primary" />
          <h3 className="text-[13px] font-bold uppercase tracking-widest text-foreground">User Trade Journal</h3>
        </div>
        <div className="flex bg-muted/50 p-1 rounded-md">
          {['ALL', 'AGREEMENT', 'DIVERGENCE'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f as 'ALL' | 'AGREEMENT' | 'DIVERGENCE')}
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

      <div className="overflow-x-auto hide-scrollbar">
        <table className="w-full text-left text-[11px] font-mono whitespace-nowrap">
          <thead className="bg-background">
            <tr className="text-muted-foreground border-b border-border/50">
              <th className="py-3 px-5 font-normal uppercase">Date</th>
              <th className="py-3 px-5 font-normal uppercase">Asset</th>
              <th className="py-3 px-5 font-normal uppercase text-center">User Action</th>
              <th className="py-3 px-5 font-normal uppercase text-center">Hydra Signal</th>
              <th className="py-3 px-5 font-normal uppercase text-center">Alignment</th>
              <th className="py-3 px-5 font-normal uppercase text-right">User PnL</th>
              <th className="py-3 px-5 font-normal uppercase text-right">Hydra PnL</th>
              <th className="py-3 px-5 font-normal uppercase w-1/3">Trader Notes</th>
            </tr>
          </thead>
          <tbody>
            {filteredEntries.map((entry) => (
              <tr key={entry.id} className="border-b border-border/20 hover:bg-muted/30 transition-colors last:border-0">
                <td className="py-3 px-5 text-muted-foreground font-bold">{entry.date}</td>
                <td className="py-3 px-5 text-foreground font-bold">{entry.ticker}</td>
                
                <td className="py-3 px-5 text-center">
                  <span className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-1 w-fit mx-auto",
                    entry.userAction === 'BUY' ? "text-positive bg-positive/10 border border-positive/20" : 
                    entry.userAction === 'SELL' ? "text-negative bg-negative/10 border border-negative/20" : 
                    "text-muted-foreground bg-muted/50 border border-border"
                  )}>
                    <User className="w-3 h-3" /> {entry.userAction}
                  </span>
                </td>
                
                <td className="py-3 px-5 text-center">
                  <span className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-1 w-fit mx-auto",
                    entry.hydraAction === 'BUY' ? "text-positive bg-positive/10 border border-positive/20" : 
                    entry.hydraAction === 'SELL' ? "text-negative bg-negative/10 border border-negative/20" : 
                    "text-muted-foreground bg-muted/50 border border-border"
                  )}>
                    <BrainCircuit className="w-3 h-3" /> {entry.hydraAction}
                  </span>
                </td>
                
                <td className="py-3 px-5 text-center">
                  {entry.matched ? (
                    <div className="flex items-center justify-center gap-1 text-positive">
                      <Check className="w-4 h-4" /> <span className="text-[10px] uppercase font-bold">Sync</span>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center gap-1 text-warning">
                      <X className="w-4 h-4" /> <span className="text-[10px] uppercase font-bold">Div</span>
                    </div>
                  )}
                </td>

                <td className={cn("py-3 px-5 text-right font-bold", entry.userPnl > 0 ? "text-positive" : entry.userPnl < 0 ? "text-negative" : "text-muted-foreground")}>
                  {entry.userPnl > 0 ? '+' : ''}{entry.userPnl.toFixed(2)}%
                </td>
                
                <td className={cn("py-3 px-5 text-right font-bold", entry.hydraPnl > 0 ? "text-primary" : entry.hydraPnl < 0 ? "text-negative" : "text-muted-foreground")}>
                  {entry.hydraPnl > 0 ? '+' : ''}{entry.hydraPnl.toFixed(2)}%
                </td>

                <td className="py-3 px-5 text-muted-foreground text-[11px] truncate max-w-[200px]" title={entry.notes}>
                  {entry.notes}
                </td>
              </tr>
            ))}
            {filteredEntries.length === 0 && (
              <tr>
                <td colSpan={8} className="py-8 text-center text-muted-foreground italic text-[12px]">
                  No journal entries match the current filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

    </div>
  );
}
