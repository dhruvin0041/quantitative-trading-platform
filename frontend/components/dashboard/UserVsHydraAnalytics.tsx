"use client";

import React from 'react';
import { Swords, User, BrainCircuit } from 'lucide-react';
import { cn } from '@/lib/utils';
export function UserVsHydraAnalytics() {
  
  // Mock aggregated analytics data
  const analytics = {
    user: {
      winRate: 52.4,
      profitFactor: 1.1,
      sharpe: 0.8,
      totalReturn: 12.5,
      trades: 145
    },
    hydra: {
      winRate: 68.5,
      profitFactor: 2.3,
      sharpe: 1.9,
      totalReturn: 28.4,
      trades: 145
    },
    alignment: {
      syncRate: 42,
      missedOpportunities: 18,
      savedByHydraVeto: 12
    }
  };

  const getWinnerClass = (userVal: number, hydraVal: number) => {
    return userVal > hydraVal ? "text-muted-foreground" : "text-primary font-black";
  };

  return (
    <div className="flex flex-col gap-6">
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* User Card */}
        <div className="flex flex-col p-5 rounded-xl border border-border bg-card">
          <div className="flex items-center gap-2 mb-4 pb-3 border-b border-border">
            <div className="w-8 h-8 rounded bg-muted flex items-center justify-center">
              <User className="w-4 h-4 text-foreground" />
            </div>
            <div className="flex flex-col">
              <span className="text-[14px] font-black uppercase text-foreground tracking-tight">Manual Trading</span>
              <span className="text-[11px] text-muted-foreground">User Executions</span>
            </div>
          </div>
          
          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">Win Rate</span>
              <span className="text-[16px] font-mono font-bold text-foreground">{analytics.user.winRate}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">Profit Factor</span>
              <span className="text-[16px] font-mono font-bold text-foreground">{analytics.user.profitFactor}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">Total Return</span>
              <span className="text-[16px] font-mono font-black text-positive">+{analytics.user.totalReturn}%</span>
            </div>
          </div>
        </div>

        {/* VS Badge / Alignment Stats */}
        <div className="flex flex-col items-center justify-center p-5 rounded-xl border border-border/50 bg-background/50 relative">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-background border border-border flex items-center justify-center shadow-sm">
            <Swords className="w-4 h-4 text-muted-foreground" />
          </div>
          
          <div className="flex flex-col items-center text-center gap-1 w-full mt-2">
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Signal Sync Rate</span>
            <div className="w-full bg-muted rounded-full h-2 mt-1 overflow-hidden flex">
              <div className="h-full bg-primary" style={{ width: `${analytics.alignment.syncRate}%` }} />
              <div className="h-full bg-warning" style={{ width: `${100 - analytics.alignment.syncRate}%` }} />
            </div>
            <div className="flex justify-between w-full mt-1">
              <span className="text-[10px] font-mono text-primary">{analytics.alignment.syncRate}% Sync</span>
              <span className="text-[10px] font-mono text-warning">{100 - analytics.alignment.syncRate}% Div</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 w-full mt-6">
            <div className="flex flex-col items-center p-2 rounded bg-muted/30 border border-border/50 text-center">
              <span className="text-[16px] font-black text-warning">{analytics.alignment.missedOpportunities}</span>
              <span className="text-[9px] uppercase font-bold text-muted-foreground mt-1">Missed<br/>Hydra Wins</span>
            </div>
            <div className="flex flex-col items-center p-2 rounded bg-muted/30 border border-border/50 text-center">
              <span className="text-[16px] font-black text-positive">{analytics.alignment.savedByHydraVeto}</span>
              <span className="text-[9px] uppercase font-bold text-muted-foreground mt-1">Losses<br/>Prevented</span>
            </div>
          </div>
        </div>

        {/* Hydra Card */}
        <div className="flex flex-col p-5 rounded-xl border border-primary/20 bg-primary/5 shadow-[0_0_15px_rgba(var(--primary),0.05)] relative overflow-hidden">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-primary/10 rounded-full blur-2xl pointer-events-none" />
          
          <div className="flex items-center gap-2 mb-4 pb-3 border-b border-primary/10 relative z-10">
            <div className="w-8 h-8 rounded bg-primary/20 flex items-center justify-center">
              <BrainCircuit className="w-4 h-4 text-primary" />
            </div>
            <div className="flex flex-col">
              <span className="text-[14px] font-black uppercase text-primary tracking-tight">Hydra Baseline</span>
              <span className="text-[11px] text-muted-foreground">System Recommendation</span>
            </div>
          </div>
          
          <div className="flex flex-col gap-3 relative z-10">
            <div className="flex justify-between items-center">
              <span className="text-[11px] font-bold text-primary/70 uppercase">Win Rate</span>
              <span className={cn("text-[16px] font-mono", getWinnerClass(analytics.user.winRate, analytics.hydra.winRate))}>{analytics.hydra.winRate}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[11px] font-bold text-primary/70 uppercase">Profit Factor</span>
              <span className={cn("text-[16px] font-mono", getWinnerClass(analytics.user.profitFactor, analytics.hydra.profitFactor))}>{analytics.hydra.profitFactor}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[11px] font-bold text-primary/70 uppercase">Total Return</span>
              <span className="text-[16px] font-mono font-black text-positive">+{analytics.hydra.totalReturn}%</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
