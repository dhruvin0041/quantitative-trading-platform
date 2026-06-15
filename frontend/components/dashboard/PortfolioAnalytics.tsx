import React from 'react';
import { ChartData } from '@/types';
import { Briefcase, Wallet, Percent, Activity, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface PortfolioAnalyticsProps {
  data: ChartData | null;
  currency?: string;
}

export function PortfolioAnalytics({ data, currency = '$' }: PortfolioAnalyticsProps) {
  if (!data || !data.portfolio) return (
    <div className="flex items-center justify-center h-48 text-muted-foreground font-mono text-[13px] uppercase tracking-widest border border-border rounded-xl bg-card">
      Awaiting Portfolio Telemetry...
    </div>
  );

  const { portfolio } = data;
  const isPositive = portfolio.return_pct >= 0;

  const totalEquity = portfolio.equity ?? 0;
  const cash = portfolio.cash ?? 0;
  const invested = totalEquity - cash;
  
  const investedPct = totalEquity > 0 ? (invested / totalEquity) * 100 : 0;
  const cashPct = totalEquity > 0 ? (cash / totalEquity) * 100 : 0;

  const activePositions = Object.entries(portfolio.positions).filter(([, pos]) => pos.shares > 0);

  return (
    <div className="flex flex-col gap-6">
      
      {/* SECTION: High Level Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <Briefcase className="w-3.5 h-3.5" /> Total Equity
          </span>
          <span className="text-[18px] font-mono font-black text-foreground">
            {currency}{totalEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        
        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <Wallet className="w-3.5 h-3.5" /> Available Cash
          </span>
          <span className="text-[18px] font-mono font-black text-foreground">
            {currency}{cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <Activity className="w-3.5 h-3.5" /> Today&apos;s PnL
          </span>
          <span className={cn(
            "text-[18px] font-mono font-black",
            (portfolio.today_pnl ?? 0) >= 0 ? "text-positive" : "text-negative"
          )}>
            {(portfolio.today_pnl ?? 0) >= 0 ? '+' : ''}{currency}{(portfolio.today_pnl ?? 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="p-4 rounded-lg bg-card border border-border flex flex-col gap-1">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-1">
            <Percent className="w-3.5 h-3.5" /> Inception Return
          </span>
          <span className={cn(
            "text-[18px] font-mono font-black",
            isPositive ? "text-positive" : "text-negative"
          )}>
            {isPositive ? '+' : ''}{(portfolio.return_pct ?? 0).toFixed(2)}%
          </span>
        </div>
      </div>

      {/* SECTION: Allocation (1D Treemap Mock) & Detailed PnL */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        <div className="flex flex-col gap-4 p-5 rounded-lg bg-card border border-border">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <Layers className="w-4 h-4 text-primary" />
            Capital Allocation
          </h3>
          <div className="flex flex-col gap-2 mt-2">
            <div className="flex justify-between items-end">
               <span className="text-[12px] font-bold text-foreground">Invested vs Cash</span>
            </div>
            <div className="h-6 w-full bg-muted rounded-lg overflow-hidden flex shadow-inner border border-border">
               <motion.div 
                 initial={{ width: 0 }}
                 animate={{ width: `${investedPct}%` }}
                 className="h-full bg-primary border-r border-card flex items-center justify-center"
               >
                 {investedPct > 15 && <span className="text-[10px] font-bold text-white uppercase tracking-widest">Inv {investedPct.toFixed(0)}%</span>}
               </motion.div>
               <motion.div 
                 initial={{ width: 0 }}
                 animate={{ width: `${cashPct}%` }}
                 className="h-full bg-border flex items-center justify-center"
               >
                 {cashPct > 15 && <span className="text-[10px] font-bold text-foreground uppercase tracking-widest">Cash {cashPct.toFixed(0)}%</span>}
               </motion.div>
            </div>
            <div className="flex justify-between text-[11px] font-mono text-muted-foreground uppercase mt-2">
               <span>Invested: {currency}{invested.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
               <span>Cash: {currency}{cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4 p-5 rounded-lg bg-card border border-border">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest border-b border-border pb-2">
            Performance Breakdown
          </h3>
          <div className="grid grid-cols-2 gap-4 mt-2">
            <div className="flex flex-col justify-between">
               <span className="text-[12px] font-bold text-muted-foreground uppercase tracking-widest">Today&apos;s P&L</span>
               <span className={cn("text-[14px] font-mono font-bold", (portfolio.mtd_pnl ?? 0) >= 0 ? "text-positive" : "text-negative")}>
                 {(portfolio.mtd_pnl ?? 0) >= 0 ? '+' : ''}{currency}{Math.abs(portfolio.mtd_pnl ?? 0).toLocaleString()}
               </span>
            </div>
            <div className="flex flex-col justify-between">
               <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">YTD PnL</span>
               <span className={cn("text-[14px] font-mono font-bold", (portfolio.ytd_pnl ?? 0) >= 0 ? "text-positive" : "text-negative")}>
                 {(portfolio.ytd_pnl ?? 0) >= 0 ? '+' : ''}{currency}{Math.abs(portfolio.ytd_pnl ?? 0).toLocaleString()}
               </span>
            </div>
            <div className="flex flex-col justify-between">
               <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Realized</span>
               <span className={cn("text-[14px] font-mono font-bold", (portfolio.realized_pnl ?? 0) >= 0 ? "text-positive" : "text-negative")}>
                 {(portfolio.realized_pnl ?? 0) >= 0 ? '+' : ''}{currency}{Math.abs(portfolio.realized_pnl ?? 0).toLocaleString()}
               </span>
            </div>
            <div className="flex flex-col justify-between">
               <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Unrealized</span>
               <span className={cn("text-[14px] font-mono font-bold", (portfolio.unrealized_pnl ?? 0) >= 0 ? "text-positive" : "text-negative")}>
                 {(portfolio.unrealized_pnl ?? 0) >= 0 ? '+' : ''}{currency}{Math.abs(portfolio.unrealized_pnl ?? 0).toLocaleString()}
               </span>
            </div>
          </div>
        </div>

      </div>

      {/* SECTION: Open Positions Table */}
      {activePositions.length > 0 && (
        <div className="border border-border rounded-lg overflow-hidden bg-card">
          <table className="w-full text-left border-collapse">
            <thead className="bg-background border-b border-border">
              <tr>
                <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest">Ticker</th>
                <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Shares</th>
                <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Avg Entry</th>
                <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">Mark</th>
                <th className="px-4 py-3 text-[11px] font-bold text-muted-foreground uppercase tracking-widest text-right">PnL</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {activePositions.map(([ticker, pos]) => {
                // Determine mock current mark based on portfolio logic (simplified here)
                const currentMark = data.current_price && ticker === data.ticker ? data.current_price : pos.avg_price;
                const posPnl = (currentMark - pos.avg_price) * pos.shares;
                const isPos = posPnl >= 0;

                return (
                  <tr key={ticker} className="hover:bg-muted/50 transition-colors">
                    <td className="px-4 py-3 text-[13px] font-bold text-foreground uppercase">{ticker}</td>
                    <td className="px-4 py-3 text-[13px] font-mono text-foreground text-right">{pos.shares.toLocaleString()}</td>
                    <td className="px-4 py-3 text-[13px] font-mono text-foreground text-right">{currency}{pos.avg_price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-[13px] font-mono text-foreground text-right">{currency}{currentMark.toFixed(2)}</td>
                    <td className={cn(
                      "px-4 py-3 text-[13px] font-mono font-bold text-right",
                      isPos ? "text-positive" : "text-negative"
                    )}>
                      {isPos ? '+' : ''}{currency}{posPnl.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

    </div>
  );
}
