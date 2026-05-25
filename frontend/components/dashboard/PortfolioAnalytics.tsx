import React from 'react';
import { ChartData } from '@/types';
import { Briefcase, Wallet, Percent, LayoutList, Activity } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface PortfolioAnalyticsProps {
  data: ChartData | null;
  currency?: string;
}

export function PortfolioAnalytics({ data, currency = '$' }: PortfolioAnalyticsProps) {
  if (!data || !data.portfolio) return null;

  const { portfolio } = data;
  const isPositive = portfolio.return_pct >= 0;

  const stats = [
    {
      label: 'Total Equity',
      value: `${currency}${(portfolio.equity ?? 0).toLocaleString()}`,
      icon: <Briefcase className="w-3.5 h-3.5 text-muted-foreground" />,
      color: 'text-foreground'
    },
    {
      label: 'Available Cash',
      value: `${currency}${(portfolio.cash ?? 0).toLocaleString()}`,
      icon: <Wallet className="w-3.5 h-3.5 text-muted-foreground" />,
      color: 'text-emerald-500'
    },
    {
      label: 'Today\'s PnL',
      value: `${(portfolio.today_pnl ?? 0) >= 0 ? '+' : ''}${currency}${(portfolio.today_pnl ?? 0).toLocaleString()}`,
      icon: <Activity className="w-3.5 h-3.5 text-muted-foreground" />,
      color: (portfolio.today_pnl ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'
    },
    {
      label: 'Inception Return',
      value: `${isPositive ? '+' : ''}${(portfolio.return_pct ?? 0).toFixed(2)}%`,
      icon: <Percent className="w-3.5 h-3.5 text-muted-foreground" />,
      color: isPositive ? 'text-green-500 dark:text-glow' : 'text-red-500 dark:text-glow'
    }
  ];

  const pnlBreakdown = [
    { label: 'MTD PnL', value: portfolio.mtd_pnl ?? 0, prefix: currency },
    { label: 'YTD PnL', value: portfolio.ytd_pnl ?? 0, prefix: currency },
    { label: 'Realized', value: portfolio.realized_pnl ?? 0, prefix: currency },
    { label: 'Unrealized', value: portfolio.unrealized_pnl ?? 0, prefix: currency }
  ];

  const activePositions = Object.entries(portfolio.positions).filter(([, pos]) => pos.shares > 0);

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300" data-tour="portfolio">
      <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Briefcase className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Live Portfolio</h3>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          <span className="text-[9px] font-mono font-bold opacity-60 uppercase">Sync_Active</span>
        </div>
      </div>
      <div className="p-4 flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-3">
          {stats.map((stat, i) => (
            <motion.div 
              key={stat.label}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.3 }}
              className="p-3 rounded-lg bg-muted/30 dark:bg-black/20 border border-border flex flex-col items-start hover:bg-muted/50 dark:hover:bg-black/40 transition-colors"
            >
              <div className="flex items-center gap-1.5 mb-1.5">
                {stat.icon}
                <span className="text-[9px] uppercase font-black tracking-wider text-muted-foreground">{stat.label}</span>
              </div>
              <span className={`text-base font-black font-mono tracking-tight ${stat.color}`}>
                {stat.value}
              </span>
            </motion.div>
          ))}
        </div>

        {/* Detailed Breakdown */}
        <div className="grid grid-cols-2 gap-2 p-2 rounded-lg bg-secondary/30 border border-border">
           {pnlBreakdown.map(item => (
             <div key={item.label} className="flex justify-between items-center px-1">
               <span className="text-[8px] font-bold text-muted-foreground uppercase">{item.label}</span>
               <span className={cn("text-[9px] font-mono font-bold", item.value >= 0 ? "text-green-500" : "text-red-500")}>
                 {item.value >= 0 ? '+' : ''}{item.prefix}{Math.abs(item.value).toLocaleString()}
               </span>
             </div>
           ))}
        </div>

        {activePositions.length > 0 && (
          <div className="overflow-hidden rounded-lg border border-border shadow-inner bg-muted/10 dark:bg-transparent">
            <table className="w-full text-xs text-left font-mono">
              <thead className="text-[9px] uppercase bg-secondary/80 dark:bg-black/40 text-muted-foreground font-black">
                <tr>
                  <th className="px-3 py-2 border-b border-border">Ticker</th>
                  <th className="px-3 py-2 text-right border-b border-border">Shares</th>
                  <th className="px-3 py-2 text-right border-b border-border">Avg Px</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-card/40 dark:bg-black/20">
                {activePositions.map(([ticker, pos]) => (
                  <tr key={ticker} className="hover:bg-primary/5 dark:hover:bg-white/5 transition-colors group/row">
                    <td className="px-3 py-2 font-black text-foreground font-sans group-hover/row:text-primary transition-colors">{ticker}</td>
                    <td className="px-3 py-2 text-right opacity-90 font-bold">{pos.shares.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right opacity-90 font-bold text-primary dark:text-foreground">{currency}{pos.avg_price.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
