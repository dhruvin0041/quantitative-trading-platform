import React from 'react';
import { ChartData } from '@/types';
import { Briefcase, Wallet, Percent, LayoutList } from 'lucide-react';
import { motion } from 'framer-motion';

interface PortfolioAnalyticsProps {
  data: ChartData | null;
}

export function PortfolioAnalytics({ data }: PortfolioAnalyticsProps) {
  if (!data || !data.portfolio) return null;

  const { portfolio } = data;
  const isPositive = portfolio.return_pct >= 0;

  const stats = [
    {
      label: 'Total Equity',
      value: `$${portfolio.equity.toLocaleString()}`,
      icon: <Briefcase className="w-3.5 h-3.5 text-muted-foreground" />,
      color: 'text-foreground'
    },
    {
      label: 'Available Cash',
      value: `$${portfolio.cash.toLocaleString()}`,
      icon: <Wallet className="w-3.5 h-3.5 text-muted-foreground" />,
      color: 'text-[var(--signal-buy)]'
    },
    {
      label: 'Net Return',
      value: `${isPositive ? '+' : ''}${portfolio.return_pct.toFixed(2)}%`,
      icon: <Percent className="w-3.5 h-3.5 text-muted-foreground" />,
      color: isPositive ? 'text-[var(--signal-buy)] text-glow' : 'text-[var(--signal-sell)] text-glow'
    },
    {
      label: 'Active Positions',
      value: Object.values(portfolio.positions).filter(p => p.shares > 0).length,
      icon: <LayoutList className="w-3.5 h-3.5 text-muted-foreground" />,
      color: 'text-primary'
    }
  ];

  const activePositions = Object.entries(portfolio.positions).filter(([, pos]) => pos.shares > 0);

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group hover:border-white/20 transition-colors" data-tour="portfolio">
      <div className="bg-black/40 border-b border-white/5 px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Briefcase className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Live Portfolio</h3>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          <span className="text-[9px] font-mono opacity-50 uppercase">Sync</span>
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
              className="p-3 rounded-lg bg-black/20 border border-white/5 flex flex-col items-start hover:bg-black/40 transition-colors"
            >
              <div className="flex items-center gap-1.5 mb-1.5">
                {stat.icon}
                <span className="text-[9px] uppercase font-bold tracking-wider text-muted-foreground">{stat.label}</span>
              </div>
              <span className={`text-base font-bold font-mono tracking-tight ${stat.color}`}>
                {stat.value}
              </span>
            </motion.div>
          ))}
        </div>

        {activePositions.length > 0 && (
          <div className="overflow-hidden rounded-lg border border-white/5">
            <table className="w-full text-xs text-left font-mono">
              <thead className="text-[9px] uppercase bg-black/40 text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-semibold border-b border-white/5">Ticker</th>
                  <th className="px-3 py-2 font-semibold text-right border-b border-white/5">Shares</th>
                  <th className="px-3 py-2 font-semibold text-right border-b border-white/5">Avg Px</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 bg-black/20">
                {activePositions.map(([ticker, pos]) => (
                  <tr key={ticker} className="hover:bg-white/5 transition-colors group/row">
                    <td className="px-3 py-2 font-bold text-foreground font-sans group-hover/row:text-primary transition-colors">{ticker}</td>
                    <td className="px-3 py-2 text-right opacity-80">{pos.shares.toLocaleString()}</td>
                    <td className="px-3 py-2 text-right opacity-80">${pos.avg_price.toFixed(2)}</td>
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
