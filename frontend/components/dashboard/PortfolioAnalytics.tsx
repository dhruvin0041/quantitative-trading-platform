import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
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
      icon: <Briefcase className="w-4 h-4 text-muted-foreground" />,
      color: 'text-foreground'
    },
    {
      label: 'Available Cash',
      value: `$${portfolio.cash.toLocaleString()}`,
      icon: <Wallet className="w-4 h-4 text-muted-foreground" />,
      color: 'text-[var(--signal-buy)]'
    },
    {
      label: 'Net Return',
      value: `${isPositive ? '+' : ''}${portfolio.return_pct.toFixed(2)}%`,
      icon: <Percent className="w-4 h-4 text-muted-foreground" />,
      color: isPositive ? 'text-[var(--signal-buy)]' : 'text-[var(--signal-sell)]'
    },
    {
      label: 'Active Positions',
      value: Object.values(portfolio.positions).filter(p => p.shares > 0).length,
      icon: <LayoutList className="w-4 h-4 text-muted-foreground" />,
      color: 'text-primary'
    }
  ];

  const activePositions = Object.entries(portfolio.positions).filter(([_, pos]) => pos.shares > 0);

  return (
    <Card className="shadow-sm border-border bg-card" data-tour="portfolio">
      <CardHeader className="pb-4">
        <CardTitle className="text-base font-semibold flex items-center gap-2 text-foreground">
          <Briefcase className="w-4 h-4 text-primary" />
          Live Portfolio
        </CardTitle>
        <CardDescription className="text-xs">Real-time paper trading status</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {stats.map((stat, i) => (
            <motion.div 
              key={stat.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.3 }}
              className="p-4 rounded-lg bg-secondary/30 border border-border/50 flex flex-col items-start"
            >
              <div className="flex items-center gap-2 mb-2">
                {stat.icon}
                <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">{stat.label}</span>
              </div>
              <span className={`text-2xl font-bold font-mono tracking-tight ${stat.color}`}>
                {stat.value}
              </span>
            </motion.div>
          ))}
        </div>

        {activePositions.length > 0 && (
          <div className="overflow-hidden rounded-md border border-border">
            <table className="w-full text-sm text-left font-mono">
              <thead className="text-[10px] uppercase bg-secondary/50 text-muted-foreground">
                <tr>
                  <th className="px-4 py-3 font-semibold border-b border-border/50">Ticker</th>
                  <th className="px-4 py-3 font-semibold text-right border-b border-border/50">Shares</th>
                  <th className="px-4 py-3 font-semibold text-right border-b border-border/50">Avg Price</th>
                  <th className="px-4 py-3 font-semibold text-right border-b border-border/50">Cost Basis</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50 bg-card">
                {activePositions.map(([ticker, pos]) => (
                  <tr key={ticker} className="hover:bg-secondary/30 transition-colors">
                    <td className="px-4 py-3 font-bold text-foreground font-sans">{ticker}</td>
                    <td className="px-4 py-3 text-right">{pos.shares.toLocaleString()}</td>
                    <td className="px-4 py-3 text-right">${pos.avg_price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right">${(pos.shares * pos.avg_price).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
