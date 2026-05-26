import React from 'react';
import { ShieldAlert } from 'lucide-react';
import { motion } from 'framer-motion';
import { ChartData } from '@/types';

interface RiskDashboardProps {
  data: ChartData | null;
  currency?: string;
}

export function RiskDashboard({ data, currency = "$" }: RiskDashboardProps) {
  if (!data || !data.risk) return null;

  const { risk } = data;

  const riskMetrics = [
    { label: 'VaR (95%)', value: `${(risk.var_95 * 100).toFixed(2)}%`, isDanger: risk.var_95 > 0.02 },
    { label: 'CVaR', value: `${(risk.cvar * 100).toFixed(2)}%`, isDanger: risk.cvar > 0.03 },
    { label: 'Beta', value: risk.beta.toFixed(2), isDanger: false },
    { label: 'Kelly Frac.', value: (risk.kelly_fraction * 100).toFixed(1) + '%', isDanger: false },
    { label: 'Target Size', value: `${currency}${risk.target_size.toLocaleString()}`, isHighlight: true },
    { label: 'Max Drawdown', value: `${risk.max_drawdown.toFixed(1)}%`, isDanger: true },
  ];

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
      <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Risk Management</h3>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-primary"></span>
          </span>
          <span className="text-[9px] font-mono font-bold opacity-60 uppercase tracking-widest text-primary dark:text-foreground">Armed</span>
        </div>
      </div>
      <div className="p-4 grid grid-cols-2 gap-3">
        {riskMetrics.map((metric, i) => (
          <motion.div 
            key={metric.label}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1, duration: 0.3 }}
            className={`flex flex-col gap-1 p-2.5 bg-muted/30 dark:bg-black/20 rounded-lg border border-border hover:bg-muted/50 dark:hover:bg-black/40 transition-colors ${metric.label === 'Max Drawdown' ? 'col-span-2' : ''}`}
          >
            <span className="text-[9px] text-muted-foreground uppercase font-black tracking-wider">{metric.label}</span>
            <span className={`text-base font-mono font-black ${
              metric.isDanger ? 'text-[var(--signal-sell)]' : 
              metric.label === 'Target Size' ? 'text-amber-500' :
              'text-primary dark:text-foreground'
            }`}>
              {metric.value}
            </span>
            
            {metric.label === 'Max Drawdown' && risk.peak_equity > 0 && (
              <div className="mt-2 pt-2 border-t border-border/50 grid grid-cols-2 gap-4 text-[8px] font-mono">
                 <div className="flex flex-col">
                    <span className="text-muted-foreground uppercase mb-0.5">Peak Balance</span>
                    <span className="font-bold text-foreground">{currency}{risk.peak_equity.toLocaleString()}</span>
                    <span className="text-[7px] text-muted-foreground opacity-70">({new Date(risk.peak_date).toLocaleDateString()})</span>
                 </div>
                 <div className="flex flex-col items-end text-right">
                    <span className="text-muted-foreground uppercase mb-0.5">Trough Balance</span>
                    <span className="font-bold text-[var(--signal-sell)]">{currency}{risk.trough_equity.toLocaleString()}</span>
                    <span className="text-[7px] text-muted-foreground opacity-70">({new Date(risk.trough_date).toLocaleDateString()})</span>
                 </div>
              </div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
