import React from 'react';
import { ShieldAlert } from 'lucide-react';
import { motion } from 'framer-motion';

export function RiskDashboard() {
  const riskMetrics = [
    { label: 'VaR (95%)', value: '-$2,450', isDanger: true },
    { label: 'CVaR', value: '-$3,100', isDanger: true },
    { label: 'Beta', value: '0.85', isDanger: false },
    { label: 'Kelly Frac.', value: '0.24', isDanger: false },
    { label: 'Max Drawdown', value: '-12.4%', isDanger: true },
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
            className={`flex flex-col gap-1 p-2.5 bg-muted/30 dark:bg-black/20 rounded-lg border border-border hover:bg-muted/50 dark:hover:bg-black/40 transition-colors ${i === riskMetrics.length - 1 ? 'col-span-2 text-center items-center' : ''}`}
          >
            <span className="text-[9px] text-muted-foreground uppercase font-black tracking-wider">{metric.label}</span>
            <span className={`text-base font-mono font-black ${metric.isDanger ? 'text-[var(--signal-sell)]' : 'text-primary dark:text-foreground'}`}>
              {metric.value}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
