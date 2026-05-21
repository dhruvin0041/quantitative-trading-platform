import React from 'react';
import { Trophy } from 'lucide-react';
import { motion } from 'framer-motion';

export function PerformanceValidation() {
  const perfMetrics = [
    { label: 'Sharpe Ratio', value: '2.14', description: 'Risk-adj return' },
    { label: 'Sortino Ratio', value: '3.42', description: 'Downside risk-adj' },
    { label: 'Calmar Ratio', value: '1.85', description: 'Return vs Drawdown' },
    { label: 'Profit Factor', value: '1.76', description: 'Gross Profit / Loss' },
    { label: 'Win Rate', value: '64.2%', description: 'Profitable trades' },
  ];

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group hover:border-white/20 transition-colors flex-1">
      <div className="bg-black/40 border-b border-white/5 px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Trophy className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">System Validation</h3>
        </div>
      </div>
      <div className="p-4 flex flex-col gap-2 flex-1 justify-center">
        {perfMetrics.map((metric, i) => (
          <motion.div 
            key={metric.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1, duration: 0.3 }}
            className="flex items-center justify-between p-2 rounded-md hover:bg-white/5 border border-transparent hover:border-white/10 transition-colors"
          >
            <div className="flex flex-col">
              <span className="text-xs font-semibold text-foreground">{metric.label}</span>
              <span className="text-[9px] uppercase tracking-wider text-muted-foreground/70">{metric.description}</span>
            </div>
            <span className="text-sm font-mono font-bold text-foreground">
              {metric.value}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
