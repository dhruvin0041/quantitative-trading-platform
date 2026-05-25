import React, { useEffect, useState } from 'react';
import { Trophy } from 'lucide-react';
import { motion } from 'framer-motion';

export function PerformanceValidation() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [metrics, setMetrics] = useState<any>(null);

  useEffect(() => {
    const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
    const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    
    fetch(`${API_URL}/performance`, {
      headers: { "X-API-Key": API_KEY }
    })
      .then(res => res.json())
      .then(data => {
        if (data.summary) {
          setMetrics([
            { label: 'Sharpe Ratio', value: data.summary.sharpe.toFixed(2), description: 'Risk-adj return' },
            { label: 'Sortino Ratio', value: data.summary.sortino.toFixed(2), description: 'Downside risk-adj' },
            { label: 'Calmar Ratio', value: data.summary.calmar.toFixed(2), description: 'Return vs Drawdown' },
            { label: 'Profit Factor', value: data.summary.profit_factor.toFixed(2), description: 'Gross Profit / Loss' },
            { label: 'Win Rate', value: `${data.summary.win_rate.toFixed(1)}%`, description: 'Profitable trades' },
          ]);
        }
      })
      .catch(err => console.error("Failed to fetch performance:", err));
  }, []);

  if (!metrics) return null;

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300 flex-1">
      <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Trophy className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">System Validation</h3>
        </div>
        <div className="text-[8px] font-mono font-bold opacity-50 uppercase tracking-tighter">Live_Telemetry</div>
      </div>
      <div className="p-4 flex flex-col gap-2 flex-1 justify-center">
        {metrics.map((metric: any, i: number) => (
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          <motion.div 
            key={metric.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1, duration: 0.3 }}
            className="flex items-center justify-between p-2 rounded-md hover:bg-muted/50 dark:hover:bg-white/5 border border-transparent transition-colors"
          >
            <div className="flex flex-col">
              <span className="text-xs font-black text-foreground font-sans">{metric.label}</span>
              <span className="text-[9px] uppercase tracking-wider text-muted-foreground font-bold">{metric.description}</span>
            </div>
            <span className="text-sm font-mono font-black text-primary dark:text-foreground">
              {metric.value}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
