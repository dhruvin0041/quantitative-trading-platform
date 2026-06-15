import React, { useEffect, useState } from 'react';
import { Trophy } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { API_KEY, getBaseUrl } from '@/lib/config';

interface ValidationMetric {
  label: string;
  value: string;
  description: string;
}

export function PerformanceValidation() {
  const [metrics, setMetrics] = useState<ValidationMetric[] | null>(null);
  const API_URL = getBaseUrl();

  useEffect(() => {
    fetch(`${API_URL}/performance`, {
      headers: { "X-API-Key": API_KEY }
    })
      .then(res => res.json())
      .then(data => {
        if (data.summary) {
          const format = (val: number | string, decimals: number, suffix: string = '') => {
            if (typeof val === 'string') return val;
            return `${val.toFixed(decimals)}${suffix}`;
          };

          setMetrics([
            { label: 'Sharpe Ratio', value: format(data.summary.sharpe, 2), description: 'Risk-adj return' },
            { label: 'Sortino Ratio', value: format(data.summary.sortino, 2), description: 'Downside risk-adj' },
            { label: 'Calmar Ratio', value: format(data.summary.calmar, 2), description: 'Return vs Drawdown' },
            { label: 'Profit Factor', value: format(data.summary.profit_factor, 2), description: 'Gross Profit / Loss' },
            { label: 'Win Rate', value: format(data.summary.win_rate, 1, '%'), description: 'Profitable trades' },
          ]);
        }
      })
      .catch(err => console.error("Failed to fetch performance:", err));
  }, [API_URL]);

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
        {metrics.map((metric: ValidationMetric, i: number) => (
          <motion.div 
            key={metric.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1, duration: 0.3 }}
            className="flex items-center justify-between p-2 rounded-md hover:bg-muted/50 dark:hover:bg-white/5 border border-transparent transition-colors"
          >
            <div className="flex flex-col flex-1 shrink min-w-0 pr-2">
              <span className="text-xs font-black text-foreground font-sans truncate">{metric.label}</span>
              <span className="text-[9px] uppercase tracking-wider text-muted-foreground font-bold truncate">{metric.description}</span>
            </div>
            <span className={cn(
              "font-mono font-black text-primary dark:text-foreground text-right shrink-0 break-words leading-tight", 
              typeof metric.value === 'string' && metric.value.includes('Insufficient') ? "text-[8px] max-w-[120px]" : "text-sm"
            )}>
              {metric.value}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
