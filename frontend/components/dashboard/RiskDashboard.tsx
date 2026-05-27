import React from 'react';
import { ShieldAlert, Info } from 'lucide-react';
import { motion } from 'framer-motion';
import { ChartData } from '@/types';
import { cn } from '@/lib/utils';

interface RiskDashboardProps {
  data: ChartData | null;
  currency?: string;
}

export function RiskDashboard({ data, currency = "$" }: RiskDashboardProps) {
  if (!data || !data.risk) return null;

  const { risk } = data;

  const getRiskSeverity = (val: number, thresholds: {low: number, high: number}, reverse = false) => {
    if (reverse) {
      if (val < thresholds.low) return { level: 'SEVERE', color: 'text-red-500', bg: 'bg-red-500/10 border-red-500/30' };
      if (val < thresholds.high) return { level: 'ELEVATED', color: 'text-amber-500', bg: 'bg-amber-500/10 border-amber-500/30' };
      return { level: 'NOMINAL', color: 'text-green-500', bg: 'bg-green-500/10 border-green-500/30' };
    }
    if (val > thresholds.high) return { level: 'SEVERE', color: 'text-red-500', bg: 'bg-red-500/10 border-red-500/30' };
    if (val > thresholds.low) return { level: 'ELEVATED', color: 'text-amber-500', bg: 'bg-amber-500/10 border-amber-500/30' };
    return { level: 'NOMINAL', color: 'text-green-500', bg: 'bg-green-500/10 border-green-500/30' };
  };

  const varRisk = getRiskSeverity(Math.abs(risk.var_95), { low: 0.02, high: 0.05 });
  const cvarRisk = getRiskSeverity(Math.abs(risk.cvar), { low: 0.03, high: 0.08 });
  const betaRisk = getRiskSeverity(Math.abs(risk.beta), { low: 1.2, high: 2.0 });

  const getBetaInterpretation = (beta: number) => {
    if (beta > 1.5) return "Hyper-sensitive to market";
    if (beta > 1) return "Aggressive vs benchmark";
    if (beta < 0) return "Inverse correlation";
    if (beta < 0.5) return "Defensive / Uncorrelated";
    return "Tracks benchmark";
  };

  const riskMetrics = [
    { 
      label: 'VaR (95%)', 
      value: `${(risk.var_95 * 100).toFixed(2)}%`, 
      interpretation: "1-Day Max Loss",
      severity: varRisk
    },
    { 
      label: 'CVaR', 
      value: `${(risk.cvar * 100).toFixed(2)}%`, 
      interpretation: "Tail Risk Loss",
      severity: cvarRisk
    },
    { 
      label: 'Beta', 
      value: risk.beta.toFixed(2), 
      interpretation: getBetaInterpretation(risk.beta),
      severity: betaRisk
    },
    { 
      label: 'Kelly Frac.', 
      value: (risk.kelly_fraction * 100).toFixed(1) + '%', 
      interpretation: "Optimal sizing limit",
      severity: { level: 'INFO', color: 'text-blue-500', bg: 'bg-blue-500/10 border-blue-500/30' }
    },
    { 
      label: 'Target Size', 
      value: `${currency}${risk.target_size.toLocaleString()}`, 
      interpretation: "Capital at risk",
      severity: { level: 'ACTION', color: 'text-amber-500', bg: 'bg-amber-500/10 border-amber-500/30' }
    },
    { 
      label: 'Max Drawdown', 
      value: `${risk.max_drawdown.toFixed(1)}%`, 
      interpretation: "Historical worst-case",
      severity: getRiskSeverity(Math.abs(risk.max_drawdown), { low: 15, high: 30 })
    },
  ];

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
      <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Institutional Risk Context</h3>
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
            className={`flex flex-col gap-1.5 p-3 bg-muted/30 dark:bg-black/20 rounded-lg border border-border hover:bg-muted/50 dark:hover:bg-black/40 transition-colors ${metric.label === 'Max Drawdown' ? 'col-span-2' : ''}`}
          >
            <div className="flex justify-between items-start">
               <span className="text-[9px] text-muted-foreground uppercase font-black tracking-wider">{metric.label}</span>
               <span className={cn("text-[7px] px-1.5 py-0.5 rounded font-black uppercase tracking-widest border", metric.severity.bg, metric.severity.color)}>
                 {metric.severity.level}
               </span>
            </div>
            
            <div className="flex items-end gap-2">
               <span className={cn("text-base font-mono font-black", metric.severity.color)}>
                 {metric.value}
               </span>
            </div>
            
            <span className="text-[8px] text-muted-foreground italic leading-tight flex items-center gap-1">
               <Info className="w-2.5 h-2.5 opacity-70" />
               {metric.interpretation}
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