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

  const getRegimeColor = (regime: string) => {
    const r = regime.toUpperCase();
    if (r.includes('STABLE')) return { color: 'text-emerald-500', bg: 'bg-emerald-500', border: 'border-emerald-500/30', glow: 'shadow-[0_0_15px_rgba(16,185,129,0.3)]' };
    if (r.includes('ELEVATED')) return { color: 'text-amber-500', bg: 'bg-amber-500', border: 'border-amber-500/30', glow: 'shadow-[0_0_15px_rgba(245,158,11,0.3)]' };
    if (r.includes('DEFENSIVE')) return { color: 'text-orange-500', bg: 'bg-orange-500', border: 'border-orange-500/30', glow: 'shadow-[0_0_15px_rgba(249,115,22,0.3)]' };
    if (r.includes('CRITICAL')) return { color: 'text-red-500', bg: 'bg-red-500', border: 'border-red-500/30', glow: 'shadow-[0_0_15px_rgba(239,68,68,0.3)]' };
    if (r.includes('PANIC')) return { color: 'text-red-600', bg: 'bg-red-600', border: 'border-red-600/30', glow: 'shadow-[0_0_20px_rgba(220,38,38,0.5)]' };
    return { color: 'text-muted-foreground', bg: 'bg-muted', border: 'border-border', glow: '' };
  };

  const regime = getRegimeColor(risk.risk_regime);

  const riskMetrics = [
    { label: 'Beta (10D)', value: risk.beta.toFixed(2), interpretation: risk.beta > 1 ? 'High Volatility' : 'Nominal' },
    { label: 'Kelly Fraction', value: `${(risk.kelly_fraction * 100).toFixed(1)}%`, interpretation: 'Suggested Sizing' },
    { label: 'Expected Value', value: `${risk.expected_value?.toFixed(2) ?? 'N/A'}%`, interpretation: 'Statistical Edge' },
    { label: 'Win Probability', value: `${(risk.win_probability * 100).toFixed(1)}%`, interpretation: 'Confidence' },
  ];

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
      <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-primary" />
          <h3 className="text-[11px] font-black uppercase tracking-widest text-foreground/80">Risk Context Engine</h3>
        </div>
        <div className={cn("text-[9px] font-mono font-bold uppercase tracking-widest px-2 py-0.5 rounded border-2 transition-all duration-500", regime.color, regime.border, regime.glow)}>
          {risk.risk_regime}
        </div>
      </div>

      <div className="p-5 flex flex-col gap-6">
        {/* Continuous Risk Meter (Phase 9) */}
        <div className="flex flex-col gap-3">
          <div className="flex justify-between items-end">
            <div className="flex flex-col gap-0.5">
               <span className="text-[9px] text-muted-foreground uppercase font-black tracking-[0.15em]">Institutional Index</span>
               <span className="text-[7px] text-muted-foreground/60 italic">Multi-weighted risk aggregation</span>
            </div>
            <span className={cn("text-3xl font-mono font-black tracking-tighter tabular-nums", regime.color)}>
              {risk.institutional_risk_index.toFixed(1)}
            </span>
          </div>
          <div className="h-3 w-full bg-muted/30 dark:bg-white/5 rounded-sm overflow-hidden flex gap-0.5 p-0.5 border border-white/5 relative">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${risk.institutional_risk_index}%` }}
              className={cn("h-full rounded-sm transition-all duration-1000 relative z-10", regime.bg)}
            />
            {/* Range markers */}
            <div className="absolute inset-0 flex justify-between px-1 items-center pointer-events-none opacity-20">
               {[20, 40, 60, 80].map(m => (
                 <div key={m} className="w-px h-1.5 bg-foreground" style={{ left: `${m}%` }} />
               ))}
            </div>
          </div>
          <div className="flex justify-between text-[7px] text-muted-foreground font-mono uppercase tracking-tighter opacity-70 font-bold">
            <span>Stable</span>
            <span>Elevated</span>
            <span>Defensive</span>
            <span>Critical</span>
            <span>Panic</span>
          </div>
        </div>


        <div className="grid grid-cols-2 gap-3">
          {riskMetrics.map((metric, i) => (
            <motion.div 
              key={metric.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="p-3 bg-muted/20 dark:bg-white/5 rounded-lg border border-border/50 flex flex-col gap-1"
            >
              <span className="text-[8px] text-muted-foreground uppercase font-bold tracking-wider">{metric.label}</span>
              <span className="text-sm font-mono font-black">{metric.value}</span>
              <span className="text-[7px] text-muted-foreground italic">{metric.interpretation}</span>
            </motion.div>
          ))}
        </div>

        {/* Drawdown Section */}
        <div className="p-3 bg-red-500/5 dark:bg-red-500/10 rounded-lg border border-red-500/20 flex flex-col gap-2">
          <div className="flex justify-between items-center text-[9px] uppercase font-black tracking-widest text-red-500/80">
            <span>Historical Max Drawdown</span>
            <span className="font-mono">{risk.max_drawdown.toFixed(1)}%</span>
          </div>
          <div className="grid grid-cols-2 gap-4 text-[8px] font-mono">
             <div className="flex flex-col">
                <span className="text-muted-foreground uppercase mb-0.5 opacity-50">Peak Balance</span>
                <span className="font-bold text-foreground/80">{currency}{risk.peak_equity.toLocaleString()}</span>
             </div>
             <div className="flex flex-col items-end text-right">
                <span className="text-muted-foreground uppercase mb-0.5 opacity-50">Trough Balance</span>
                <span className="font-bold text-red-400">{currency}{risk.trough_equity.toLocaleString()}</span>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}