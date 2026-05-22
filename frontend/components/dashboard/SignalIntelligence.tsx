import React from 'react';
import { ChartData } from '@/types';
import { Cpu, AlertTriangle, Newspaper, Activity, AlertCircle } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

import { Variants } from 'framer-motion';

interface SignalIntelligenceProps {
  data: ChartData | null;
}

export function SignalIntelligence({ data }: SignalIntelligenceProps) {
  if (!data || !data.models) return (
    <div className="h-full w-full flex items-center justify-center text-muted-foreground font-mono text-xs uppercase tracking-widest border border-white/5 rounded-xl bg-card/20">
      Awaiting Signal Telemetry...
    </div>
  );

  const { signal, confidence_score, market_regime, volatility_state, volume_ratio, models, projections, technical_snapshot, signal_note, qualitative_alpha } = data;

  const getSignalColor = (action: string) => {
    if (action.includes('BUY')) return 'text-[var(--signal-buy)] border-[var(--signal-buy)]/30 bg-[var(--signal-buy)]/10 dark:text-[var(--signal-buy)] dark:border-[var(--signal-buy)]/30 dark:bg-[var(--signal-buy)]/10';
    if (action.includes('SELL')) return 'text-[var(--signal-sell)] border-[var(--signal-sell)]/30 bg-[var(--signal-sell)]/10 dark:text-[var(--signal-sell)] dark:border-[var(--signal-sell)]/30 dark:bg-[var(--signal-sell)]/10';
    return 'text-[var(--signal-hold)] border-[var(--signal-hold)]/30 bg-[var(--signal-hold)]/10 dark:text-[var(--signal-hold)] dark:border-[var(--signal-hold)]/30 dark:bg-[var(--signal-hold)]/10';
  };

  const getRegimeColor = (regime: string) => {
    if (regime === 'BULL') return 'bg-green-500/20 text-green-500 border-green-500/30';
    if (regime === 'BEAR') return 'bg-red-500/20 text-red-500 border-red-500/30';
    return 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30';
  };

  const getVolatilityColor = (state: string) => {
    if (state === 'HIGH') return 'bg-orange-500/20 text-orange-500 border-orange-500/30';
    if (state === 'MEDIUM') return 'bg-yellow-500/20 text-yellow-500 border-yellow-500/30';
    return 'bg-blue-500/20 text-blue-500 border-blue-500/30';
  };

  const getRSIColor = (rsi: number) => {
    if (rsi > 70) return 'text-red-500';
    if (rsi < 30) return 'text-green-500';
    return 'text-foreground';
  };

  const getADXColor = (adx: number) => {
    if (adx > 25) return 'text-orange-500';
    if (adx < 20) return 'text-muted-foreground';
    return 'text-foreground';
  };

  const container: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 10 },
    show: { 
      opacity: 1, 
      y: 0, 
      transition: { type: "spring", stiffness: 300, damping: 24 } 
    }
  };

  return (
    <motion.div 
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-col gap-4 h-full"
      data-tour="intelligence"
    >
      {/* Warning Banner */}
      {signal_note && (
        <motion.div variants={item} className="flex items-center gap-2 px-4 py-2 bg-orange-500/10 border border-orange-500/30 rounded-lg text-orange-500 text-xs font-bold">
          <AlertCircle className="w-4 h-4" />
          <span>⚠️ {signal_note}</span>
        </motion.div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-1">
        {/* Model Consensus */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
          <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Model Consensus</h3>
            </div>
            <div className="flex gap-1.5">
              <span className={cn("text-[8px] font-black px-1.5 py-0.5 rounded border uppercase", getRegimeColor(market_regime))}>{market_regime}</span>
              <span className={cn("text-[8px] font-black px-1.5 py-0.5 rounded border uppercase", getVolatilityColor(volatility_state))}>{volatility_state} VOL</span>
            </div>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-2 justify-center">
            {Object.entries(models).map(([name, pred]) => (
              <div key={name} className="flex items-center justify-between p-2 rounded border border-border bg-muted/30 dark:bg-black/20 group-hover:bg-muted/50 dark:group-hover:bg-black/40 transition-colors">
                <span className="text-[10px] font-mono text-muted-foreground uppercase font-bold">{name}</span>
                <div className="flex items-center gap-2">
                  <span className={cn("text-[9px] font-black uppercase px-2 py-0.5 rounded border", getSignalColor(pred.signal))}>
                    {pred.signal}
                  </span>
                  <span className="text-[10px] font-mono font-bold text-primary dark:text-foreground w-10 text-right">{Math.round(pred.probability * 100)}%</span>
                </div>
              </div>
            ))}
            <div className="mt-2 pt-2 border-t border-border/50 flex justify-between items-center">
              <span className="text-[9px] font-bold text-muted-foreground uppercase">Volume Ratio</span>
              <span className={cn("text-[10px] font-mono font-bold", volume_ratio < 0.7 ? "text-red-500" : "text-green-500")}>{volume_ratio.toFixed(2)}</span>
            </div>
          </div>
        </motion.div>

        {/* Risk Projections */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
          <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">10-Day Projections</h3>
            </div>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-center gap-4">
            <div className="flex items-center justify-between font-mono text-sm relative">
              <div className="flex flex-col">
                <span className="text-muted-foreground text-[10px] font-bold uppercase mb-1 text-center">Floor</span>
                <span className="text-[var(--signal-sell)] font-black text-xl">${projections.floor.toFixed(2)}</span>
              </div>
              <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1/3 h-px bg-border opacity-50"></div>
              <div className="flex flex-col">
                <span className="text-muted-foreground text-[10px] font-bold uppercase mb-1 text-center">Ceiling</span>
                <span className="text-[var(--signal-buy)] font-black text-xl">${projections.ceiling.toFixed(2)}</span>
              </div>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed border-l-4 border-primary/30 pl-3 bg-primary/5 py-2 rounded-r-md italic">
              <span className="text-primary font-black uppercase text-[9px] not-italic mr-1">Signal:</span> {signal} ({confidence_score.toFixed(1)}%)
            </p>
          </div>
        </motion.div>

        {/* Technical Snapshot */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
          <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Technical Snapshot</h3>
            </div>
          </div>
          <div className="p-4 flex-1 grid grid-cols-3 gap-3">
            <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
              <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">RSI</span>
              <span className={cn("text-xs font-mono font-black", getRSIColor(technical_snapshot.RSI))}>{technical_snapshot.RSI}</span>
            </div>
            <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
              <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">MACD</span>
              <span className="text-xs font-mono font-black">{technical_snapshot.MACD}</span>
            </div>
            <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
              <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">ATR</span>
              <span className="text-xs font-mono font-black">{technical_snapshot.ATR}</span>
            </div>
            <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
              <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">BB Pos</span>
              <span className="text-xs font-mono font-black">{technical_snapshot.BB_Position}</span>
            </div>
            <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
              <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">ADX</span>
              <span className={cn("text-xs font-mono font-black", getADXColor(technical_snapshot.ADX))}>{technical_snapshot.ADX}</span>
            </div>
            <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
              <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">Vol Ratio</span>
              <span className="text-xs font-mono font-black">{technical_snapshot.Volume_Ratio}</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Qualitative Alpha / NLP */}
      <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300 h-24">
        <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-1 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Newspaper className="w-3 h-3 text-primary" />
            <h3 className="text-[9px] font-black uppercase tracking-widest text-muted-foreground">Qualitative Alpha (Gemini)</h3>
          </div>
        </div>
        <div className="p-3 flex-1 overflow-y-auto">
          <p className="text-[10px] text-foreground font-medium leading-relaxed italic">
            &quot;{qualitative_alpha || "No qualitative data available for this cycle."}&quot;
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}
