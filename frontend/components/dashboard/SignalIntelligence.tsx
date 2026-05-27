import React from 'react';
import { ChartData } from '@/types';
import { 
  Cpu, Clock, Unlock, Eye, Lock, ArrowRightCircle, Target, Newspaper, Zap
} from 'lucide-react';
import { motion, Variants } from 'framer-motion';
import { cn } from '@/lib/utils';

interface SignalIntelligenceProps {
  data: ChartData | null;
  currency?: string;
}

export function SignalIntelligence({ data, currency = '$' }: SignalIntelligenceProps) {
  if (!data || !data.models) return (
    <div className="min-h-[280px] w-full flex items-center justify-center text-muted-foreground font-mono text-xs uppercase tracking-widest border border-white/5 rounded-xl bg-card/20">
      Awaiting Signal Telemetry...
    </div>
  );

  const { 
    market_regime, structural_regime,
    volatility_state, models, projections, 
    explainable_confidence, confidence_breakdown,
    execution_state, execution_reasoning, signal_bias,
    forecast_interpretation, forecast_explanation, consensus_intelligence,
    timing_reason, qualitative_alpha, sentiment_score
  } = data;

  // PRIORITY 1: Execution Authority & Confidence
  const isSuppressed = execution_state === 'BLOCKED' || execution_state === 'VETOED' || execution_state === 'OBSERVE ONLY' || execution_state === 'CALIBRATION UNSTABLE';

  const getExecutionColor = (state: string) => {
    if (state.includes('EXECUTE')) return 'text-emerald-500 border-emerald-500/30 bg-emerald-500/10';
    if (state === 'REDUCED SIZE') return 'text-amber-500 border-amber-500/30 bg-amber-500/10';
    if (state === 'BLOCKED' || state === 'VETOED') return 'text-red-500 border-red-500/30 bg-red-500/10';
    return 'text-zinc-500 border-zinc-500/30 bg-zinc-500/10';
  };

  const getExecutionIcon = (state: string) => {
    if (state.includes('EXECUTE')) return <Unlock className="w-4 h-4" />;
    if (state === 'REDUCED SIZE') return <ArrowRightCircle className="w-4 h-4" />;
    if (state === 'BLOCKED' || state === 'VETOED') return <Lock className="w-4 h-4" />;
    return <Eye className="w-4 h-4" />;
  };

  const getBiasColor = (bias: string) => {
     if (bias === 'BULLISH') return 'text-emerald-500';
     if (bias === 'BEARISH') return 'text-red-500';
     return 'text-zinc-500';
  };

  const container: Variants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.05 } }
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
  };

  const holdingTime = typeof data.trade_parameters?.holding_time_estimate === 'number' 
    ? data.trade_parameters.holding_time_estimate 
    : 10;

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4 h-full" data-tour="intelligence">
      
      {/* PRIORITY 1: INSTITUTIONAL EXECUTION PANEL */}
      <motion.div variants={item} className={cn(
        "p-4 rounded-xl border-2 flex flex-col md:flex-row gap-6 items-center transition-all duration-500",
        isSuppressed ? "bg-zinc-900/40 border-zinc-800" : "bg-emerald-950/10 border-emerald-500/20 shadow-[0_0_20px_rgba(16,185,129,0.05)]"
      )}>
        <div className="flex flex-col gap-2 items-center md:items-start min-w-[200px]">
           <div className={cn("px-3 py-1 rounded-full border text-[10px] font-black uppercase tracking-[0.2em] flex items-center gap-2", getExecutionColor(execution_state))}>
              {getExecutionIcon(execution_state)} {execution_state}
           </div>
           <p className="text-[11px] font-medium text-muted-foreground leading-tight text-center md:text-left italic">
             &quot;{execution_reasoning}&quot;
           </p>
        </div>

        <div className="h-px w-full md:h-12 md:w-px bg-border/40" />

        <div className="flex flex-1 justify-around w-full gap-4">
           <div className="flex flex-col items-center gap-1">
              <span className="text-[8px] font-black uppercase text-muted-foreground tracking-tighter">Probabilistic Confidence</span>
              <div className="flex items-end gap-1">
                 <span className="text-2xl font-mono font-black text-primary leading-none">{explainable_confidence?.toFixed(1)}%</span>
                 <span className="text-[9px] font-bold opacity-40 mb-1">AGG</span>
              </div>
           </div>
           
           <div className="flex flex-col items-center gap-1">
              <span className="text-[8px] font-black uppercase text-muted-foreground tracking-tighter">Expected Value</span>
              <div className="flex items-end gap-1">
                 <span className={cn("text-2xl font-mono font-black leading-none", (data.expected_value?.ev_pct ?? 0) > 0 ? "text-emerald-500" : "text-red-500")}>
                   {(data.expected_value?.ev_pct ?? 0).toFixed(2)}%
                 </span>
                 <span className="text-[9px] font-bold opacity-40 mb-1">EV</span>
              </div>
           </div>

           <div className="flex flex-col items-center gap-1">
              <span className="text-[8px] font-black uppercase text-muted-foreground tracking-tighter">Predictive Bias</span>
              <div className="flex items-end gap-1">
                 <span className={cn("text-xl font-black uppercase leading-none", getBiasColor(signal_bias))}>
                    {signal_bias}
                 </span>
              </div>
              <span className="text-[8px] font-bold opacity-60 uppercase">{forecast_interpretation}</span>
           </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-1">
        {/* PRIORITY 2: ENSEMBLE INTELLIGENCE */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group border-border hover:border-primary/30 transition-all">
          <div className="bg-secondary/30 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Consensus Engine</h3>
            </div>
            <span className="text-[8px] font-mono font-bold opacity-50 uppercase tracking-tighter">{consensus_intelligence}</span>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-1.5 justify-center">
            {Object.entries(models).map(([name, pred]) => {
              if (name.includes("META") || name.includes("ENSEMBLE")) return null;
              return (
                <div key={name} className="flex items-center justify-between p-1.5 rounded bg-muted/20 border border-transparent hover:border-border transition-colors">
                  <span className="text-[9px] font-mono text-muted-foreground uppercase font-bold">{name}</span>
                  <div className="flex items-center gap-2">
                    <span className={cn(
                      "text-[8px] font-black uppercase px-1.5 py-0.5 rounded border",
                      pred.signal === 'BUY' ? "text-emerald-500 border-emerald-500/20 bg-emerald-500/5" :
                      pred.signal === 'SELL' ? "text-red-500 border-red-500/20 bg-red-500/5" :
                      "text-zinc-500 border-zinc-500/20 bg-zinc-500/5"
                    )}>{pred.signal}</span>
                    <span className="text-[9px] font-mono font-bold text-primary dark:text-foreground w-8 text-right">
                      {Math.round(pred.probability * 100)}%
                    </span>
                  </div>
                </div>
              );
            })}
            
            {confidence_breakdown && (
              <div className="grid grid-cols-3 gap-1 mt-2 p-1.5 rounded bg-black/10">
                {Object.entries(confidence_breakdown).filter(([k]) => k !== 'Total_Raw_Score').map(([key, val]) => (
                  <div key={key} className="flex flex-col items-center p-0.5">
                     <span className="text-[5px] font-black uppercase text-muted-foreground text-center truncate w-full">{key.replace('_', ' ')}</span>
                     <span className={cn("text-[8px] font-mono font-bold", val > 0 ? "text-emerald-500" : val < 0 ? "text-red-500" : "text-zinc-500")}>
                       {val > 0 ? '+' : ''}{val.toFixed(0)}
                     </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>

        {/* PRIORITY 2: TIMING & REGIME */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group border-border">
          <div className="bg-secondary/30 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-3.5 h-3.5 text-amber-500" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Timing & Regime</h3>
            </div>
            <span className="text-[8px] font-mono font-bold opacity-50 uppercase tracking-tighter">Predictive_V2.1</span>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-4 justify-center">
            <div className="flex flex-col gap-1">
               <div className="flex justify-between items-center px-1">
                  <span className="text-[9px] font-black uppercase text-muted-foreground">Market Structure</span>
                  <span className={cn(
                    "text-[9px] font-black px-2 py-0.5 rounded-full uppercase",
                    market_regime === 'BULL' ? "bg-emerald-500/10 text-emerald-500" : market_regime === 'BEAR' ? "bg-red-500/10 text-red-500" : "bg-zinc-500/10 text-zinc-500"
                  )}>{structural_regime}</span>
               </div>
               <div className="h-px bg-border/20 w-full" />
            </div>

            <div className="flex flex-col gap-2">
               <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded bg-amber-500/10"><Clock className="w-3 h-3 text-amber-500" /></div>
                  <div className="flex flex-col">
                     <span className="text-[8px] font-black uppercase text-muted-foreground">Velocity Intelligence</span>
                     <p className="text-[10px] font-medium leading-tight">{timing_reason}</p>
                  </div>
               </div>
            </div>

            <div className="mt-auto pt-2 flex items-center justify-between border-t border-border/30">
               <div className="flex flex-col">
                  <span className="text-[7px] font-black uppercase text-muted-foreground opacity-50">Volume Dynamics</span>
                  <span className={cn("text-[10px] font-mono font-bold", data.volume_ratio > 1.2 ? "text-emerald-500" : "text-foreground")}>
                    {data.volume_ratio.toFixed(2)}x Baseline
                  </span>
               </div>
               <div className="flex flex-col items-end">
                  <span className="text-[7px] font-black uppercase text-muted-foreground opacity-50">Volatility State</span>
                  <span className={cn(
                    "text-[10px] font-mono font-black",
                    volatility_state === 'HIGH' ? "text-red-500" : volatility_state === 'LOW' ? "text-emerald-500" : "text-foreground"
                  )}>{volatility_state}</span>
               </div>
            </div>
          </div>
        </motion.div>

        {/* PRIORITY 2: FORECAST & RISK */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group border-border">
          <div className="bg-secondary/30 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Trade Constraints</h3>
            </div>
            <span className="text-[8px] font-mono font-bold opacity-50 uppercase tracking-tighter">Validated_Risk</span>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-4 justify-center">
            <div className="flex items-center justify-between font-mono text-sm relative z-10 px-2">
                <div className="flex flex-col items-center">
                  <span className="text-muted-foreground text-[8px] font-bold uppercase mb-1">P10 Floor</span>
                  <span className="text-red-500 font-black text-sm">{currency}{projections.floor.toFixed(2)}</span>
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-muted-foreground text-[8px] font-bold uppercase mb-1">P50 Median</span>
                  <span className="text-foreground font-black text-base">{currency}{projections.median?.toFixed(2)}</span>
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-muted-foreground text-[8px] font-bold uppercase mb-1">P90 Ceiling</span>
                  <span className="text-emerald-500 font-black text-sm">{currency}{projections.ceiling.toFixed(2)}</span>
                </div>
            </div>

            <div className="flex flex-col gap-1.5 p-2 rounded bg-muted/30 border border-border">
               <div className="flex justify-between items-center">
                  <span className="text-[8px] font-black uppercase text-muted-foreground">RR Ratio</span>
                  <span className={cn("text-[10px] font-mono font-bold", (data.risk?.risk_reward_ratio ?? 0) >= 1.5 ? "text-emerald-500" : "text-red-500")}>
                    {(data.risk?.risk_reward_ratio ?? 0).toFixed(2)}
                  </span>
               </div>
               <div className="flex justify-between items-center">
                  <span className="text-[8px] font-black uppercase text-muted-foreground">Target Horizon</span>
                  <span className="text-[10px] font-mono font-bold">{holdingTime} Days</span>
               </div>
            </div>

            <p className="text-[9px] text-center text-muted-foreground leading-tight italic px-2">{forecast_explanation}</p>
          </div>
        </motion.div>
      </div>
      
      {/* PRIORITY 3: QUALITATIVE ALPHA & FUNDAMENTAL CONTEXT */}
      <motion.div variants={item} className="p-3 bg-card/10 border border-border/40 rounded-xl flex items-center gap-4">
        <Newspaper className="w-4 h-4 text-muted-foreground shrink-0" />
        <p className="text-[10px] text-foreground/80 font-medium leading-relaxed italic flex-1 truncate">
          &quot;{qualitative_alpha || "Awaiting Gemini fundamental context layer synchronization."}&quot;
        </p>
        <div className="flex items-center gap-4 border-l border-border/40 pl-4 shrink-0">
           <div className="flex flex-col">
              <span className="text-[7px] font-black text-muted-foreground uppercase leading-none mb-1">Sentiment</span>
              <span className={cn("text-[10px] font-mono font-bold", (sentiment_score ?? 0) > 0 ? "text-emerald-500" : "text-red-500")}>
                {(sentiment_score ?? 0).toFixed(2)}
              </span>
           </div>
           <div className="flex flex-col">
              <span className="text-[7px] font-black text-muted-foreground uppercase leading-none mb-1">Alpha Source</span>
              <span className="text-[10px] font-mono font-bold text-primary italic">Live_Institutional</span>
           </div>
        </div>
      </motion.div>

    </motion.div>
  );
}
