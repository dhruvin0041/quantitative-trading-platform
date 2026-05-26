import React from 'react';
import { ChartData } from '@/types';
import { Cpu, AlertTriangle, Newspaper, AlertCircle, BrainCircuit, Clock, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { formatDistanceToNow } from 'date-fns';

import { Variants } from 'framer-motion';

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
    signal, market_regime, 
    volatility_state, models, projections, 
    is_point_forecast, model_agreement, bullish_models, 
    bearish_models, neutral_models, timestamp,
    signal_note, qualitative_alpha, xai, sentiment_score 
  } = data;

  const isDivergent = (market_regime === 'BULL' && signal.includes('SELL')) || (market_regime === 'BEAR' && signal.includes('BUY'));

  const getSignalColor = (action: string) => {
    if (action.includes('BUY')) return 'text-[var(--signal-buy)] border-[var(--signal-buy)]/30 bg-[var(--signal-buy)]/10';
    if (action.includes('SELL')) return 'text-[var(--signal-sell)] border-[var(--signal-sell)]/30 bg-[var(--signal-sell)]/10';
    return 'text-[var(--signal-hold)] border-[var(--signal-hold)]/30 bg-[var(--signal-hold)]/10';
  };

  const getRegimeColor = (regime: string) => {
    if (regime === 'BULL') return 'bg-green-500 text-white border-green-600';
    if (regime === 'BEAR') return 'bg-red-500 text-white border-red-600';
    return 'bg-zinc-500 text-white border-zinc-600';
  };

  const getVolatilityColor = (state: string) => {
    if (state === 'HIGH') return 'bg-red-500 text-white border-red-600';
    if (state === 'MEDIUM') return 'bg-orange-500 text-white border-orange-600';
    return 'bg-green-500 text-white border-green-600';
  };

  const container: Variants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05 }
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

  const signalAge = timestamp ? formatDistanceToNow(new Date(timestamp), { addSuffix: true }) : 'Unknown';

  // PRIORITY #6: Model Disagreement Intelligence
  const getDisagreementNote = () => {
    const counts = { BULL: bullish_models || 0, BEAR: bearish_models || 0, NEUT: neutral_models || 0 };
    if (model_agreement >= 100) return "Full Consensus: Unified direction confirmed across all agent architectures.";
    if (model_agreement >= 66) {
      const majority = counts.BULL > counts.BEAR ? "Bullish" : "Bearish";
      return `Strong Consensus: ${majority} edge identified with minor technical divergence.`;
    }
    return "Conflicting Consensus: Tree agents and Neural Fusion are reporting divergent structural patterns. Caution advised.";
  };

  return (
    <motion.div 
      variants={container}
      initial="hidden"
      animate="show"
      className="flex flex-col gap-4 h-full"
      data-tour="intelligence"
    >
      {/* Prediction Timestamp Header */}
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2 text-[9px] font-mono font-bold text-muted-foreground uppercase tracking-widest">
           <Clock className="w-3 h-3" />
           Generated: {new Date(timestamp).toLocaleString()} UTC ({signalAge})
        </div>
        <div className="flex items-center gap-3">
           <div className="flex items-center gap-2 text-[9px] font-mono font-bold text-primary uppercase tracking-widest">
              <CheckCircle2 className="w-3 h-3" />
              Agreement: {model_agreement?.toFixed(1)}%
           </div>
           <div className="flex gap-1">
              <span className="text-[8px] font-black px-1.5 py-0.5 rounded bg-green-500/10 text-green-500 border border-green-500/30">B:{bullish_models || 0}</span>
              <span className="text-[8px] font-black px-1.5 py-0.5 rounded bg-red-500/10 text-red-500 border border-red-500/30">S:{bearish_models || 0}</span>
              <span className="text-[8px] font-black px-1.5 py-0.5 rounded bg-zinc-500/10 text-zinc-400 border border-zinc-500/30">H:{neutral_models || 0}</span>
           </div>
        </div>
      </div>

      {/* Warning Banner */}
      {signal_note && (
        <motion.div variants={item} className="flex items-center gap-2 px-4 py-2 bg-orange-100 border border-orange-500 dark:bg-orange-950/50 dark:border-orange-600 rounded-lg text-orange-700 dark:text-orange-400 text-xs font-bold w-full">
          <AlertCircle className="w-4 h-4" />
          <span>⚠️ {signal_note}</span>
        </motion.div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-1">
        {/* Model Consensus */}
        <motion.div 
          variants={item} 
          className={cn(
            "glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300 border-2",
            isDivergent ? "border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.2)]" : "border-border"
          )}
        >
          <div className={cn(
            "border-b px-4 py-2 flex items-center justify-between transition-colors",
            isDivergent ? "bg-amber-500/10 border-amber-500/30" : "bg-secondary/50 dark:bg-black/40 border-border"
          )}>
            <div className="flex items-center gap-2">
              <Cpu className={cn("w-3.5 h-3.5", isDivergent ? "text-amber-500" : "text-primary")} />
              <h3 className={cn(
                "text-[10px] font-black uppercase tracking-widest",
                isDivergent ? "text-amber-500" : "text-muted-foreground"
              )}>Model Consensus</h3>
            </div>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-2 justify-center">
            {Object.entries(models).map(([name, pred]) => {
              const isMeta = name.includes("META") || name.includes("ENSEMBLE");
              if (isMeta) return null;
              
              const prob = pred.probability;
              const probText = prob === 0 ? "N/A" : `${Math.round(prob * 100)}%`;
              
              return (
                <div key={name} className="flex items-center justify-between p-2 rounded border border-border bg-muted/30 dark:bg-black/20 group-hover:bg-muted/50 dark:group-hover:bg-black/40 transition-colors">
                  <span className="text-[10px] font-mono text-muted-foreground uppercase font-bold">{name}</span>
                  <div className="flex items-center gap-2">
                    <span className={cn("text-[9px] font-black uppercase px-2 py-0.5 rounded border", getSignalColor(pred.signal))}>
                      {pred.signal}
                    </span>
                    <span className="text-[10px] font-mono font-bold text-primary dark:text-foreground w-10 text-right">{probText}</span>
                  </div>
                </div>
              );
            })}
            
            <div className="mt-2 p-2 rounded bg-secondary/30 border border-border/50">
               <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1 block">CONSENSUS INTELLIGENCE</span>
               <p className="text-[9px] text-foreground leading-tight italic">{getDisagreementNote()}</p>
            </div>

            <div className="flex gap-1.5 mt-2 justify-center">
              <span className={cn("text-[8px] font-black px-2 py-1 rounded-full uppercase shadow-sm", getRegimeColor(market_regime))}>
                {market_regime === 'BULL' ? '🐂' : market_regime === 'BEAR' ? '🐻' : '⚖️'} {market_regime}
              </span>
              <span className={cn("text-[8px] font-black px-2 py-1 rounded-full uppercase shadow-sm", getVolatilityColor(volatility_state))}>
                {volatility_state} VOL
              </span>
            </div>
          </div>
        </motion.div>

        {/* PRIORITY #2: XAI Signal Drivers with Visual Hierarchy */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
          <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">XAI Signal Drivers</h3>
            </div>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-center gap-3">
            {xai && xai.top_drivers && xai.top_drivers.length > 0 ? (
              <>
                <div className="flex flex-col gap-2.5">
                  {xai.top_drivers.slice(0, 3).map((driver, idx) => {
                    const maxImpact = Math.max(...xai.top_drivers.map(d => Math.abs(d.impact)));
                    const barWidth = (Math.abs(driver.impact) / maxImpact) * 100;
                    
                    return (
                      <div key={idx} className="flex flex-col gap-1">
                        <div className="flex justify-between items-end">
                          <span className="text-[10px] font-black text-foreground uppercase tracking-tight flex items-center gap-1.5">
                            <span className="text-[8px] opacity-30">#{idx + 1}</span> {driver.feature}
                          </span>
                          <span className={cn(
                            "text-[9px] font-mono font-black",
                            driver.direction === 'bullish' ? "text-green-500" : "text-red-500"
                          )}>
                            {driver.impact > 0 ? '+' : ''}{driver.impact.toFixed(3)}
                          </span>
                        </div>
                        <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden flex">
                          <div 
                            className={cn(
                              "h-full rounded-full transition-all duration-1000 ease-out",
                              driver.direction === 'bullish' ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.4)]" : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.4)]"
                            )}
                            style={{ width: `${barWidth}%` }}
                          />
                        </div>
                        <span className={cn(
                          "text-[7px] font-black uppercase tracking-widest",
                          driver.direction === 'bullish' ? "text-green-600/70" : "text-red-600/70"
                        )}>Contribution: {driver.direction}</span>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-1 pt-2 border-t border-border/30">
                  <p className="text-[10px] text-muted-foreground italic leading-relaxed text-center">
                    {xai.explanation}
                  </p>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <span className="text-[10px] text-muted-foreground italic uppercase tracking-tighter opacity-50">SHAP Inference Unavailable</span>
              </div>
            )}
          </div>
        </motion.div>

        {/* Risk Projections */}
        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
          <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">10-Day Projections (TFT)</h3>
            </div>
            {is_point_forecast && (
               <span className="text-[8px] font-black px-1.5 py-0.5 rounded bg-blue-500 text-white uppercase tracking-tighter">Point Forecast</span>
            )}
          </div>
          <div className="p-4 flex-1 flex flex-col justify-center gap-4">
            <div className="flex items-center justify-between font-mono text-sm relative z-10">
              {is_point_forecast ? (
                <div className="flex flex-col items-center justify-center w-full py-2 opacity-70">
                  <span className="text-foreground font-black text-lg">{currency}{projections.median?.toFixed(2)}</span>
                  <span className="text-[10px] uppercase font-black tracking-widest mt-1 text-muted-foreground">Point Forecast Only</span>
                </div>
              ) : (
                <>
                  <div className="flex flex-col items-center">
                    <span className="text-muted-foreground text-[9px] font-bold uppercase mb-1">P10 Floor</span>
                    <span className="text-[var(--signal-sell)] font-black text-lg">{currency}{projections.floor.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-muted-foreground text-[9px] font-bold uppercase mb-1">P50 Median</span>
                    <span className="text-foreground font-black text-lg">{currency}{projections.median?.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-muted-foreground text-[9px] font-bold uppercase mb-1">P90 Ceiling</span>
                    <span className="text-[var(--signal-buy)] font-black text-lg">{currency}{projections.ceiling.toFixed(2)}</span>
                  </div>
                </>
              )}
            </div>
            
            {/* Kelly Transparency & Distribution */}
            <div className="mt-4 pt-4 border-t border-border/50">
               <div className="flex items-center justify-between mb-3">
                  <span className="text-[9px] font-black uppercase text-muted-foreground tracking-widest">Risk Allocation (Kelly)</span>
                  <span className={cn(
                    "text-[10px] font-mono font-black px-2 py-0.5 rounded",
                    (data.risk?.kelly_fraction || 0) > 0 ? "bg-emerald-500/10 text-emerald-500" : "bg-muted text-muted-foreground opacity-50"
                  )}>
                    {((data.risk?.kelly_fraction || 0) * 100).toFixed(1)}% Allocation
                  </span>
               </div>
               
               <div className="grid grid-cols-3 gap-2">
                  <div className="flex flex-col p-2 rounded bg-secondary/30 border border-border/50">
                    <span className="text-[7px] font-black text-muted-foreground uppercase opacity-70">Win Prob</span>
                    <span className="text-xs font-mono font-bold text-foreground">{(data.risk?.win_probability || 0).toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col p-2 rounded bg-secondary/30 border border-border/50">
                    <span className="text-[7px] font-black text-muted-foreground uppercase opacity-70">Exp. Value</span>
                    <span className={cn(
                      "text-xs font-mono font-bold",
                      (data.risk?.expected_value || 0) >= 0 ? "text-emerald-500" : "text-red-500"
                    )}>
                      {(data.risk?.expected_value || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col p-2 rounded bg-secondary/30 border border-border/50">
                    <span className="text-[7px] font-black text-muted-foreground uppercase opacity-70">R/R Ratio</span>
                    <span className="text-xs font-mono font-bold text-foreground">{(data.risk?.risk_reward_ratio || 0).toFixed(2)}</span>
                  </div>
               </div>

               {(data.risk?.expected_value || 0) < 0 && (
                 <div className="mt-2 flex items-start gap-2 p-2 rounded bg-red-500/5 border border-red-500/20">
                    <AlertCircle className="w-3 h-3 text-red-500 shrink-0 mt-0.5" />
                    <p className="text-[8px] text-red-500 leading-tight font-medium italic">
                      Zero allocation enforced by RiskAgent due to negative Expected Value (EV).
                    </p>
                 </div>
               )}
            </div>
          </div>
        </motion.div>

      </div>

      {/* PRIORITY #4: NLP Panel Compression */}
      <motion.div variants={item} className={cn(
        "glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-500",
        (!qualitative_alpha || qualitative_alpha.includes("unavailable")) ? "min-h-[40px] opacity-80" : "min-h-[96px]"
      )}>
        <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-1 flex items-center justify-between h-8 shrink-0">
          <div className="flex items-center gap-2">
            <Newspaper className="w-3 h-3 text-primary" />
            <h3 className="text-[9px] font-black uppercase tracking-widest text-muted-foreground">Qualitative Alpha (Gemini)</h3>
          </div>
          {(!qualitative_alpha || qualitative_alpha.includes("unavailable")) && (
             <div className="flex items-center gap-2">
                <span className="text-[8px] font-black text-amber-500 uppercase tracking-widest px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20">Module Disabled</span>
                <span className="text-[8px] text-muted-foreground hidden sm:inline">Set GOOGLE_API_KEY to enable Fundamental Analysis</span>
             </div>
          )}
        </div>
        
        {qualitative_alpha && !qualitative_alpha.includes("unavailable") ? (
          <div className="p-3 flex-1 flex items-center gap-4 animate-in fade-in slide-in-from-bottom-1 duration-500">
            <p className="text-[10px] text-foreground font-medium leading-relaxed italic flex-1">
              &quot;{qualitative_alpha}&quot;
            </p>
            
            {sentiment_score !== undefined && sentiment_score !== null && (
              <div className="w-48 shrink-0 flex flex-col gap-1 border-l border-border/50 pl-4">
                <span className="text-[8px] font-bold text-muted-foreground uppercase text-center">Sentiment Score</span>
                <div className="flex justify-between text-[8px] font-black text-muted-foreground px-1">
                  <span className="text-red-500">BEARISH</span>
                  <span className="text-green-500">BULLISH</span>
                </div>
                <div className="relative h-1.5 w-full rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 opacity-80 mt-1">
                  <div 
                    className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 bg-white border border-black rounded-full shadow-sm"
                    style={{ left: `calc(${((sentiment_score + 1) / 2) * 100}% - 5px)` }}
                  />
                </div>
                <span className="text-[9px] font-mono text-center mt-0.5">{sentiment_score.toFixed(2)}</span>
              </div>
            )}
          </div>
        ) : (
          <div className="px-4 py-1.5 flex items-center">
             <span className="text-[9px] text-muted-foreground italic font-medium">Fundamental telemetry currently suppressed by rate-limits or missing credentials.</span>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
