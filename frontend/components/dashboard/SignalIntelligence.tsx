import React from 'react';
import { ChartData } from '@/types';
import { Cpu, AlertTriangle, AlertCircle, BrainCircuit, Clock, CheckCircle2, ShieldAlert, Zap, Info, Newspaper } from 'lucide-react';
import { motion, Variants } from 'framer-motion';
import { cn } from '@/lib/utils';
import { formatDistanceToNow } from 'date-fns';

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
    is_point_forecast, timestamp,
    signal_note, xai,
    explainable_confidence, confidence_breakdown,
    signal_reasoning, veto_reason, timing_reason, forecast_reason, rr_reason,
    qualitative_alpha, sentiment_score
  } = data;

  const isSuppressed = signal === 'VETOED' || signal === 'HOLD' || (data.risk?.expected_value ?? 0) <= 0 || (explainable_confidence ?? 0) < 40;
  const isDivergent = (market_regime === 'BULL' && signal.includes('SELL')) || (market_regime === 'BEAR' && signal.includes('BUY'));

  const getSignalColor = (action: string) => {
    if (action.includes('BUY')) return isSuppressed ? 'text-zinc-500 border-zinc-500/30 bg-zinc-500/10' : 'text-[var(--signal-buy)] border-[var(--signal-buy)]/30 bg-[var(--signal-buy)]/10';
    if (action.includes('SELL')) return isSuppressed ? 'text-zinc-500 border-zinc-500/30 bg-zinc-500/10' : 'text-[var(--signal-sell)] border-[var(--signal-sell)]/30 bg-[var(--signal-sell)]/10';
    return 'text-[var(--signal-hold)] border-[var(--signal-hold)]/30 bg-[var(--signal-hold)]/10';
  };

  const getRegimeColor = (regime: string) => {
    if (isSuppressed) return 'bg-zinc-600 text-white border-zinc-700 opacity-60';
    if (regime === 'BULL') return 'bg-green-500 text-white border-green-600';
    if (regime === 'BEAR') return 'bg-red-500 text-white border-red-600';
    return 'bg-zinc-500 text-white border-zinc-600';
  };

  const getVolatilityColor = (state: string) => {
    if (isSuppressed) return 'bg-zinc-600 text-white border-zinc-700 opacity-60';
    if (state === 'HIGH') return 'bg-red-500 text-white border-red-600';
    if (state === 'MEDIUM') return 'bg-orange-500 text-white border-orange-600';
    return 'bg-green-500 text-white border-green-600';
  };

  const container: Variants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.05 } }
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
  };

  const signalAge = timestamp ? formatDistanceToNow(new Date(timestamp), { addSuffix: true }) : 'Unknown';

  const renderConfidenceMatrix = () => {
    if (!confidence_breakdown) return null;
    return (
      <div className="grid grid-cols-3 gap-1 mt-2 p-2 rounded bg-black/20 border border-border/50">
        {Object.entries(confidence_breakdown).filter(([k]) => k !== 'Total_Raw_Score').map(([key, val]) => (
          <div key={key} className="flex flex-col items-center justify-center p-1 rounded bg-secondary/30">
             <span className="text-[6px] font-black uppercase text-muted-foreground text-center leading-tight">{key.replace('_', ' ')}</span>
             <span className={cn("text-[9px] font-mono font-bold", val > 0 ? "text-green-500" : val < 0 ? "text-red-500" : "text-zinc-500")}>
               {val > 0 ? '+' : ''}{val.toFixed(1)}
             </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4 h-full" data-tour="intelligence">
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2 text-[9px] font-mono font-bold text-muted-foreground uppercase tracking-widest">
           <Clock className="w-3 h-3" /> Generated: {new Date(timestamp).toLocaleString()} UTC ({signalAge})
        </div>
        <div className="flex items-center gap-3">
           <div className="flex items-center gap-2 text-[9px] font-mono font-bold text-primary uppercase tracking-widest">
              <CheckCircle2 className="w-3 h-3" /> Conf: {explainable_confidence?.toFixed(1)}%
           </div>
        </div>
      </div>

      {signal === 'VETOED' && (
        <motion.div variants={item} className="flex flex-col gap-1 px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-lg w-full">
          <div className="flex items-center gap-2 text-red-500 text-xs font-black uppercase tracking-wider mb-1">
            <ShieldAlert className="w-4 h-4" /> Institutional Veto Enforced
          </div>
          <p className="text-[10px] text-red-400 font-medium leading-tight"><span className="font-bold text-red-500">Primary Cause:</span> {veto_reason}</p>
          <p className="text-[9px] text-red-400/80 leading-tight italic mt-1"><span className="font-bold">Context:</span> {signal_reasoning}</p>
        </motion.div>
      )}

      {signal_note && signal !== 'VETOED' && (
        <motion.div variants={item} className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-500 text-xs font-bold w-full">
          <AlertCircle className="w-4 h-4" /> <span>{signal_note}</span>
        </motion.div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-1">
        <motion.div variants={item} className={cn("glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300 border-2", isDivergent && !isSuppressed ? "border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.2)]" : "border-border")}>
          <div className={cn("border-b px-4 py-2 flex items-center justify-between transition-colors", isDivergent && !isSuppressed ? "bg-amber-500/10 border-amber-500/30" : "bg-secondary/50 dark:bg-black/40 border-border")}>
            <div className="flex items-center gap-2">
              <Cpu className={cn("w-3.5 h-3.5", isDivergent && !isSuppressed ? "text-amber-500" : "text-primary")} />
              <h3 className={cn("text-[10px] font-black uppercase tracking-widest", isDivergent && !isSuppressed ? "text-amber-500" : "text-muted-foreground")}>Consensus Matrix</h3>
            </div>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-1 justify-center">
            {Object.entries(models).map(([name, pred]) => {
              if (name.includes("META") || name.includes("ENSEMBLE")) return null;
              const probText = pred.probability === 0 ? "N/A" : `${Math.round(pred.probability * 100)}%`;
              return (
                <div key={name} className="flex items-center justify-between p-1.5 rounded border border-border bg-muted/30 group-hover:bg-muted/50 transition-colors">
                  <span className="text-[9px] font-mono text-muted-foreground uppercase font-bold">{name}</span>
                  <div className="flex items-center gap-2">
                    <span className={cn("text-[8px] font-black uppercase px-1.5 py-0.5 rounded border", getSignalColor(pred.signal))}>{pred.signal}</span>
                    <span className="text-[9px] font-mono font-bold text-primary dark:text-foreground w-8 text-right">{probText}</span>
                  </div>
                </div>
              );
            })}
            
            {renderConfidenceMatrix()}

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

        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
          <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">XAI & Timing</h3>
            </div>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-center gap-3">
            {xai && xai.top_drivers && xai.top_drivers.length > 0 ? (
              <div className="flex flex-col gap-2.5">
                {xai.top_drivers.slice(0, 3).map((driver, idx) => {
                  const maxImpact = Math.max(...xai.top_drivers.map(d => Math.abs(d.impact)));
                  const barWidth = (Math.abs(driver.impact) / maxImpact) * 100;
                  const isBullishDriver = driver.direction === 'bullish';
                  const dirColor = (isBullishDriver && isSuppressed) ? "bg-zinc-500" : isBullishDriver ? "bg-green-500" : "bg-red-500";
                  const textColor = (isBullishDriver && isSuppressed) ? "text-zinc-500" : isBullishDriver ? "text-green-500" : "text-red-500";
                  
                  return (
                    <div key={idx} className="flex flex-col gap-1">
                      <div className="flex justify-between items-end">
                        <span className="text-[9px] font-black text-foreground uppercase tracking-tight flex items-center gap-1.5">
                          <span className="text-[7px] opacity-30">#{idx + 1}</span> {driver.feature}
                        </span>
                        <span className={cn("text-[8px] font-mono font-black", textColor)}>
                          {driver.impact > 0 ? '+' : ''}{driver.impact.toFixed(3)}
                        </span>
                      </div>
                      <div className="h-1 w-full bg-secondary rounded-full overflow-hidden flex">
                        <div className={cn("h-full rounded-full transition-all duration-1000 ease-out", dirColor)} style={{ width: `${barWidth}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : null}
            
            <div className="mt-auto pt-2 border-t border-border/30">
              <div className="flex items-center gap-1 mb-1">
                <Zap className="w-3 h-3 text-amber-500" />
                <span className="text-[8px] font-black uppercase text-muted-foreground">Predictive Timing</span>
              </div>
              <p className="text-[9px] text-foreground leading-tight italic">{timing_reason || "Awaiting timing decomposition."}</p>
            </div>
          </div>
        </motion.div>

        <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
          <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-primary" />
              <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">10-Day Projections (TFT)</h3>
            </div>
          </div>
          <div className="p-4 flex-1 flex flex-col justify-center gap-3">
            <div className="flex items-center justify-between font-mono text-sm relative z-10">
              {is_point_forecast ? (
                <div className="flex flex-col items-center justify-center w-full py-2 opacity-70">
                  <span className="text-foreground font-black text-lg">{currency}{projections.median?.toFixed(2)}</span>
                </div>
              ) : (
                <>
                  <div className="flex flex-col items-center">
                    <span className="text-muted-foreground text-[8px] font-bold uppercase mb-1">P10</span>
                    <span className={cn("font-black text-sm", isSuppressed ? "text-zinc-500" : "text-[var(--signal-sell)]")}>{currency}{projections.floor.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-muted-foreground text-[8px] font-bold uppercase mb-1">P50</span>
                    <span className="text-foreground font-black text-base">{currency}{projections.median?.toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <span className="text-muted-foreground text-[8px] font-bold uppercase mb-1">P90</span>
                    <span className={cn("font-black text-sm", isSuppressed ? "text-zinc-500" : "text-[var(--signal-buy)]")}>{currency}{projections.ceiling.toFixed(2)}</span>
                  </div>
                </>
              )}
            </div>
            
            <p className="text-[8px] text-center text-muted-foreground leading-tight italic px-2">{forecast_reason}</p>
            
            <div className="mt-2 pt-3 border-t border-border/50">
               <div className="grid grid-cols-3 gap-2">
                  <div className="flex flex-col p-1.5 rounded bg-secondary/30 border border-border/50">
                    <span className="text-[7px] font-black text-muted-foreground uppercase opacity-70">Win Prob</span>
                    <span className="text-[10px] font-mono font-bold text-foreground">{(data.risk?.win_probability || 0).toFixed(2)}</span>
                  </div>
                  <div className="flex flex-col p-1.5 rounded bg-secondary/30 border border-border/50">
                    <span className="text-[7px] font-black text-muted-foreground uppercase opacity-70">Exp. Value</span>
                    <span className={cn("text-[10px] font-mono font-bold", (data.risk?.expected_value || 0) >= 0 ? "text-emerald-500" : "text-red-500")}>
                      {(data.risk?.expected_value || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-col p-1.5 rounded bg-secondary/30 border border-border/50">
                    <span className="text-[7px] font-black text-muted-foreground uppercase opacity-70">R/R Ratio</span>
                    <span className="text-[10px] font-mono font-bold text-foreground">{(data.risk?.risk_reward_ratio || 0).toFixed(2)}</span>
                  </div>
               </div>
               {rr_reason && !rr_reason.includes("HOLD") && (
                 <p className="text-[8px] mt-1.5 text-muted-foreground italic leading-tight"><Info className="inline w-2 h-2 mr-1" />{rr_reason}</p>
               )}
            </div>
          </div>
        </motion.div>
      </div>
      
      {/* Qualitative Alpha */}
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