import React from 'react';
import { ChartData } from '@/types';
import { 
  Cpu, Clock, Unlock, Eye, Lock, ArrowRightCircle, Target, Newspaper, Zap,
  BarChart3, Activity, ShieldCheck, AlertCircle, ChevronRight
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
    structural_regime,
    volatility_state, models, projections, 
    explainable_confidence, confidence_breakdown,
    execution_state, execution_reasoning, signal_bias,
    forecast_interpretation, forecast_explanation, consensus_intelligence,
    timing_reason, qualitative_alpha, sentiment_score,
    decision_tree, model_weights,
    xai
  } = data;

  const isVetoed = execution_state === 'VETOED' || execution_state === 'BLOCKED';

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

  const container: Variants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.05 } }
  };

  const item: Variants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
  };

  return (
    <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-4 h-full" data-tour="intelligence">
      
      {/* PHASE 2 & 3: INSTITUTIONAL EXECUTION AUTHORITY */}
      <motion.div variants={item} className={cn(
        "p-5 rounded-xl border-2 flex flex-col lg:flex-row gap-8 items-center transition-all duration-500 shadow-2xl backdrop-blur-md",
        isVetoed ? "bg-red-950/5 border-red-500/20" : "bg-emerald-950/10 border-emerald-500/20 shadow-[0_0_30px_rgba(16,185,129,0.05)]"
      )}>
        <div className="flex flex-col gap-3 items-center lg:items-start min-w-[240px]">
           <div className={cn("px-4 py-1.5 rounded-full border text-[11px] font-black uppercase tracking-[0.2em] flex items-center gap-2 shadow-sm", getExecutionColor(execution_state))}>
              {getExecutionIcon(execution_state)} {execution_state}
           </div>
           <div className="space-y-1">
             <p className="text-[12px] font-bold text-foreground leading-tight text-center lg:text-left">
               {isVetoed ? "Veto Reason:" : "Execution Thesis:"}
             </p>
             <p className="text-[11px] font-medium text-muted-foreground leading-relaxed text-center lg:text-left italic opacity-80 max-w-xs">
               &quot;{execution_reasoning}&quot;
             </p>
           </div>
        </div>

        <div className="h-px w-full lg:h-16 lg:w-px bg-border/40" />

        <div className="flex flex-1 justify-between w-full gap-6 px-4">
           <div className="flex flex-col items-center gap-1 group">
              <span className="text-[9px] font-black uppercase text-muted-foreground tracking-widest opacity-60 group-hover:opacity-100 transition-opacity">Probabilistic Confidence</span>
              <div className="flex items-end gap-1.5">
                 <span className="text-3xl font-mono font-black text-primary leading-none tracking-tighter">{explainable_confidence?.toFixed(1)}%</span>
                 <span className="text-[10px] font-black text-muted-foreground mb-1">Cnf</span>
              </div>
           </div>
           
           <div className="flex flex-col items-center gap-1 group">
              <span className="text-[9px] font-black uppercase text-muted-foreground tracking-widest opacity-60 group-hover:opacity-100 transition-opacity">Expected Value</span>
              <div className="flex items-end gap-1.5">
                 <span className={cn("text-3xl font-mono font-black leading-none tracking-tighter", (data.expected_value?.ev_pct ?? 0) > 0 ? "text-emerald-500" : "text-red-500")}>
                   {(data.expected_value?.ev_pct ?? 0).toFixed(2)}%
                 </span>
                 <span className="text-[10px] font-black text-muted-foreground mb-1">EV</span>
              </div>
           </div>

           <div className="flex flex-col items-center gap-1 group">
              <span className="text-[9px] font-black uppercase text-muted-foreground tracking-widest opacity-60 group-hover:opacity-100 transition-opacity">Predictive Bias</span>
              <div className="flex items-end gap-1.5">
                 <span className={cn(
                   "text-2xl font-black uppercase leading-none tracking-tight", 
                   signal_bias === 'BULLISH' ? 'text-emerald-500' : signal_bias === 'BEARISH' ? 'text-red-500' : 'text-zinc-500'
                 )}>
                    {signal_bias}
                 </span>
              </div>
              <span className="text-[9px] font-bold text-muted-foreground opacity-50 uppercase tracking-tighter">{forecast_interpretation}</span>
           </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1">
        
        {/* PHASE 4: INSTITUTIONAL CONSENSUS MATRIX */}
        <motion.div variants={item} className="lg:col-span-5 glass-panel rounded-xl flex flex-col overflow-hidden border-border/60 hover:border-primary/30 transition-all bg-card/30">
          <div className="bg-secondary/40 border-b border-border/60 px-4 py-2.5 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-primary" />
              <h3 className="text-[11px] font-black uppercase tracking-widest text-foreground/80">Consensus Matrix</h3>
            </div>
            <span className="text-[9px] font-mono font-black text-primary/80 uppercase tracking-tighter">{consensus_intelligence}</span>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-2">
            <div className="grid grid-cols-12 px-2 pb-1 text-[8px] font-black text-muted-foreground uppercase tracking-widest border-b border-border/20">
              <span className="col-span-4 text-left">Agent Node</span>
              <span className="col-span-3 text-center">Signal</span>
              <span className="col-span-2 text-center">Weight</span>
              <span className="col-span-3 text-right">Rel.</span>
            </div>
            <div className="space-y-1">
              {Object.entries(models).map(([name, pred]) => {
                if (name.includes("META") || name.includes("ENSEMBLE")) return null;
                const intel = model_weights?.[name] || { weight: 0.25, recent_accuracy: 0.75 };
                
                return (
                  <div key={name} className="grid grid-cols-12 items-center p-2.5 rounded bg-muted/10 border border-transparent hover:border-border/40 hover:bg-muted/20 transition-all group/model">
                    <span className="col-span-4 text-[10px] font-mono text-muted-foreground uppercase font-bold group-hover/model:text-foreground transition-colors">{name.replace('_AGENT', '').replace('DL_', '')}</span>
                    <div className="col-span-3 flex justify-center">
                      <span className={cn(
                        "text-[9px] font-black uppercase px-2 py-0.5 rounded border leading-none",
                        pred.signal === 'BUY' ? "text-emerald-500 border-emerald-500/20 bg-emerald-500/10" :
                        pred.signal === 'SELL' ? "text-red-500 border-red-500/20 bg-red-500/10" :
                        "text-zinc-500 border-zinc-500/20 bg-zinc-500/10"
                      )}>{pred.signal}</span>
                    </div>
                    <div className="col-span-2 flex flex-col items-center">
                       <span className="text-[10px] font-mono font-bold text-primary/70">{(intel.weight * 100).toFixed(0)}%</span>
                       <span className="text-[6px] text-muted-foreground uppercase">Wgt</span>
                    </div>
                    <div className="col-span-3 flex justify-end items-center gap-1.5">
                       <div className="flex flex-col items-end">
                          <span className="text-[10px] font-mono font-black text-foreground">{Math.round(intel.recent_accuracy * 100)}%</span>
                          <span className="text-[6px] text-muted-foreground uppercase leading-none">Reliability</span>
                       </div>
                       <div className={cn("w-1.5 h-1.5 rounded-full shadow-[0_0_5px_rgba(16,185,129,0.5)]", intel.recent_accuracy > 0.8 ? "bg-emerald-500" : "bg-amber-500")} />
                    </div>
                  </div>
                );
              })}
            </div>

            
            {/* Confidence Attribution Scorecard */}
            {confidence_breakdown && (
              <div className="mt-auto grid grid-cols-4 gap-2 pt-4 border-t border-border/20">
                {Object.entries(confidence_breakdown).filter(([k]) => k !== 'Total_Raw_Score').map(([key, val]) => (
                  <div key={key} className="flex flex-col items-center p-1.5 rounded bg-black/20 border border-white/5">
                     <span className="text-[6px] font-black uppercase text-muted-foreground text-center truncate w-full mb-1">{key.replace('_', ' ')}</span>
                     <span className={cn("text-[10px] font-mono font-black", val > 0 ? "text-emerald-500" : val < 0 ? "text-red-500" : "text-zinc-500")}>
                       {val > 0 ? '+' : ''}{val.toFixed(0)}
                     </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </motion.div>

        {/* PHASE 3: SIGNAL GOVERNANCE DECISION TREE */}
        <motion.div variants={item} className="lg:col-span-4 glass-panel rounded-xl flex flex-col overflow-hidden border-border/60 bg-card/30">
          <div className="bg-secondary/40 border-b border-border/60 px-4 py-2.5 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-500" />
              <h3 className="text-[11px] font-black uppercase tracking-widest text-foreground/80">Governance Audit</h3>
            </div>
            <span className="text-[9px] font-mono font-black text-emerald-500 uppercase tracking-tighter">Live_Audit</span>
          </div>
          <div className="p-4 flex-1 flex flex-col gap-3">
            {decision_tree?.map((node, i) => (
              <div key={node.node} className="flex items-center gap-3 relative">
                {i < decision_tree.length - 1 && (
                  <div className="absolute left-[7.5px] top-4 w-px h-6 bg-border/40" />
                )}
                <div className={cn(
                  "w-4 h-4 rounded-full flex items-center justify-center shrink-0 z-10 shadow-sm",
                  node.status === 'PASS' ? "bg-emerald-500/20 text-emerald-500" : "bg-red-500/20 text-red-500"
                )}>
                  {node.status === 'PASS' ? <Zap className="w-2 h-2 fill-current" /> : <AlertCircle className="w-2 h-2" />}
                </div>
                <div className="flex flex-1 items-center justify-between p-2 rounded bg-muted/10 border border-border/20 group/node hover:border-primary/20 transition-all">
                   <div className="flex flex-col">
                      <span className="text-[10px] font-black uppercase text-foreground/70 group-hover/node:text-foreground transition-colors">{node.node}</span>
                      <span className="text-[9px] font-medium text-muted-foreground opacity-70">{node.detail}</span>
                   </div>
                   <span className={cn(
                     "text-[8px] font-black uppercase px-1.5 py-0.5 rounded leading-none border",
                     node.status === 'PASS' ? "text-emerald-500 border-emerald-500/20" : "text-red-500 border-red-500/20"
                   )}>{node.status}</span>
                </div>
              </div>
            ))}
            
            <div className="mt-auto p-2.5 rounded-lg bg-black/20 border border-white/5 flex items-center justify-between group cursor-help">
               <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-[9px] font-black uppercase text-muted-foreground tracking-tighter">Entropy Safety Gate</span>
               </div>
               <span className="text-[10px] font-mono font-bold text-emerald-400">ACTIVE</span>
            </div>
          </div>
        </motion.div>

        {/* TIMING, REGIME & FORECAST */}
        <motion.div variants={item} className="lg:col-span-3 flex flex-col gap-4">
          
          {/* Market Regime Node */}
          <div className="glass-panel rounded-xl flex flex-col overflow-hidden border-border/60 bg-card/30 flex-1">
             <div className="bg-secondary/40 border-b border-border/60 px-4 py-2 flex items-center justify-between">
                <span className="text-[10px] font-black uppercase tracking-widest text-foreground/60">Regime Dynamics</span>
                <Activity className="w-3.5 h-3.5 text-amber-500" />
             </div>
             <div className="p-4 flex flex-col gap-4">
                <div className="flex flex-col gap-1">
                  <span className="text-[8px] font-black uppercase text-muted-foreground/60 tracking-widest">Structural State</span>
                  <div className="flex items-center justify-between">
                    <span className={cn(
                      "text-sm font-black uppercase tracking-tight",
                      structural_regime.includes('BULL') ? 'text-emerald-500' : structural_regime.includes('BEAR') ? 'text-red-500' : 'text-zinc-500'
                    )}>{structural_regime}</span>
                    <span className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-[8px] font-mono font-bold uppercase">{volatility_state} VOL</span>
                  </div>
                </div>

                <div className="space-y-2">
                   <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded bg-primary/10 border border-primary/20"><Clock className="w-3 h-3 text-primary" /></div>
                      <div className="flex flex-col max-w-[140px]">
                         <span className="text-[8px] font-black uppercase text-muted-foreground tracking-tighter">Timing Engine</span>
                         <p className="text-[10px] font-medium leading-[1.1] text-foreground/80">{timing_reason}</p>
                      </div>
                   </div>
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-border/20">
                   <div className="flex flex-col">
                      <span className="text-[7px] font-black uppercase text-muted-foreground opacity-50">Vol Ratio</span>
                      <span className={cn("text-[11px] font-mono font-black", data.volume_ratio > 1.2 ? "text-emerald-500" : "text-foreground")}>
                        {data.volume_ratio.toFixed(2)}x
                      </span>
                   </div>
                   <div className="flex flex-col items-end">
                      <span className="text-[7px] font-black uppercase text-muted-foreground opacity-50">Bias Conf.</span>
                      <span className="text-[11px] font-mono font-black text-primary">{explainable_confidence?.toFixed(0)}%</span>
                   </div>
                </div>
             </div>
          </div>

          {/* Trade Projections Node */}
          <div className="glass-panel rounded-xl flex flex-col overflow-hidden border-border/60 bg-card/30 flex-1">
             <div className="p-4 flex flex-col gap-3">
                <div className="flex items-center justify-between text-[10px] font-black uppercase tracking-widest text-foreground/60 mb-1">
                  <span>Forecast Cone</span>
                  <div className="flex gap-2">
                    <span className={cn(
                      "text-[8px] font-mono px-1.5 rounded border leading-none py-0.5",
                      projections.reliability === "HIGH" ? "text-emerald-500 border-emerald-500/20 bg-emerald-500/10" :
                      projections.reliability === "MEDIUM" ? "text-amber-500 border-amber-500/20 bg-amber-500/10" :
                      "text-red-500 border-red-500/20 bg-red-500/10"
                    )}>{projections.reliability} REL</span>
                    <BarChart3 className="w-3.5 h-3.5 text-primary" />
                  </div>
                </div>
                
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-end">
                    <div className="flex flex-col">
                      <span className="text-[8px] font-black text-muted-foreground uppercase opacity-60">P10 Floor</span>
                      <span className="text-red-500 font-mono font-black text-sm">{currency}{projections.floor.toFixed(2)}</span>
                    </div>
                    <div className="flex flex-col items-center">
                      <span className="text-[8px] font-black text-muted-foreground uppercase opacity-60">Drift</span>
                      <span className={cn(
                        "font-mono font-black text-[10px]",
                        (projections.drift ?? 0) > 0 ? "text-emerald-500" : "text-red-500"
                      )}>{((projections.drift ?? 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex flex-col items-end">
                      <span className="text-[8px] font-black text-muted-foreground uppercase opacity-60">P90 Ceiling</span>
                      <span className="text-emerald-500 font-mono font-black text-sm">{currency}{projections.ceiling.toFixed(2)}</span>
                    </div>
                  </div>
                  
                  <div className="h-4 w-full bg-black/20 rounded-sm border border-white/5 relative overflow-hidden flex items-center px-1">
                    <div className="absolute left-[10%] right-[10%] h-1.5 bg-gradient-to-r from-red-500/20 via-primary/20 to-emerald-500/20 rounded-full blur-[1px]" />
                    <motion.div 
                      initial={{ left: '50%' }}
                      animate={{ left: '48%' }}
                      className="absolute w-1 h-3 bg-white/80 rounded-full shadow-[0_0_5px_white]" 
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between gap-4 mt-2">
                   <div className="flex flex-col">
                      <span className="text-[7px] font-black uppercase text-muted-foreground opacity-50">Conf. Cone</span>
                      <span className="text-[10px] font-mono font-black text-primary">
                        {(projections.confidence ?? 0).toFixed(1)}%
                      </span>
                   </div>
                   <div className="flex flex-col items-end">
                      <span className="text-[7px] font-black uppercase text-muted-foreground opacity-50">Exp. Move</span>
                      <span className="text-[10px] font-mono font-black text-foreground">{((projections.expected_move ?? 0) * 100).toFixed(1)}%</span>
                   </div>
                </div>
             </div>
          </div>
        </motion.div>
      </div>
      
      {/* PHASE 5: XAI & QUALITATIVE ALPHA BAR */}
      <motion.div variants={item} className="p-4 bg-black/40 border border-border/40 rounded-xl flex flex-col lg:flex-row gap-6 items-center">
        <div className="flex items-center gap-4 flex-1 w-full lg:w-auto">
          <Newspaper className="w-5 h-5 text-primary shrink-0 opacity-80" />
          <div className="flex flex-col gap-0.5">
            <span className="text-[8px] font-black text-primary uppercase tracking-[0.2em] leading-none mb-1">Qualitative Alpha Proxy</span>
            <p className="text-[11px] text-foreground/90 font-medium leading-relaxed italic line-clamp-1">
              &quot;{qualitative_alpha || "Syncing Gemini fundamental context layer..."}&quot;
            </p>
          </div>
        </div>

        <div className="h-px w-full lg:h-10 lg:w-px bg-border/40" />

        {/* Mini XAI Drivers Visual (Phase 5) */}
        <div className="flex items-center gap-6 shrink-0 w-full lg:w-auto justify-between lg:justify-end px-2">
           {xai?.top_drivers?.map((driver) => (
             <div key={driver.feature} className="flex flex-col items-center gap-1 min-w-[65px] group/xai relative">
                <span className="text-[7px] font-black text-muted-foreground uppercase group-hover/xai:text-primary transition-colors truncate w-full text-center">
                  {driver.feature.replace('_', ' ').replace('_vs_', '/')}
                </span>
                <div className="flex items-center gap-1.5">
                   <div className="h-1.5 w-12 bg-white/5 rounded-full overflow-hidden flex items-center justify-center p-0.5 border border-white/5">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(100, driver.impact * 250)}%` }}
                        className={cn(
                          "h-full rounded-full",
                          driver.direction === 'bullish' ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]" : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.4)]"
                        )}
                      />
                   </div>
                   <div className={cn(
                     "w-1.5 h-1.5 rounded-full",
                     driver.stability > 0.85 ? "bg-emerald-500/40" : "bg-amber-500/40"
                   )} title={`Stability: ${Math.round(driver.stability * 100)}%`} />
                </div>
                {/* Driver Tooltip (Phase 5) */}
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-black/90 border border-white/10 px-2 py-1 rounded text-[6px] font-mono whitespace-nowrap opacity-0 group-hover/xai:opacity-100 transition-opacity z-50 pointer-events-none">
                  Impact: {(driver.impact * 10).toFixed(2)} | Stability: {(driver.stability * 100).toFixed(0)}%
                </div>
             </div>
           ))}
           
           <div className="flex flex-col border-l border-border/40 pl-6 ml-2">
              <span className="text-[7px] font-black text-muted-foreground uppercase leading-none mb-1.5 tracking-widest">Sentiment</span>
              <div className="flex items-center gap-2">
                <span className={cn("text-xl font-mono font-black leading-none tracking-tighter", (sentiment_score ?? 0) > 0 ? "text-emerald-500" : "text-red-500")}>
                  {(sentiment_score ?? 0).toFixed(2)}
                </span>
                <div className={cn("w-2 h-2 rounded-full", (sentiment_score ?? 0) > 0 ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]" : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.6)]")} />
              </div>
           </div>
        </div>
      </motion.div>

    </motion.div>
  );
}
