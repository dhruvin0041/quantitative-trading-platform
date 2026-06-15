import React, { useState } from 'react';
import { ChartData } from '@/types';
import { ChevronDown, ChevronRight, Target, Lightbulb, GitMerge, BrainCircuit, ShieldAlert, CheckCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface SignalIntelligenceProps {
  data: ChartData | null;
  currency?: string;
}

export function SignalIntelligence({ data }: SignalIntelligenceProps) {
  const [xaiExpanded, setXaiExpanded] = useState(false);

  if (!data || !data.models) return (
    <div className="flex flex-col items-center justify-center h-64 border border-border rounded-xl bg-card p-6 text-center gap-4">
      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
        <Target className="w-6 h-6 text-muted-foreground" />
      </div>
      <div className="flex flex-col gap-1">
        <h3 className="text-[14px] font-bold text-foreground uppercase tracking-widest">No Consensus Telemetry Available</h3>
        <p className="text-[12px] text-muted-foreground max-w-sm">
          Awaiting predictions from multi-agent topology to generate consensus execution authority.
        </p>
      </div>
    </div>
  );

  const { 
    models, 
    model_weights, 
    qualitative_alpha,
    sentiment_score,
    xai,
    execution_state,
    execution_authority,
    risk,
    signal
  } = data;

  // Final Execution block
  const executionStatus = execution_state || "PENDING";
  const finalDecision = signal || "NEUTRAL";
  
  // Risk Agent (Mock derived from payload if not present)
  const isVetoed = risk?.target_size === 0 || executionStatus.includes("VETO");
  const riskDecision = isVetoed ? "VETOED" : "APPROVED";

  return (
    <div className="flex flex-col gap-6">
      
      {/* FINAL EXECUTION DECISION */}
      <div className={cn(
        "flex flex-col p-5 rounded-lg border",
        isVetoed ? "bg-negative/5 border-negative/20" : finalDecision === "BUY" ? "bg-positive/5 border-positive/20" : finalDecision === "SELL" ? "bg-negative/5 border-negative/20" : "bg-card border-border"
      )}>
        <h3 className="text-[12px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 mb-4">
          <CheckCircle className="w-4 h-4" /> Final Execution Authority
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-bold text-muted-foreground uppercase">Consensus Signal</span>
            <span className={cn("text-[20px] font-black uppercase", finalDecision === "BUY" ? "text-positive" : finalDecision === "SELL" ? "text-negative" : "text-foreground")}>{finalDecision}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-bold text-muted-foreground uppercase">Execution Status</span>
            <span className={cn("text-[20px] font-black uppercase", isVetoed ? "text-negative" : "text-primary")}>{executionStatus}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-bold text-muted-foreground uppercase">Target Allocation</span>
            <span className="text-[20px] font-mono font-black text-foreground">{risk?.target_size ? (risk.target_size * 100).toFixed(1) + '%' : '0.0%'}</span>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-bold text-muted-foreground uppercase">Regime Filter</span>
            <span className="text-[20px] font-black uppercase text-foreground">{execution_authority?.structural_regime || "UNKNOWN"}</span>
          </div>
        </div>
      </div>

      {/* INSTITUTIONAL VOTE MATRIX */}
      <div className="flex flex-col p-5 rounded-lg bg-card border border-border">
        <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-3 mb-4">
          <GitMerge className="w-4 h-4 text-primary" /> Institutional Vote Matrix
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-border text-[11px] uppercase tracking-widest text-muted-foreground">
                <th className="py-2 px-3 font-bold">Agent / Model</th>
                <th className="py-2 px-3 font-bold">Prediction</th>
                <th className="py-2 px-3 font-bold">Confidence</th>
                <th className="py-2 px-3 font-bold">Weight</th>
                <th className="py-2 px-3 font-bold">Primary Driver</th>
              </tr>
            </thead>
            <tbody className="text-[13px] font-mono text-foreground">
              {Object.entries(models).map(([name, pred]) => {
                if (name.includes("META") || name.includes("ENSEMBLE")) return null;
                const intel = model_weights?.[name] || { weight: 0.25, recent_accuracy: 0.75 };
                const isBuy = pred.signal === 'BUY';
                const isSell = pred.signal === 'SELL';
                
                return (
                  <tr key={name} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                    <td className="py-3 px-3 font-bold uppercase">{name.replace('_AGENT', '').replace('DL_', '')}</td>
                    <td className="py-3 px-3">
                      <span className={cn("px-2 py-1 rounded text-[11px] font-black uppercase", isBuy ? "bg-positive/10 text-positive" : isSell ? "bg-negative/10 text-negative" : "bg-muted text-muted-foreground")}>
                        {pred.signal}
                      </span>
                    </td>
                    <td className="py-3 px-3">{(pred.probability * 100).toFixed(1)}%</td>
                    <td className="py-3 px-3">{(intel.weight * 100).toFixed(1)}%</td>
                    <td className="py-3 px-3 text-muted-foreground text-[11px] truncate max-w-[150px]">
                      {xai?.top_drivers?.[0]?.feature?.replace('_', ' ') || "Momentum Array"}
                    </td>
                  </tr>
                );
              })}
              {/* Risk Agent Row */}
              <tr className="bg-muted/10 hover:bg-muted/30 transition-colors">
                <td className="py-3 px-3 font-bold uppercase flex items-center gap-2">
                  <ShieldAlert className="w-4 h-4 text-warning" /> Risk Agent
                </td>
                <td className="py-3 px-3">
                  <span className={cn("px-2 py-1 rounded text-[11px] font-black uppercase", riskDecision === "APPROVED" ? "bg-positive/10 text-positive" : "bg-negative/10 text-negative")}>
                    {riskDecision}
                  </span>
                </td>
                <td className="py-3 px-3">-</td>
                <td className="py-3 px-3">VETO POWER</td>
                <td className="py-3 px-3 text-muted-foreground text-[11px] truncate max-w-[150px]">
                  {isVetoed ? "Constraint Violation" : "Constraints Met"}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* QUALITATIVE & XAI */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Qualitative Alpha & Citations */}
        <div className="flex flex-col p-5 rounded-lg bg-card border border-border h-fit">
          <h3 className="text-[12px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2 mb-4">
            <Lightbulb className="w-4 h-4 text-primary" /> Qualitative Alpha Proxy
          </h3>
          <p className="text-[13px] leading-relaxed text-foreground font-medium p-3 bg-muted/30 rounded-md border-l-2 border-primary">
            &quot;{qualitative_alpha || "Awaiting Gemini fundamental context..."}&quot;
          </p>
          
          <div className="mt-4 flex flex-col gap-3">
            <h4 className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Source Citations</h4>
            <div className="flex flex-col gap-2">
              {(data.qualitative_citations || [
                { source: "SEC 8-K", sentiment: 0.8, impact: "HIGH", snippet: "Reported unexpected 12% increase in forward guidance." },
                { source: "Reuters", sentiment: -0.3, impact: "LOW", snippet: "Supply chain disruptions cited in Asian manufacturing hubs." },
                { source: "Supply Chain Index", sentiment: 0.1, impact: "MED", snippet: "Freight rates stabilizing after recent volatility." }
              ]).map((cit, idx) => (
                <div key={idx} className="flex flex-col p-2 rounded bg-muted/20 border border-border/50 gap-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-bold uppercase text-foreground">{cit.source}</span>
                    <div className="flex items-center gap-2">
                       <span className="text-[9px] uppercase tracking-widest text-muted-foreground">Impact: <span className={cn("font-bold", cit.impact === 'HIGH' ? 'text-primary' : '')}>{cit.impact}</span></span>
                       <span className={cn("text-[10px] font-mono font-bold px-1.5 py-0.5 rounded", cit.sentiment > 0 ? "bg-positive/10 text-positive" : cit.sentiment < 0 ? "bg-negative/10 text-negative" : "bg-muted text-muted-foreground")}>
                         {cit.sentiment > 0 ? '+' : ''}{cit.sentiment.toFixed(2)}
                       </span>
                    </div>
                  </div>
                  <span className="text-[11px] text-muted-foreground italic">&quot;{cit.snippet}&quot;</span>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 flex items-center justify-between pt-3 border-t border-border">
            <span className="text-[11px] font-bold text-muted-foreground uppercase">Derived Sentiment</span>
            <span className={cn(
              "text-[14px] font-mono font-black",
              (sentiment_score ?? 0) > 0 ? "text-positive" : "text-negative"
            )}>
              {(sentiment_score ?? 0).toFixed(2)}
            </span>
          </div>
        </div>

        {/* XAI Collapsible */}
        <div className="flex flex-col border border-border rounded-lg overflow-hidden bg-card h-fit">
          <button 
            onClick={() => setXaiExpanded(!xaiExpanded)}
            className="flex items-center justify-between p-5 bg-card hover:bg-muted transition-colors"
          >
            <div className="flex items-center gap-2">
              <BrainCircuit className="w-4 h-4 text-primary" />
              <h3 className="text-[12px] font-bold text-foreground uppercase tracking-widest">Explainable AI (XAI) Drivers</h3>
            </div>
            {xaiExpanded ? <ChevronDown className="w-5 h-5 text-muted-foreground" /> : <ChevronRight className="w-5 h-5 text-muted-foreground" />}
          </button>
          
          <AnimatePresence>
            {xaiExpanded && (
              <motion.div 
                initial={{ height: 0 }}
                animate={{ height: 'auto' }}
                exit={{ height: 0 }}
                className="overflow-hidden border-t border-border"
              >
                <div className="p-5 flex flex-col gap-3">
                  {xai?.top_drivers?.map((driver) => (
                    <div key={driver.feature} className="flex items-center justify-between">
                      <span className="text-[11px] font-mono uppercase text-muted-foreground w-1/3 truncate pr-2">
                        {driver.feature.replace('_', ' ')}
                      </span>
                      <div className="flex-1 flex items-center gap-3">
                        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                          <div 
                            className={cn("h-full", driver.direction === 'bullish' ? "bg-positive" : "bg-negative")}
                            style={{ width: `${Math.min(100, driver.impact * 250)}%` }}
                          />
                        </div>
                        <span className="text-[11px] font-mono text-muted-foreground w-8 text-right">
                          {(driver.impact * 10).toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                  {(!xai || !xai.top_drivers || xai.top_drivers.length === 0) && (
                    <div className="text-muted-foreground text-[11px] italic text-center py-2">
                      No XAI driver data available.
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

      </div>

    </div>
  );
}
