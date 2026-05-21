import React from 'react';
import { ChartData } from '@/types';
import { Cpu, AlertTriangle, Newspaper } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface SignalIntelligenceProps {
  data: ChartData | null;
}

export function SignalIntelligence({ data }: SignalIntelligenceProps) {
  if (!data || !data.ai_report) return (
    <div className="h-full w-full flex items-center justify-center text-muted-foreground font-mono text-xs uppercase tracking-widest border border-white/5 rounded-xl bg-card/20">
      Awaiting Signal Telemetry...
    </div>
  );

  const { Models, Context, Risk_Management } = data.ai_report;

  const getSignalColor = (action: string) => {
    if (action.includes('BUY')) return 'text-[var(--signal-buy)] border-[var(--signal-buy)]/30 bg-[var(--signal-buy)]/10';
    if (action.includes('SELL')) return 'text-[var(--signal-sell)] border-[var(--signal-sell)]/30 bg-[var(--signal-sell)]/10';
    return 'text-[var(--signal-hold)] border-[var(--signal-hold)]/30 bg-[var(--signal-hold)]/10';
  };

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  return (
    <motion.div 
      variants={container}
      initial="hidden"
      animate="show"
      className="grid grid-cols-1 md:grid-cols-3 gap-4 h-full"
      data-tour="intelligence"
    >
      {/* Model Consensus */}
      <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group hover:border-primary/30 transition-colors">
        <div className="bg-black/40 border-b border-white/5 px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-primary" />
            <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Model Consensus</h3>
          </div>
        </div>
        <div className="p-4 flex-1 flex flex-col gap-3 justify-center">
          <div className="flex items-center justify-between p-2.5 rounded border border-white/5 bg-black/20 group-hover:bg-black/40 transition-colors">
            <span className="text-xs font-mono text-muted-foreground">DL_LSTM_V4</span>
            <div className="flex items-center gap-2">
              <span className={cn("text-[10px] font-black uppercase px-2 py-0.5 rounded border", getSignalColor(Models.Primary_Deep_Learning.Suggested_Action))}>
                {Models.Primary_Deep_Learning.Suggested_Action}
              </span>
              <span className="text-xs font-mono opacity-70 w-12 text-right">{Models.Primary_Deep_Learning.Confidence}</span>
            </div>
          </div>
          <div className="flex items-center justify-between p-2.5 rounded border border-white/5 bg-black/20 group-hover:bg-black/40 transition-colors">
            <span className="text-xs font-mono text-muted-foreground">XGB_AGENT</span>
            <div className="flex items-center gap-2">
              <span className={cn("text-[10px] font-black uppercase px-2 py-0.5 rounded border", getSignalColor(Models.Secondary_XGBoost.Suggested_Action))}>
                {Models.Secondary_XGBoost.Suggested_Action}
              </span>
              <span className="text-xs font-mono opacity-70 w-12 text-right">{Models.Secondary_XGBoost.Confidence}</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Risk Projections */}
      <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group hover:border-[var(--signal-buy)]/30 transition-colors">
        <div className="bg-black/40 border-b border-white/5 px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-[var(--signal-hold)]" />
            <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">10-Day Projections</h3>
          </div>
        </div>
        <div className="p-4 flex-1 flex flex-col justify-center gap-4">
          <div className="flex items-center justify-between font-mono text-sm relative">
            <div className="flex flex-col">
              <span className="text-muted-foreground text-[10px] uppercase mb-1">Floor</span>
              <span className="text-[var(--signal-sell)] font-bold text-lg">${Risk_Management.Dynamic_10_Day_Range.Low.toFixed(2)}</span>
            </div>
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1/3 h-px bg-gradient-to-r from-[var(--signal-sell)]/20 via-white/20 to-[var(--signal-buy)]/20"></div>
            <div className="flex flex-col text-right">
              <span className="text-muted-foreground text-[10px] uppercase mb-1">Ceiling</span>
              <span className="text-[var(--signal-buy)] font-bold text-lg">${Risk_Management.Dynamic_10_Day_Range.High.toFixed(2)}</span>
            </div>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed border-l border-primary/30 pl-3 bg-primary/5 py-2 rounded-r-md">
            <span className="text-primary font-bold">XAI:</span> {Risk_Management.Meta_Model_Status}
          </p>
        </div>
      </motion.div>

      {/* Market Context */}
      <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group hover:border-white/20 transition-colors">
        <div className="bg-black/40 border-b border-white/5 px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Newspaper className="w-3.5 h-3.5 text-muted-foreground" />
            <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">NLP Analysis</h3>
          </div>
        </div>
        <div className="p-4 flex-1 flex flex-col">
          <p className="text-xs text-foreground/80 leading-relaxed font-sans line-clamp-6">
            {Context.Top_Headline_Processed}
          </p>
          <div className="mt-auto pt-4 flex justify-between items-center opacity-50">
            <span className="text-[9px] font-mono">SOURCE: SEC EDGAR / NEWS</span>
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
          </div>
        </div>
      </motion.div>

    </motion.div>
  );
}
