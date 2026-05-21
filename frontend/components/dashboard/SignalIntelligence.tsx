import React from 'react';
import { ChartData } from '@/types';
import { Cpu, AlertTriangle, Newspaper } from 'lucide-react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

import { Variants } from 'framer-motion';

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
    if (action.includes('BUY')) return 'text-[var(--signal-buy)] border-[var(--signal-buy)]/30 bg-[var(--signal-buy)]/10 dark:text-[var(--signal-buy)] dark:border-[var(--signal-buy)]/30 dark:bg-[var(--signal-buy)]/10';
    if (action.includes('SELL')) return 'text-[var(--signal-sell)] border-[var(--signal-sell)]/30 bg-[var(--signal-sell)]/10 dark:text-[var(--signal-sell)] dark:border-[var(--signal-sell)]/30 dark:bg-[var(--signal-sell)]/10';
    return 'text-[var(--signal-hold)] border-[var(--signal-hold)]/30 bg-[var(--signal-hold)]/10 dark:text-[var(--signal-hold)] dark:border-[var(--signal-hold)]/30 dark:bg-[var(--signal-hold)]/10';
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
      className="grid grid-cols-1 md:grid-cols-3 gap-4 h-full"
      data-tour="intelligence"
    >
      {/* Model Consensus */}
      <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
        <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-primary" />
            <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Model Consensus</h3>
          </div>
        </div>
        <div className="p-4 flex-1 flex flex-col gap-3 justify-center">
          <div className="flex items-center justify-between p-2.5 rounded border border-border bg-muted/30 dark:bg-black/20 group-hover:bg-muted/50 dark:group-hover:bg-black/40 transition-colors">
            <span className="text-xs font-mono text-muted-foreground uppercase font-bold">DL_LSTM_V4</span>
            <div className="flex items-center gap-2">
              <span className={cn("text-[10px] font-black uppercase px-2 py-0.5 rounded border", getSignalColor(Models.Primary_Deep_Learning.Suggested_Action))}>
                {Models.Primary_Deep_Learning.Suggested_Action}
              </span>
              <span className="text-xs font-mono font-bold text-primary dark:text-foreground w-12 text-right">{Models.Primary_Deep_Learning.Confidence}</span>
            </div>
          </div>
          <div className="flex items-center justify-between p-2.5 rounded border border-border bg-muted/30 dark:bg-black/20 group-hover:bg-muted/50 dark:group-hover:bg-black/40 transition-colors">
            <span className="text-xs font-mono text-muted-foreground uppercase font-bold">XGB_AGENT</span>
            <div className="flex items-center gap-2">
              <span className={cn("text-[10px] font-black uppercase px-2 py-0.5 rounded border", getSignalColor(Models.Secondary_XGBoost.Suggested_Action))}>
                {Models.Secondary_XGBoost.Suggested_Action}
              </span>
              <span className="text-xs font-mono font-bold text-primary dark:text-foreground w-12 text-right">{Models.Secondary_XGBoost.Confidence}</span>
            </div>
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
              <span className="text-muted-foreground text-[10px] font-bold uppercase mb-1">Floor</span>
              <span className="text-[var(--signal-sell)] font-black text-xl">${Risk_Management.Dynamic_10_Day_Range.Low.toFixed(2)}</span>
            </div>
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-1/3 h-px bg-border opacity-50"></div>
            <div className="flex flex-col text-right">
              <span className="text-muted-foreground text-[10px] font-bold uppercase mb-1">Ceiling</span>
              <span className="text-[var(--signal-buy)] font-black text-xl">${Risk_Management.Dynamic_10_Day_Range.High.toFixed(2)}</span>
            </div>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed border-l-4 border-primary/30 pl-3 bg-primary/5 py-2 rounded-r-md italic">
            <span className="text-primary font-black uppercase text-[9px] not-italic mr-1">Status:</span> {Risk_Management.Meta_Model_Status}
          </p>
        </div>
      </motion.div>

      {/* Market Context */}
      <motion.div variants={item} className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
        <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Newspaper className="w-3.5 h-3.5 text-primary" />
            <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">NLP Analysis</h3>
          </div>
        </div>
        <div className="p-4 flex-1 flex flex-col">
          <p className="text-xs text-foreground font-medium leading-relaxed font-sans line-clamp-6">
            {Context.Top_Headline_Processed}
          </p>
          <div className="mt-auto pt-4 flex justify-between items-center opacity-60">
            <span className="text-[9px] font-mono font-bold uppercase">Source: SEC EDGAR / News</span>
            <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
          </div>
        </div>
      </motion.div>

    </motion.div>
  );
}
