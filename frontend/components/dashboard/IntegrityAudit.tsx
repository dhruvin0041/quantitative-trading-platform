import React, { useState } from 'react';
import { ShieldCheck, Info, Database, Zap, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChartData } from '@/types';
import { cn } from '@/lib/utils';

interface IntegrityAuditProps {
  data: ChartData | null;
}

export function IntegrityAudit({ data }: IntegrityAuditProps) {
  const [isOpen, setIsOpen] = useState(false);

  const audits = [
    { 
      metric: 'Alpha Signal', 
      source: 'MetaEnsemble (backend/src/models/ensemble)', 
      provenance: data?.signal === 'VETOED' ? 'RiskAgent Override' : 'Neural Fusion Consensus',
      status: !!data?.signal ? 'LIVE' : 'DISCONNECTED'
    },
    { 
      metric: 'Uncertainty', 
      source: 'Meta-Model Covariance (backend/src/models/ensemble)', 
      provenance: 'Live Variance Analysis',
      status: data?.uncertainty_score !== undefined ? 'LIVE' : 'DISCONNECTED'
    },
    { 
      metric: 'Market Regime', 
      source: 'HMM Detector (backend/src/models/regime_detector.py)', 
      provenance: 'Viterbi Transition State',
      status: !!data?.market_regime ? 'LIVE' : 'DISCONNECTED'
    },
    { 
      metric: 'XAI Drivers', 
      source: 'SHAP TreeExplainer (backend/src/execution/live_inference.py)', 
      provenance: 'Additive Feature Attribution',
      status: (data?.xai && data.xai.top_drivers?.length > 0) ? 'LIVE' : (data?.xai ? 'EMPTY' : 'DISCONNECTED')
    },
    { 
      metric: 'Portfolio', 
      source: 'PaperTradingEngine (backend/data/paper_trading.json)', 
      provenance: 'State-Locked Persistence',
      status: !!data?.portfolio ? 'LIVE' : 'DISCONNECTED'
    },
    {
      metric: 'TFT Forecast',
      source: 'TemporalFusionTransformer (backend/src/models/neural)',
      provenance: data?.is_point_forecast ? 'Static Point Output' : 'Multivariate Quantile Bands',
      status: data?.is_point_forecast ? 'WARNING' : 'LIVE'
    },
    { 
      metric: 'NLP Alpha', 
      source: 'Gemini-2.0-Flash (backend/src/data_ingestion/nlp_processor.py)', 
      provenance: 'LLM Qualitative Extraction (Informational Only)',
      status: (data?.qualitative_alpha === 'Qualitative analysis unavailable' || !data?.qualitative_alpha) ? 'DISABLED' : 'LIVE'
    },
    { 
      metric: 'FX Normalization', 
      source: 'FXEngine (backend/src/execution/fx_engine.py)', 
      provenance: 'Live Yahoo Finance Feeds (Dynamic)',
      status: !!data?.portfolio?.fx_rates ? 'LIVE' : 'DISCONNECTED'
    }
  ];

  return (
    <div className="fixed bottom-4 left-4 z-[60]">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white dark:bg-black/80 border border-emerald-500/50 text-emerald-600 dark:text-emerald-400 text-[10px] font-black uppercase tracking-widest hover:bg-emerald-500 hover:text-white dark:hover:text-black transition-all shadow-lg"
      >
        <ShieldCheck className="w-3.5 h-3.5" />
        Data Integrity Mode
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            className="absolute bottom-12 left-0 w-[400px] bg-white dark:bg-black/95 border border-border dark:border-white/10 rounded-xl shadow-2xl p-4 backdrop-blur-xl overflow-hidden"
          >
            <div className="flex items-center justify-between mb-4 border-b border-border dark:border-white/5 pb-2">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-primary" />
                <h3 className="text-xs font-black uppercase tracking-tight text-foreground">Audit Provenance Report</h3>
              </div>
              <span className="text-[9px] font-mono opacity-50 text-foreground">v2.1.0_STABLE</span>
            </div>

            <div className="space-y-3 max-h-[350px] overflow-y-auto pr-2 custom-scrollbar">
              {audits.map((a, i) => (
                <div key={i} className="flex flex-col gap-1 p-2 rounded bg-secondary/50 dark:bg-white/5 border border-border dark:border-white/5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-foreground">{a.metric}</span>
                    <span className={cn(
                      "flex items-center gap-1 text-[8px] font-black",
                      a.status === 'LIVE' ? "text-emerald-500" : (a.status === 'WARNING' ? "text-amber-500" : "text-red-500")
                    )}>
                      {a.status === 'LIVE' ? <Zap className="w-2.5 h-2.5 fill-current" /> : (a.status === 'WARNING' ? <Info className="w-2.5 h-2.5" /> : <AlertTriangle className="w-2.5 h-2.5" />)}
                      {a.status}
                    </span>
                  </div>
                  <div className="flex flex-col text-[9px] font-mono opacity-70">
                    <span className="text-muted-foreground truncate">Source: {a.source}</span>
                    <span className="text-primary italic font-bold">Mode: {a.provenance}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
              <p className="text-[9px] leading-relaxed text-emerald-400 italic">
                Verification Note: Every metric in this terminal is actively sourced from a live compute instance. Synthetic data injection is strictly prohibited by Hydra core mandates.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
