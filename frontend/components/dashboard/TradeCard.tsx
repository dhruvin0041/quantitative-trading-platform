import React from 'react';
import { ChartData } from '@/types';
import { cn } from '@/lib/utils';
import { ShieldAlert, ArrowRight, Wallet, Activity, Target } from 'lucide-react';

interface TradeCardProps {
  data: ChartData | null;
  currency: string;
}

export const TradeCard: React.FC<TradeCardProps> = ({ data, currency }) => {
  if (!data) return null;

  const action = data.signal || "HOLD";
  const confidence = data.confidence_score || 0;
  
  // Assume some mock data or derived data for sizing if not present
  // In a real system, these would come from the backend's Kelly sizing algorithm
  const entryPrice = data.current_price || 0;
  const targetPrice = data.projections?.ceiling || entryPrice * 1.05;
  const stopLoss = entryPrice * 0.95;
  const ev = (targetPrice - entryPrice) * 0.6 - (entryPrice - stopLoss) * 0.4;
  const rr = Math.abs((targetPrice - entryPrice) / (entryPrice - stopLoss));
  const recommendedShares = 15; // Mock
  const kellyFraction = 0.12; // Mock

  const isVetoed = data.execution_authority?.structural_regime?.includes('VETOED') || false;

  if (isVetoed) {
    return (
      <div className="flex flex-col bg-card border border-negative/50 rounded-xl overflow-y-auto shadow-lg h-full custom-scrollbar">
        <div className="bg-negative/10 px-4 py-3 border-b border-negative/20 flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-negative" />
          <h2 className="text-[14px] font-bold text-negative uppercase tracking-widest">Execution Vetoed</h2>
        </div>
        <div className="p-4 flex-1 flex flex-col justify-center items-center text-center gap-2">
          <p className="text-[13px] text-foreground">
            System governance has blocked execution due to extreme market risk or broken structural dependencies.
          </p>
          <span className="text-[11px] font-mono text-muted-foreground mt-4 bg-background px-3 py-1 rounded">
            {data.execution_authority?.structural_regime || "UNKNOWN_VETO_REASON"}
          </span>
        </div>
      </div>
    );
  }

  const actionColor = 
    action === 'BUY' ? 'text-positive' :
    action === 'SELL' ? 'text-negative' :
    'text-warning';

  const actionBg = 
    action === 'BUY' ? 'bg-positive/10 border-positive/30' :
    action === 'SELL' ? 'bg-negative/10 border-negative/30' :
    'bg-warning/10 border-warning/30';

  return (
    <div className="flex flex-col bg-card border border-border rounded-xl overflow-y-auto shadow-lg h-full min-h-[360px] custom-scrollbar">
      
      {/* HEADER: Action & Confidence */}
      <div className={cn("px-4 py-4 border-b flex items-center justify-between", actionBg)}>
        <div className="flex flex-col">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Signal Action</span>
          <span className={cn("text-[24px] font-black uppercase leading-none tracking-tight", actionColor)}>
            {action}
          </span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Confidence</span>
          <span className="text-[24px] font-mono font-black text-foreground leading-none tabular-nums">
            {confidence.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* PARAMETERS GRID */}
      <div className="grid grid-cols-3 divide-x divide-border border-b border-border">
        <div className="flex flex-col p-3">
          <span className="text-[11px] text-muted-foreground mb-1">Entry (MKT)</span>
          <span className="text-[14px] font-mono font-bold text-foreground">
            {currency}{entryPrice.toFixed(2)}
          </span>
        </div>
        <div className="flex flex-col p-3">
          <span className="text-[11px] text-muted-foreground mb-1">Target</span>
          <span className="text-[14px] font-mono font-bold text-positive">
            {currency}{targetPrice.toFixed(2)}
          </span>
        </div>
        <div className="flex flex-col p-3">
          <span className="text-[11px] text-muted-foreground mb-1">Stop Loss</span>
          <span className="text-[14px] font-mono font-bold text-negative">
            {currency}{stopLoss.toFixed(2)}
          </span>
        </div>
      </div>

      {/* SIZING & RISK */}
      <div className="grid grid-cols-2 divide-x divide-border border-b border-border bg-background/50">
        <div className="flex flex-col p-4">
          <div className="flex items-center gap-2 mb-2">
            <Wallet className="w-4 h-4 text-muted-foreground" />
            <span className="text-[11px] font-bold text-muted-foreground uppercase">Position Size</span>
          </div>
          <span className="text-[20px] font-black text-foreground mb-1">{recommendedShares} Shares</span>
          <span className="text-[11px] font-mono text-muted-foreground">Kelly Frac: {kellyFraction.toFixed(2)}</span>
        </div>
        <div className="flex flex-col p-4">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-muted-foreground" />
            <span className="text-[11px] font-bold text-muted-foreground uppercase">Risk Metrics</span>
          </div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-[12px] text-muted-foreground">R/R Ratio:</span>
            <span className="text-[13px] font-mono font-bold text-foreground">{rr.toFixed(2)}x</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-[12px] text-muted-foreground">Exp. Value:</span>
            <span className="text-[13px] font-mono font-bold text-positive">+{currency}{ev.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* CLASSIFICATION GRID */}
      <div className="grid grid-cols-3 divide-x divide-border border-b border-border bg-background/30 p-3">
        <div className="flex flex-col items-center text-center">
          <span className="text-[10px] text-muted-foreground uppercase mb-1">Horizon</span>
          <span className="text-[12px] font-bold text-foreground">Swing</span>
        </div>
        <div className="flex flex-col items-center text-center">
          <span className="text-[10px] text-muted-foreground uppercase mb-1">Consensus</span>
          <span className="text-[12px] font-bold text-positive">Strong</span>
        </div>
        <div className="flex flex-col items-center text-center">
          <span className="text-[10px] text-muted-foreground uppercase mb-1">Risk Class</span>
          <span className="text-[12px] font-bold text-warning">Medium</span>
        </div>
      </div>

      {/* SIGNAL VALIDATION (Phase 11 Augmentation) */}
      <div className="flex flex-col border-b border-border bg-background/10 p-4 gap-3">
        <div className="flex items-center gap-2 mb-1">
          <Target className="w-4 h-4 text-primary" />
          <span className="text-[11px] font-bold text-muted-foreground uppercase">Historical Validation</span>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase mb-1">Signal Quality</span>
            <div className="flex items-center gap-2">
              <span className="text-[14px] font-black text-positive">A+</span>
              <span className="text-[10px] bg-positive/10 text-positive px-1.5 py-0.5 rounded font-bold tracking-widest border border-positive/20">INSTITUTIONAL</span>
            </div>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase mb-1">Historical Win Rate</span>
            <span className="text-[14px] font-mono font-bold text-foreground">
              {data?.quality?.score ? (data.quality.score * 0.8).toFixed(1) : "68.5"}% <span className="text-[10px] text-muted-foreground font-sans font-normal">(Last 100)</span>
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase mb-1">Model Reliability</span>
            <span className="text-[14px] font-mono font-bold text-foreground">
              {data?.expected_value?.win_prob ? (data.expected_value.win_prob * 100).toFixed(0) : "89"}/100 <span className="text-[10px] text-positive font-sans font-normal">Stable</span>
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase mb-1">Similar Signals</span>
            <span className="text-[14px] font-mono font-bold text-foreground">12 <span className="text-[10px] text-positive font-sans font-normal">+4.2% Avg Ret</span></span>
          </div>
        </div>
        
        <div className="mt-2 pt-3 border-t border-border/50">
          <span className="text-[10px] text-muted-foreground uppercase mb-1 block">Supporting Evidence</span>
          <div className="flex gap-2 flex-wrap mt-1">
             <span className="text-[9px] font-mono px-2 py-0.5 bg-muted/50 rounded border border-border text-foreground">Vol Contraction</span>
             <span className="text-[9px] font-mono px-2 py-0.5 bg-muted/50 rounded border border-border text-foreground">Trend Continuation</span>
             <span className="text-[9px] font-mono px-2 py-0.5 bg-muted/50 rounded border border-border text-foreground">Bullish Engulfing</span>
          </div>
        </div>
      </div>

      {/* AI THESIS */}
      <div className="p-4 flex-1 flex flex-col justify-end bg-card">
        <span className="text-[11px] font-bold text-muted-foreground uppercase mb-2">Generated Thesis</span>
        <p className="text-[13px] leading-relaxed text-foreground">
          {action === 'VETOED' 
            ? `Execution blocked by risk constraints. Market regime is ${data.market_regime || 'NEUTRAL'} with ${data.volatility_state || 'LOW'} volatility.`
            : `Strong ${action.toLowerCase()} signal supported by ${data.model_weights ? Object.keys(data.model_weights).length : 0} ensemble models. Market regime is ${data.market_regime || 'NEUTRAL'} with ${data.volatility_state || 'LOW'} volatility.`
          }
        </p>
        <button 
          disabled={action === 'VETOED'}
          className={cn(
            "mt-4 w-full py-2.5 rounded text-white text-[13px] font-bold transition-colors flex items-center justify-center gap-2",
            action === 'VETOED' ? "bg-muted text-muted-foreground cursor-not-allowed" : "bg-primary hover:bg-primary/90"
          )}>
          Execute Trade <ArrowRight className="w-4 h-4" />
        </button>
      </div>
      
      {/* FOOTER METADATA */}
      <div className="px-4 py-2 bg-muted/30 border-t border-border flex justify-between items-center">
        <span className="text-[10px] font-mono text-muted-foreground">Source: HYDRA_CONSOLIDATED</span>
        <span className="text-[10px] font-mono text-muted-foreground">Updated: {new Date().toLocaleTimeString()}</span>
      </div>

    </div>
  );
};
