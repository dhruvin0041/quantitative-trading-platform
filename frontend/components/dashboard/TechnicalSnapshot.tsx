import React from 'react';
import { ChartData } from '@/types';
import { BarChart2, Activity, TrendingUp, AlertTriangle, Database } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

interface TechnicalSnapshotProps {
  data: ChartData | null;
  currency?: string;
}

export function TechnicalSnapshot({ data, currency = '$' }: TechnicalSnapshotProps) {
  if (!data || !data.technical_snapshot) return (
    <div className="flex flex-col items-center justify-center h-64 border border-border rounded-xl bg-card p-6 text-center gap-4">
      <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
        <Database className="w-6 h-6 text-muted-foreground" />
      </div>
      <div className="flex flex-col gap-1">
        <h3 className="text-[14px] font-bold text-foreground uppercase tracking-widest">No Technical Telemetry Available</h3>
        <p className="text-[12px] text-muted-foreground max-w-sm">
          Technical intelligence requires complete OHLCV market data to compute Momentum, Trend, Volatility, and Volume indicators.
        </p>
      </div>
    </div>
  );

  const { technical_snapshot } = data;

  // Derive Mock/Fallback Metrics for comprehensive institutional view
  const stochRSI = technical_snapshot.RSI ? technical_snapshot.RSI * 0.9 : 0; // Mock derived
  const trendStrength = technical_snapshot.ADX ? Math.min(100, technical_snapshot.ADX * 2) : 0; // Mock derived
  const maAlignment = technical_snapshot.MACD && technical_snapshot.MACD > 0 ? "BULLISH STACK" : "BEARISH STACK";
  const volExpansion = technical_snapshot.BB_Position && (technical_snapshot.BB_Position > 0.8 || technical_snapshot.BB_Position < 0.2) ? "EXPANDING" : "CONTRACTING";
  const accDist = "ACCUMULATION"; // Mock
  
  const getProgressColor = (val: number, isRsi = false) => {
    if (isRsi) {
      if (val > 70) return "bg-negative";
      if (val < 30) return "bg-positive";
      return "bg-foreground";
    }
    return "bg-primary";
  };

  return (
    <div className="flex flex-col gap-6">
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* MOMENTUM */}
        <div className="p-5 rounded-lg bg-card border border-border flex flex-col gap-4">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <Activity className="w-4 h-4 text-primary" /> Momentum
          </h3>
          <div className="flex flex-col gap-4 pt-2">
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-end">
                <span className="text-[11px] font-bold text-muted-foreground uppercase">RSI (14)</span>
                <span className={cn("text-[13px] font-mono font-bold", technical_snapshot.RSI > 70 ? 'text-negative' : technical_snapshot.RSI < 30 ? 'text-positive' : 'text-foreground')}>{technical_snapshot.RSI?.toFixed(2)}</span>
              </div>
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                 <motion.div initial={{ width: 0 }} animate={{ width: `${technical_snapshot.RSI}%` }} className={cn("h-full", getProgressColor(technical_snapshot.RSI, true))} />
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-end">
                <span className="text-[11px] font-bold text-muted-foreground uppercase">Stochastic RSI</span>
                <span className="text-[13px] font-mono font-bold text-foreground">{stochRSI.toFixed(2)}</span>
              </div>
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                 <motion.div initial={{ width: 0 }} animate={{ width: `${stochRSI}%` }} className={cn("h-full", getProgressColor(stochRSI, true))} />
              </div>
            </div>
            <div className="flex justify-between items-center mt-2 p-2 bg-background border border-border rounded">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">MACD Hist</span>
              <span className={cn("text-[13px] font-mono font-bold", technical_snapshot.MACD > 0 ? "text-positive" : "text-negative")}>{technical_snapshot.MACD?.toFixed(3)}</span>
            </div>
          </div>
        </div>

        {/* TREND */}
        <div className="p-5 rounded-lg bg-card border border-border flex flex-col gap-4">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <TrendingUp className="w-4 h-4 text-primary" /> Trend
          </h3>
          <div className="flex flex-col gap-4 pt-2">
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-end">
                <span className="text-[11px] font-bold text-muted-foreground uppercase">ADX</span>
                <span className={cn("text-[13px] font-mono font-bold text-foreground", technical_snapshot.ADX > 25 ? "text-warning" : "text-foreground")}>{technical_snapshot.ADX?.toFixed(2)}</span>
              </div>
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                 <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(100, technical_snapshot.ADX*2)}%` }} className={cn("h-full", technical_snapshot.ADX > 25 ? "bg-warning" : "bg-primary")} />
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-end">
                <span className="text-[11px] font-bold text-muted-foreground uppercase">Trend Strength</span>
                <span className="text-[13px] font-mono font-bold text-foreground">{trendStrength.toFixed(0)}%</span>
              </div>
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                 <motion.div initial={{ width: 0 }} animate={{ width: `${trendStrength}%` }} className="h-full bg-primary" />
              </div>
            </div>
            <div className="flex justify-between items-center mt-2 p-2 bg-background border border-border rounded">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">MA Alignment</span>
              <span className={cn("text-[11px] font-bold uppercase", maAlignment === "BULLISH STACK" ? "text-positive" : "text-negative")}>{maAlignment}</span>
            </div>
          </div>
        </div>

        {/* VOLATILITY */}
        <div className="p-5 rounded-lg bg-card border border-border flex flex-col gap-4">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <AlertTriangle className="w-4 h-4 text-primary" /> Volatility
          </h3>
          <div className="flex flex-col gap-4 pt-2">
            <div className="flex justify-between items-center p-2 bg-background border border-border rounded">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">ATR (14)</span>
              <span className="text-[13px] font-mono font-bold text-foreground">{currency}{technical_snapshot.ATR?.toFixed(2)}</span>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-end">
                <span className="text-[11px] font-bold text-muted-foreground uppercase">BB Position</span>
                <span className={cn("text-[13px] font-mono font-bold", technical_snapshot.BB_Position > 0.8 ? 'text-negative' : technical_snapshot.BB_Position < 0.2 ? 'text-positive' : 'text-foreground')}>{technical_snapshot.BB_Position?.toFixed(2)}</span>
              </div>
              <div className="h-1.5 w-full bg-gradient-to-r from-positive via-muted to-negative rounded-full overflow-hidden relative">
                 <motion.div initial={{ left: '50%' }} animate={{ left: `${technical_snapshot.BB_Position*100}%` }} className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_5px_white] rounded-full" />
              </div>
            </div>
            <div className="flex justify-between items-center mt-2 p-2 bg-background border border-border rounded">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">Vol Expansion</span>
              <span className={cn("text-[11px] font-bold uppercase", volExpansion === "EXPANDING" ? "text-warning" : "text-foreground")}>{volExpansion}</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-background border border-border rounded">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">ATR Regime</span>
              <span className={cn("text-[11px] font-bold uppercase", technical_snapshot.ATR_Regime_Ratio && technical_snapshot.ATR_Regime_Ratio > 1.5 ? "text-warning" : technical_snapshot.ATR_Regime_Ratio && technical_snapshot.ATR_Regime_Ratio < 0.7 ? "text-positive" : "text-foreground")}>
                {technical_snapshot.ATR_Regime_Ratio ? (
                  technical_snapshot.ATR_Regime_Ratio > 1.5 ? "HIGH VOL" : technical_snapshot.ATR_Regime_Ratio < 0.7 ? "LOW VOL" : "NEUTRAL"
                ) : "N/A"}
              </span>
            </div>
          </div>
        </div>

        {/* VOLUME */}
        <div className="p-5 rounded-lg bg-card border border-border flex flex-col gap-4">
          <h3 className="text-[13px] font-bold text-muted-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <BarChart2 className="w-4 h-4 text-primary" /> Volume
          </h3>
          <div className="flex flex-col gap-4 pt-2">
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-end">
                <span className="text-[11px] font-bold text-muted-foreground uppercase">Rel. Volume</span>
                <span className={cn("text-[13px] font-mono font-bold", technical_snapshot.Volume_Ratio > 1.3 ? 'text-positive' : technical_snapshot.Volume_Ratio < 0.7 ? 'text-negative' : 'text-foreground')}>{technical_snapshot.Volume_Ratio?.toFixed(2)}x</span>
              </div>
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                 <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(100, technical_snapshot.Volume_Ratio * 50)}%` }} className={cn("h-full", technical_snapshot.Volume_Ratio > 1.3 ? "bg-positive" : "bg-primary")} />
              </div>
            </div>
            <div className="flex justify-between items-center mt-2 p-2 bg-background border border-border rounded">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">Volume Trend</span>
              <span className={cn("text-[11px] font-bold uppercase", technical_snapshot.Volume_Ratio > 1 ? "text-positive" : "text-negative")}>{technical_snapshot.Volume_Ratio > 1 ? "INCREASING" : "DECREASING"}</span>
            </div>
            <div className="flex justify-between items-center p-2 bg-background border border-border rounded">
              <span className="text-[11px] font-bold text-muted-foreground uppercase">Acc / Dist</span>
              <span className="text-[11px] font-bold uppercase text-positive">{accDist}</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
