import React from 'react';
import { ChartData } from '@/types';
import { Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TechnicalSnapshotProps {
  data: ChartData | null;
  currency?: string;
}

export function TechnicalSnapshot({ data, currency = '$' }: TechnicalSnapshotProps) {
  if (!data || !data.technical_snapshot) return null;

  const { technical_snapshot } = data;

  const getRSIColor = (rsi: number) => {
    if (rsi > 70) return 'text-red-500';
    if (rsi < 30) return 'text-green-500';
    return 'text-foreground';
  };

  const getRSILabel = (rsi: number) => {
    if (rsi > 80) return 'EXTREME OVERBOUGHT';
    if (rsi > 70) return 'OVERBOUGHT';
    if (rsi < 20) return 'EXTREME OVERSOLD';
    if (rsi < 30) return 'OVERSOLD';
    return 'NEUTRAL';
  };

  const getADXColor = (adx: number) => {
    if (adx > 35) return 'text-orange-600 dark:text-orange-400';
    if (adx > 25) return 'text-orange-500';
    if (adx < 20) return 'text-muted-foreground';
    return 'text-foreground';
  };

  const getADXLabel = (adx: number) => {
    if (adx > 35) return 'EXTREME TREND';
    if (adx > 25) return 'STRONG TREND';
    if (adx < 20) return 'WEAK/NO TREND';
    return 'TRENDING';
  };

  return (
    <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300">
      <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-primary" />
          <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Technical Snapshot</h3>
        </div>
      </div>
      <div className="p-4 flex-1 grid grid-cols-2 gap-3">
        <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50 relative overflow-hidden group/tech">
          <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">RSI (14)</span>
          <span className={cn("text-xs font-mono font-black", getRSIColor(technical_snapshot.RSI))}>{technical_snapshot.RSI?.toFixed(2)}</span>
          <span className={cn("text-[7px] font-black mt-1 uppercase", getRSIColor(technical_snapshot.RSI))}>{getRSILabel(technical_snapshot.RSI)}</span>
          {technical_snapshot.RSI > 70 && <span className="absolute bottom-0 inset-x-0 h-0.5 bg-red-500/50" />}
          {technical_snapshot.RSI < 30 && <span className="absolute bottom-0 inset-x-0 h-0.5 bg-green-500/50" />}
        </div>
        <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
          <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">MACD Hist</span>
          <span className={cn("text-xs font-mono font-black", technical_snapshot.MACD > 0 ? "text-green-500" : "text-red-500")}>{technical_snapshot.MACD?.toFixed(3)}</span>
          <span className="text-[7px] font-black mt-1 text-muted-foreground uppercase">{technical_snapshot.MACD > 0 ? 'BULLISH' : 'BEARISH'}</span>
        </div>
        <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50 relative">
          <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">BB Position</span>
          <span className={cn("text-xs font-mono font-black", technical_snapshot.BB_Position > 0.8 ? "text-red-500" : technical_snapshot.BB_Position < 0.2 ? "text-green-500" : "text-foreground")}>{technical_snapshot.BB_Position?.toFixed(2)}</span>
          <span className="text-[7px] font-black mt-1 text-muted-foreground uppercase">{technical_snapshot.BB_Position > 0.8 ? 'OVEREXTENDED' : technical_snapshot.BB_Position < 0.2 ? 'UNDEREXTENDED' : 'WITHIN BANDS'}</span>
          {technical_snapshot.BB_Position > 0.8 && <span className="absolute bottom-0 inset-x-0 h-0.5 bg-red-500/50" />}
          {technical_snapshot.BB_Position < 0.2 && <span className="absolute bottom-0 inset-x-0 h-0.5 bg-green-500/50" />}
        </div>
        <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
          <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">ATR (14)</span>
          <span className="text-xs font-mono font-black">{currency}{technical_snapshot.ATR?.toFixed(2)}</span>
          <span className="text-[7px] font-black mt-1 text-muted-foreground uppercase">VOLATILITY</span>
        </div>
        <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50">
          <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">ADX (Trend)</span>
          <span className={cn("text-xs font-mono font-black", getADXColor(technical_snapshot.ADX))}>{technical_snapshot.ADX?.toFixed(2)}</span>
          <span className={cn("text-[7px] font-black mt-1 uppercase", getADXColor(technical_snapshot.ADX))}>{getADXLabel(technical_snapshot.ADX)}</span>
        </div>
        <div className="flex flex-col items-center justify-center p-2 rounded bg-muted/20 border border-border/50 relative">
          <span className="text-[8px] font-bold text-muted-foreground uppercase mb-1">Volume Ratio</span>
          <span className={cn("text-xs font-mono font-black", technical_snapshot.Volume_Ratio > 1.3 ? "text-green-500" : technical_snapshot.Volume_Ratio < 0.7 ? "text-red-500" : "text-foreground")}>{technical_snapshot.Volume_Ratio?.toFixed(2)}</span>
          <span className="text-[7px] font-black mt-1 text-muted-foreground uppercase">{technical_snapshot.Volume_Ratio > 1.3 ? 'HEAVY' : technical_snapshot.Volume_Ratio < 0.7 ? 'LIGHT' : 'NORMAL'}</span>
          {technical_snapshot.Volume_Ratio > 1.3 && <span className="absolute bottom-0 inset-x-0 h-0.5 bg-green-500/50" />}
          {technical_snapshot.Volume_Ratio < 0.7 && <span className="absolute bottom-0 inset-x-0 h-0.5 bg-red-500/50" />}
        </div>
      </div>
    </div>
  );
}
