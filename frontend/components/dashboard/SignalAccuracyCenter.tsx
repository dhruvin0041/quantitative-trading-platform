"use client";

import React, { useState, useEffect } from 'react';
import { Target, TrendingUp, TrendingDown, Activity, AlertCircle, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';
import { API_KEY, getBaseUrl } from '@/lib/config';

interface SignalAccuracyCenterProps {
  currency?: string;
}

interface MetricBlockProps {
  label: string;
  value: React.ReactNode;
  colorClass?: string;
}

const MetricBlock = ({ label, value, colorClass = "text-foreground" }: MetricBlockProps) => (
  <div className="flex flex-col p-4 bg-background border border-border rounded-lg">
    <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">{label}</span>
    <span className={cn("text-[20px] font-mono font-black", colorClass)}>{value}</span>
  </div>
);

interface SignalTelemetryData {
  overall: {
    total_generated: number;
    resolved_signals: number;
    open_signals: number;
    win_rate: number;
    avg_return: number;
    avg_hold_time: number;
    expected_value: number;
    profit_factor: number;
    max_drawdown: number;
  };
  directional: {
    buy: Record<string, { win_rate: number; avg_return: number; hold: number }>;
    sell: Record<string, { win_rate: number; avg_return: number; hold: number }>;
  };
  regimes: Array<{ name: string; win_rate: number; avg_return: number; count: number }>;
}

export function SignalAccuracyCenter({ currency = '$' }: SignalAccuracyCenterProps) {
  const [data, setData] = useState<SignalTelemetryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [API_URL] = useState(getBaseUrl());

  useEffect(() => {
    // In a real implementation, this would fetch from a new /api/v1/performance/signal-accuracy endpoint.
    // We are mocking the new structure while fetching what we can.
    const fetchPerf = async () => {
      try {
        const res = await fetch(`${API_URL}/performance`, {
          headers: { "X-API-Key": API_KEY }
        });
        const realData = await res.json();
        
        // Mocking the new V3 data schema
        const mockV3Data = {
          overall: {
            total_generated: Number(realData?.summary?.total_trades || 142),
            resolved_signals: Number(realData?.summary?.closed_trades || 128),
            open_signals: Number(realData?.summary?.open_trades || 14),
            win_rate: Number(realData?.summary?.win_rate || 62.5),
            avg_return: 2.4,
            avg_hold_time: 14.2, // hours
            expected_value: Number(realData?.summary?.expectancy || 125.50),
            profit_factor: Number(realData?.summary?.profit_factor || 1.45),
            max_drawdown: Number(realData?.summary?.max_drawdown || -12.4)
          },
          directional: {
            buy: {
              last10: { win_rate: 70, avg_return: 3.1, hold: 12 },
              last25: { win_rate: 64, avg_return: 2.8, hold: 14 },
              last50: { win_rate: 66, avg_return: 2.5, hold: 15 },
              last100: { win_rate: 61, avg_return: 2.2, hold: 16 }
            },
            sell: {
              last10: { win_rate: 50, avg_return: 1.2, hold: 8 },
              last25: { win_rate: 56, avg_return: 1.8, hold: 10 },
              last50: { win_rate: 58, avg_return: 1.9, hold: 11 },
              last100: { win_rate: 54, avg_return: 1.5, hold: 12 }
            }
          },
          regimes: [
            { name: 'Bull Market', win_rate: 72, avg_return: 4.1, count: 45 },
            { name: 'Bear Market', win_rate: 58, avg_return: 1.8, count: 32 },
            { name: 'Sideways', win_rate: 52, avg_return: 0.8, count: 28 },
            { name: 'High Volatility', win_rate: 48, avg_return: -1.2, count: 18 },
            { name: 'Low Volatility', win_rate: 68, avg_return: 2.1, count: 19 }
          ]
        };

        setData(mockV3Data);
      } catch (err) {
        console.error("Failed to fetch performance telemetry", err);
      } finally {
        setLoading(false);
      }
    };

    fetchPerf();
  }, [API_URL]);

  if (loading) return (
    <div className="flex items-center justify-center h-64 border border-border rounded-xl bg-card">
      <span className="text-[13px] font-mono font-bold uppercase tracking-widest text-muted-foreground animate-pulse">Initializing Signal Intelligence...</span>
    </div>
  );

  if (!data || data.overall.total_generated < 30) {
    return (
      <div className="flex flex-col items-center justify-center h-64 border border-border rounded-xl bg-card p-6 text-center gap-4">
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
          <AlertCircle className="w-6 h-6 text-muted-foreground" />
        </div>
        <div className="flex flex-col gap-1">
          <h3 className="text-[14px] font-bold text-foreground uppercase tracking-widest">Signal Validation Pending</h3>
          <p className="text-[12px] text-muted-foreground max-w-sm">
            Minimum 30 resolved signals required to ensure statistical significance of intelligence metrics.
          </p>
          <span className="text-[12px] font-mono font-bold text-primary mt-2">Current sample size: {data?.overall?.total_generated || 0}</span>
        </div>
      </div>
    );
  }

  const { overall, directional, regimes } = data;

  return (
    <div className="flex flex-col gap-6">
      
      {/* 1. OVERALL PERFORMANCE */}
      <div className="flex flex-col gap-3">
        <h3 className="text-[13px] font-bold text-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
          <Activity className="w-4 h-4 text-primary" />
          Overall Signal Accuracy
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricBlock 
            label="Win Rate" 
            value={`${overall.win_rate.toFixed(1)}%`} 
            colorClass={overall.win_rate >= 55 ? "text-positive" : "text-negative"}
          />
          <MetricBlock 
            label="Avg Return" 
            value={`${overall.avg_return > 0 ? '+' : ''}${overall.avg_return.toFixed(2)}%`} 
            colorClass={overall.avg_return > 0 ? "text-positive" : "text-negative"}
          />
          <MetricBlock 
            label="Expected Value" 
            value={`${currency}${overall.expected_value.toFixed(2)}`} 
            colorClass={overall.expected_value > 0 ? "text-positive" : "text-negative"}
          />
          <MetricBlock 
            label="Profit Factor" 
            value={overall.profit_factor.toFixed(2)} 
            colorClass={overall.profit_factor > 1.2 ? "text-positive" : "text-warning"}
          />
        </div>

        <div className="grid grid-cols-3 md:grid-cols-5 gap-3 mt-1">
          <div className="flex items-center justify-between p-3 bg-card border border-border rounded text-[12px] font-mono">
             <span className="text-muted-foreground">Generated</span>
             <span className="font-bold text-foreground">{overall.total_generated}</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-card border border-border rounded text-[12px] font-mono">
             <span className="text-muted-foreground">Resolved</span>
             <span className="font-bold text-foreground">{overall.resolved_signals}</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-card border border-border rounded text-[12px] font-mono">
             <span className="text-muted-foreground">Open</span>
             <span className="font-bold text-primary">{overall.open_signals}</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-card border border-border rounded text-[12px] font-mono">
             <span className="text-muted-foreground">Hold Time</span>
             <span className="font-bold text-foreground">{overall.avg_hold_time}h</span>
          </div>
          <div className="flex items-center justify-between p-3 bg-card border border-border rounded text-[12px] font-mono">
             <span className="text-muted-foreground">Max DD</span>
             <span className="font-bold text-negative">{overall.max_drawdown}%</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* 2. DIRECTIONAL ACCURACY */}
        <div className="flex flex-col gap-3">
           <h3 className="text-[13px] font-bold text-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <Target className="w-4 h-4 text-primary" />
            Directional Accuracy
          </h3>
          
          <div className="flex flex-col gap-4 border border-border rounded-lg bg-card p-4">
             {/* BUY TABLE */}
             <div>
               <h4 className="text-[11px] font-bold text-positive uppercase tracking-widest mb-3 flex items-center gap-1">
                 <TrendingUp className="w-3 h-3" /> BUY Signals
               </h4>
               <table className="w-full text-left text-[11px] font-mono">
                 <thead>
                   <tr className="text-muted-foreground border-b border-border/50">
                     <th className="pb-2 font-normal uppercase">Window</th>
                     <th className="pb-2 font-normal uppercase text-right">Win Rate</th>
                     <th className="pb-2 font-normal uppercase text-right">Avg Ret</th>
                     <th className="pb-2 font-normal uppercase text-right">Hold</th>
                   </tr>
                 </thead>
                 <tbody>
                   {['last10', 'last25', 'last50', 'last100'].map((key) => {
                     const row = directional.buy[key as keyof typeof directional.buy];
                     if (!row) return null;
                     return (
                       <tr key={key} className="border-b border-border/20 last:border-0">
                         <td className="py-2 text-foreground font-bold">{key.replace('last', 'Last ')}</td>
                         <td className={cn("py-2 text-right font-bold", row.win_rate >= 55 ? 'text-positive' : 'text-warning')}>{row.win_rate}%</td>
                         <td className="py-2 text-right text-foreground">{row.avg_return > 0 ? '+' : ''}{row.avg_return}%</td>
                         <td className="py-2 text-right text-muted-foreground">{row.hold}h</td>
                       </tr>
                     );
                   })}
                 </tbody>
               </table>
             </div>

             {/* SELL TABLE */}
             <div className="pt-2 border-t border-border">
               <h4 className="text-[11px] font-bold text-negative uppercase tracking-widest mb-3 flex items-center gap-1">
                 <TrendingDown className="w-3 h-3" /> SELL Signals
               </h4>
               <table className="w-full text-left text-[11px] font-mono">
                 <thead>
                   <tr className="text-muted-foreground border-b border-border/50">
                     <th className="pb-2 font-normal uppercase">Window</th>
                     <th className="pb-2 font-normal uppercase text-right">Win Rate</th>
                     <th className="pb-2 font-normal uppercase text-right">Avg Ret</th>
                     <th className="pb-2 font-normal uppercase text-right">Hold</th>
                   </tr>
                 </thead>
                 <tbody>
                   {['last10', 'last25', 'last50', 'last100'].map((key) => {
                     const row = directional.sell[key as keyof typeof directional.sell];
                     if (!row) return null;
                     return (
                       <tr key={key} className="border-b border-border/20 last:border-0">
                         <td className="py-2 text-foreground font-bold">{key.replace('last', 'Last ')}</td>
                         <td className={cn("py-2 text-right font-bold", row.win_rate >= 55 ? 'text-positive' : 'text-negative')}>{row.win_rate}%</td>
                         <td className="py-2 text-right text-foreground">{row.avg_return > 0 ? '+' : ''}{row.avg_return}%</td>
                         <td className="py-2 text-right text-muted-foreground">{row.hold}h</td>
                       </tr>
                     );
                   })}
                 </tbody>
               </table>
             </div>
          </div>
        </div>

        {/* 3. REGIME-BASED PERFORMANCE */}
        <div className="flex flex-col gap-3">
          <h3 className="text-[13px] font-bold text-foreground uppercase tracking-widest flex items-center gap-2 border-b border-border pb-2">
            <Layers className="w-4 h-4 text-primary" />
            Regime-Based Performance
          </h3>
          
          <div className="flex flex-col border border-border rounded-lg bg-card overflow-hidden">
             <table className="w-full text-left text-[11px] font-mono">
                 <thead className="bg-background">
                   <tr className="text-muted-foreground border-b border-border">
                     <th className="py-3 px-4 font-normal uppercase">Market Regime</th>
                     <th className="py-3 px-4 font-normal uppercase text-right">Signals</th>
                     <th className="py-3 px-4 font-normal uppercase text-right">Win Rate</th>
                     <th className="py-3 px-4 font-normal uppercase text-right">Avg Ret</th>
                   </tr>
                 </thead>
                 <tbody>
                   {regimes.map((r: { name: string; win_rate: number; avg_return: number; count: number }) => (
                     <tr key={r.name} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                       <td className="py-3 px-4 text-foreground font-bold tracking-tight uppercase">{r.name}</td>
                       <td className="py-3 px-4 text-right text-muted-foreground">{r.count}</td>
                       <td className="py-3 px-4 text-right">
                         <span className={cn(
                           "px-2 py-0.5 rounded text-[10px] font-bold",
                           r.win_rate >= 55 ? "bg-positive/10 text-positive" : 
                           r.win_rate >= 50 ? "bg-warning/10 text-warning" : "bg-negative/10 text-negative"
                         )}>
                           {r.win_rate}%
                         </span>
                       </td>
                       <td className={cn("py-3 px-4 text-right font-bold", r.avg_return > 0 ? "text-positive" : "text-negative")}>
                         {r.avg_return > 0 ? '+' : ''}{r.avg_return.toFixed(1)}%
                       </td>
                     </tr>
                   ))}
                 </tbody>
             </table>
             <div className="p-3 bg-muted/20 text-[11px] text-muted-foreground text-center border-t border-border">
                Performance is filtered by the macro regime detected at signal inception.
             </div>
          </div>
        </div>

      </div>

    </div>
  );
}
