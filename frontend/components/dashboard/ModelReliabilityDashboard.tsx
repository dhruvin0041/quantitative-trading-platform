"use client";

import React, { useState, useEffect } from 'react';
import { Cpu, TrendingUp, TrendingDown, Minus, Trophy, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getBaseUrl } from '@/lib/config';

interface ModelReliabilityDashboardProps {
  currency?: string;
}

interface ModelData {
  id: string;
  name: string;
  acc_10: number;
  acc_25: number;
  acc_50: number;
  acc_100: number;
  avg_return: number;
  win_rate: number;
  reliability_score: number;
  trend: string;
}

export function ModelReliabilityDashboard({}: ModelReliabilityDashboardProps) {
  const [data, setData] = useState<ModelData[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [API_URL] = useState(getBaseUrl());

  useEffect(() => {
    // Mocking the new V3 Model Reliability schema
    const fetchPerf = async () => {
      try {
        await new Promise(resolve => setTimeout(resolve, 600)); // Simulate latency
        
        const mockModels = [
          {
            id: 'fusion', name: 'Fusion Engine', 
            acc_10: 70, acc_25: 68, acc_50: 71, acc_100: 69,
            avg_return: 3.2, win_rate: 69.5, reliability_score: 92,
            trend: 'Improving'
          },
          {
            id: 'xgb', name: 'XGBoost Alpha', 
            acc_10: 60, acc_25: 64, acc_50: 66, acc_100: 62,
            avg_return: 2.1, win_rate: 63.0, reliability_score: 78,
            trend: 'Declining'
          },
          {
            id: 'dl', name: 'Deep Q-Network (DQN)', 
            acc_10: 50, acc_25: 56, acc_50: 54, acc_100: 58,
            avg_return: 1.1, win_rate: 54.5, reliability_score: 61,
            trend: 'Stable'
          },
          {
            id: 'lgbm', name: 'LightGBM Core', 
            acc_10: 80, acc_25: 72, acc_50: 68, acc_100: 65,
            avg_return: 2.8, win_rate: 71.2, reliability_score: 88,
            trend: 'Improving'
          },
          {
            id: 'consensus', name: 'Consensus Baseline', 
            acc_10: 60, acc_25: 60, acc_50: 62, acc_100: 61,
            avg_return: 1.8, win_rate: 60.7, reliability_score: 75,
            trend: 'Stable'
          }
        ];

        // Sort by reliability
        mockModels.sort((a, b) => b.reliability_score - a.reliability_score);

        setData(mockModels);
      } catch (err) {
        console.error("Failed to fetch model reliability", err);
      } finally {
        setLoading(false);
      }
    };

    fetchPerf();
  }, [API_URL]);

  if (loading) return (
    <div className="flex items-center justify-center h-64 border border-border rounded-xl bg-card">
      <span className="text-[13px] font-mono font-bold uppercase tracking-widest text-muted-foreground animate-pulse">Initializing Reliability Engine...</span>
    </div>
  );

  if (!data) return null;

  const bestModel = data[0];
  const worstModel = data[data.length - 1];

  const TrendIcon = ({ trend }: { trend: string }) => {
    if (trend === 'Improving') return <TrendingUp className="w-4 h-4 text-positive" />;
    if (trend === 'Declining') return <TrendingDown className="w-4 h-4 text-negative" />;
    return <Minus className="w-4 h-4 text-muted-foreground" />;
  };

  return (
    <div className="flex flex-col gap-6">
      
      {/* HIGHLIGHTS */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="flex items-start gap-4 p-4 rounded-xl bg-positive/5 border border-positive/20">
           <div className="w-10 h-10 rounded-full bg-positive/10 flex items-center justify-center shrink-0 mt-1">
             <Trophy className="w-5 h-5 text-positive" />
           </div>
           <div className="flex flex-col">
             <span className="text-[11px] font-bold uppercase tracking-widest text-positive mb-1">Highest Reliability</span>
             <span className="text-[16px] font-black text-foreground">{bestModel.name}</span>
             <span className="text-[12px] text-muted-foreground mt-1">
               Currently leading with a {bestModel.reliability_score}/100 score and {bestModel.acc_10}% win rate over the last 10 trades.
             </span>
           </div>
        </div>

        <div className="flex items-start gap-4 p-4 rounded-xl bg-negative/5 border border-negative/20">
           <div className="w-10 h-10 rounded-full bg-negative/10 flex items-center justify-center shrink-0 mt-1">
             <AlertTriangle className="w-5 h-5 text-negative" />
           </div>
           <div className="flex flex-col">
             <span className="text-[11px] font-bold uppercase tracking-widest text-negative mb-1">Performance Drift Detected</span>
             <span className="text-[16px] font-black text-foreground">{worstModel.name}</span>
             <span className="text-[12px] text-muted-foreground mt-1">
               Model is exhibiting structural decay. Accuracy dropped to {worstModel.acc_10}% in the current regime.
             </span>
           </div>
        </div>
      </div>

      {/* RELIABILITY MATRIX */}
      <div className="flex flex-col border border-border rounded-xl bg-card overflow-hidden">
        <div className="px-4 py-3 border-b border-border bg-background flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" />
          <h3 className="text-[13px] font-bold uppercase tracking-widest text-foreground">Model Reliability Rankings</h3>
        </div>
        
        <div className="overflow-x-auto hide-scrollbar">
          <table className="w-full text-left text-[11px] font-mono whitespace-nowrap">
            <thead className="bg-background">
              <tr className="text-muted-foreground border-b border-border/50">
                <th className="py-3 px-4 font-normal uppercase">Rank</th>
                <th className="py-3 px-4 font-normal uppercase">Intelligence Engine</th>
                <th className="py-3 px-4 font-normal uppercase text-right">Last 10</th>
                <th className="py-3 px-4 font-normal uppercase text-right">Last 25</th>
                <th className="py-3 px-4 font-normal uppercase text-right">Last 50</th>
                <th className="py-3 px-4 font-normal uppercase text-right">Last 100</th>
                <th className="py-3 px-4 font-normal uppercase text-right">Avg Ret</th>
                <th className="py-3 px-4 font-normal uppercase text-center">Trend</th>
                <th className="py-3 px-4 font-normal uppercase text-right">Reliability</th>
              </tr>
            </thead>
            <tbody>
              {data.map((model: ModelData, idx: number) => (
                <tr key={model.id} className="border-b border-border/20 hover:bg-muted/30 transition-colors last:border-0">
                  <td className="py-3 px-4 text-muted-foreground font-bold">#{idx + 1}</td>
                  <td className="py-3 px-4 text-foreground font-bold tracking-tight uppercase">{model.name}</td>
                  <td className={cn("py-3 px-4 text-right font-bold", model.acc_10 >= 55 ? 'text-positive' : 'text-negative')}>{model.acc_10}%</td>
                  <td className="py-3 px-4 text-right text-muted-foreground">{model.acc_25}%</td>
                  <td className="py-3 px-4 text-right text-muted-foreground">{model.acc_50}%</td>
                  <td className="py-3 px-4 text-right text-muted-foreground">{model.acc_100}%</td>
                  <td className={cn("py-3 px-4 text-right", model.avg_return > 0 ? "text-positive" : "text-negative")}>
                    {model.avg_return > 0 ? '+' : ''}{model.avg_return.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-center">
                    <div className="flex items-center justify-center gap-1">
                       <TrendIcon trend={model.trend} />
                       <span className="text-[10px] text-muted-foreground uppercase">{model.trend}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right">
                     <span className={cn(
                       "px-2 py-1 rounded text-[11px] font-black tracking-widest",
                       model.reliability_score >= 80 ? "bg-positive/10 text-positive border border-positive/20" : 
                       model.reliability_score >= 65 ? "bg-warning/10 text-warning border border-warning/20" : "bg-negative/10 text-negative border border-negative/20"
                     )}>
                       {model.reliability_score}/100
                     </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
