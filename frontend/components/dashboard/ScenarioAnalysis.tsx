"use client";

import React, { useState } from 'react';
import { ActivitySquare, Zap, TrendingDown, Target, Skull, CloudLightning } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ChartData } from '@/types';

interface ScenarioAnalysisProps {
  data?: ChartData | null;
}

export function ScenarioAnalysis({ data }: ScenarioAnalysisProps) {
  const [activeScenario, setActiveScenario] = useState('base');

  const baselineProb = data?.confidence_score || 72.5;
  const currentPrice = data?.current_price || 150.00;

  const scenarios = [
    {
      id: 'base',
      name: 'Baseline',
      icon: Target,
      description: 'Current market conditions persist. Normal volatility.',
      probShift: 0,
      priceTarget: currentPrice * 1.05,
      impact: 'NEUTRAL'
    },
    {
      id: 'spy_drop',
      name: 'SPY -5% Shock',
      icon: TrendingDown,
      description: 'Broad market liquidity contraction. High correlation impact.',
      probShift: -15.2,
      priceTarget: currentPrice * 0.92,
      impact: 'NEGATIVE'
    },
    {
      id: 'vol_spike',
      name: 'VIX > 30 Spike',
      icon: CloudLightning,
      description: 'Volatility regime shift. Multi-factor deleveraging.',
      probShift: -22.5,
      priceTarget: currentPrice * 0.88,
      impact: 'NEGATIVE'
    },
    {
      id: 'sector_rotation',
      name: 'Sector Rotation (Inflow)',
      icon: Zap,
      description: 'Capital rotation into asset class. Momentum amplification.',
      probShift: +12.4,
      priceTarget: currentPrice * 1.12,
      impact: 'POSITIVE'
    },
    {
      id: 'black_swan',
      name: 'Black Swan (Tail Risk)',
      icon: Skull,
      description: 'Generative GAN stress test. Unprecedented exogenous shock.',
      probShift: -45.0,
      priceTarget: currentPrice * 0.65,
      impact: 'SEVERE'
    }
  ];

  const active = scenarios.find(s => s.id === activeScenario) || scenarios[0];
  const activeProb = Math.max(0, Math.min(100, baselineProb + active.probShift));

  return (
    <div className="flex flex-col border border-border rounded-xl bg-card overflow-hidden">
      
      <div className="px-5 py-4 border-b border-border bg-background flex items-center gap-2">
        <ActivitySquare className="w-4 h-4 text-primary" />
        <h3 className="text-[13px] font-bold uppercase tracking-widest text-foreground">Generative Scenario Analysis</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12">
        
        {/* Scenario Selector */}
        <div className="col-span-1 md:col-span-4 border-b md:border-b-0 md:border-r border-border bg-muted/10 p-3 flex flex-col gap-2">
          {scenarios.map(s => {
            const Icon = s.icon;
            return (
              <button
                key={s.id}
                onClick={() => setActiveScenario(s.id)}
                className={cn(
                  "flex items-center gap-3 p-3 rounded-lg text-left transition-all border",
                  activeScenario === s.id 
                    ? "bg-background border-primary shadow-sm" 
                    : "bg-transparent border-transparent hover:bg-muted/50 text-muted-foreground"
                )}
              >
                <div className={cn(
                  "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                  activeScenario === s.id ? "bg-primary/10" : "bg-muted"
                )}>
                  <Icon className={cn("w-4 h-4", activeScenario === s.id ? "text-primary" : "text-muted-foreground")} />
                </div>
                <div className="flex flex-col">
                  <span className={cn("text-[12px] font-bold uppercase", activeScenario === s.id ? "text-foreground" : "text-muted-foreground")}>{s.name}</span>
                  <span className="text-[10px] text-muted-foreground line-clamp-1">{s.description}</span>
                </div>
              </button>
            )
          })}
        </div>

        {/* Scenario Output */}
        <div className="col-span-1 md:col-span-8 p-6 flex flex-col justify-center">
          
          <div className="flex items-center justify-between mb-8">
            <div className="flex flex-col">
              <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Simulated Scenario</span>
              <span className="text-[20px] font-black text-foreground uppercase">{active.name}</span>
            </div>
            <span className={cn(
              "px-3 py-1 rounded text-[11px] font-black uppercase tracking-widest border",
              active.impact === 'POSITIVE' ? "bg-positive/10 text-positive border-positive/20" :
              active.impact === 'NEGATIVE' ? "bg-negative/10 text-negative border-negative/20" :
              active.impact === 'SEVERE' ? "bg-warning/10 text-warning border-warning/20" :
              "bg-muted/50 text-muted-foreground border-border"
            )}>
              Impact: {active.impact}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-6 relative">
            {/* Connector Line */}
            <div className="absolute top-1/2 left-1/4 right-1/4 h-[1px] bg-border border-dashed z-0 hidden md:block" />
            
            <div className="flex flex-col items-center text-center p-5 bg-background border border-border rounded-xl relative z-10 shadow-sm">
              <span className="text-[11px] font-bold text-muted-foreground uppercase mb-2 tracking-widest">Baseline Signal Conf</span>
              <span className="text-[32px] font-mono font-black text-foreground leading-none">{baselineProb.toFixed(1)}%</span>
            </div>
            
            <div className={cn(
              "flex flex-col items-center text-center p-5 border rounded-xl relative z-10 shadow-sm transition-colors",
              active.impact === 'POSITIVE' ? "bg-positive/5 border-positive/30" :
              active.impact === 'NEGATIVE' ? "bg-negative/5 border-negative/30" :
              active.impact === 'SEVERE' ? "bg-warning/5 border-warning/30" :
              "bg-background border-border"
            )}>
              <span className="text-[11px] font-bold text-muted-foreground uppercase mb-2 tracking-widest">Stressed Confidence</span>
              <div className="flex items-center gap-3">
                <span className={cn(
                  "text-[32px] font-mono font-black leading-none",
                  active.probShift > 0 ? "text-positive" : active.probShift < 0 ? "text-negative" : "text-foreground"
                )}>
                  {activeProb.toFixed(1)}%
                </span>
                {active.probShift !== 0 && (
                  <span className={cn(
                    "text-[12px] font-mono font-bold px-1.5 py-0.5 rounded",
                    active.probShift > 0 ? "bg-positive/20 text-positive" : "bg-negative/20 text-negative"
                  )}>
                    {active.probShift > 0 ? '+' : ''}{active.probShift.toFixed(1)}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="mt-8 p-4 rounded-lg bg-muted/30 border border-border">
            <div className="flex justify-between items-center">
              <span className="text-[12px] font-bold text-muted-foreground uppercase">Projected Price Target</span>
              <span className={cn(
                "text-[16px] font-mono font-black",
                active.priceTarget > currentPrice ? "text-positive" : "text-negative"
              )}>
                ${active.priceTarget.toFixed(2)}
              </span>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
