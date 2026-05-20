"use client";

import React, { useState, useEffect } from 'react';
import { HeroHeader } from '@/components/dashboard/HeroHeader';
import { PriceChart } from '@/components/dashboard/PriceChart';
import { SignalIntelligence } from '@/components/dashboard/SignalIntelligence';
import { PortfolioAnalytics } from '@/components/dashboard/PortfolioAnalytics';
import { RiskDashboard } from '@/components/dashboard/RiskDashboard';
import { PerformanceValidation } from '@/components/dashboard/PerformanceValidation';
import { OnboardingTour } from '@/components/OnboardingTour';
import { ChartData, UniverseStock } from '@/types';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

type LayoutMode = 'standard' | 'analytics' | 'compact';

export default function HydraDashboard() {
  const [ticker, setTicker] = useState<string>("AAPL"); 
  const [universe, setUniverse] = useState<UniverseStock[]>([]);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('standard');
  
  // 1. Load Universe
  useEffect(() => {
    fetch("http://localhost:8000/universe", {
      headers: { "X-API-Key": "dev-secret-key-1234" }
    })
      .then(res => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
      })
      .then(data => {
        if (data.universe) setUniverse(data.universe);
      })
      .catch(err => {
        console.error("Failed to load universe", err);
        // Fallback mock universe if backend is down
        setUniverse([
          { ticker: "AAPL", name: "Apple Inc." },
          { ticker: "MSFT", name: "Microsoft Corp." },
          { ticker: "GOOGL", name: "Alphabet Inc." },
          { ticker: "NVDA", name: "NVIDIA Corp." },
          { ticker: "TSLA", name: "Tesla Inc." }
        ]);
      });

    // Load layout preference
    const savedLayout = localStorage.getItem('hydra-layout') as LayoutMode;
    if (savedLayout) setLayoutMode(savedLayout);
  }, []);

  // 2. Fetch AI Predictions
  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`http://localhost:8000/predict?ticker=${ticker}`, {
      headers: { "X-API-Key": "dev-secret-key-1234" }
    })
      .then(res => {
        if (!res.ok) throw new Error('Network response was not ok');
        return res.json();
      })
      .then(data => {
        setChartData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Inference Engine Failed", err);
        setError("Failed to connect to inference engine. The backend might be offline.");
        setLoading(false);
      });
  }, [ticker]); 

  const updateLayout = (mode: LayoutMode) => {
    setLayoutMode(mode);
    localStorage.setItem('hydra-layout', mode);
  };

  return (
    <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/20 overflow-x-hidden">
      <OnboardingTour />
      
      <div id="dashboard-container" className="max-w-[1800px] mx-auto px-4 md:px-8 lg:px-12 py-8 flex flex-col gap-6 md:gap-8">
        
        {/* HERO HEADER */}
        <HeroHeader 
          ticker={ticker} 
          universe={universe} 
          chartData={chartData} 
          loading={loading}
          onTickerChange={setTicker}
        />

        {error && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-destructive/10 border border-destructive/20 text-destructive px-4 py-3 rounded-md text-sm font-medium flex items-center shadow-sm"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            {error}
          </motion.div>
        )}

        {/* MAIN DASHBOARD GRID */}
        <div className={cn(
          "grid grid-cols-1 gap-6 md:gap-8 transition-all duration-500",
          layoutMode === 'standard' && "lg:grid-cols-12",
          layoutMode === 'analytics' && "lg:grid-cols-3",
          layoutMode === 'compact' && "lg:grid-cols-4"
        )}>
          
          {/* LEFT COLUMN - Chart & Core Intelligence */}
          <motion.div 
            className={cn(
              "flex flex-col gap-6 md:gap-8",
              layoutMode === 'standard' && "lg:col-span-8",
              layoutMode === 'analytics' && "lg:col-span-2",
              layoutMode === 'compact' && "lg:col-span-3"
            )}
            layout
          >
            <PriceChart data={chartData} loading={loading} />
            <SignalIntelligence data={chartData} />
            {layoutMode !== 'compact' && <RiskDashboard />}
          </motion.div>

          {/* RIGHT COLUMN - Analytics */}
          <motion.div 
            className={cn(
              "flex flex-col gap-6 md:gap-8",
              layoutMode === 'standard' && "lg:col-span-4",
              layoutMode === 'analytics' && "lg:col-span-1",
              layoutMode === 'compact' && "lg:col-span-1"
            )}
            layout
          >
            <PortfolioAnalytics data={chartData} />
            <PerformanceValidation />
            
            {layoutMode === 'compact' && <RiskDashboard />}

            {/* System Status Card */}
            <Card className="p-6 rounded-xl bg-secondary/30 border border-border/50 shadow-none">
              <h4 className="text-[10px] uppercase tracking-widest font-black text-muted-foreground mb-4">Core Telemetry</h4>
              <ul className="space-y-3 text-sm">
                <li className="flex justify-between items-center">
                  <span className="text-muted-foreground font-medium">Market Regime</span>
                  <Badge variant="outline" className="font-mono text-[10px] bg-background border-border/50">Risk-On</Badge>
                </li>
                <li className="flex justify-between items-center">
                  <span className="text-muted-foreground font-medium">Model Drift</span>
                  <span className="font-mono text-xs font-bold text-[var(--signal-buy)]">0.002% (Stable)</span>
                </li>
                <li className="flex justify-between items-center">
                  <span className="text-muted-foreground font-medium">Inference Latency</span>
                  <span className="font-mono text-xs">24ms</span>
                </li>
              </ul>
            </Card>
          </motion.div>
        </div>
        
        {/* LAYOUT CONTROLS (Floating Bottom) */}
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 bg-background/80 backdrop-blur-md p-1 rounded-full border border-border shadow-xl overflow-hidden">
          {(['standard', 'analytics', 'compact'] as LayoutMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => updateLayout(mode)}
              className={cn(
                "px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest transition-all",
                layoutMode === mode 
                  ? "bg-primary text-primary-foreground shadow-sm" 
                  : "text-muted-foreground hover:bg-secondary"
              )}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

