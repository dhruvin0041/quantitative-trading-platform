"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { PriceChart } from '@/components/dashboard/PriceChart';
import { SignalIntelligence } from '@/components/dashboard/SignalIntelligence';
import { PortfolioAnalytics } from '@/components/dashboard/PortfolioAnalytics';
import { RiskDashboard } from '@/components/dashboard/RiskDashboard';
import { PerformanceValidation } from '@/components/dashboard/PerformanceValidation';
import { ChartData, UniverseStock } from '@/types';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Activity, Cpu, Search, Menu, ChevronRight } from 'lucide-react';
import { CommandMenu } from '@/components/CommandMenu';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function HydraTerminal() {
  const [ticker, setTicker] = useState<string>("AAPL"); 
  const [universe, setUniverse] = useState<UniverseStock[]>([]);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  
  // 1. Load Universe
  useEffect(() => {
    fetch("http://localhost:8000/universe", {
      headers: { "X-API-Key": "dev-secret-key-1234" }
    })
      .then(res => res.json())
      .then(data => { if (data.universe) setUniverse(data.universe); })
      .catch(() => {
        setUniverse([
          { ticker: "AAPL", name: "Apple Inc." },
          { ticker: "MSFT", name: "Microsoft Corp." },
          { ticker: "GOOGL", name: "Alphabet Inc." },
          { ticker: "NVDA", name: "NVIDIA Corp." },
          { ticker: "TSLA", name: "Tesla Inc." },
          { ticker: "AMZN", name: "Amazon.com" },
          { ticker: "META", name: "Meta Platforms" }
        ]);
      });
  }, []);

  // 2. Fetch AI Predictions
  useEffect(() => {
    let isMounted = true;
    
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`http://localhost:8000/predict?ticker=${ticker}`, {
          headers: { "X-API-Key": "dev-secret-key-1234" }
        });
        if (!res.ok) throw new Error('Network response was not ok');
        const data = await res.json();
        if (isMounted) {
          setChartData(data);
          setLoading(false);
        }
      } catch (err) {
        console.error("Inference Engine Failed", err);
        if (isMounted) {
          setError("Failed to connect to inference engine.");
          setLoading(false);
        }
      }
    };

    fetchPrediction();
    return () => { isMounted = false; };
  }, [ticker]); 

  const tickerData = useMemo(() => {
    return universe.map((stock, i) => ({
      ...stock,
      // Use stable pseudo-random values based on ticker string/index for terminal feel
      isPositive: (stock.ticker.charCodeAt(0) + i) % 2 === 0,
      change: ((stock.ticker.charCodeAt(1) % 20) / 10).toFixed(2)
    }));
  }, [universe]);

  const primaryAction = chartData?.ai_report?.Models?.Primary_Deep_Learning?.Suggested_Action || "HOLD";
  const confidence = chartData?.ai_report?.Models?.Primary_Deep_Learning?.Confidence || "0%";

  return (
    <div className="h-screen w-screen overflow-hidden bg-background text-foreground font-sans flex flex-col selection:bg-primary/20">
      
      {/* TOP NAVIGATION BAR */}
      <header className="h-14 shrink-0 glass-header flex items-center justify-between px-4 z-50">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setSidebarOpen(!isSidebarOpen)}
            className="p-1.5 hover:bg-white/5 rounded-md transition-colors"
          >
            <Menu className="w-5 h-5 opacity-70" />
          </button>
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            <span className="font-mono font-bold tracking-tight text-lg hidden sm:inline-block">HYDRA<span className="text-muted-foreground font-light">|TERMINAL</span></span>
          </div>
          
          <div 
            className="hidden md:flex ml-6 h-8 bg-secondary/50 rounded-md border border-border/50 flex items-center px-3 gap-2 w-64 cursor-text hover:bg-secondary transition-colors" 
            onClick={() => window.dispatchEvent(new CustomEvent('hydra-open-command'))}
          >
            <Search className="w-4 h-4 opacity-50 text-foreground" />
            <span className="text-xs opacity-50 font-mono text-foreground">⌘K to search assets...</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {loading ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary">
              <Activity className="w-3 h-3 animate-pulse" />
              <span className="text-[10px] font-mono uppercase tracking-widest font-bold">Processing</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-500">
              <div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.8)]" />
              <span className="text-[10px] font-mono uppercase tracking-widest font-bold">System Online</span>
            </div>
          )}
          <CommandMenu />
          <ThemeToggle />
        </div>
      </header>

      {/* LIVE TICKER TAPE */}
      <div className="h-8 shrink-0 border-b border-border bg-card/40 flex items-center overflow-hidden relative">
        <div className="flex gap-8 whitespace-nowrap animate-ticker text-[11px] font-mono font-medium opacity-80">
          {tickerData.map((stock, i) => (
            <span key={i} className="flex items-center gap-2">
              <span className="text-foreground font-bold">{stock.ticker}</span>
              <span className={stock.isPositive ? "text-[var(--signal-buy)]" : "text-[var(--signal-sell)]"}>
                {stock.isPositive ? "+" : "-"}{stock.change}%
              </span>
              <span className="text-muted-foreground mx-4">|</span>
            </span>
          ))}
          {/* Duplicate for infinite scroll illusion */}
          {tickerData.map((stock, i) => (
            <span key={`dup-${i}`} className="flex items-center gap-2">
              <span className="text-foreground font-bold">{stock.ticker}</span>
              <span className={stock.isPositive ? "text-[var(--signal-buy)]" : "text-[var(--signal-sell)]"}>
                {stock.isPositive ? "+" : "-"}{stock.change}%
              </span>
              <span className="text-muted-foreground mx-4">|</span>
            </span>
          ))}
        </div>
        <div className="absolute top-0 left-0 h-full w-24 bg-gradient-to-r from-background to-transparent z-10 pointer-events-none" />
        <div className="absolute top-0 right-0 h-full w-24 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none" />
      </div>

      {/* MAIN WORKSPACE */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* LEFT SIDEBAR - WATCHLIST */}
        <motion.aside 
          initial={false}
          animate={{ width: isSidebarOpen ? 280 : 0, opacity: isSidebarOpen ? 1 : 0 }}
          className="shrink-0 border-r border-white/5 bg-card/30 overflow-y-auto hide-scrollbar flex flex-col"
        >
          <div className="p-4 w-[280px]">
            <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-4">Coverage Universe</h3>
            <div className="flex flex-col gap-1">
              {universe.map(stock => (
                <button
                  key={stock.ticker}
                  onClick={() => setTicker(stock.ticker)}
                  className={cn(
                    "flex items-center justify-between px-3 py-2 rounded-md text-sm transition-all border border-transparent",
                    ticker === stock.ticker 
                      ? "bg-primary/10 border-primary/30 text-primary font-bold shadow-[inset_2px_0_0_var(--primary)]" 
                      : "hover:bg-white/5 text-muted-foreground hover:text-foreground"
                  )}
                >
                  <div className="flex flex-col items-start text-left">
                    <span className="font-mono">{stock.ticker}</span>
                    <span className="text-[10px] opacity-70 truncate w-32">{stock.name}</span>
                  </div>
                  {ticker === stock.ticker && <ChevronRight className="w-4 h-4 opacity-50" />}
                </button>
              ))}
            </div>

            <div className="mt-8">
               <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-4">Macro Environment</h3>
               <div className="p-3 rounded-lg border border-border/50 bg-black/20 flex flex-col gap-2">
                 <div className="flex justify-between items-center">
                   <span className="text-xs text-muted-foreground">Regime</span>
                   <span className="text-xs font-mono font-bold text-[var(--signal-buy)]">RISK-ON</span>
                 </div>
                 <div className="flex justify-between items-center">
                   <span className="text-xs text-muted-foreground">VIX</span>
                   <span className="text-xs font-mono font-bold">14.22</span>
                 </div>
               </div>
            </div>
          </div>
        </motion.aside>

        {/* CENTER WORKSPACE - CHART & SIGNALS */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden bg-[#020202]">
          
          {/* Header Status Strip */}
          <div className="h-16 border-b border-white/5 shrink-0 flex items-center justify-between px-6">
            <div className="flex items-center gap-4">
              <h2 className="text-3xl font-bold font-sans tracking-tight">{ticker}</h2>
              {chartData && !loading && (
                <div className="flex items-center gap-2">
                  <span className={cn(
                    "px-2 py-0.5 text-xs font-bold rounded-sm uppercase tracking-wider",
                    primaryAction === 'BUY' ? "bg-[var(--signal-buy)]/20 text-[var(--signal-buy)]" :
                    primaryAction === 'SELL' ? "bg-[var(--signal-sell)]/20 text-[var(--signal-sell)]" :
                    "bg-[var(--signal-hold)]/20 text-[var(--signal-hold)]"
                  )}>
                    {primaryAction}
                  </span>
                  <span className="font-mono text-sm opacity-70">{confidence} CONF</span>
                </div>
              )}
            </div>
            <div className="text-right">
               <div className="font-mono text-2xl font-bold">${chartData?.price?.toFixed(2) || '0.00'}</div>
            </div>
          </div>

          {error && (
            <div className="m-4 bg-destructive/10 border border-destructive/30 text-destructive p-3 rounded-md text-xs font-mono">
              {error}
            </div>
          )}

          {/* Chart Section */}
          <div className="flex-1 min-h-[400px] relative p-4 pb-0">
            <div className="absolute inset-4 rounded-xl border border-white/5 overflow-hidden glass-surface shadow-2xl">
               <PriceChart data={chartData} loading={loading} />
            </div>
          </div>

          {/* Signal Intelligence Section */}
          <div className="h-[280px] shrink-0 p-4">
             <SignalIntelligence data={chartData} />
          </div>

        </main>

        {/* RIGHT SIDEBAR - ANALYTICS */}
        <aside className="w-[360px] shrink-0 border-l border-white/5 bg-card/30 overflow-y-auto hide-scrollbar flex flex-col p-4 gap-4">
          <PortfolioAnalytics data={chartData} />
          <RiskDashboard />
          <PerformanceValidation />
        </aside>

      </div>
    </div>
  );
}
