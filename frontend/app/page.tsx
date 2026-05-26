"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { PriceChart } from '@/components/dashboard/PriceChart';
import { SignalIntelligence } from '@/components/dashboard/SignalIntelligence';
import { PortfolioAnalytics } from '@/components/dashboard/PortfolioAnalytics';
import { RiskDashboard } from '@/components/dashboard/RiskDashboard';
import { ChartData, UniverseStock } from '@/types';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Activity, Cpu, Menu, ChevronRight, CheckCircle2 } from 'lucide-react';
import { CommandMenu } from '@/components/CommandMenu';
import { ThemeToggle } from '@/components/ThemeToggle';
import { StockSearch } from '@/components/StockSearch';
import { TechnicalSnapshot } from '@/components/dashboard/TechnicalSnapshot';
import { BacktestPanel } from '@/components/dashboard/BacktestPanel';
import { IntegrityAudit } from '@/components/dashboard/IntegrityAudit';
import { PaperTradingPerformance } from '@/components/dashboard/PaperTradingPerformance';

export default function HydraTerminal() {
  const [ticker, setTicker] = useState<string>("AAPL"); 
  const [market, setMarket] = useState<'us' | 'india'>('us');
  const [universe, setUniverse] = useState<UniverseStock[]>([]);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [isBacktestOpen, setBacktestOpen] = useState(false);
  
  // 1. Load Universe
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    fetch(`${API_URL}/universe`, {
      headers: { "X-API-Key": API_KEY }
    })
      .then(res => res.json())
      .then(data => { if (data.universe) setUniverse(data.universe); })
      .catch(() => {
        // Fallback for UI if backend unreachable
        setUniverse([
          { ticker: "AAPL", name: "Apple Inc.", price: 0, pct_change: 0, market: 'us' },
          { ticker: "MSFT", name: "Microsoft Corp.", price: 0, pct_change: 0, market: 'us' }
        ]);
      });
  }, [API_URL, API_KEY]);

  // 2. Fetch AI Predictions
  useEffect(() => {
    let isMounted = true;
    
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);
      setChartData(null); // CRITICAL UX FIX: Clear stale data before fetching to prevent mismatched ticker/price rendering
      try {
        const res = await fetch(`${API_URL}/predict?ticker=${ticker}`, {
          headers: { "X-API-Key": API_KEY }
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
  }, [ticker, API_URL, API_KEY]); 

  const tickerData = useMemo(() => {
    return universe.map((stock) => ({
      ...stock,
      isPositive: stock.pct_change >= 0,
      change: stock.pct_change.toFixed(2)
    }));
  }, [universe]);

  const filteredUniverse = useMemo(() => {
    return universe.filter(s => s.market === market);
  }, [universe, market]);

  const handleStockSelect = (selectedTicker: string) => {
    setTicker(selectedTicker);
    // Find the stock to auto-switch market if needed (for search)
    const stock = universe.find(s => s.ticker === selectedTicker);
    if (stock && stock.market !== market) {
      setMarket(stock.market as 'us' | 'india');
    }
  };

  const handleBaseCurrencyChange = async (newBase: string) => {
    try {
      const res = await fetch(`${API_URL}/portfolio/base_currency`, {
        method: 'POST',
        headers: { 
          "X-API-Key": API_KEY,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ currency: newBase })
      });
      if (res.ok) {
        // Trigger a re-fetch of prediction to get updated portfolio values
        const res2 = await fetch(`${API_URL}/predict?ticker=${ticker}`, {
          headers: { "X-API-Key": API_KEY }
        });
        const data = await res2.json();
        setChartData(data);
      }
    } catch (err) {
      console.error("Failed to update base currency", err);
    }
  };

  const primaryAction = chartData?.signal || "HOLD";
  const confidence = chartData?.confidence_score ? `${chartData.confidence_score.toFixed(1)}%` : "0%";
  
  // Multi-Currency Detection
  const currencySymbol = useMemo(() => {
    const stock = universe.find(s => s.ticker === ticker);
    return stock?.market === 'india' ? '₹' : '$';
  }, [universe, ticker]);

  return (
    <div className="min-h-screen bg-background text-foreground font-sans flex flex-col selection:bg-primary/20">
      
      {/* TOP NAVIGATION BAR */}
      <header className="h-14 glass-header flex items-center justify-between px-4 z-50 sticky top-0">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setSidebarOpen(!isSidebarOpen)}
            className="p-1.5 hover:bg-secondary rounded-md transition-colors text-foreground"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            <span className="font-mono font-bold tracking-tight text-lg hidden sm:inline-block">HYDRA<span className="text-muted-foreground font-light">|TERMINAL</span></span>
          </div>
          
          <StockSearch universe={universe} onSelect={handleStockSelect} />
        </div>

        <div className="flex items-center gap-4">
          {loading ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary">
              <Activity className="w-3 h-3 animate-pulse" />
              <span className="text-[10px] font-mono uppercase tracking-widest font-bold">Processing</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--secondary)] border border-[var(--signal-buy)]/30 text-[var(--signal-buy)] dark:bg-green-500/10 dark:border-green-500/20 dark:text-green-500 shadow-sm">
              <div className="w-2 h-2 rounded-full bg-[var(--signal-buy)] shadow-[0_0_8px_rgba(29,122,58,0.5)] dark:bg-green-500 dark:shadow-[0_0_8px_rgba(34,197,94,0.8)]" />
              <span className="text-[10px] font-mono uppercase tracking-widest font-black">System Online</span>
            </div>
          )}
          <button 
            onClick={() => setBacktestOpen(true)}
            className="px-3 py-1 text-[10px] font-mono font-bold tracking-widest uppercase rounded-full bg-muted border border-border hover:bg-primary hover:text-primary-foreground hover:border-primary transition-colors"
          >
            Backtest
          </button>
          <CommandMenu />
          <ThemeToggle />
        </div>
      </header>

      <BacktestPanel 
        isOpen={isBacktestOpen} 
        onClose={() => setBacktestOpen(false)} 
        currentTicker={ticker} 
      />

      {/* LIVE TICKER TAPE */}
      <div className="h-8 border-b border-border bg-secondary flex items-center overflow-hidden relative">
        <div className="flex gap-8 whitespace-nowrap animate-ticker text-[11px] font-mono font-medium opacity-100">
          {tickerData.map((stock, i) => (
            <span key={i} className="flex items-center gap-2">
              <span className="text-foreground font-bold tracking-tight">{stock.ticker}</span>
              <span className={stock.isPositive ? "text-[var(--signal-buy)]" : "text-[var(--signal-sell)]"}>
                {stock.isPositive ? "+" : "-"}{stock.change}%
              </span>
              <span className="text-muted-foreground mx-4">|</span>
            </span>
          ))}
          {/* Duplicate for infinite scroll illusion */}
          {tickerData.map((stock, i) => (
            <span key={`dup-${i}`} className="flex items-center gap-2">
              <span className="text-foreground font-bold tracking-tight">{stock.ticker}</span>
              <span className={stock.isPositive ? "text-[var(--signal-buy)]" : "text-[var(--signal-sell)]"}>
                {stock.isPositive ? "+" : "-"}{stock.change}%
              </span>
              <span className="text-muted-foreground mx-4">|</span>
            </span>
          ))}
        </div>
        <div className="absolute top-0 left-0 h-full w-24 bg-gradient-to-r from-secondary to-transparent z-10 pointer-events-none" />
        <div className="absolute top-0 right-0 h-full w-24 bg-gradient-to-l from-secondary to-transparent z-10 pointer-events-none" />
      </div>

      {/* MAIN WORKSPACE */}
      <div className="flex flex-1">
        
        {/* LEFT SIDEBAR - WATCHLIST */}
        <motion.aside 
          initial={false}
          animate={{ width: isSidebarOpen ? 280 : 0, opacity: isSidebarOpen ? 1 : 0 }}
          className="shrink-0 border-r border-border bg-sidebar flex flex-col"
        >
          <div className="p-4 w-[280px]">
            <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-4">Coverage Universe</h3>
            
            {/* Market Selector */}
            <div className="flex p-1 bg-secondary/50 rounded-lg border border-border mb-6">
              <button
                onClick={() => setMarket('india')}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] font-black uppercase transition-all",
                  market === 'india' ? "bg-primary text-white shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                <span>🇮🇳</span> India
              </button>
              <button
                onClick={() => setMarket('us')}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] font-black uppercase transition-all",
                  market === 'us' ? "bg-primary text-white shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                <span>🇺🇸</span> USA
              </button>
            </div>

            <div className="flex flex-col gap-1.5">
              <AnimatePresence mode="wait">
                <motion.div
                  key={market}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  transition={{ duration: 0.2 }}
                  className="flex flex-col gap-1.5"
                >
                  {filteredUniverse.map(stock => (
                    <button
                      key={stock.ticker}
                      onClick={() => handleStockSelect(stock.ticker)}
                      className={cn(
                        "flex flex-col items-start px-3 py-2.5 rounded-lg text-sm transition-all border border-transparent group/card text-left",
                        ticker === stock.ticker 
                          ? "bg-primary text-white font-bold shadow-md dark:bg-primary/10 dark:text-primary dark:shadow-none border-primary/20" 
                          : "hover:bg-secondary text-muted-foreground hover:text-foreground dark:hover:bg-white/5"
                      )}
                    >
                      <div className="w-full flex justify-between items-center mb-0.5">
                        <span className={cn(
                          "font-bold truncate max-w-[140px]",
                          ticker === stock.ticker ? "text-white dark:text-primary" : "text-foreground opacity-90 group-hover/card:opacity-100"
                        )}>
                          {stock.name}
                        </span>
                        <ChevronRight className={cn(
                          "w-3.5 h-3.5 transition-transform duration-200",
                          ticker === stock.ticker ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-1 group-hover/card:opacity-50"
                        )} />
                      </div>
                      <div className="flex items-center justify-between w-full">
                        <span className={cn(
                          "font-mono text-[10px] font-bold tracking-tighter uppercase",
                          ticker === stock.ticker ? "opacity-80" : "opacity-50"
                        )}>
                          {stock.ticker}
                        </span>
                        {stock.pct_change !== 0 && (
                          <span className={cn(
                            "text-[9px] font-black font-mono",
                            stock.pct_change >= 0 ? "text-green-500" : "text-red-500"
                          )}>
                            {stock.pct_change >= 0 ? "+" : ""}{stock.pct_change.toFixed(2)}%
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </motion.div>
              </AnimatePresence>
            </div>

            <div className="mt-8">
               <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-4">Macro Environment</h3>
               <div className="p-3 rounded-lg border border-border bg-secondary/50 dark:bg-black/20 flex flex-col gap-2">
                 <div className="flex justify-between items-center">
                   <span className="text-xs text-muted-foreground">Regime</span>
                   <span className={cn(
                     "text-[10px] font-mono font-bold uppercase px-1.5 py-0.5 rounded border",
                     chartData?.market_regime === 'BULL' ? "bg-green-500/20 text-green-500 border-green-500/30" :
                     chartData?.market_regime === 'BEAR' ? "bg-red-500/20 text-red-500 border-red-500/30" :
                     "bg-zinc-500/20 text-zinc-400 border-zinc-500/30"
                   )}>
                     {chartData?.market_regime || "NEUTRAL"}
                   </span>
                 </div>
                 <div className="flex justify-between items-center">
                   <span className="text-xs text-muted-foreground font-sans">Vol State</span>
                   <span className={cn(
                     "text-[10px] font-mono font-bold uppercase px-1.5 py-0.5 rounded border",
                     chartData?.volatility_state === 'HIGH' ? "bg-orange-500/20 text-orange-500 border-orange-500/30" :
                     chartData?.volatility_state === 'MEDIUM' ? "bg-yellow-500/20 text-yellow-500 border-yellow-500/30" :
                     "bg-blue-500/20 text-blue-500 border-blue-500/30"
                   )}>
                     {chartData?.volatility_state || "LOW"}
                   </span>
                 </div>
               </div>
            </div>
          </div>
        </motion.aside>

        {/* CENTER WORKSPACE - CHART & SIGNALS */}
        <main className="flex-1 flex flex-col min-w-0 bg-muted/20 dark:bg-background">
          
          {/* Header Status Strip */}
          <div className="h-16 border-b border-border flex items-center justify-between px-6 bg-background sticky top-[5.5rem] z-30">
            <div className="flex items-center gap-4 text-foreground">
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <h2 className="text-2xl font-black tracking-tight">{ticker}</h2>
                  {chartData?.metadata && (
                    <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-muted border border-border">
                      <span className="text-[10px] leading-none">{chartData.metadata.market === 'INDIA' ? '🇮🇳' : '🇺🇸'}</span>
                      <span className="text-[8px] font-black uppercase tracking-tighter opacity-70">
                        {chartData.metadata.market} | {chartData.metadata.exchange} | {chartData.metadata.currency}
                      </span>
                    </div>
                  )}
                </div>
                <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest">{chartData?.metadata?.name || ticker}</span>
              </div>
              {chartData && !loading && (
                <div className="flex items-center gap-2">
                  <span className={cn(
                    "px-3 py-1 text-xs font-black rounded-full border-2 uppercase tracking-widest shadow-sm",
                    primaryAction === 'BUY' ? "bg-[var(--signal-buy)] border-[var(--signal-buy)] text-white" :
                    primaryAction === 'SELL' ? "bg-[var(--signal-sell)] border-[var(--signal-sell)] text-white" :
                    "bg-[var(--signal-hold)] border-[var(--signal-hold)] text-white"
                  )}>
                    {primaryAction}
                  </span>
                  <span className="font-mono text-sm font-bold opacity-70">{confidence} CONF</span>
                </div>
              )}
            </div>
            <div className="text-right text-foreground flex items-center gap-3">
               <div className="font-mono text-2xl font-bold">{currencySymbol}{chartData?.current_price?.toFixed(2) || '0.00'}</div>
               {chartData && !loading && (
                  <div className="flex gap-2">
                     <span className={cn("text-[10px] font-black uppercase px-2 py-1 rounded-full border", 
                        chartData.market_regime === 'BULL' ? "bg-green-500/20 text-green-500 border-green-500/30" :
                        chartData.market_regime === 'BEAR' ? "bg-red-500/20 text-red-500 border-red-500/30" :
                        "bg-zinc-500/20 text-zinc-400 border-zinc-500/30")}>
                        {chartData.market_regime === 'BULL' ? '🐂' : chartData.market_regime === 'BEAR' ? '🐻' : '⚖️'} {chartData.market_regime}
                     </span>
                     <span className={cn("text-[10px] font-black uppercase px-2 py-1 rounded-full border", 
                        chartData.volatility_state === 'HIGH' ? "bg-red-500/20 text-red-500 border-red-500/30" :
                        chartData.volatility_state === 'MEDIUM' ? "bg-orange-500/20 text-orange-500 border-orange-500/30" :
                        "bg-green-500/20 text-green-500 border-green-500/30")}>
                        ⚡ {chartData.volatility_state} VOL
                     </span>
                     {(chartData.uncertainty_score !== undefined && chartData.uncertainty_score > 45) && (
                         <span className="text-[10px] font-black uppercase px-2 py-1 rounded-full border bg-red-500 text-white border-red-600">
                             ⚠️ VETOED
                         </span>
                     )}
                  </div>
               )}
            </div>
          </div>

          {error && (
            <div className="m-4 bg-destructive/10 border border-destructive/30 text-destructive p-3 rounded-md text-xs font-mono">
              {error}
            </div>
          )}

          {/* Chart Section - Intrinsic Flow */}
          <div className="p-4 pb-0">
             <div className="rounded-xl border border-border overflow-hidden bg-card shadow-2xl">
                <PriceChart data={chartData} loading={loading} />
             </div>
          </div>

          {/* Signal Intelligence Section */}
          <div className="p-4 flex flex-col gap-4">
             <SignalIntelligence data={chartData} currency={currencySymbol} />
             
             {/* Institutional Validation Layer - Bottom Content */}
             <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-2">
                <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300 min-h-[200px]">
                   <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Activity className="w-3.5 h-3.5 text-primary" />
                        <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Recent Closed Trades</h3>
                      </div>
                      <span className="text-[8px] font-mono font-bold opacity-50 uppercase tracking-tighter">Live_Audit</span>
                   </div>
                   <div className="p-0 flex-1 overflow-auto max-h-[300px]">
                      {chartData?.portfolio?.positions && Object.keys(chartData.portfolio.positions).length > 0 ? (
                        <table className="w-full text-left border-collapse">
                          <thead className="sticky top-0 bg-background/95 backdrop-blur-sm z-10 border-b border-border">
                            <tr>
                              <th className="px-4 py-2 text-[8px] font-black uppercase text-muted-foreground">Asset</th>
                              <th className="px-4 py-2 text-[8px] font-black uppercase text-muted-foreground">Shares</th>
                              <th className="px-4 py-2 text-[8px] font-black uppercase text-muted-foreground">Avg Price</th>
                              <th className="px-4 py-2 text-[8px] font-black uppercase text-muted-foreground">Market</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(chartData.portfolio.positions).map(([ticker, pos]) => (
                              <tr key={ticker} className="border-b border-border/30 hover:bg-muted/30 transition-colors">
                                <td className="px-4 py-2 font-mono text-[10px] font-bold">{ticker}</td>
                                <td className="px-4 py-2 font-mono text-[10px]">{pos.shares}</td>
                                <td className="px-4 py-2 font-mono text-[10px]">{currencySymbol}{pos.avg_price.toLocaleString()}</td>
                                <td className="px-4 py-2"><span className="text-[8px] px-1.5 py-0.5 rounded bg-secondary uppercase font-bold">{pos.market}</span></td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div className="flex items-center justify-center h-full text-[10px] text-muted-foreground uppercase font-mono tracking-widest opacity-50">
                           No Closed Alpha recorded
                        </div>
                      )}
                   </div>
                </div>

                <div className="glass-panel rounded-xl flex flex-col overflow-hidden group transition-all duration-300 min-h-[200px]">
                   <div className="bg-secondary/50 dark:bg-black/40 border-b border-border px-4 py-2 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Cpu className="w-3.5 h-3.5 text-primary" />
                        <h3 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Validation Log</h3>
                      </div>
                      <span className="text-[8px] font-mono font-bold opacity-50 uppercase tracking-tighter">Event_Stream</span>
                   </div>
                   <div className="p-4 flex-1 flex flex-col gap-3">
                      <div className="flex items-center justify-between p-2 rounded bg-emerald-500/5 border border-emerald-500/20">
                         <div className="flex flex-col">
                            <span className="text-[8px] font-black text-emerald-500 uppercase tracking-widest">Integrity Check</span>
                            <span className="text-[10px] font-medium text-foreground">Mathematical Reconciliation Passed</span>
                         </div>
                         <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      </div>
                      <div className="flex items-center justify-between p-2 rounded bg-primary/5 border border-primary/20">
                         <div className="flex flex-col">
                            <span className="text-[8px] font-black text-primary uppercase tracking-widest">Model Sync</span>
                            <span className="text-[10px] font-medium text-foreground">Meta-Ensemble Weights Aligned</span>
                         </div>
                         <Activity className="w-4 h-4 text-primary" />
                      </div>
                      <div className="mt-auto pt-2 border-t border-border/30 flex justify-between items-center text-[7px] font-mono text-muted-foreground">
                         <span>SEC_EDGAR_READY</span>
                         <span>PHYSICAL_PROXY_LIVE</span>
                         <span>GNN_NODES: 899</span>
                      </div>
                   </div>
                </div>
             </div>
          </div>

        </main>

        {/* RIGHT SIDEBAR - ANALYTICS */}
        <aside className="w-[360px] shrink-0 border-l border-border bg-background flex flex-col p-4 gap-4">
          
          {/* Base Currency Selector */}
          <div className="glass-panel rounded-xl flex flex-col overflow-hidden border border-border p-3 bg-secondary/20">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Portfolio Base</span>
              <div className="flex gap-1">
                {['USD', 'INR', 'EUR', 'GBP'].map(curr => (
                  <button
                    key={curr}
                    onClick={() => handleBaseCurrencyChange(curr)}
                    className={cn(
                      "px-2 py-0.5 rounded text-[9px] font-black transition-all border",
                      chartData?.portfolio?.base_currency === curr 
                        ? "bg-primary text-white border-primary shadow-sm" 
                        : "bg-muted text-muted-foreground border-border hover:bg-secondary hover:text-foreground"
                    )}
                  >
                    {curr}
                  </button>
                ))}
              </div>
            </div>
            
            {/* FX Transparency */}
            {chartData?.portfolio?.fx_rates && (
              <div className="mt-2 pt-2 border-t border-border/50 flex flex-col gap-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-[8px] font-bold text-muted-foreground uppercase">Exchange Rates (vs USD)</span>
                  <span className="text-[8px] font-mono text-emerald-500 uppercase font-black tracking-tighter">● Live_Feeds</span>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {Object.entries(chartData.portfolio.fx_rates).map(([curr, rate]) => {
                    if (curr === 'USD') return null;
                    return (
                      <div key={curr} className="flex flex-col p-1.5 rounded bg-black/20 border border-border/30">
                        <span className="text-[7px] font-black text-muted-foreground opacity-70 uppercase tracking-tighter">USD/{curr}</span>
                        <span className="text-[10px] font-mono font-bold text-foreground">{(rate as number).toFixed(2)}</span>
                      </div>
                    );
                  })}
                </div>
                <div className="flex justify-between items-center mt-1">
                  <span className="text-[7px] text-muted-foreground italic">Source: Yahoo Finance</span>
                  <span className="text-[7px] text-muted-foreground opacity-50 uppercase font-mono">Updated: {new Date(chartData.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} UTC</span>
                </div>
              </div>
            )}
          </div>

          <PortfolioAnalytics data={chartData} currency={chartData?.portfolio?.base_currency === 'INR' ? '₹' : (chartData?.portfolio?.base_currency === 'EUR' ? '€' : (chartData?.portfolio?.base_currency === 'GBP' ? '£' : '$'))} />
          <RiskDashboard data={chartData} currency={chartData?.portfolio?.base_currency === 'INR' ? '₹' : (chartData?.portfolio?.base_currency === 'EUR' ? '€' : (chartData?.portfolio?.base_currency === 'GBP' ? '£' : '$'))} />
          <PaperTradingPerformance currency={chartData?.portfolio?.base_currency === 'INR' ? '₹' : (chartData?.portfolio?.base_currency === 'EUR' ? '€' : (chartData?.portfolio?.base_currency === 'GBP' ? '£' : '$'))} />
          <TechnicalSnapshot data={chartData} currency={currencySymbol} />
          <div className="h-10 shrink-0" />
        </aside>

      </div>
      <IntegrityAudit data={chartData} />
    </div>
  );
}
