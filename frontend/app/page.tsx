"use client";

import React, { useState, useEffect, useMemo } from 'react';
import { PriceChart } from '@/components/dashboard/PriceChart';
import { TradeCard } from '@/components/dashboard/TradeCard';
import { SignalIntelligence } from '@/components/dashboard/SignalIntelligence';
import { PortfolioAnalytics } from '@/components/dashboard/PortfolioAnalytics';
import { RiskDashboard } from '@/components/dashboard/RiskDashboard';
import { TechnicalSnapshot } from '@/components/dashboard/TechnicalSnapshot';
import { PaperTradingPerformance } from '@/components/dashboard/PaperTradingPerformance';
import { IntegrityAudit } from '@/components/dashboard/IntegrityAudit';
import { ChartData, UniverseStock } from '@/types';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Activity, Cpu, Menu, Shield, Briefcase, ActivitySquare, Target, BarChart2, ShieldCheck, X, CheckCircle2 } from 'lucide-react';
import { CommandMenu } from '@/components/CommandMenu';
import { ThemeToggle } from '@/components/ThemeToggle';
import { StockSearch } from '@/components/StockSearch';
import { API_KEY, getBaseUrl } from '@/lib/config';

type TabType = 'CONSENSUS' | 'RISK' | 'PORTFOLIO' | 'PERFORMANCE' | 'TECHNICAL';

export default function HydraTerminal() {
  const [ticker, setTicker] = useState<string>("AAPL"); 
  const [market, setMarket] = useState<string>('us');
  const [universe, setUniverse] = useState<UniverseStock[]>([]);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isSidebarOpen, setSidebarOpen] = useState(true);
  const [isDebugModalOpen, setDebugModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>('CONSENSUS');
  const [API_URL] = useState(getBaseUrl());

  useEffect(() => {
    fetch(`${API_URL}/active_ticker`, { headers: { "X-API-Key": API_KEY } })
      .then(res => res.json())
      .then(data => {
        if (data.ticker) setTicker(data.ticker);
        if (data.market) setMarket(data.market);
      })
      .catch(err => console.error("Failed to fetch active ticker", err));

    fetch(`${API_URL}/universe`, { headers: { "X-API-Key": API_KEY } })
      .then(res => res.json())
      .then(data => { if (data.universe) setUniverse(data.universe); })
      .catch(() => {
        setUniverse([
          { ticker: "AAPL", name: "Apple Inc.", price: 0, pct_change: 0, market: 'us' },
          { ticker: "MSFT", name: "Microsoft Corp.", price: 0, pct_change: 0, market: 'us' }
        ]);
      });
  }, [API_URL]);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    
    const fetchPrediction = async () => {
      if (!ticker) return;

      setLoading(true);
      setError(null);
      setChartData(null); 

      const requestUrl = `${API_URL}/predict?ticker=${ticker}`;
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      try {
        const res = await fetch(requestUrl, {
          headers: { "X-API-Key": API_KEY },
          signal: controller.signal,
          cache: 'no-store'
        });

        clearTimeout(timeoutId);

        if (!res.ok) throw new Error(`Server responded with ${res.status}`);

        const data = await res.json();

        if (isMounted && data.ticker === ticker) {
          setChartData(data);
          setLoading(false);
        }
      } catch (err) {
        clearTimeout(timeoutId);
        if (isMounted) {
          setError((err as Error).message || "Failed to fetch AI predictions.");
          setLoading(false);
        }
      }
    };

    fetchPrediction();
    return () => { isMounted = false; controller.abort(); };
  }, [ticker, API_URL]); 

  const filteredUniverse = useMemo(() => {
    return universe.filter(s => s.market === market);
  }, [universe, market]);

  const handleStockSelect = (selectedTicker: string) => {
    setTicker(selectedTicker);
    const stock = universe.find(s => s.ticker === selectedTicker);
    if (stock && stock.market !== market) {
      setMarket(stock.market);
    }
  };

  const currencySymbol = useMemo(() => {
    const stock = universe.find(s => s.ticker === ticker);
    return stock?.market === 'india' ? '₹' : '$';
  }, [universe, ticker]);

  return (
    <div className="min-h-screen bg-background text-foreground font-sans flex flex-col selection:bg-primary/30">
      
      {/* UNIVERSAL HEADER */}
      <header className="h-14 bg-card border-b border-border flex items-center justify-between px-4 z-50 sticky top-0">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setSidebarOpen(!isSidebarOpen)}
            className="p-1.5 hover:bg-muted rounded-md transition-colors text-foreground"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            <span className="font-mono font-bold tracking-tight text-lg hidden sm:inline-block">HYDRA<span className="text-muted-foreground font-light">|V2</span></span>
          </div>
          <StockSearch universe={universe} onSelect={handleStockSelect} />
        </div>

        <div className="flex items-center gap-4">
          {loading ? (
            <div className="flex items-center gap-2 px-3 py-1 rounded bg-primary/10 border border-primary/20 text-primary">
              <Activity className="w-3 h-3 animate-spin" />
              <span className="text-[11px] font-mono uppercase font-bold">Syncing</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 rounded bg-positive/10 border border-positive/20 text-positive shadow-sm">
              <div className="w-2 h-2 rounded-full bg-positive" />
              <span className="text-[11px] font-mono uppercase font-bold">Connected</span>
            </div>
          )}
          <button
            onClick={() => setDebugModalOpen(true)}
            className="p-1.5 hover:bg-muted rounded-md transition-colors text-muted-foreground hover:text-foreground"
            title="System Integrity Audit"
          >
            <ShieldCheck className="w-5 h-5" />
          </button>
          <CommandMenu />
          <ThemeToggle />
        </div>
      </header>

      {/* MAIN WORKSPACE: 3-COLUMN LAYOUT */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* COLUMN 1: WATCHLIST */}
        <motion.aside 
          initial={false}
          animate={{ width: isSidebarOpen ? 320 : 0, opacity: isSidebarOpen ? 1 : 0 }}
          className="shrink-0 border-r border-border bg-card flex flex-col overflow-hidden"
        >
          <div className="p-4 w-[320px] h-full flex flex-col">
            <div className="flex p-1 bg-background rounded-lg border border-border mb-4 flex-wrap gap-1">
              <button
                onClick={() => setMarket('us')}
                className={cn(
                  "flex-1 min-w-[45px] py-1.5 rounded-md text-[11px] font-bold uppercase transition-all",
                  market === 'us' ? "bg-muted text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                🇺🇸 US
              </button>
              <button
                onClick={() => setMarket('india')}
                className={cn(
                  "flex-1 min-w-[45px] py-1.5 rounded-md text-[11px] font-bold uppercase transition-all",
                  market === 'india' ? "bg-muted text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                🇮🇳 IND
              </button>
              <button
                onClick={() => setMarket('crypto')}
                className={cn(
                  "flex-1 min-w-[45px] py-1.5 rounded-md text-[11px] font-bold uppercase transition-all",
                  market === 'crypto' ? "bg-muted text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                ₿ CRYPTO
              </button>
              <button
                onClick={() => setMarket('forex')}
                className={cn(
                  "flex-1 min-w-[45px] py-1.5 rounded-md text-[11px] font-bold uppercase transition-all",
                  market === 'forex' ? "bg-muted text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                💱 FX
              </button>
              <button
                onClick={() => setMarket('commodities')}
                className={cn(
                  "flex-1 min-w-[45px] py-1.5 rounded-md text-[11px] font-bold uppercase transition-all",
                  market === 'commodities' ? "bg-muted text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                )}
              >
                🥇 COM
              </button>
            </div>

            <div className="flex-1 overflow-y-auto hide-scrollbar">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-card z-10 border-b border-border">
                  <tr>
                    <th className="pb-2 text-[11px] font-bold text-muted-foreground uppercase">Symbol</th>
                    <th className="pb-2 text-[11px] font-bold text-muted-foreground uppercase text-right">Price</th>
                    <th className="pb-2 text-[11px] font-bold text-muted-foreground uppercase text-right">Chg%</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredUniverse.map(stock => (
                    <tr 
                      key={stock.ticker}
                      onClick={() => handleStockSelect(stock.ticker)}
                      className={cn(
                        "cursor-pointer transition-colors border-b border-border/50",
                        ticker === stock.ticker ? "bg-primary/10" : "hover:bg-muted/50"
                      )}
                    >
                      <td className="py-2.5">
                        <div className="flex flex-col">
                          <span className={cn("font-bold text-[13px]", ticker === stock.ticker ? "text-primary" : "text-foreground")}>{stock.ticker}</span>
                          <span className="text-[11px] text-muted-foreground truncate max-w-[120px]">{stock.name}</span>
                        </div>
                      </td>
                      <td className="py-2.5 text-right font-mono text-[13px] text-foreground">
                        {stock.price.toFixed(2)}
                      </td>
                      <td className={cn(
                        "py-2.5 text-right font-mono text-[13px] font-bold",
                        stock.pct_change >= 0 ? "text-positive" : "text-negative"
                      )}>
                        {stock.pct_change > 0 ? '+' : ''}{stock.pct_change.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.aside>

        {/* COLUMN 2 & 3: CENTER EXECUTION AREA */}
        <main className="flex-1 flex flex-col min-w-0 bg-background overflow-y-auto hide-scrollbar">
          
          {error && (
            <div className="m-4 bg-negative/10 border border-negative/30 text-negative p-3 rounded text-[13px] font-mono">
              {error}
            </div>
          )}

          {/* TOP SECTION: Chart + Trade Card */}
          <div className="flex flex-col xl:flex-row gap-4 p-4 xl:h-[500px]">
            {/* Chart Container */}
            <div className="flex-1 bg-card border border-border rounded-xl overflow-hidden min-h-[400px] shadow-lg flex flex-col">
              {/* Context Header */}
              <div className="px-4 py-3 border-b border-border flex justify-between items-center bg-card shrink-0">
                 <div className="flex items-center gap-3">
                   <h2 className="text-[20px] font-bold">{ticker}</h2>
                   <span className="text-[13px] text-muted-foreground">{chartData?.metadata?.name || ''}</span>
                   <span className="px-2 py-0.5 rounded bg-muted text-[11px] text-foreground font-mono">
                     {chartData?.current_price ? `${currencySymbol}${chartData.current_price.toFixed(2)}` : '---'}
                   </span>
                 </div>
              </div>
              <div className="flex-1 relative w-full h-full min-h-[300px]">
                <PriceChart data={chartData} loading={loading} />
              </div>
            </div>

            {/* Trade Card Container */}
            <div className="w-full xl:w-[360px] shrink-0">
               <TradeCard data={chartData} currency={currencySymbol} />
            </div>
          </div>

          {/* BOTTOM SECTION: Tabbed Interface */}
          <div className="px-4 pb-4 flex-1 flex flex-col">
            <div className="bg-card border border-border rounded-xl shadow-lg flex-1 flex flex-col overflow-hidden">
              
              {/* Tab Headers */}
              <div className="flex border-b border-border bg-card overflow-x-auto hide-scrollbar">
                {[
                  { id: 'CONSENSUS', icon: Target, label: 'Model Consensus' },
                  { id: 'RISK', icon: Shield, label: 'Risk Engine' },
                  { id: 'PORTFOLIO', icon: Briefcase, label: 'Portfolio' },
                  { id: 'PERFORMANCE', icon: CheckCircle2, label: 'Validation Center' },
                  { id: 'TECHNICAL', icon: BarChart2, label: 'Technicals' },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as TabType)}
                    className={cn(
                      "flex items-center gap-2 px-6 py-3 text-[13px] font-bold uppercase tracking-wider transition-colors border-b-2 whitespace-nowrap",
                      activeTab === tab.id 
                        ? "border-primary text-primary bg-primary/5" 
                        : "border-transparent text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    )}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="p-4 flex-1 overflow-y-auto bg-background/50">
                {activeTab === 'CONSENSUS' && (
                  <div className="max-w-4xl">
                    <SignalIntelligence data={chartData} currency={currencySymbol} />
                  </div>
                )}
                {activeTab === 'RISK' && (
                  <div className="max-w-4xl">
                    <RiskDashboard data={chartData} currency={currencySymbol} />
                  </div>
                )}
                {activeTab === 'PORTFOLIO' && (
                  <div className="max-w-4xl">
                    <PortfolioAnalytics data={chartData} currency={currencySymbol} />
                  </div>
                )}
                {activeTab === 'PERFORMANCE' && (
                  <div className="max-w-4xl">
                    <PaperTradingPerformance currency={currencySymbol} />
                  </div>
                )}
                {activeTab === 'TECHNICAL' && (
                  <div className="max-w-4xl">
                    <TechnicalSnapshot data={chartData} currency={currencySymbol} />
                  </div>
                )}
              </div>

            </div>
          </div>

        </main>
      </div>

      {/* DEBUG MODAL: Integrity Audit */}
      {isDebugModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-card border border-border rounded-xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden"
          >
            <div className="flex items-center justify-between p-4 border-b border-border bg-background">
              <h2 className="text-[14px] font-bold uppercase tracking-widest text-foreground flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-positive" />
                System Integrity Audit
              </h2>
              <button 
                onClick={() => setDebugModalOpen(false)}
                className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto">
              <IntegrityAudit data={chartData} />
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
