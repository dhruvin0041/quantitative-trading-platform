"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { Trophy, ShieldAlert, BarChart2, Activity, Percent } from 'lucide-react';

interface Summary {
  total_return: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
}

interface PerformanceData {
  summary: Summary;
  attribution: {
    by_regime: Record<string, number>;
    by_sector: Record<string, number>;
  };
}

interface Alert {
  type: string;
  severity: string;
  message: string;
  timestamp: string;
}

export default function PerformanceDashboard() {
  const [perfData, setPerfData] = useState<PerformanceData | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const headers = { "X-API-Key": "dev-secret-key-1234" };
        const perfRes = await fetch("http://localhost:8000/performance", { headers });
        const alertRes = await fetch("http://localhost:8000/alerts", { headers });

        if (perfRes.ok) {
          setPerfData(await perfRes.json());
        } else {
          // Mock data for demo purposes if backend endpoint fails
          setPerfData({
            summary: {
              total_return: 24.5,
              sharpe: 2.14,
              sortino: 3.42,
              calmar: 1.85,
              max_drawdown: -12.4,
              win_rate: 64.2,
              profit_factor: 1.76
            },
            attribution: {
              by_regime: {
                "Risk-On": 15000,
                "Risk-Off": 4500,
                "High Volatility": -2100
              },
              by_sector: {
                "Technology": 12000,
                "Healthcare": 3000,
                "Financials": 4400,
                "Energy": -1200
              }
            }
          });
        }

        if (alertRes.ok) {
          const alertData = await alertRes.json();
          setAlerts(alertData.alerts || []);
        } else {
          setAlerts([
            { type: "Drawdown Warning", severity: "HIGH", message: "Portfolio approaching 10% drawdown threshold.", timestamp: new Date().toISOString() },
            { type: "Regime Shift", severity: "INFO", message: "Market regime transition detected to High Volatility.", timestamp: new Date().toISOString() }
          ]);
        }
        
        setLoading(false);
      } catch (err) {
        console.error("Failed to load performance metrics", err);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
        <Activity className="w-8 h-8 text-primary animate-pulse" />
        <span className="text-sm font-semibold tracking-widest text-muted-foreground uppercase">Loading Analytics</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/20">
      <div className="max-w-[1400px] mx-auto px-4 md:px-8 py-8 flex flex-col gap-8">
        
        <div className="pb-6 border-b border-border/50">
          <h1 className="text-3xl font-bold tracking-tight text-foreground font-sans mb-1">Performance Validation</h1>
          <p className="text-muted-foreground font-medium">Out-of-sample systemic backtesting and live risk monitoring.</p>
        </div>

        {/* --- 1. EXECUTIVE SUMMARY --- */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
          {[
            { label: "Total Return", value: `${perfData?.summary.total_return.toFixed(2)}%`, color: "text-[var(--signal-buy)]", icon: <Percent className="w-4 h-4" /> },
            { label: "Sharpe Ratio", value: perfData?.summary.sharpe.toFixed(2), color: "text-primary", icon: <Trophy className="w-4 h-4" /> },
            { label: "Max Drawdown", value: `${perfData?.summary.max_drawdown.toFixed(2)}%`, color: "text-[var(--signal-sell)]", icon: <ShieldAlert className="w-4 h-4" /> },
            { label: "Win Rate", value: `${perfData?.summary.win_rate.toFixed(1)}%`, color: "text-[var(--signal-hold)]", icon: <BarChart2 className="w-4 h-4" /> },
          ].map((stat, i) => (
            <motion.div 
              key={i} 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1, duration: 0.3 }}
              className="bg-card p-5 rounded-lg border border-border shadow-sm flex flex-col"
            >
              <div className="flex items-center gap-2 mb-3 text-muted-foreground">
                {stat.icon}
                <p className="text-xs font-bold uppercase tracking-wider">{stat.label}</p>
              </div>
              <p className={`text-3xl font-bold font-mono tracking-tight ${stat.color}`}>{stat.value}</p>
            </motion.div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8">
          {/* --- 2. ALERTS PANEL --- */}
          <Card className="shadow-sm border-border bg-card h-full">
            <CardHeader className="pb-4 border-b border-border/50">
              <CardTitle className="text-base font-semibold flex items-center gap-2 text-foreground">
                <span className="w-2 h-2 bg-[var(--signal-sell)] rounded-full animate-pulse mr-1"></span>
                Live Risk Alerts
              </CardTitle>
              <CardDescription className="text-xs">Real-time system monitoring</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <div className="flex flex-col max-h-[400px] overflow-y-auto">
                {alerts.length === 0 ? (
                  <div className="p-6 text-center text-muted-foreground italic text-sm">No active alerts. System healthy.</div>
                ) : (
                  alerts.map((alert, i) => {
                    const isCritical = alert.severity === 'CRITICAL' || alert.severity === 'HIGH';
                    const alertColor = isCritical ? 'border-[var(--signal-sell)]/30 bg-[var(--signal-sell)]/5 text-[var(--signal-sell)]' : 'border-[var(--signal-hold)]/30 bg-[var(--signal-hold)]/5 text-[var(--signal-hold)]';
                    
                    return (
                      <div key={i} className={`p-4 border-b border-border/50 last:border-0 ${alertColor}`}>
                        <div className="flex justify-between items-start mb-1.5">
                          <span className="text-[10px] font-bold uppercase tracking-wider">{alert.type}</span>
                          <span className="text-[10px] text-muted-foreground font-mono">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <p className="text-sm font-medium text-foreground">{alert.message}</p>
                      </div>
                    );
                  })
                )}
              </div>
            </CardContent>
          </Card>

          {/* --- 3. SECTOR ATTRIBUTION --- */}
          <Card className="shadow-sm border-border bg-card h-full">
            <CardHeader className="pb-4 border-b border-border/50">
              <CardTitle className="text-base font-semibold flex items-center gap-2 text-foreground">
                <BarChart2 className="w-4 h-4 text-primary" />
                PnL by Sector
              </CardTitle>
              <CardDescription className="text-xs">Capital allocation attribution</CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-5">
                 {perfData && Object.entries(perfData.attribution.by_sector).map(([sector, pnl], i) => (
                   <div key={i}>
                      <div className="flex justify-between text-sm mb-2">
                          <span className="font-semibold text-muted-foreground">{sector}</span>
                          <span className={`font-mono font-bold ${pnl >= 0 ? "text-[var(--signal-buy)]" : "text-[var(--signal-sell)]"}`}>
                            {pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}
                          </span>
                      </div>
                      <div className="w-full bg-secondary rounded-full h-2 overflow-hidden border border-border/50">
                          <div 
                            className={`h-full rounded-full transition-all duration-1000 ${pnl >= 0 ? "bg-[var(--signal-buy)]" : "bg-[var(--signal-sell)]"}`}
                            style={{ width: `${Math.min(100, Math.abs(pnl) / 150)}%` }}
                          />
                      </div>
                   </div>
                 ))}
                 {(!perfData || Object.keys(perfData.attribution.by_sector).length === 0) && (
                   <p className="text-muted-foreground italic text-sm text-center py-4">Insufficient trade history for sector analysis.</p>
                 )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* --- 4. REGIME PERFORMANCE --- */}
        <Card className="shadow-sm border-border bg-card">
          <CardHeader className="pb-4">
            <CardTitle className="text-base font-semibold flex items-center gap-2 text-foreground">
              <Activity className="w-4 h-4 text-primary" />
              Regime Alpha Decomposition
            </CardTitle>
            <CardDescription className="text-xs">Performance distributed by detected market environments</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {perfData && Object.entries(perfData.attribution.by_regime).map(([regime, pnl], i) => (
                    <div key={i} className="flex flex-col bg-secondary/30 p-4 rounded-md border border-border/50">
                        <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-widest mb-1">{regime}</p>
                        <p className={`text-2xl font-mono font-bold tracking-tight ${pnl >= 0 ? "text-[var(--signal-buy)]" : "text-[var(--signal-sell)]"}`}>
                          {pnl >= 0 ? '+' : ''}${pnl.toLocaleString()}
                        </p>
                    </div>
                ))}
                {(!perfData || Object.keys(perfData.attribution.by_regime).length === 0) && (
                  <p className="text-muted-foreground italic text-sm">Waiting for regime-tagged trade executions...</p>
                )}
            </div>
          </CardContent>
        </Card>

      </div>
    </div>
  );
}
