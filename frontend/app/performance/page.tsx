"use client";

import React, { useState, useEffect } from 'react';

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
        
        setPerfData(await perfRes.json());
        setAlerts((await alertRes.json()).alerts);
        setLoading(false);
      } catch (err) {
        console.error("Failed to load performance metrics", err);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <div className="p-8 text-white">Loading Performance Metrics...</div>;

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8 border-l-4 border-cyan-500 pl-4">HYDRA: Performance Validation</h1>

        {/* --- 1. EXECUTIVE SUMMARY --- */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {[
            { label: "Total Return", value: `${perfData?.summary.total_return.toFixed(2)}%`, color: "text-emerald-400" },
            { label: "Sharpe Ratio", value: perfData?.summary.sharpe.toFixed(2), color: "text-cyan-400" },
            { label: "Max Drawdown", value: `${perfData?.summary.max_drawdown.toFixed(2)}%`, color: "text-rose-400" },
            { label: "Win Rate", value: `${perfData?.summary.win_rate.toFixed(1)}%`, color: "text-amber-400" },
          ].map((stat, i) => (
            <div key={i} className="bg-slate-800 p-6 rounded-lg border border-slate-700">
              <p className="text-slate-400 text-sm font-medium">{stat.label}</p>
              <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* --- 2. ALERTS PANEL --- */}
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <span className="w-2 h-2 bg-rose-500 rounded-full mr-2"></span>
              Live Risk Alerts
            </h2>
            <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
              {alerts.length === 0 ? (
                <p className="text-slate-500 italic">No active alerts. System healthy.</p>
              ) : (
                alerts.map((alert, i) => (
                  <div key={i} className={`p-3 rounded border ${
                    alert.severity === 'CRITICAL' ? 'bg-rose-900/20 border-rose-500/50 text-rose-200' :
                    alert.severity === 'HIGH' ? 'bg-amber-900/20 border-amber-500/50 text-amber-200' :
                    'bg-slate-700 border-slate-600 text-slate-300'
                  }`}>
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-xs font-bold uppercase tracking-wider">{alert.type}</span>
                      <span className="text-[10px] opacity-60">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <p className="text-sm">{alert.message}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* --- 3. SECTOR ATTRIBUTION --- */}
          <div className="bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h2 className="text-xl font-semibold mb-4">PnL by Sector</h2>
            <div className="space-y-4">
               {perfData && Object.entries(perfData.attribution.by_sector).map(([sector, pnl], i) => (
                 <div key={i}>
                    <div className="flex justify-between text-sm mb-1">
                        <span>{sector}</span>
                        <span className={pnl >= 0 ? "text-emerald-400" : "text-rose-400"}>${pnl.toLocaleString()}</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-1.5">
                        <div className={`h-1.5 rounded-full ${pnl >= 0 ? "bg-emerald-500" : "bg-rose-500"}`} 
                             style={{ width: `${Math.min(100, Math.abs(pnl) / 1000)}%` }}></div>
                    </div>
                 </div>
               ))}
               {(!perfData || Object.keys(perfData.attribution.by_sector).length === 0) && <p className="text-slate-500 italic">Insufficient trade history for sector analysis.</p>}
            </div>
          </div>
        </div>

        {/* --- 4. REGIME PERFORMANCE --- */}
        <div className="mt-8 bg-slate-800 p-6 rounded-lg border border-slate-700">
            <h2 className="text-xl font-semibold mb-4">Regime Alpha Decomposition</h2>
            <div className="flex flex-wrap gap-4">
                {perfData && Object.entries(perfData.attribution.by_regime).map(([regime, pnl], i) => (
                    <div key={i} className="flex-1 min-w-[200px] bg-slate-900 p-4 rounded border border-slate-700">
                        <p className="text-xs text-slate-500 uppercase font-bold tracking-widest">{regime}</p>
                        <p className={`text-xl font-bold ${pnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>${pnl.toLocaleString()}</p>
                    </div>
                ))}
                {(!perfData || Object.keys(perfData.attribution.by_regime).length === 0) && <p className="text-slate-500 italic">Waiting for regime-tagged trade executions...</p>}
            </div>
        </div>
      </div>
    </div>
  );
}
