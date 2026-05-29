"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Trophy, LineChart, Layers, Crosshair, CheckCircle2, AlertTriangle, AlertCircle, TrendingDown } from 'lucide-react';

interface ValidationMetrics {
  win_rate: number;
  total_trades: number;
  profit_factor: number;
}

interface ValidationData {
  status?: string;
  performance?: Record<string, number>;
  calibration?: {
    ece?: number;
    reliability_curve?: { bin: string; count: number; predicted_conf: number; actual_win_rate: number }[];
  };
  strategy_health?: {
    status?: string;
    historical_win_rate?: number;
    rolling_win_rate?: number;
    drift?: number;
  };
  market_segmentation?: Record<string, ValidationMetrics>;
  regime_segmentation?: Record<string, ValidationMetrics>;
  recent_signals?: { asset: string; confidence: number; outcome: string }[];
}

export default function ValidationDashboard() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<ValidationData | null>(null);

  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${API_URL}/validation`, {
          headers: { "X-API-Key": API_KEY }
        });
        const valData = await res.json();
        setData(valData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [API_KEY, API_URL]);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Activity className="w-8 h-8 text-indigo-500 animate-pulse" />
          <span className="text-xs font-mono tracking-widest text-zinc-500 uppercase">Loading Empirical Validation Data...</span>
        </div>
      </div>
    );
  }

  if (data?.status === "NO_DATA") {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center max-w-md">
          <AlertCircle className="w-12 h-12 text-yellow-500" />
          <h2 className="text-xl font-bold text-zinc-200">Insufficient Trade History</h2>
          <p className="text-sm text-zinc-500">The Empirical Validation Framework requires closed paper trades to generate statistical significance. Please run the Paper Trading Engine to accumulate historical signal outcomes.</p>
        </div>
      </div>
    );
  }

  const p = data?.performance || ({} as Record<string, number>);
  const c = data?.calibration || ({} as NonNullable<ValidationData['calibration']>);
  const h = data?.strategy_health || ({} as NonNullable<ValidationData['strategy_health']>);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 font-sans p-6 pb-20">
      <div className="max-w-[1600px] mx-auto space-y-6">
        
        {/* HEADER */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between pb-4 border-b border-zinc-800 gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Empirical Validation Framework</h1>
            <p className="text-xs font-mono text-zinc-500 mt-1">LIVE OUT-OF-SAMPLE (OOS) TRADE JOURNAL & STATISTICAL AUDIT</p>
          </div>
          <div className="flex gap-4">
            <div className={`px-4 py-2 rounded flex items-center gap-2 border ${h.status === 'HEALTHY' ? 'bg-green-500/10 border-green-500/30 text-green-400' : 'bg-red-500/10 border-red-500/30 text-red-400'}`}>
              {h.status === 'HEALTHY' ? <CheckCircle2 className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
              <span className="text-xs font-mono font-bold">SYSTEM {h.status || 'UNKNOWN'}</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          
          {/* LEFT COLUMN: CORE PERFORMANCE */}
          <div className="md:col-span-8 space-y-6">
            
            {/* OOS METRICS */}
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <Trophy className="w-4 h-4 text-yellow-500" />
                  Live Out-Of-Sample Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricBox label="Win Rate" value={`${p.win_rate?.toFixed(1) || 0}%`} />
                <MetricBox label="Profit Factor" value={p.profit_factor?.toFixed(2) || '0.0'} />
                <MetricBox label="Sharpe (Trade Proxy)" value={p.sharpe_proxy?.toFixed(2) || '0.0'} />
                <MetricBox label="Sortino (Proxy)" value={p.sortino_proxy?.toFixed(2) || '0.0'} />
                <MetricBox label="Total Closed Trades" value={p.total_trades || 0} />
                <MetricBox label="Avg Holding Days" value={p.avg_holding_days?.toFixed(1) || '0.0'} />
                <MetricBox label="Expectancy" value={`$${p.expectancy?.toFixed(2) || '0.0'}`} color={p.expectancy > 0 ? 'text-green-500' : 'text-red-500'} />
                <MetricBox label="Risk / Reward" value={`1 : ${p.risk_reward?.toFixed(2) || '0.0'}`} />
              </CardContent>
            </Card>

            {/* CONFIDENCE CALIBRATION */}
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <Crosshair className="w-4 h-4 text-blue-500" />
                  Confidence Calibration Analysis
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="flex items-center justify-between mb-6">
                  <span className="text-xs text-zinc-400">Comparing Model Predicted Confidence vs. Actual Win Rate. Expected Calibration Error (ECE):</span>
                  <span className="text-lg font-mono font-bold text-blue-400">{c.ece?.toFixed(2)}%</span>
                </div>
                
                <div className="space-y-4">
                  {c.reliability_curve && c.reliability_curve.length > 0 ? (
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    c.reliability_curve.map((bin: any, idx: number) => (
                      <div key={idx} className="flex flex-col">
                        <div className="flex justify-between text-xs font-mono text-zinc-500 mb-1">
                          <span>BIN {bin.bin} (n={bin.count})</span>
                          <span>Pred: {bin.predicted_conf.toFixed(1)}% | Actual: {bin.actual_win_rate.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-zinc-800 rounded overflow-hidden flex">
                          <div className="bg-blue-500 h-full" style={{ width: `${bin.predicted_conf}%` }}></div>
                        </div>
                        <div className="h-2 w-full bg-zinc-800 rounded overflow-hidden flex mt-1">
                          <div className="bg-green-500 h-full" style={{ width: `${bin.actual_win_rate}%` }}></div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center text-zinc-500 py-4 text-xs font-mono">NO CALIBRATION DATA AVAILABLE</div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* STRATEGY HEALTH & DRIFT */}
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <TrendingDown className="w-4 h-4 text-orange-500" />
                  Degradation & Drift Monitoring
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricBox label="Status" value={h.status || 'UNKNOWN'} color={h.status === 'HEALTHY' ? 'text-green-500' : 'text-red-500'} />
                <MetricBox label="Historical Win Rate" value={`${h.historical_win_rate?.toFixed(1) || 0}%`} />
                <MetricBox label="Recent Win Rate" value={`${h.rolling_win_rate?.toFixed(1) || 0}%`} />
                <MetricBox label="Performance Drift" value={`${(h.drift || 0) > 0 ? '-' : '+'}${Math.abs(h.drift || 0).toFixed(1)}%`} color={(h.drift || 0) > 0 ? 'text-red-500' : 'text-green-500'} />
              </CardContent>
            </Card>

          </div>

          {/* RIGHT COLUMN: SEGMENTATION & RECENT SIGNALS */}
          <div className="md:col-span-4 space-y-6">
            
            {/* MARKET SEGMENTATION */}
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <Layers className="w-4 h-4 text-indigo-500" />
                  Market Segmentation
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
  {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                {(data as any).market_segmentation && Object.keys((data as any).market_segmentation).length > 0 ? (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  (Object.entries((data as any).market_segmentation) as any).map(([market, metrics]: [string, any]) => (
                    <div key={market} className="flex justify-between items-center border-b border-zinc-800 pb-2 last:border-0">
                      <span className="text-xs font-mono text-zinc-400">{market.toUpperCase()}</span>
                      <div className="text-right">
                        <div className="text-sm font-mono font-bold text-zinc-200">WR: {metrics.win_rate.toFixed(1)}%</div>
                        <div className="text-[10px] text-zinc-500">n={metrics.total_trades} | PF: {metrics.profit_factor.toFixed(2)}</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-zinc-500 py-2 text-xs font-mono">NO DATA</div>
                )}
              </CardContent>
            </Card>

            {/* REGIME SEGMENTATION */}
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <Activity className="w-4 h-4 text-purple-500" />
                  Regime Segmentation
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
  {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                {(data as any).regime_segmentation && Object.keys((data as any).regime_segmentation).length > 0 ? (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  (Object.entries((data as any).regime_segmentation) as any).map(([regime, metrics]: [string, any]) => (
                    <div key={regime} className="flex justify-between items-center border-b border-zinc-800 pb-2 last:border-0">
                      <span className="text-xs font-mono text-zinc-400">{regime}</span>
                      <div className="text-right">
                        <div className="text-sm font-mono font-bold text-zinc-200">WR: {metrics.win_rate.toFixed(1)}%</div>
                        <div className="text-[10px] text-zinc-500">n={metrics.total_trades} | PF: {metrics.profit_factor.toFixed(2)}</div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-zinc-500 py-2 text-xs font-mono">NO DATA</div>
                )}
              </CardContent>
            </Card>

            {/* RECENT SIGNAL JOURNAL */}
            <Card className="bg-[#111] border-zinc-800 h-[300px] flex flex-col">
              <CardHeader className="pb-2 border-b border-zinc-800/50 shrink-0">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <LineChart className="w-4 h-4 text-emerald-500" />
                  Recent Signal Journal
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0 overflow-y-auto grow p-0">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-zinc-900 sticky top-0">
                    <tr>
                      <th className="p-2 font-normal text-zinc-500">TICKER</th>
                      <th className="p-2 font-normal text-zinc-500">CONF</th>
                      <th className="p-2 font-normal text-zinc-500 text-right">OUTCOME</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data?.recent_signals?.map((sig: { asset: string; confidence: number; outcome: string }, idx: number) => (
                      <tr key={idx} className="border-b border-zinc-800 hover:bg-zinc-900/50">
                        <td className="p-2 font-bold text-zinc-300">{sig.asset}</td>
                        <td className="p-2 text-zinc-400">{sig.confidence?.toFixed(1)}%</td>
                        <td className={`p-2 text-right font-bold ${
                          sig.outcome === 'WIN' ? 'text-green-500' : 
                          sig.outcome === 'LOSS' ? 'text-red-500' : 'text-zinc-500'
                        }`}>
                          {sig.outcome}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>

          </div>
        </div>
      </div>
    </div>
  );
}

function MetricBox({ label, value, color = "text-zinc-100" }: { label: string, value: string | number, color?: string }) {
  return (
    <div className="flex flex-col p-3 bg-zinc-900/30 border border-zinc-800 rounded">
      <span className="text-[10px] uppercase font-bold text-zinc-500 mb-1">{label}</span>
      <span className={`text-xl font-mono font-bold ${color}`}>{value}</span>
    </div>
  );
}
