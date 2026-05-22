"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Activity, BarChart2, ShieldAlert, Trophy, Percent, Target, Zap, LineChart, TrendingUp, TrendingDown, Layers } from 'lucide-react';
import { motion } from 'framer-motion';

export default function InstitutionalDashboard() {
  const [loading, setLoading] = useState(true);

  // Mocked state for presentation since this requires live DB connection
  const [data] = useState({
    trading: {
      equity_curve: [100000, 100500, 101200, 100800, 102000, 103500, 103100, 104500, 105200, 106000],
      daily_pnl: 800,
      monthly_pnl: 6000,
      win_rate: 62.5,
      profit_factor: 1.85,
      drawdown: -2.1
    },
    models: {
      lstm: 58.4,
      xgboost: 56.2,
      lightgbm: 57.1,
      dqn: 54.8,
      tft: 59.2,
      informer: 58.8,
      patchtst: 60.1,
      ensemble: 63.5,
      consensus_rate: 42.0
    },
    risk: {
      beta: 0.85,
      alpha: 4.2,
      exposure: 65.0,
      kelly: 18.5,
      volatility: 12.4,
      var_95: 1.2
    },
    signals: {
      active: 4,
      historical: 128,
      regime: "BULL / LOW VOLATILITY"
    }
  });

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Activity className="w-8 h-8 text-blue-500 animate-pulse" />
          <span className="text-xs font-mono tracking-widest text-zinc-500 uppercase">Loading Institutional Telemetry...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 font-sans p-6">
      <div className="max-w-[1600px] mx-auto space-y-6">
        
        {/* HEADER */}
        <div className="flex items-center justify-between pb-4 border-b border-zinc-800">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Quantitative Analytics Terminal</h1>
            <p className="text-xs font-mono text-zinc-500 mt-1">SYSTEM ONLINE • MULTI-AGENT CONSENSUS ACTIVE</p>
          </div>
          <div className="flex gap-4 font-mono text-xs">
            <div className="flex flex-col items-end">
              <span className="text-zinc-500">MARKET REGIME</span>
              <span className="text-green-500 font-bold">{data.signals.regime}</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          
          {/* TRADING PERFORMANCE */}
          <div className="md:col-span-8 space-y-6">
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <LineChart className="w-4 h-4 text-blue-500" />
                  Trading Performance
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 grid grid-cols-3 md:grid-cols-6 gap-4">
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase font-bold text-zinc-500 mb-1">Win Rate</span>
                  <span className="text-xl font-mono font-bold text-zinc-100">{data.trading.win_rate}%</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase font-bold text-zinc-500 mb-1">Profit Factor</span>
                  <span className="text-xl font-mono font-bold text-zinc-100">{data.trading.profit_factor}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase font-bold text-zinc-500 mb-1">Drawdown</span>
                  <span className="text-xl font-mono font-bold text-red-500">{data.trading.drawdown}%</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase font-bold text-zinc-500 mb-1">Daily PnL</span>
                  <span className="text-xl font-mono font-bold text-green-500">+${data.trading.daily_pnl}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase font-bold text-zinc-500 mb-1">Monthly PnL</span>
                  <span className="text-xl font-mono font-bold text-green-500">+${data.trading.monthly_pnl}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase font-bold text-zinc-500 mb-1">Total Trades</span>
                  <span className="text-xl font-mono font-bold text-zinc-100">{data.signals.historical}</span>
                </div>
              </CardContent>
            </Card>

            {/* MODEL MONITORING */}
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <Layers className="w-4 h-4 text-purple-500" />
                  Model Monitoring (Out-of-Sample Accuracy)
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { name: "PatchTST", acc: data.models.patchtst, color: "text-purple-400" },
                  { name: "TFT", acc: data.models.tft, color: "text-purple-400" },
                  { name: "Informer", acc: data.models.informer, color: "text-purple-400" },
                  { name: "LSTM V4", acc: data.models.lstm, color: "text-blue-400" },
                  { name: "LightGBM", acc: data.models.lightgbm, color: "text-orange-400" },
                  { name: "XGBoost", acc: data.models.xgboost, color: "text-orange-400" },
                  { name: "DQN Policy", acc: data.models.dqn, color: "text-green-400" },
                  { name: "META-ENSEMBLE", acc: data.models.ensemble, color: "text-zinc-100" }
                ].map(m => (
                  <div key={m.name} className="flex justify-between items-center p-3 bg-zinc-900/50 border border-zinc-800 rounded">
                    <span className="text-[10px] font-bold text-zinc-400 uppercase">{m.name}</span>
                    <span className={`text-sm font-mono font-bold ${m.color}`}>{m.acc}%</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          {/* RISK & SIGNALS SIDEBAR */}
          <div className="md:col-span-4 space-y-6">
            
            {/* RISK PANEL */}
            <Card className="bg-[#111] border-zinc-800 h-full">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <ShieldAlert className="w-4 h-4 text-orange-500" />
                  Portfolio Risk Limits
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
                <div className="flex justify-between items-center border-b border-zinc-800 pb-2">
                  <span className="text-xs font-mono text-zinc-500">Jensen's Alpha</span>
                  <span className="text-sm font-mono font-bold text-green-500">+{data.risk.alpha}%</span>
                </div>
                <div className="flex justify-between items-center border-b border-zinc-800 pb-2">
                  <span className="text-xs font-mono text-zinc-500">Portfolio Beta</span>
                  <span className="text-sm font-mono font-bold text-zinc-300">{data.risk.beta}</span>
                </div>
                <div className="flex justify-between items-center border-b border-zinc-800 pb-2">
                  <span className="text-xs font-mono text-zinc-500">Value at Risk (95%)</span>
                  <span className="text-sm font-mono font-bold text-red-400">{data.risk.var_95}%</span>
                </div>
                <div className="flex justify-between items-center border-b border-zinc-800 pb-2">
                  <span className="text-xs font-mono text-zinc-500">Kelly Allocation Limit</span>
                  <span className="text-sm font-mono font-bold text-blue-400">{data.risk.kelly}% max</span>
                </div>
                <div className="flex justify-between items-center pb-2">
                  <span className="text-xs font-mono text-zinc-500">Current Exposure</span>
                  <span className="text-sm font-mono font-bold text-zinc-300">{data.risk.exposure}%</span>
                </div>
              </CardContent>
            </Card>

            {/* SIGNAL INTELLIGENCE */}
            <Card className="bg-[#111] border-zinc-800">
              <CardHeader className="pb-2 border-b border-zinc-800/50">
                <CardTitle className="text-sm font-bold flex items-center gap-2 text-zinc-300">
                  <Target className="w-4 h-4 text-green-500" />
                  Signal Intelligence
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4 space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-mono text-zinc-500">Active Live Signals</span>
                  <span className="text-lg font-mono font-bold text-zinc-100">{data.signals.active}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs font-mono text-zinc-500">Consensus Rate</span>
                  <span className="text-lg font-mono font-bold text-blue-500">{data.models.consensus_rate}%</span>
                </div>
              </CardContent>
            </Card>

          </div>
        </div>
      </div>
    </div>
  );
}
