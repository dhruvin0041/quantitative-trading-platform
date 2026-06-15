import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Play, Loader2, TrendingUp, BarChart3, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { API_KEY, getBaseUrl } from '@/lib/config';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { BacktestSummary } from '@/types';

interface BacktestPanelProps {
  isOpen: boolean;
  onClose: () => void;
  currentTicker: string;
}

export function BacktestPanel({ isOpen, onClose, currentTicker }: BacktestPanelProps) {
  const [ticker, setTicker] = useState(currentTicker);
  const [period, setPeriod] = useState('1y');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BacktestSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    const API_URL = getBaseUrl();
    
    try {
      const res = await fetch(`${API_URL}/backtest?ticker=${ticker}&period=${period}`, {
        headers: { "X-API-Key": API_KEY }
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to run backtest');
      }
      const data = await res.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Failed to run backtest. Ensure backend is running and data is available.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: '100%', opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: '100%', opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="fixed inset-y-0 right-0 w-full md:w-[480px] bg-background border-l border-border shadow-2xl z-[100] flex flex-col"
        >
          <div className="h-14 border-b border-border flex items-center justify-between px-5 bg-card">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-primary" />
              <h2 className="font-bold tracking-widest text-[13px] text-foreground uppercase">Backtest Engine</h2>
            </div>
            <button onClick={onClose} className="p-1.5 hover:bg-muted rounded-md text-muted-foreground hover:text-foreground transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="p-6 border-b border-border space-y-5 bg-card">
            <div className="flex flex-col gap-2">
              <label className="text-[12px] font-bold uppercase tracking-widest text-muted-foreground">Target Asset</label>
              <input 
                type="text" 
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="bg-background border border-border rounded-md p-3 text-[14px] font-mono text-foreground uppercase focus:outline-none focus:border-primary transition-colors"
              />
            </div>
            
            <div className="flex flex-col gap-2">
              <label className="text-[12px] font-bold uppercase tracking-widest text-muted-foreground">Evaluation Period</label>
              <div className="flex gap-2">
                {['3m', '6m', '1y', '2y'].map(p => (
                  <button
                    key={p}
                    onClick={() => setPeriod(p)}
                    className={cn(
                      "flex-1 py-2 text-[13px] font-bold rounded-md uppercase transition-all border",
                      period === p ? "bg-primary text-white border-primary shadow-lg shadow-blue-500/20" : "bg-background border-border text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <button 
              onClick={runBacktest}
              disabled={loading}
              className="w-full mt-2 py-3 bg-emerald-600 text-white rounded-md font-bold text-[13px] uppercase tracking-widest hover:bg-positive transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {loading ? "Running Simulation..." : "Run Backtest"}
            </button>
            
            {error && (
              <div className="p-4 bg-negative/10 border border-negative/30 rounded-md text-negative text-[13px] flex gap-2 items-center font-mono">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-6 bg-background custom-scrollbar">
            {!results && !loading && (
              <div className="h-full flex flex-col items-center justify-center text-muted-foreground gap-4">
                <BarChart3 className="w-12 h-12 opacity-50" />
                <p className="text-[14px] font-bold uppercase tracking-widest">Ready to simulate strategies</p>
              </div>
            )}

            {results && !loading && (results.total_trades === undefined || results.total_trades >= 30) && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                  <div className="p-4 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Win Rate</span>
                    <span className="text-[20px] font-mono font-black text-positive">{results.win_rate}%</span>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Profit Factor</span>
                    <span className="text-[20px] font-mono font-black text-foreground">{results.profit_factor}</span>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Sharpe</span>
                    <span className="text-[20px] font-mono font-black text-foreground">{results.sharpe_ratio}</span>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Max Drawdown</span>
                    <span className="text-[20px] font-mono font-black text-negative">{results.max_drawdown}%</span>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Vetoed Rate</span>
                    <span className="text-[20px] font-mono font-black text-warning">{results.vetoed_rate}%</span>
                  </div>
                  <div className="p-4 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[11px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Coverage</span>
                    <span className="text-[20px] font-mono font-black text-primary">{results.coverage || results.signal_coverage || 0}%</span>
                  </div>
                </div>

                {results.monthly_win_rates && (
                  <div className="p-5 bg-card border border-border rounded-lg">
                    <h3 className="text-[12px] font-bold text-muted-foreground uppercase tracking-widest mb-4">Monthly Win Rate</h3>
                    <div className="h-40">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={results.monthly_win_rates}>
                          <XAxis dataKey="month" fontSize={11} tickLine={false} axisLine={false} tick={{fill: '#64748b'}} />
                          <YAxis fontSize={11} tickLine={false} axisLine={false} tickFormatter={val => `${val}%`} tick={{fill: '#64748b'}} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', fontSize: '12px', color: '#f8fafc' }}
                            itemStyle={{ color: '#f8fafc' }}
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            formatter={(value: any) => [`${value}%`, 'Win Rate']}
                          />
                          <Bar dataKey="win_rate" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {results.best_signal && (
                    <div className="p-4 bg-card border border-positive/30 rounded-lg">
                      <h4 className="text-[12px] font-bold text-positive uppercase tracking-widest mb-3">Best Signal</h4>
                      <div className="text-[13px] text-foreground space-y-2">
                        <div className="flex justify-between items-center"><span className="text-muted-foreground">Date</span><span className="font-mono">{results.best_signal.date}</span></div>
                        <div className="flex justify-between items-center"><span className="text-muted-foreground">Signal</span><span className="font-black text-positive">{results.best_signal.signal}</span></div>
                        <div className="flex justify-between items-center"><span className="text-muted-foreground">Conf</span><span className="font-mono">{results.best_signal.confidence}%</span></div>
                        <div className="flex justify-between items-center mt-3 pt-3 border-t border-border"><span className="font-bold">Return</span><span className="font-mono font-black text-positive">+{results.best_signal.actual_return}%</span></div>
                      </div>
                    </div>
                  )}
                  {results.worst_signal && (
                    <div className="p-4 bg-card border border-negative/30 rounded-lg">
                      <h4 className="text-[12px] font-bold text-negative uppercase tracking-widest mb-3">Worst Signal</h4>
                      <div className="text-[13px] text-foreground space-y-2">
                        <div className="flex justify-between items-center"><span className="text-muted-foreground">Date</span><span className="font-mono">{results.worst_signal.date}</span></div>
                        <div className="flex justify-between items-center"><span className="text-muted-foreground">Signal</span><span className="font-black text-negative">{results.worst_signal.signal}</span></div>
                        <div className="flex justify-between items-center"><span className="text-muted-foreground">Conf</span><span className="font-mono">{results.worst_signal.confidence}%</span></div>
                        <div className="flex justify-between items-center mt-3 pt-3 border-t border-border"><span className="font-bold">Return</span><span className="font-mono font-black text-negative">{results.worst_signal.actual_return}%</span></div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            {results && !loading && results.total_trades !== undefined && results.total_trades < 30 && (
              <div className="flex flex-col items-center justify-center h-full text-center gap-4 p-6 border border-border rounded-lg bg-card">
                <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
                  <AlertCircle className="w-6 h-6 text-muted-foreground" />
                </div>
                <div className="flex flex-col gap-1">
                  <h3 className="text-[14px] font-bold text-foreground uppercase tracking-widest">Backtest Analytics Unavailable</h3>
                  <p className="text-[12px] text-muted-foreground max-w-sm">
                    Minimum 30 completed trades required to ensure statistical significance of performance metrics.
                  </p>
                  <span className="text-[12px] font-mono font-bold text-primary mt-2">Current sample size: {results.total_trades} trades</span>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
