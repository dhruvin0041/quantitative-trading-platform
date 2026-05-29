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
          <div className="h-14 border-b border-border flex items-center justify-between px-4 bg-secondary/30">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-primary" />
              <h2 className="font-black tracking-widest text-sm">BACKTEST ENGINE</h2>
            </div>
            <button onClick={onClose} className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-foreground">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="p-6 border-b border-border space-y-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Target Asset</label>
              <input 
                type="text" 
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                className="bg-muted/50 border border-border rounded p-2 text-sm font-mono uppercase focus:outline-none focus:border-primary"
              />
            </div>
            
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Evaluation Period</label>
              <div className="flex gap-2">
                {['3m', '6m', '1y', '2y'].map(p => (
                  <button
                    key={p}
                    onClick={() => setPeriod(p)}
                    className={cn(
                      "flex-1 py-1.5 text-xs font-bold rounded uppercase transition-colors border",
                      period === p ? "bg-primary text-white border-primary" : "bg-muted/30 border-border text-muted-foreground hover:bg-muted"
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
              className="w-full mt-2 py-3 bg-primary text-white rounded font-black text-sm uppercase tracking-widest hover:bg-primary/90 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
              {loading ? "Running Simulation..." : "Run Backtest"}
            </button>
            
            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-red-500 text-xs flex gap-2 items-center">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-6 bg-muted/10">
            {!results && !loading && (
              <div className="h-full flex flex-col items-center justify-center text-muted-foreground gap-4 opacity-50">
                <BarChart3 className="w-12 h-12" />
                <p className="text-sm font-medium">Ready to simulate strategies.</p>
              </div>
            )}

            {results && !loading && (
              <div className="space-y-6">
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Win Rate</span>
                    <span className="text-lg font-black text-green-500">{results.win_rate}%</span>
                  </div>
                  <div className="p-3 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Profit Factor</span>
                    <span className="text-lg font-black">{results.profit_factor}</span>
                  </div>
                  <div className="p-3 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Sharpe</span>
                    <span className="text-lg font-black">{results.sharpe_ratio}</span>
                  </div>
                  <div className="p-3 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Max Drawdown</span>
                    <span className="text-lg font-black text-red-500">{results.max_drawdown}%</span>
                  </div>
                  <div className="p-3 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Vetoed Rate</span>
                    <span className="text-lg font-black text-amber-500">{results.vetoed_rate}%</span>
                  </div>
                  <div className="p-3 bg-card border border-border rounded-lg flex flex-col items-center">
                    <span className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Coverage</span>
                    <span className="text-lg font-black text-primary">{results.coverage || results.signal_coverage || 0}%</span>
                  </div>
                </div>

                {results.monthly_win_rates && (
                  <div className="p-4 bg-card border border-border rounded-lg">
                    <h3 className="text-[10px] font-bold text-muted-foreground uppercase mb-4">Monthly Win Rate</h3>
                    <div className="h-32">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={results.monthly_win_rates}>
                          <XAxis dataKey="month" fontSize={10} tickLine={false} axisLine={false} />
                          <YAxis fontSize={10} tickLine={false} axisLine={false} tickFormatter={val => `${val}%`} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: 'var(--background)', borderColor: 'var(--border)', fontSize: '12px' }}
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            formatter={(value: any) => [`${value}%`, 'Win Rate']}
                          />
                          <Bar dataKey="win_rate" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  {results.best_signal && (
                    <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                      <h4 className="text-[10px] font-bold text-green-500 uppercase mb-2">Best Signal</h4>
                      <div className="text-xs text-foreground space-y-1">
                        <div className="flex justify-between"><span className="text-muted-foreground">Date</span><span className="font-mono">{results.best_signal.date}</span></div>
                        <div className="flex justify-between"><span className="text-muted-foreground">Signal</span><span className="font-black text-green-500">{results.best_signal.signal}</span></div>
                        <div className="flex justify-between"><span className="text-muted-foreground">Conf</span><span>{results.best_signal.confidence}%</span></div>
                        <div className="flex justify-between mt-2 pt-2 border-t border-green-500/20"><span className="font-bold">Return</span><span className="font-black text-green-500">+{results.best_signal.actual_return}%</span></div>
                      </div>
                    </div>
                  )}
                  {results.worst_signal && (
                    <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                      <h4 className="text-[10px] font-bold text-red-500 uppercase mb-2">Worst Signal</h4>
                      <div className="text-xs text-foreground space-y-1">
                        <div className="flex justify-between"><span className="text-muted-foreground">Date</span><span className="font-mono">{results.worst_signal.date}</span></div>
                        <div className="flex justify-between"><span className="text-muted-foreground">Signal</span><span className="font-black text-red-500">{results.worst_signal.signal}</span></div>
                        <div className="flex justify-between"><span className="text-muted-foreground">Conf</span><span>{results.worst_signal.confidence}%</span></div>
                        <div className="flex justify-between mt-2 pt-2 border-t border-red-500/20"><span className="font-bold">Return</span><span className="font-black text-red-500">{results.worst_signal.actual_return}%</span></div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
