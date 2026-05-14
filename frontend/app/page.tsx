"use client";

import React, { useState, useEffect, useRef, ChangeEvent } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, createSeriesMarkers, LineSeries, SeriesMarker, Time } from 'lightweight-charts';

interface UniverseStock {
  ticker: string;
  name: string;
}

interface AIReport {
  Models: {
    Primary_Deep_Learning: {
      Suggested_Action: string;
      Confidence: string;
    };
    Secondary_XGBoost: {
      Suggested_Action: string;
      Confidence: string;
    };
  };
  Risk_Management: {
    Meta_Model_Status: string;
    Dynamic_10_Day_Range: {
      Low: number;
      High: number;
    };
  };
  Context: {
    Top_Headline_Processed: string;
  };
}

interface ChartData {
  candles: { time: string; open: number; high: number; low: number; close: number }[];
  clouds: { time: string; ribbon_upper: number; ribbon_lower: number; bb_upper: number; bb_lower: number }[];
  ai_report: AIReport;
  historical_markers: { time: string; action: string; probability: number; label?: string }[];
}

export default function HydraDashboard() {
  const [ticker, setTicker] = useState<string>("AAPL"); 
  const [universe, setUniverse] = useState<UniverseStock[]>([]);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // --- 1. LOAD THE S&P 500 ---
  useEffect(() => {
    fetch("http://localhost:8000/universe")
      .then(res => res.json())
      .then(data => {
        if (data.universe) setUniverse(data.universe);
      })
      .catch(err => console.error("Failed to load universe", err));
  }, []);

  // --- 2. FETCH AI PREDICTIONS ---
  useEffect(() => {
    fetch(`http://localhost:8000/predict?ticker=${ticker}`)
      .then(res => res.json())
      .then(data => {
        setChartData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Inference Engine Failed", err);
        setLoading(false);
      });
  }, [ticker]); 

  // --- 3. RENDER TRADINGVIEW CHART ---
  useEffect(() => {
    if (!chartContainerRef.current || !chartData || !chartData.candles) return;

    // Initialize Chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1e293b' }, // Tailwind slate-800
        textColor: '#cbd5e1', // Tailwind slate-300
      },
      grid: {
        vertLines: { color: '#334155' },
        horzLines: { color: '#334155' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981', 
      downColor: '#ef4444', 
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    // --- NEW: Add the Trend Ribbon (The Cloud) and Bollinger Bands ---
    if (chartData.clouds && chartData.clouds.length > 0) {
      // 1. BB Outer Bands (The blue dashed lines)
      const bbUpperSeries = chart.addSeries(LineSeries, { color: '#3b82f6', lineWidth: 1, lineStyle: 2 });
      const bbLowerSeries = chart.addSeries(LineSeries, { color: '#3b82f6', lineWidth: 1, lineStyle: 2 });
      
      bbUpperSeries.setData(chartData.clouds.map(c => ({ time: c.time, value: c.bb_upper })));
      bbLowerSeries.setData(chartData.clouds.map(c => ({ time: c.time, value: c.bb_lower })));

      // 2. The Ribbon (Fast/Slow EMA)
      const ribbonUpperSeries = chart.addSeries(LineSeries, { color: '#10b981', lineWidth: 1 });
      const ribbonLowerSeries = chart.addSeries(LineSeries, { color: '#ef4444', lineWidth: 1 });
      
      ribbonUpperSeries.setData(chartData.clouds.map(c => ({ time: c.time, value: c.ribbon_upper })));
      ribbonLowerSeries.setData(chartData.clouds.map(c => ({ time: c.time, value: c.ribbon_lower })));
    }

    // Set Price Data
    candlestickSeries.setData(chartData.candles);

    // Map AI Signals to Chart Markers (Arrow Style with Labels)
    if (chartData.historical_markers && chartData.historical_markers.length > 0) {
      const markers = chartData.historical_markers.map((marker) => ({
        time: marker.time,
        position: marker.action === 'BUY' ? 'belowBar' : 'aboveBar',
        color: marker.action === 'BUY' ? '#10b981' : '#ef4444',
        shape: marker.action === 'BUY' ? 'arrowUp' : 'arrowDown',
        text: marker.action, // Just "BUY" or "SELL"
      }));
      
      // TradingView requires markers to be sorted by time chronologically
      markers.sort((a, b) => new Date(a.time as string).getTime() - new Date(b.time as string).getTime());
      createSeriesMarkers(candlestickSeries, markers as SeriesMarker<Time>[]);
    }

    chart.timeScale().fitContent();

    // Cleanup chart on unmount or data refresh
    return () => chart.remove();
  }, [chartData]); // Re-draw chart whenever AI data changes

  const handleTickerChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setLoading(true);
    setTicker(e.target.value);
  };

  return (
    <div className="bg-slate-900 min-h-screen text-white p-8 font-sans">
      
      {/* THE CONTROL PANEL */}
      <div className="flex items-center justify-between mb-8 bg-slate-800 p-4 rounded-lg shadow-lg border border-slate-700">
        <h1 className="text-3xl font-bold tracking-wider text-cyan-400">HYDRA TERMINAL</h1>
        
        <div className="flex items-center space-x-4">
          <label className="text-gray-400 font-semibold text-sm tracking-widest uppercase">Target Asset:</label>
          <select 
            className="bg-slate-900 text-cyan-400 font-bold p-3 rounded-md focus:outline-none focus:ring-2 focus:ring-cyan-500 border border-slate-600 shadow-inner"
            value={ticker}
            onChange={handleTickerChange}
          >
            {universe.map((stock) => (
              <option key={stock.ticker} value={stock.ticker}>
                {stock.ticker} - {stock.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* THE CHART VIEW */}
      <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700 flex flex-col items-center justify-center relative min-h-[500px]">
        
        {loading && (
          <div className="absolute inset-0 z-10 bg-slate-900/80 flex flex-col items-center justify-center rounded-lg backdrop-blur-sm">
            <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-cyan-500 mb-4"></div>
            <div className="animate-pulse text-cyan-500 text-xl font-bold tracking-widest">
              HYDRA INFERENCE ENGINE RUNNING...
            </div>
          </div>
        )}

        <div className="w-full text-left mb-4 flex justify-between items-end">
           <p className="text-gray-400 text-sm tracking-widest uppercase">Live AI Analysis</p>
           <h2 className="text-3xl font-bold text-white">{ticker}</h2>
        </div>

        {/* This div is where TradingView injects the canvas chart */}
        <div ref={chartContainerRef} className="w-full h-[400px] border border-slate-700 rounded overflow-hidden" />
        
      </div>

      {/* AI QUANT SYSTEM REPORT */}
      {chartData && chartData.ai_report && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* PRIMARY & SECONDARY MODELS */}
          <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
            <h3 className="text-cyan-400 font-bold mb-4 flex items-center">
              <span className="mr-2">⚡</span> MODEL CONFIDENCE
            </h3>
            
            <div className="space-y-4">
              <div className="p-3 bg-slate-900 rounded border border-slate-700">
                <p className="text-xs text-gray-500 uppercase font-bold mb-1">Primary LSTM</p>
                <div className="flex justify-between items-center">
                  <span className={`text-lg font-bold ${chartData.ai_report.Models.Primary_Deep_Learning.Suggested_Action === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {chartData.ai_report.Models.Primary_Deep_Learning.Suggested_Action}
                  </span>
                  <span className="text-cyan-500 font-mono">{chartData.ai_report.Models.Primary_Deep_Learning.Confidence}</span>
                </div>
              </div>

              <div className="p-3 bg-slate-900 rounded border border-slate-700">
                <p className="text-xs text-gray-500 uppercase font-bold mb-1">XGBoost Ensemble</p>
                <div className="flex justify-between items-center">
                  <span className={`text-lg font-bold ${chartData.ai_report.Models.Secondary_XGBoost.Suggested_Action === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {chartData.ai_report.Models.Secondary_XGBoost.Suggested_Action}
                  </span>
                  <span className="text-cyan-500 font-mono">{chartData.ai_report.Models.Secondary_XGBoost.Confidence}</span>
                </div>
              </div>
            </div>
          </div>

          {/* RISK MANAGEMENT */}
          <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
            <h3 className="text-rose-400 font-bold mb-4 flex items-center">
              <span className="mr-2">🛡️</span> RISK STATUS
            </h3>
            <p className="text-sm text-gray-300 leading-relaxed italic mb-4">
              {chartData.ai_report.Risk_Management.Meta_Model_Status}
            </p>
            <div className="p-3 bg-slate-900 rounded border border-slate-700">
              <p className="text-xs text-gray-500 uppercase font-bold mb-2">10-Day Forecast Range</p>
              <div className="flex justify-between text-sm">
                <span>Low: <span className="text-rose-400 font-bold">${chartData.ai_report.Risk_Management.Dynamic_10_Day_Range.Low}</span></span>
                <span>High: <span className="text-emerald-400 font-bold">${chartData.ai_report.Risk_Management.Dynamic_10_Day_Range.High}</span></span>
              </div>
            </div>
          </div>

          {/* NEWS CONTEXT */}
          <div className="bg-slate-800 p-6 rounded-lg shadow-lg border border-slate-700">
            <h3 className="text-amber-400 font-bold mb-4 flex items-center">
              <span className="mr-2">📰</span> NEWS SENTIMENT
            </h3>
            <div className="h-[120px] overflow-y-auto pr-2 custom-scrollbar text-sm text-gray-400 leading-relaxed">
              {chartData.ai_report.Context.Top_Headline_Processed}
            </div>
          </div>

        </div>
      )}
    </div>
  );
}