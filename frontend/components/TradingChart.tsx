'use client';
import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries, createSeriesMarkers } from 'lightweight-charts';

export default function TradingChart() {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const [report, setReport] = useState<any>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        // 1. Initialize TradingView Lightweight Chart
        const chart = createChart(chartContainerRef.current, {
            layout: { 
                background: { type: ColorType.Solid, color: '#111827' }, 
                textColor: '#9CA3AF' 
            },
            grid: { 
                vertLines: { color: '#1F2937' }, 
                horzLines: { color: '#1F2937' } 
            },
            width: chartContainerRef.current.clientWidth,
            height: 500,
            timeScale: { timeVisible: true, borderColor: '#374151' }
        });

        const candlestickSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#10B981', 
            downColor: '#EF4444', 
            borderVisible: false, 
            wickUpColor: '#10B981', 
            wickDownColor: '#EF4444',
        });

        // 2. Fetch Data from the Python API
        const fetchSystemData = async () => {
            try {
                // The 't' parameter acts as a cache-buster
                const res = await fetch(`http://localhost:8000/predict?ticker=AAPL&t=${new Date().getTime()}`);
                const data = await res.json();
                
                // Save report for the UI Panel
                setReport(data.ai_report);

                // Draw the historical candles
                candlestickSeries.setData(data.candles);

                // 3. Draw All AI Markers (Historical + Live)
                const markers: any[] = [];
                const lastCandle = data.candles[data.candles.length - 1];

                // Plot Historical Backtest Signals
                if (data.historical_markers) {
                    data.historical_markers.forEach((marker: any) => {
                        // Prevent drawing a historical marker on today's live candle
                        if (marker.time !== lastCandle.time) {
                            markers.push({
                                time: marker.time,
                                position: marker.action === 'BUY' ? 'belowBar' : 'aboveBar',
                                color: marker.action === 'BUY' ? '#10B981' : '#EF4444',
                                shape: marker.action === 'BUY' ? 'arrowUp' : 'arrowDown',
                                text: `${marker.action} (${marker.probability}%)`
                            });
                        }
                    });
                }

                // Plot Today's Live Execution Signal
                const action = data.ai_report.Final_Execution.Action;
                if (action.includes('BUY')) {
                    markers.push({ 
                        time: lastCandle.time, 
                        position: 'belowBar', color: '#10B981', shape: 'arrowUp', 
                        text: `LIVE BUY @ ${data.ai_report.Current_Price}` 
                    });
                } else if (action.includes('SELL')) {
                    markers.push({ 
                        time: lastCandle.time, 
                        position: 'aboveBar', color: '#EF4444', shape: 'arrowDown', 
                        text: `LIVE SELL @ ${data.ai_report.Current_Price}` 
                    });
                }

                // Render all markers simultaneously
                createSeriesMarkers(candlestickSeries, markers);
                chart.timeScale().fitContent();

            } catch (error) {
                console.error("Hydra Engine Connection Failed:", error);
            }
        };

        fetchSystemData();

        // 4. Handle Window Resizing and Memory Cleanup
        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);

        return () => { 
            window.removeEventListener('resize', handleResize);
            chart.remove(); 
        };
    }, []);

    return (
        <div className="p-6 bg-gray-900 text-white rounded-2xl shadow-2xl max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
                Project Hydra Terminal
            </h2>
            
            {/* The TradingView Chart Canvas */}
            <div ref={chartContainerRef} className="w-full rounded-xl overflow-hidden border border-gray-700 shadow-inner" />
            
            {/* The AI Analysis Panel */}
            {report && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-gray-800 p-6 rounded-xl mt-6 border border-gray-700">
                    <div>
                        <p className="text-gray-400 uppercase tracking-wider text-xs font-semibold mb-1">Current Price</p>
                        <p className="text-4xl font-mono">${report.Current_Price}</p>
                    </div>
                    <div>
                        <p className="text-gray-400 uppercase tracking-wider text-xs font-semibold mb-1">System Execution</p>
                        <p className={`text-2xl font-bold ${
                            report.Final_Execution.Action.includes('BUY') ? 'text-emerald-400' : 
                            report.Final_Execution.Action.includes('SELL') ? 'text-red-400' : 'text-amber-400'
                        }`}>
                            {report.Final_Execution.Action}
                        </p>
                        <p className="text-xs text-gray-400 mt-2 leading-relaxed">{report.Risk_Management.Meta_Model_Status}</p>
                    </div>
                    <div>
                        <p className="text-gray-400 uppercase tracking-wider text-xs font-semibold mb-1">15-Day Dynamic Range</p>
                        <div className="flex items-end gap-2">
                            <span className="text-red-400 font-mono">${report.Risk_Management.Dynamic_10_Day_Range.Low}</span>
                            <span className="text-gray-600 mb-1">/</span>
                            <span className="text-emerald-400 font-mono">${report.Risk_Management.Dynamic_10_Day_Range.High}</span>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}