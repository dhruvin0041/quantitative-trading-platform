import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, LineSeries, createSeriesMarkers, IChartApi, ISeriesApi } from 'lightweight-charts';
import { ChartData } from '@/types';
import { Skeleton } from '@/components/ui/skeleton';

interface PriceChartProps {
  data: ChartData | null;
  loading: boolean;
}

export function PriceChart({ data, loading }: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const bbUpperSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bbLowerSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const ribbonUpperSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const ribbonLowerSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Use CSS variables for theming to automatically support light/dark mode
    const isDark = document.documentElement.classList.contains('dark');
    
    // Light theme defaults based on the new Design System
    const bgColor = isDark ? '#000000' : '#FFFFFF';
    const textColor = isDark ? '#94A3B8' : '#5C3D1E';
    const gridColor = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(240,217,188,0.2)';
    const buyColor = isDark ? '#10B981' : '#1D7A3A';
    const sellColor = isDark ? '#F43F5E' : '#C0380A';
    const accentColor = isDark ? '#3B82F6' : '#E8650A';

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: bgColor },
        textColor: textColor,
        fontFamily: "'Fira Code', monospace",
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: accentColor, labelBackgroundColor: accentColor },
        horzLine: { color: accentColor, labelBackgroundColor: accentColor },
      },
      timeScale: {
        borderColor: gridColor,
      },
      rightPriceScale: {
        borderColor: gridColor,
      },
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
    });
    
    chartRef.current = chart;

    candlestickSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: buyColor, 
      downColor: sellColor, 
      borderVisible: false,
      wickUpColor: buyColor,
      wickDownColor: sellColor,
    });

    bbUpperSeriesRef.current = chart.addSeries(LineSeries, { color: accentColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });
    bbLowerSeriesRef.current = chart.addSeries(LineSeries, { color: accentColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });
    ribbonUpperSeriesRef.current = chart.addSeries(LineSeries, { color: buyColor, lineWidth: 1, crosshairMarkerVisible: false });
    ribbonLowerSeriesRef.current = chart.addSeries(LineSeries, { color: sellColor, lineWidth: 1, crosshairMarkerVisible: false });

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ 
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight 
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!data || !data.candles || !chartRef.current || !candlestickSeriesRef.current) return;

    candlestickSeriesRef.current.setData(data.candles);

    if (data.clouds && data.clouds.length > 0 && bbUpperSeriesRef.current && bbLowerSeriesRef.current && ribbonUpperSeriesRef.current && ribbonLowerSeriesRef.current) {
      bbUpperSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.bb_upper })));
      bbLowerSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.bb_lower })));
      ribbonUpperSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.ribbon_upper })));
      ribbonLowerSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.ribbon_lower })));
    }

    if (data.historical_markers && data.historical_markers.length > 0) {
      const markers = data.historical_markers.map((marker) => ({
        time: marker.time,
        position: (marker.action === 'BUY' ? 'belowBar' : 'aboveBar') as "belowBar" | "aboveBar",
        color: marker.action === 'BUY' ? '#10B981' : '#F43F5E',
        shape: (marker.action === 'BUY' ? 'arrowUp' : 'arrowDown') as "arrowUp" | "arrowDown",
        text: marker.action,
        size: 1,
      }));
      
      markers.sort((a, b) => new Date(a.time as string).getTime() - new Date(b.time as string).getTime());
      // lightweight-charts v5+ uses createSeriesMarkers standalone function
      createSeriesMarkers(candlestickSeriesRef.current, markers);
    }

    chartRef.current.timeScale().fitContent();
  }, [data]);

  return (
    <div className="w-full h-full relative" data-tour="chart">
      {/* Loading Overlay */}
      {(loading && !data) && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50 backdrop-blur-md">
          <Skeleton className="w-full h-full bg-white/5" />
        </div>
      )}
      
      {/* Chart Container */}
      <div 
        ref={chartContainerRef} 
        className="w-full h-full absolute inset-0" 
      />
    </div>
  );
}
