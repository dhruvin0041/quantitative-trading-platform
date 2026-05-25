import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, LineSeries, createSeriesMarkers, IChartApi, ISeriesApi } from 'lightweight-charts';
import { ChartData } from '@/types';
import { Skeleton } from '@/components/ui/skeleton';
import { useTheme } from 'next-themes';

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
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const isDark = resolvedTheme === 'dark';
    
    // Theme-specific colors
    const bgColor = isDark ? 'rgba(15, 15, 15, 1)' : '#FDFAF5';
    const textColor = isDark ? '#94A3B8' : '#5C3D1E';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0,0,0,0.06)';
    const buyColor = isDark ? '#00E676' : '#1D7A3A';
    const sellColor = isDark ? '#FF5252' : '#C0380A';
    const maOrange = '#FF8C38';
    const maBlue = '#4FC3F7';
    const supportLineColor = isDark ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.3)';

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
        vertLine: { color: isDark ? maBlue : '#E8650A', labelBackgroundColor: isDark ? maBlue : '#E8650A' },
        horzLine: { color: isDark ? maBlue : '#E8650A', labelBackgroundColor: isDark ? maBlue : '#E8650A' },
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

    bbUpperSeriesRef.current = chart.addSeries(LineSeries, { color: supportLineColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });
    bbLowerSeriesRef.current = chart.addSeries(LineSeries, { color: supportLineColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });
    ribbonUpperSeriesRef.current = chart.addSeries(LineSeries, { color: maOrange, lineWidth: 1, crosshairMarkerVisible: false });
    ribbonLowerSeriesRef.current = chart.addSeries(LineSeries, { color: maBlue, lineWidth: 1, crosshairMarkerVisible: false });

    // Re-apply data if available
    if (data && data.candles) {
      candlestickSeriesRef.current.setData(data.candles);
      
      if (data.clouds && data.clouds.length > 0) {
        bbUpperSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.bb_upper })));
        bbLowerSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.bb_lower })));
        ribbonUpperSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.ribbon_upper })));
        ribbonLowerSeriesRef.current.setData(data.clouds.map(c => ({ time: c.time, value: c.ribbon_lower })));
      }

      if (data.historical_markers && data.historical_markers.length > 0) {
        const markers = data.historical_markers.map((marker) => ({
          time: marker.time,
          position: (marker.action === 'BUY' ? 'belowBar' : 'aboveBar') as "belowBar" | "aboveBar",
          color: marker.action === 'BUY' ? buyColor : sellColor,
          shape: (marker.action === 'BUY' ? 'arrowUp' : 'arrowDown') as "arrowUp" | "arrowDown",
          text: marker.action,
          size: 1,
        }));
        markers.sort((a, b) => new Date(a.time as string).getTime() - new Date(b.time as string).getTime());
        createSeriesMarkers(candlestickSeriesRef.current, markers);
      }
      chart.timeScale().fitContent();
    }

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
  }, [resolvedTheme, data, loading]);

  useEffect(() => {
    if (!data || !data.candles || !chartRef.current || !candlestickSeriesRef.current) return;

    const isDark = resolvedTheme === 'dark';
    const buyColor = isDark ? '#00E676' : '#1D7A3A';
    const sellColor = isDark ? '#FF5252' : '#C0380A';

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
        color: marker.action === 'BUY' ? buyColor : sellColor,
        shape: (marker.action === 'BUY' ? 'arrowUp' : 'arrowDown') as "arrowUp" | "arrowDown",
        text: marker.action,
        size: 1,
      }));
      
      markers.sort((a, b) => new Date(a.time as string).getTime() - new Date(b.time as string).getTime());
      createSeriesMarkers(candlestickSeriesRef.current, markers);
    }

    chartRef.current.timeScale().fitContent();
  }, [data, resolvedTheme]);

  return (
    <div className="w-full relative aspect-[21/9]" data-tour="chart">
      {/* Loading Overlay */}
      {(loading && !data) && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50 backdrop-blur-md rounded-xl">
          <Skeleton className="w-full h-full bg-white/5" />
        </div>
      )}
      
      {/* Chart Container */}
      <div 
        ref={chartContainerRef} 
        className="w-full h-full" 
      />
    </div>
  );
}

