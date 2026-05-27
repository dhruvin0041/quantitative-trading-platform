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
  }, [resolvedTheme]);

  useEffect(() => {
    if (!data || !data.candles || !chartRef.current || !candlestickSeriesRef.current) return;

    const isDark = resolvedTheme === 'dark';
    const buyColor = isDark ? '#00E676' : '#1D7A3A';
    const sellColor = isDark ? '#FF5252' : '#C0380A';

    candlestickSeriesRef.current.setData(data.candles);

    if (data.clouds && data.clouds.length > 0 && bbUpperSeriesRef.current && bbLowerSeriesRef.current && ribbonUpperSeriesRef.current && ribbonLowerSeriesRef.current) {
      bbUpperSeriesRef.current.setData(data.clouds.filter(c => c.bb_upper !== null).map(c => ({ time: c.time, value: c.bb_upper as number })));
      bbLowerSeriesRef.current.setData(data.clouds.filter(c => c.bb_lower !== null).map(c => ({ time: c.time, value: c.bb_lower as number })));
      ribbonUpperSeriesRef.current.setData(data.clouds.filter(c => c.ribbon_upper !== null).map(c => ({ time: c.time, value: c.ribbon_upper as number })));
      ribbonLowerSeriesRef.current.setData(data.clouds.filter(c => c.ribbon_lower !== null).map(c => ({ time: c.time, value: c.ribbon_lower as number })));
    }

    if (data.historical_markers && data.historical_markers.length > 0) {
      // FIX 9: Chart Signal Saturation
      const filteredMarkers = data.historical_markers.filter(m => {
        // Drop VETOED/HOLD markers entirely from chart to reduce noise
        if (m.action === 'VETOED' || m.action === 'HOLD') return false;
        // Require institutional minimum confidence
        if (m.probability && m.probability < 70) return false;
        return true;
      }).sort((a,b) => new Date(a.time as string).getTime() - new Date(b.time as string).getTime());
      
      // Duplicate suppression / Minimum spacing (drop if same action within 5 days)
      const cleanedMarkers: typeof filteredMarkers = [];
      let lastAction = null;
      let lastTime = 0;
      
      for (const m of filteredMarkers) {
        const timeVal = new Date(m.time as string).getTime();
        if (lastAction === m.action && (timeVal - lastTime) < (5 * 24 * 60 * 60 * 1000)) {
           continue;
        }
        cleanedMarkers.push(m);
        lastAction = m.action;
        lastTime = timeVal;
      }

      const markers = cleanedMarkers.map((marker) => ({
        time: marker.time,
        position: (marker.action === 'BUY' ? 'belowBar' : 'aboveBar') as "belowBar" | "aboveBar",
        color: marker.action === 'BUY' ? buyColor : sellColor,
        shape: (marker.action === 'BUY' ? 'arrowUp' : 'arrowDown') as "arrowUp" | "arrowDown",
        text: marker.action,
        size: 1,
      }));
      
      createSeriesMarkers(candlestickSeriesRef.current, markers);
    } else {
      createSeriesMarkers(candlestickSeriesRef.current, []);
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