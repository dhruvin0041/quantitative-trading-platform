import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, LineSeries, SeriesMarker, Time } from 'lightweight-charts';
import { ChartData } from '@/types';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';

interface PriceChartProps {
  data: ChartData | null;
  loading: boolean;
}

export function PriceChart({ data, loading }: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const bbUpperSeriesRef = useRef<any>(null);
  const bbLowerSeriesRef = useRef<any>(null);
  const ribbonUpperSeriesRef = useRef<any>(null);
  const ribbonLowerSeriesRef = useRef<any>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Use CSS variables for theming to automatically support light/dark mode
    const isDark = document.documentElement.classList.contains('dark');
    
    // Light theme defaults based on the new Design System
    const bgColor = isDark ? '#0F172A' : '#FFFFFF';
    const textColor = isDark ? '#64748B' : '#64748B';
    const gridColor = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(15,23,42,0.05)';
    const buyColor = '#10B981';
    const sellColor = '#F43F5E';
    const bbColor = '#6366F1'; // Analytics Blue

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: bgColor },
        textColor: textColor,
        fontFamily: 'var(--font-geist-mono), monospace',
      },
      grid: {
        vertLines: { color: gridColor },
        horzLines: { color: gridColor },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      timeScale: {
        borderColor: gridColor,
      },
      rightPriceScale: {
        borderColor: gridColor,
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
    });
    
    chartRef.current = chart;

    candlestickSeriesRef.current = chart.addSeries(CandlestickSeries, {
      upColor: buyColor, 
      downColor: sellColor, 
      borderVisible: false,
      wickUpColor: buyColor,
      wickDownColor: sellColor,
    });

    bbUpperSeriesRef.current = chart.addSeries(LineSeries, { color: bbColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });
    bbLowerSeriesRef.current = chart.addSeries(LineSeries, { color: bbColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false });
    ribbonUpperSeriesRef.current = chart.addSeries(LineSeries, { color: buyColor, lineWidth: 1, crosshairMarkerVisible: false });
    ribbonLowerSeriesRef.current = chart.addSeries(LineSeries, { color: sellColor, lineWidth: 1, crosshairMarkerVisible: false });

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
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
    if (!data || !data.candles || !chartRef.current) return;

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
        position: marker.action === 'BUY' ? 'belowBar' : 'aboveBar',
        color: marker.action === 'BUY' ? '#10B981' : '#F43F5E',
        shape: marker.action === 'BUY' ? 'arrowUp' : 'arrowDown',
        text: marker.action,
        size: 1,
      }));
      
      markers.sort((a, b) => new Date(a.time as string).getTime() - new Date(b.time as string).getTime());
      // lightweight-charts handles markers via createSeriesMarkers in vanilla, but since v4 it's setMarkers
      candlestickSeriesRef.current.setMarkers(markers);
    }

    chartRef.current.timeScale().fitContent();
  }, [data]);

  return (
    <Card className="w-full relative shadow-sm border-border overflow-hidden" data-tour="chart">
      <CardContent className="p-0 relative" aria-label="Interactive price chart with AI signals">
        {/* Loading Overlay */}
        {(loading && !data) && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/50 backdrop-blur-sm p-6">
            <Skeleton className="w-full h-full rounded-md" />
          </div>
        )}
        
        {/* Chart Container */}
        <div 
          ref={chartContainerRef} 
          className="w-full h-[400px]" 
        />
      </CardContent>
    </Card>
  );
}
