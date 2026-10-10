import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, LineSeries, HistogramSeries, createSeriesMarkers, IChartApi, ISeriesApi, Time, SeriesMarker, ISeriesMarkersPluginApi } from 'lightweight-charts';
import { ChartData } from '@/types';
import { Skeleton } from '@/components/ui/skeleton';
import { useTheme } from 'next-themes';
import { cn } from '@/lib/utils';

interface PriceChartProps {
  data: ChartData | null;
  loading: boolean;
}

export function PriceChart({ data, loading }: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const bbUpperSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bbLowerSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const ribbonUpperSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const ribbonLowerSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  // const trailingStopSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const forecastP90Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const forecastP50Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const forecastP10Ref = useRef<ISeriesApi<"Line"> | null>(null);
  const seriesMarkersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const { resolvedTheme } = useTheme();
  const [legendContent, setLegendContent] = React.useState<React.ReactNode>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const isDark = resolvedTheme === 'dark';
    
    // Theme-specific colors - preserving existing structural colors
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
        scaleMargins: {
          top: 0.08,
          bottom: 0.22,
        },
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

    volumeSeriesRef.current = chart.addSeries(HistogramSeries, {
      color: '#26a69a',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume_scale',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    chart.priceScale('volume_scale').applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
      visible: false,
    });

    bbUpperSeriesRef.current = chart.addSeries(LineSeries, { color: supportLineColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
    bbLowerSeriesRef.current = chart.addSeries(LineSeries, { color: supportLineColor, lineWidth: 1, lineStyle: 2, crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
    ribbonUpperSeriesRef.current = chart.addSeries(LineSeries, { color: maOrange, lineWidth: 1, crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
    ribbonLowerSeriesRef.current = chart.addSeries(LineSeries, { color: maBlue, lineWidth: 1, crosshairMarkerVisible: false, autoscaleInfoProvider: () => null });
    // trailingStopSeriesRef.current = chart.addSeries(LineSeries, { color: '#FF5722', lineWidth: 2, lineStyle: 3, crosshairMarkerVisible: true, autoscaleInfoProvider: () => null });

    forecastP90Ref.current = chart.addSeries(LineSeries, { color: isDark ? 'rgba(0, 230, 118, 0.6)' : 'rgba(29, 122, 58, 0.6)', lineWidth: 2, lineStyle: 2, crosshairMarkerVisible: true });
    forecastP50Ref.current = chart.addSeries(LineSeries, { color: isDark ? 'rgba(79, 195, 247, 0.8)' : 'rgba(79, 195, 247, 0.8)', lineWidth: 2, lineStyle: 0, crosshairMarkerVisible: true });
    forecastP10Ref.current = chart.addSeries(LineSeries, { color: isDark ? 'rgba(255, 82, 82, 0.6)' : 'rgba(192, 56, 10, 0.6)', lineWidth: 2, lineStyle: 2, crosshairMarkerVisible: true });

    const resizeObserver = new ResizeObserver((entries) => {
      if (chartRef.current && entries.length > 0) {
        const newRect = entries[0].contentRect;
        chartRef.current.applyOptions({ 
          width: newRect.width,
          height: newRect.height 
        });
      }
    });

    const container = chartContainerRef.current;
    if (container) resizeObserver.observe(container);

    return () => {
      if (container) resizeObserver.unobserve(container);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesMarkersRef.current = null;
    };
  }, [resolvedTheme]);

  useEffect(() => {
    if (!data || !data.candles || !chartRef.current || !candlestickSeriesRef.current) return;

    // Deduplicate and chronologically order candles to guarantee stability
    type CandleItem = NonNullable<ChartData['candles']>[number];
    type CloudItem = NonNullable<ChartData['clouds']>[number];
    type MarkerItem = NonNullable<ChartData['historical_markers']>[number];

    const candleMap = new Map<string, CandleItem>();
    for (const c of data.candles) {
      if (c && c.time) candleMap.set(String(c.time), c);
    }
    const sortedCandles = Array.from(candleMap.values()).sort(
      (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime()
    );

    candlestickSeriesRef.current.setData(sortedCandles);

    if (volumeSeriesRef.current) {
      const volumeData = sortedCandles.map((c: CandleItem) => ({
        time: c.time,
        value: c.volume || 0,
        color: (c.close >= c.open) ? 'rgba(0, 230, 118, 0.35)' : 'rgba(255, 82, 82, 0.35)',
      }));
      volumeSeriesRef.current.setData(volumeData);
    }

    if (data.clouds && data.clouds.length > 0 && bbUpperSeriesRef.current && bbLowerSeriesRef.current && ribbonUpperSeriesRef.current && ribbonLowerSeriesRef.current) {
      const cloudMap = new Map<string, CloudItem>();
      for (const cl of data.clouds) {
        if (cl && cl.time) cloudMap.set(String(cl.time), cl);
      }
      const sortedClouds = Array.from(cloudMap.values()).sort(
        (a, b) => new Date(a.time).getTime() - new Date(b.time).getTime()
      );

      bbUpperSeriesRef.current.setData(sortedClouds.filter(c => c.bb_upper !== null && c.bb_upper !== undefined).map(c => ({ time: c.time, value: c.bb_upper as number })));
      bbLowerSeriesRef.current.setData(sortedClouds.filter(c => c.bb_lower !== null && c.bb_lower !== undefined).map(c => ({ time: c.time, value: c.bb_lower as number })));
      ribbonUpperSeriesRef.current.setData(sortedClouds.filter(c => c.ribbon_upper !== null && c.ribbon_upper !== undefined).map(c => ({ time: c.time, value: c.ribbon_upper as number })));
      ribbonLowerSeriesRef.current.setData(sortedClouds.filter(c => c.ribbon_lower !== null && c.ribbon_lower !== undefined).map(c => ({ time: c.time, value: c.ribbon_lower as number })));
    }

    if (data.forecast_fan && forecastP90Ref.current && forecastP50Ref.current && forecastP10Ref.current) {
      forecastP90Ref.current.setData(data.forecast_fan.map((f: { time: string; p90: number }) => ({ time: f.time, value: f.p90 })));
      forecastP50Ref.current.setData(data.forecast_fan.map((f: { time: string; p50: number }) => ({ time: f.time, value: f.p50 })));
      forecastP10Ref.current.setData(data.forecast_fan.map((f: { time: string; p10: number }) => ({ time: f.time, value: f.p10 })));
    }

    // Zero-Repainting Mandate: Confirmed markers ONLY from closed candles
    const rawMarkers = (Array.isArray(data.historical_markers) && data.historical_markers.length > 0)
      ? data.historical_markers
      : (Array.isArray(data.markers) && data.markers.length > 0 ? data.markers : []);

    const markerMap = new Map<string, MarkerItem>();
    for (const m of rawMarkers) {
      const act = m?.action || m?.signal;
      if ((act === 'BUY' || act === 'SELL') && !m?.is_provisional && m?.time) {
        markerMap.set(String(m.time), m);
      }
    }
    const validMarkers = Array.from(markerMap.values())
      .sort((a, b) => new Date(a.time as string).getTime() - new Date(b.time as string).getTime());

    const markers: SeriesMarker<Time>[] = validMarkers.map((marker: { action?: string; signal?: string; time: string | number }) => {
      const act = marker.action || marker.signal;
      const isBuy = act === 'BUY';
      return {
        time: String(marker.time) as Time,
        position: (isBuy ? "belowBar" : "aboveBar") as "belowBar" | "aboveBar",
        color: isBuy ? "#10B981" : "#EF4444",
        shape: (isBuy ? "arrowUp" : "arrowDown") as "arrowUp" | "arrowDown",
        text: "",
        size: 1.2,
      };
    });

    // If an intraday forming candle has an estimate, render it separately with explicit PROVISIONAL styling
    if (data.provisional_marker && data.provisional_marker.action) {
      const prov = data.provisional_marker;
      markers.push({
        time: String(prov.time) as Time,
        position: prov.position as "belowBar" | "aboveBar",
        color: "#F59E0B", // Amber warning color
        shape: prov.shape as "arrowUp" | "arrowDown",
        text: "PROV",
        size: 1,
      });
    }

    if (!seriesMarkersRef.current) {
      seriesMarkersRef.current = createSeriesMarkers(candlestickSeriesRef.current, markers);
    } else {
      seriesMarkersRef.current.setMarkers(markers);
    }

    chartRef.current.timeScale().fitContent();

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleCrosshairMove = (param: any) => {
      if (!param.time || param.point.x < 0 || param.point.y < 0 || param.point.x > chartContainerRef.current!.clientWidth || param.point.y > chartContainerRef.current!.clientHeight) {
        setLegendContent(null);
        return;
      }
      // Check provisional marker first
      if (data.provisional_marker && data.provisional_marker.time === param.time) {
        const prov = data.provisional_marker;
        setLegendContent(
          <div className="flex flex-col gap-1 border-l-2 border-amber-500 pl-2">
            <span className="text-[12px] font-bold uppercase text-amber-500 tracking-wider">
              PROVISIONAL {prov.action} (UNCONFIRMED)
            </span>
            <span className="text-[10px] font-mono text-amber-400/90">
              STATE: INTRADAY FORMING CANDLE
            </span>
            <span className="text-[10px] font-mono text-muted-foreground">
              Source: Live temporary tick {prov.source_candle_timestamp || ""}
            </span>
            <span className="text-[10px] text-muted-foreground italic">
              * Non-binding estimate. Confirmed only at 16:00 EST close.
            </span>
          </div>
        );
        return;
      }

      const allMarkersList = (Array.isArray(data.historical_markers) && data.historical_markers.length > 0)
        ? data.historical_markers
        : (Array.isArray(data.markers) && data.markers.length > 0 ? data.markers : []);
      if (allMarkersList.length > 0) {
        const marker = allMarkersList.find((m: { time: string | number }) => m.time === param.time);
        const action = marker ? (marker.action || marker.signal) : null;
        if (marker && action && action !== 'HOLD') {
          const isVeto = typeof action === 'string' && (action.includes('VETO') || action === 'VAR_LIMIT_BREACH');
          const isBuy = action === 'BUY';
          setLegendContent(
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <span className={cn("text-[12px] font-bold uppercase", isVeto ? "text-warning" : (isBuy ? "text-positive" : "text-negative"))}>
                  {isVeto ? 'RISK AGENT VETO' : `CONFIRMED ${action} SIGNAL`}
                </span>
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-positive/10 text-positive font-mono uppercase">
                  LOCKED
                </span>
              </div>
              <span className="text-[11px] font-mono text-muted-foreground">Confidence: {marker.probability || Math.round((marker.confidence || 0) * 100)}%</span>
              {marker.source_candle_timestamp && (
                <span className="text-[10px] font-mono text-muted-foreground">Source Bar: {marker.source_candle_timestamp}</span>
              )}
              {marker.execution_timestamp && (
                <span className="text-[10px] font-mono text-muted-foreground">Execution: {marker.execution_timestamp}</span>
              )}
              {marker.execution_price && (
                <span className="text-[10px] font-mono text-muted-foreground">Exec Price: ${marker.execution_price}</span>
              )}
              {marker.label && <span className="text-[11px] text-muted-foreground">{marker.label}</span>}
              {isVeto && <span className="text-[10px] text-negative font-mono mt-1 border-t border-border pt-1">Reason: Sector Crowding / VaR</span>}
            </div>
          );
          return;
        }
      }
      setLegendContent(null);
    };

    chartRef.current.subscribeCrosshairMove(handleCrosshairMove);

    return () => {
      if (chartRef.current) {
        chartRef.current.unsubscribeCrosshairMove(handleCrosshairMove);
      }
    };
  }, [data, resolvedTheme]);

  return (
    <div className="absolute inset-0 w-full h-full" data-tour="chart">
      {/* Loading Overlay */}
      {(loading && !data) && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50 backdrop-blur-md rounded-xl">
          <Skeleton className="w-full h-full bg-white/5" />
        </div>
      )}
      
      {/* Chart Container */}
      <div ref={chartContainerRef} className="w-full h-full" />
      
      {/* Legend Container */}
      {legendContent && (
        <div className="absolute top-4 left-4 z-20 bg-card/90 border border-border p-3 rounded-lg shadow-xl backdrop-blur-sm pointer-events-none transition-opacity">
          {legendContent}
        </div>
      )}
    </div>
  );
}