import { useEffect, useRef } from "react";
import {
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  ColorType,
} from "lightweight-charts";
import { TIMEFRAMES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { CandleData } from "@/hooks/useChartStream";

export interface OverlayLine {
  data: { time: number; value: number }[];
  color: string;
  label?: string;
}

export interface ChartMarker {
  time: number;
  position: "aboveBar" | "belowBar" | "inBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle";
  text: string;
  size?: number;
}

interface CandlestickChartProps {
  pair: string;
  timeframe?: string;
  historical: CandleData[];
  liveBar: CandleData | null;
  isLoading?: boolean;
  height?: number;
  onTimeframeChange?: (tf: string) => void;
  overlayLines?: OverlayLine[];
  showVolume?: boolean;
  showToolbar?: boolean;
  chartMarkers?: ChartMarker[];
  onRequestOlder?: () => void;
  hasOlder?: boolean;
}

export function CandlestickChart({
  pair,
  timeframe = "M30",
  historical,
  liveBar,
  isLoading = false,
  height,
  onTimeframeChange,
  overlayLines,
  showVolume = true,
  showToolbar = true,
  chartMarkers,
  onRequestOlder,
  hasOlder = false,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const lineSeriesRefs = useRef<ISeriesApi<"Line">[]>([]);
  const seriesMarkersRef = useRef<ISeriesMarkersPluginApi<number> | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);
  const markersRef = useRef<ChartMarker[]>([]);
  const didInitialLoad = useRef(false);
  const onRequestOlderRef = useRef(onRequestOlder);

  useEffect(() => {
    onRequestOlderRef.current = onRequestOlder;
  });

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const chartHeight = height ?? container.clientHeight;

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: "#0F172A" },
        textColor: "#787B86",
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#334155" },
        horzLines: { color: "#334155" },
      },
      width: container.clientWidth,
      height: chartHeight,
      crosshair: { mode: 0 },
      timeScale: {
        borderColor: "#334155",
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "#334155",
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10B981",
      downColor: "#F43F5E",
      borderUpColor: "#10B981",
      borderDownColor: "#F43F5E",
      wickUpColor: "#10B981",
      wickDownColor: "#F43F5E",
    });
    candleSeriesRef.current = candleSeries;

    const markersPlugin = createSeriesMarkers(candleSeries, []);
    seriesMarkersRef.current = markersPlugin;

    if (showVolume) {
      const volSeries = chart.addSeries(HistogramSeries, {
        color: "#10B98160",
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
        lastValueVisible: false,
        priceLineVisible: false,
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
        visible: false,
      });
      volSeriesRef.current = volSeries;
    }

    const observer = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
          ...(height == null ? { height: containerRef.current.clientHeight } : {}),
        });
      }
    });
    observer.observe(container);
    observerRef.current = observer;

    chartRef.current = chart;

    return () => {
      markersPlugin.detach();
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volSeriesRef.current = null;
      lineSeriesRefs.current = [];
      seriesMarkersRef.current = null;
      didInitialLoad.current = false;
    };
  }, []);

  useEffect(() => {
    if (!candleSeriesRef.current || isLoading || historical.length === 0) return;

    const seen = new Set<number>();
    const ohlc = historical
      .map((c) => ({
        time: c.time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
      .filter((c) => {
        if (seen.has(c.time)) return false;
        seen.add(c.time);
        return true;
      })
      .sort((a, b) => a.time - b.time);
    candleSeriesRef.current.setData(ohlc);

    if (!didInitialLoad.current) {
      chartRef.current?.timeScale().fitContent();
      didInitialLoad.current = true;
    }

    if (showVolume && volSeriesRef.current) {
      const volSeen = new Set<number>();
      volSeriesRef.current.setData(
        historical
          .map((c) => ({
            time: c.time,
            value: c.volume || 0,
            color: c.close >= c.open ? "#10B98140" : "#F43F5E40",
          }))
          .filter((v) => {
            if (volSeen.has(v.time)) return false;
            volSeen.add(v.time);
            return true;
          })
          .sort((a, b) => a.time - b.time),
      );
    }
  }, [historical, isLoading, showVolume]);

  useEffect(() => {
    if (!candleSeriesRef.current || !liveBar) return;
    candleSeriesRef.current.update({
      time: liveBar.time,
      open: liveBar.open,
      high: liveBar.high,
      low: liveBar.low,
      close: liveBar.close,
    });
  }, [liveBar]);

  useEffect(() => {
    if (!chartRef.current || !hasOlder) return;

    const timeScale = chartRef.current.timeScale();
    let fetching = false;
    let debounce: ReturnType<typeof setTimeout> | null = null;

    const handler = (range: { from: number; to: number } | null) => {
      if (!range || fetching) return;
      if (range.from < 50) {
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(async () => {
          fetching = true;
          await onRequestOlderRef.current?.();
          fetching = false;
        }, 300);
      }
    };

    timeScale.subscribeVisibleLogicalRangeChange(handler);

    return () => {
      timeScale.unsubscribeVisibleLogicalRangeChange(handler);
      if (debounce) clearTimeout(debounce);
    };
  }, [hasOlder]);

  useEffect(() => {
    if (!chartRef.current) return;

    for (const ls of lineSeriesRefs.current) {
      chartRef.current.removeSeries(ls);
    }
    lineSeriesRefs.current = [];

    if (overlayLines && overlayLines.length > 0) {
      for (const line of overlayLines) {
        const lineSeries = chartRef.current.addSeries(LineSeries, {
          color: line.color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
          priceScaleId: "overlay",
        });
        lineSeries.setData(line.data);
        lineSeriesRefs.current.push(lineSeries);
      }
      chartRef.current.priceScale("overlay").applyOptions({
        visible: true,
        borderColor: "#334155",
        scaleMargins: { top: 0.3, bottom: 0 },
        mode: 0,
      });
    }
  }, [overlayLines]);

  useEffect(() => {
    if (!chartMarkers) return;
    markersRef.current = chartMarkers;
    seriesMarkersRef.current?.setMarkers(chartMarkers);
  }, [chartMarkers]);

  useEffect(() => {
    if (chartRef.current && containerRef.current && height != null) {
      chartRef.current.applyOptions({ height });
    }
  }, [height]);

  const bars = (historical?.length ?? 0) + (liveBar ? 1 : 0);

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {showToolbar && (
        <div className="flex items-center gap-1.5">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.key}
              onClick={() => onTimeframeChange?.(tf.key)}
              className={cn(
                "rounded-md border px-2.5 py-0.5 font-mono text-[10px] font-medium uppercase transition-all duration-200",
                timeframe === tf.key
                  ? "border-(--color-brand) bg-(--color-brand-glow) text-(--color-brand)"
                  : "border-(--color-glass-border) bg-transparent text-(--color-text-muted) hover:border-(--color-text-dim)",
              )}
            >
              {tf.label}
            </button>
          ))}
          <span className="ml-2 font-mono text-[10px] text-(--color-text-muted) tabular-nums">
            {pair} {timeframe}
          </span>
        </div>
      )}

      <div
        ref={containerRef}
        className="w-full overflow-hidden rounded-sm border border-(--color-glass-border) bg-(--color-app)"
        style={height != null ? { minHeight: height } : { flex: 1, minHeight: 200 }}
      >
        {isLoading && (
          <div className="flex items-center justify-center" style={{ height: height ?? "100%" }}>
            <span className="text-xs text-(--color-text-muted)">Loading candles...</span>
          </div>
        )}
        {!isLoading && bars === 0 && (
          <div className="flex flex-col items-center justify-center gap-1" style={{ height: height ?? "100%" }}>
            <span className="text-xs text-(--color-text-muted)">
              No data for {pair} at {timeframe}
            </span>
            <span className="text-[10px] text-(--color-text-dim)">
              Try a higher timeframe in the toolbar above
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
