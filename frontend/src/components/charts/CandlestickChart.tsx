import { useEffect, useRef, useState } from "react";
import { createChart, type IChartApi, CandlestickSeries, HistogramSeries, LineSeries, ColorType } from "lightweight-charts";
import { useCandles } from "@/api/queries";
import { TIMEFRAMES } from "@/lib/constants";

export interface OverlayLine {
  data: { time: number; value: number }[];
  color: string;
  label?: string;
}

interface CandlestickChartProps {
  pair: string;
  timeframe?: string;
  limit?: number;
  height?: number;
  onTimeframeChange?: (tf: string) => void;
  overlayLines?: OverlayLine[];
  showVolume?: boolean;
  showToolbar?: boolean;
}

export function CandlestickChart({
  pair,
  timeframe = "M30",
  limit = 200,
  height = 460,
  onTimeframeChange,
  overlayLines,
  showVolume = true,
  showToolbar = true,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [activeTf, setActiveTf] = useState(timeframe);

  const { data, isLoading } = useCandles(pair, activeTf, limit);

  const handleTimeframeChange = (tf: string) => {
    setActiveTf(tf);
    onTimeframeChange?.(tf);
  };

  useEffect(() => {
    if (!containerRef.current) return;
    if (isLoading) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const candles = data?.candles ?? [];
    if (candles.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#131722" },
        textColor: "#80899F",
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#2A2E39" },
        horzLines: { color: "#2A2E39" },
      },
      width: containerRef.current.clientWidth,
      height,
      crosshair: { mode: 0 },
      timeScale: {
        borderColor: "#2A2E39",
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "#2A2E39",
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderUpColor: "#26a69a",
      borderDownColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });

    candleSeries.setData(
      candles.map((c) => ({
        time: c.t as number,
        open: c.o,
        high: c.h,
        low: c.l,
        close: c.c,
      })),
    );

    if (overlayLines && overlayLines.length > 0) {
      for (const line of overlayLines) {
        const lineSeries = chart.addSeries(LineSeries, {
          color: line.color,
          lineWidth: 2,
          priceLineVisible: false,
          lastValueVisible: false,
          crosshairMarkerVisible: false,
        });
        lineSeries.setData(line.data);
      }
    }

    if (showVolume) {
      const volSeries = chart.addSeries(HistogramSeries, {
        color: "#26a69a60",
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });

      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
        visible: false,
      });

      volSeries.setData(
        candles.map((c) => ({
          time: c.t as number,
          value: c.volume || 0,
          color: c.c >= c.o ? "#26a69a40" : "#ef535040",
        })),
      );
    }

    chart.timeScale().fitContent();

    const observer = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    observer.observe(containerRef.current);

    chartRef.current = chart;

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, isLoading, height, pair, overlayLines, showVolume]);

  return (
    <div className="flex flex-col gap-2">
      {showToolbar && (
        <div className="flex items-center gap-1.5">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.key}
              onClick={() => handleTimeframeChange(tf.key)}
              className="rounded-md border px-2.5 py-0.5 text-[10px] font-medium uppercase transition-all duration-200"
              style={{
                borderColor: activeTf === tf.key ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: activeTf === tf.key ? "var(--color-brand-glow)" : "transparent",
                color: activeTf === tf.key ? "var(--color-brand)" : "var(--color-text-muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {tf.label}
            </button>
          ))}
          <span className="text-[10px] ml-2" style={{ color: "var(--color-text-muted)" }}>
            {pair} {activeTf}
          </span>
        </div>
      )}

      <div
        ref={containerRef}
        className="w-full rounded-sm overflow-hidden border"
        style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-app)", minHeight: height }}
      >
        {isLoading && (
          <div className="flex items-center justify-center" style={{ height }}>
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Loading candles...</span>
          </div>
        )}
        {!isLoading && (data?.candles ?? []).length === 0 && (
          <div className="flex items-center justify-center" style={{ height }}>
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              No data for {pair} at {activeTf}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}