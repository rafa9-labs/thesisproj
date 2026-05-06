import { useEffect, useRef } from "react";
import { createChart, type IChartApi, CandlestickSeries, HistogramSeries, LineSeries, ColorType } from "lightweight-charts";
import { useTradeChartData } from "@/api/queries";

interface BacktestChartProps {
  jobId: string;
  model: string;
  height?: number;
}

export function BacktestChart({ jobId, model, height = 520 }: BacktestChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const { data, isLoading } = useTradeChartData(jobId, model);

  useEffect(() => {
    if (!containerRef.current) return;
    if (isLoading || !data) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const candles = data.candles ?? [];
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
      },
      rightPriceScale: {
        borderColor: "#2A2E39",
        scaleMargins: { top: 0.05, bottom: 0.3 },
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

    const markers = (data.trades ?? []).map((t) => ({
      time: t.direction === "BUY" ? t.entry_time : t.exit_time,
      position: t.direction === "BUY" ? ("belowBar" as const) : ("aboveBar" as const),
      color: t.direction === "BUY" ? "#26a69a" : "#ef5350",
      shape: t.direction === "BUY" ? ("arrowUp" as const) : ("arrowDown" as const),
      text: `${(t.pnl_pct ?? 0) > 0 ? "+" : ""}${(t.pnl_pct ?? 0).toFixed(2)}%`,
      size: 2,
    }));

    if (markers.length > 0) {
      candleSeries.setMarkers(markers);
    }

    chart.timeScale().fitContent();

    const equity = data.equity_curve ?? [];
    if (equity.length > 0) {
      const equitySeries = chart.addSeries(LineSeries, {
        color: "#42a5f5",
        lineWidth: 2,
        priceScaleId: "equity",
      });

      chart.priceScale("equity").applyOptions({
        scaleMargins: { top: 0.8, bottom: 0.05 },
        visible: true,
        borderColor: "#2A2E39",
      });

      equitySeries.setData(
        equity.map((e) => ({
          time: e.time as number,
          value: e.value,
        })),
      );
    }

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
  }, [data, isLoading, height]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>
          Trade Visualization
        </span>
        <span className="text-[10px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
          {model} — {data?.pair} {data?.timeframe} · {data?.trades?.length ?? 0} trades
        </span>
      </div>

      <div
        ref={containerRef}
        className="w-full rounded-lg overflow-hidden border"
        style={{ borderColor: "var(--color-glass-border)", backgroundColor: "#131722", minHeight: height }}
      >
        {isLoading && (
          <div className="flex items-center justify-center" style={{ height }}>
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Loading trade chart...</span>
          </div>
        )}
        {!isLoading && (data?.candles ?? []).length === 0 && (
          <div className="flex items-center justify-center" style={{ height }}>
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>No chart data available</span>
          </div>
        )}
      </div>
    </div>
  );
}