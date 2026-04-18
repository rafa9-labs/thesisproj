import { useEffect, useRef } from "react";
import { createChart, type IChartApi, ColorType, LineSeries, HistogramSeries } from "lightweight-charts";

interface EquityCurveChartProps {
  data: { time: number; value: number }[];
  buyHoldData?: { time: number; value: number }[];
  drawdownData?: { time: number; value: number }[];
  height?: number;
}

export function EquityCurveChart({
  data,
  buyHoldData,
  drawdownData,
  height = 480,
}: EquityCurveChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

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
      crosshair: {
        mode: 0,
        vertLine: { color: "#363A45", width: 1, style: 2 },
        horzLine: { color: "#363A45", width: 1, style: 2 },
      },
      rightPriceScale: { borderColor: "#363A45" },
      timeScale: { borderColor: "#363A45", timeVisible: false },
    });

    chartRef.current = chart;

    const equitySeries = chart.addSeries(LineSeries, {
      color: "#089981",
      lineWidth: 2,
      title: "Equity",
    });
    equitySeries.setData(data as never[]);

    if (buyHoldData && buyHoldData.length > 0) {
      const bhSeries = chart.addSeries(LineSeries, {
        color: "#787B86",
        lineWidth: 1,
        lineStyle: 2,
        title: "Buy & Hold",
      });
      bhSeries.setData(buyHoldData as never[]);
    }

    if (drawdownData && drawdownData.length > 0) {
      const ddSeries = chart.addSeries(HistogramSeries, {
        color: "rgba(242, 54, 69, 0.3)",
        priceFormat: { type: "percent" },
        title: "Drawdown",
      });
      ddSeries.setData(drawdownData as never[]);
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [data, buyHoldData, drawdownData, height]);

  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border"
        style={{
          height,
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        No equity curve data
      </div>
    );
  }

  return (
    <div
      className="rounded-lg border overflow-hidden"
      style={{ borderColor: "var(--color-border)" }}
    >
      <div ref={containerRef} />
    </div>
  );
}
