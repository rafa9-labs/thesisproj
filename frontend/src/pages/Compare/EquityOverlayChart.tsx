import { useEffect, useRef } from "react";
import { createChart, type IChartApi, ColorType, LineSeries } from "lightweight-charts";
import type { EquityPoint } from "@/api/schemas";

interface ModelCurve {
  model: string;
  data: EquityPoint[];
}

interface EquityOverlayChartProps {
  curves: ModelCurve[];
  height?: number;
}

const PALETTE = [
  "#089981",
  "#2962FF",
  "#7C3AED",
  "#F59E0B",
  "#EC4899",
  "#06B6D4",
  "#FF9800",
  "#E91E63",
];

export function EquityOverlayChart({ curves, height = 400 }: EquityOverlayChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const validCurves = curves.filter((c) => c.data.length > 0);

  useEffect(() => {
    if (!containerRef.current || validCurves.length === 0) return;

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

    for (let i = 0; i < validCurves.length; i++) {
      const curve = validCurves[i];
      const color = PALETTE[i % PALETTE.length];
      const series = chart.addSeries(LineSeries, {
        color,
        lineWidth: 2,
        title: curve.model,
      });
      series.setData(curve.data as never[]);
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
  }, [validCurves, height]);

  if (validCurves.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-sm border p-8"
        style={{
          height,
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        <span className="text-sm" style={{ fontFamily: "var(--font-mono)" }}>
          No equity curve data per model
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <h3
        className="text-xs font-semibold uppercase tracking-[0.08em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Equity Overlay
      </h3>
      <div
        className="rounded-sm border overflow-hidden relative"
        style={{ borderColor: "var(--color-border)" }}
      >
        <div
          className="absolute top-3 left-3 z-10 flex flex-col gap-1 rounded-md px-3 py-2"
          style={{
            backgroundColor: "rgba(19,23,34,0.85)",
            border: "1px solid var(--color-border)",
          }}
        >
          {validCurves.map((curve, i) => (
            <div key={curve.model} className="flex items-center gap-2">
              <span
                className="inline-block h-0.5 w-4 rounded"
                style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
              />
              <span className="text-[10px]" style={{ color: "var(--color-text-secondary)" }}>
                {curve.model}
              </span>
            </div>
          ))}
        </div>
        <div ref={containerRef} />
      </div>
    </div>
  );
}
