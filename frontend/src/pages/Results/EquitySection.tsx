import { forwardRef, useState, useMemo } from "react";
import { EquityCurveChart, type EquityCurveChartHandle } from "@/components/charts/EquityCurveChart";
import { useNewsEvents } from "@/api/queries";
import type { EquityPoint } from "@/api/schemas";

interface EquitySectionProps {
  equityCurve: EquityPoint[] | null;
  buyHoldCurve: EquityPoint[] | null;
  drawdownCurve: EquityPoint[] | null;
  chartRef?: React.Ref<EquityCurveChartHandle>;
}

export const EquitySection = forwardRef<EquityCurveChartHandle, Omit<EquitySectionProps, "chartRef">>(function EquitySection(
  { equityCurve, buyHoldCurve, drawdownCurve },
  ref,
) {
  const data = equityCurve ?? [];
  const buyHold = buyHoldCurve ?? undefined;
  const drawdown = drawdownCurve ?? undefined;

  const [showEvents, setShowEvents] = useState(false);

  const startTime = data.length > 0 ? data[0].time : null;
  const endTime = data.length > 0 ? data[data.length - 1].time : null;

  const { data: newsData } = useNewsEvents(
    showEvents && startTime ? startTime : null,
    showEvents && endTime ? endTime : null,
    showEvents ? "high,medium" : undefined,
  );

  const eventMarkers = useMemo(() => {
    if (!showEvents || !newsData?.events) return undefined;
    return newsData.events;
  }, [showEvents, newsData]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Equity Curve
        </h3>
        <div className="flex items-center gap-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded"
              style={{ backgroundColor: "var(--color-accent-success)" }}
            />
            Equity
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded"
              style={{ backgroundColor: "var(--color-text-muted)", borderStyle: "dashed" }}
            />
            Buy & hold
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-4 rounded-sm"
              style={{ backgroundColor: "rgba(242, 54, 69, 0.3)" }}
            />
            Drawdown
          </span>
          <button
            onClick={() => setShowEvents(!showEvents)}
            className="rounded-md border px-2 py-0.5 text-[10px] transition-colors"
            style={{
              borderColor: showEvents ? "var(--color-accent)" : "var(--color-border)",
              backgroundColor: showEvents ? "rgba(41,98,255,0.1)" : "var(--color-surface)",
              color: showEvents ? "var(--color-accent)" : "var(--color-text-muted)",
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
            }}
          >
            {showEvents ? "Events ON" : "Events OFF"}
          </button>
          {showEvents && (
            <>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: "#EF4444" }} />
                High
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: "#F59E0B" }} />
                Med
              </span>
            </>
          )}
        </div>
      </div>
      <EquityCurveChart
        ref={ref}
        data={data}
        buyHoldData={buyHold}
        drawdownData={drawdown}
        eventMarkers={eventMarkers}
        height={420}
      />
    </div>
  );
});