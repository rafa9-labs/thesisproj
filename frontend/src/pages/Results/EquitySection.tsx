import { forwardRef, useState, useMemo } from "react";
import {
  EquityCurveChart,
  type EquityCurveChartHandle,
} from "@/components/charts/EquityCurveChart";
import { SegmentedControl } from "@/components/shared/SegmentedControl";
import { useNewsEvents } from "@/api/queries";
import type { EquityPoint, TradeRecord } from "@/api/schemas";
import type { UTCTimestamp, SeriesMarker } from "lightweight-charts";

interface EquitySectionProps {
  equityCurve: EquityPoint[] | null;
  buyHoldCurve: EquityPoint[] | null;
  drawdownCurve: EquityPoint[] | null;
  trades: TradeRecord[] | null;
  chartRef?: React.Ref<EquityCurveChartHandle>;
}

type ViewMode = "equity" | "return";

function transformForView(data: EquityPoint[], mode: ViewMode): EquityPoint[] {
  if (mode === "equity" || data.length === 0) return data;
  const base = data[0].value;
  if (base === 0) return data;
  return data.map((d) => ({ time: d.time, value: ((d.value / base) - 1) * 100 }));
}

function toggleBtn(label: string, active: boolean, onClick: () => void) {
  return (
    <button
      onClick={onClick}
      className="rounded-md border px-2 py-0.5 font-mono text-[10px] transition-colors"
      style={{
        borderColor: active ? "var(--color-accent)" : "var(--color-border)",
        backgroundColor: active ? "rgba(41,98,255,0.1)" : "var(--color-surface)",
        color: active ? "var(--color-accent)" : "var(--color-text-muted)",
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
}

export const EquitySection = forwardRef<
  EquityCurveChartHandle,
  Omit<EquitySectionProps, "chartRef">
>(function EquitySection({ equityCurve, buyHoldCurve, drawdownCurve, trades }, ref) {
  const rawEquity = equityCurve ?? [];
  const rawBuyHold = buyHoldCurve ?? [];
  const drawdown = drawdownCurve ?? [];

  const [viewMode, setViewMode] = useState<ViewMode>("equity");
  const [showBuyHold, setShowBuyHold] = useState(true);
  const [showTrades, setShowTrades] = useState(false);
  const [showEvents, setShowEvents] = useState(false);

  const data = useMemo(() => transformForView(rawEquity, viewMode), [rawEquity, viewMode]);
  const buyHold = useMemo(() => {
    if (!showBuyHold || rawBuyHold.length === 0) return undefined;
    return transformForView(rawBuyHold, viewMode);
  }, [rawBuyHold, viewMode, showBuyHold]);

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

  const seriesMarkers = useMemo<SeriesMarker<UTCTimestamp>[] | undefined>(() => {
    const result: SeriesMarker<UTCTimestamp>[] = [];

    // News event markers
    if (eventMarkers && eventMarkers.length > 0) {
      for (const m of eventMarkers) {
        const onChart = data.some(
          (d) => d.time === m.time || Math.abs(d.time - m.time) < 86400,
        );
        if (!onChart) continue;
        result.push({
          time: m.time as UTCTimestamp,
          position: "belowBar",
          color:
            m.impact === "high"
              ? "var(--color-event-high)"
              : m.impact === "medium"
                ? "var(--color-accent-warning)"
                : "var(--color-text-muted)",
          shape: "arrowUp",
          text: m.event,
          size: 1,
        });
      }
    }

    // Trade entry markers
    if (showTrades && trades && trades.length > 0) {
      for (const t of trades) {
        if (!t.entry_date) continue;
        const tTime = (new Date(t.entry_date).getTime() / 1000) as UTCTimestamp;
        const isBuy = t.direction === "BUY";
        result.push({
          time: tTime,
          position: isBuy ? "belowBar" : "aboveBar",
          color: isBuy ? "var(--color-accent-success)" : "var(--color-accent-danger)",
          shape: isBuy ? "arrowUp" : "arrowDown",
          text: `#${t.trade_id} ${isBuy ? "Buy" : "Sell"}`,
          size: 1,
        });
      }
    }

    return result.length > 0 ? result : undefined;
  }, [showTrades, trades, eventMarkers, data]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
          Equity Curve
        </h3>
        <div className="flex items-center gap-3">
          <SegmentedControl
            segments={[
              { key: "return", label: "% Return" },
              { key: "equity", label: "Multiple" },
            ]}
            active={viewMode}
            onChange={(k) => setViewMode(k as ViewMode)}
          />
          {toggleBtn("BH", showBuyHold, () => setShowBuyHold(!showBuyHold))}
          {toggleBtn("Trades", showTrades, () => setShowTrades(!showTrades))}
          {toggleBtn("Events", showEvents, () => setShowEvents(!showEvents))}
        </div>
      </div>
      <div className="flex items-center gap-4 text-xs text-(--color-text-muted)">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 rounded bg-(--color-accent-success)" />
          Equity
        </span>
        {showBuyHold && (
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded bg-(--color-text-muted)"
              style={{ borderStyle: "dashed" }}
            />
            Buy & hold
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-4 rounded-sm"
            style={{ backgroundColor: "rgba(242, 54, 69, 0.3)" }}
          />
          Drawdown
        </span>
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
        {showTrades && (
          <>
            <span className="flex items-center gap-1">
              <span style={{ color: "var(--color-accent-success)", fontSize: 9 }}>▲</span>
              Buy
            </span>
            <span className="flex items-center gap-1">
              <span style={{ color: "var(--color-accent-danger)", fontSize: 9 }}>▼</span>
              Sell
            </span>
          </>
        )}
      </div>
      <EquityCurveChart
        ref={ref}
        data={data}
        buyHoldData={buyHold}
        drawdownData={drawdown}
        seriesMarkers={seriesMarkers}
        height={420}
      />
    </div>
  );
});
