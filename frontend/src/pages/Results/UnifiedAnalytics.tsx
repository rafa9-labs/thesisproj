import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SegmentedControl } from "@/components/shared/SegmentedControl";
import {
  computeRollingReturn,
  computeRollingSharpe,
  cumulativePnlFromTrades,
} from "@/lib/chartUtils";
import type { EquityPoint, TradeRecord } from "@/api/schemas";

export interface UnifiedAnalyticsHandle {
  takeScreenshot: () => void;
}

interface ModelSeries {
  model: string;
  equityCurve: EquityPoint[] | null;
  drawdownCurve: EquityPoint[] | null;
  trades: TradeRecord[] | null;
}

interface Props {
  models: ModelSeries[];
  buyHoldCurve: EquityPoint[] | null;
  height?: number;
}

type View = "equity" | "drawdown" | "cumPnl" | "rolling";

// Blue/teal palette variations per spec
const MODEL_COLORS = ["#00E5FF", "#2962FF", "#06B6D4", "#10B981", "#0EA5E9"];
const BH_COLOR = "#787B86";
const ROLLING_RETURN_COLOR = "#a78bfa";

function formatDate(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" });
}

function toPercent(data: EquityPoint[]): EquityPoint[] {
  if (data.length === 0) return data;
  const base = data[0].value;
  if (base === 0) return data;
  return data.map((d) => ({ time: d.time, value: ((d.value / base) - 1) * 100 }));
}

function mergeByTime(
  series: { key: string; data: EquityPoint[] }[],
  transform?: (v: number) => number,
): Array<Record<string, number | string | null>> {
  const map = new Map<number, Record<string, number | string | null>>();
  for (const s of series) {
    for (const p of s.data) {
      if (p.value == null || !Number.isFinite(p.value)) continue;
      if (!map.has(p.time)) map.set(p.time, { time: p.time, date: formatDate(p.time) });
      const row = map.get(p.time)!;
      row[s.key] = transform ? transform(p.value) : p.value;
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => (a.time as number) - (b.time as number),
  );
}

function mergeRolling(
  series: { key: string; data: { time: number; value: number | null }[] }[],
): Array<Record<string, number | string | null>> {
  const map = new Map<number, Record<string, number | string | null>>();
  for (const s of series) {
    for (const p of s.data) {
      if (p.value == null || !Number.isFinite(p.value)) continue;
      if (!map.has(p.time)) map.set(p.time, { time: p.time, date: formatDate(p.time) });
      map.get(p.time)![s.key] = p.value;
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => (a.time as number) - (b.time as number),
  );
}

function mergeCumPnl(
  series: { key: string; data: { tradeNum: number; cumPnl: number }[] }[],
): Array<Record<string, number | string | null>> {
  const maxLen = Math.max(...series.map((s) => s.data.length), 0);
  const rows: Array<Record<string, number | string | null>> = [];
  for (let i = 0; i < maxLen; i++) {
    const row: Record<string, number | string | null> = { tradeNum: i + 1 };
    for (const s of series) {
      row[s.key] = s.data[i]?.cumPnl ?? null;
    }
    rows.push(row);
  }
  return rows;
}

const tooltipStyle = {
  backgroundColor: "var(--color-elevated)",
  border: "1px solid var(--color-glass-border)",
  borderRadius: 6,
  fontSize: 11,
  fontFamily: "var(--font-mono)",
} as const;

export const UnifiedAnalytics = forwardRef<UnifiedAnalyticsHandle, Props>(
  function UnifiedAnalytics({ models, buyHoldCurve, height = 440 }, ref) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [view, setView] = useState<View>("equity");
    const [showBH, setShowBH] = useState(true);
    // Default first 5 models visible
    const [visible, setVisible] = useState<Set<string>>(new Set());

    // Sync visible set when the model roster changes (job load, model switch)
    const modelKey = models.map((m) => m.model).join("|");
    useEffect(() => {
      setVisible(new Set(models.slice(0, 5).map((m) => m.model)));
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [modelKey]);

    useImperativeHandle(ref, () => ({
      takeScreenshot: () => {
        const svg = containerRef.current?.querySelector("svg");
        if (!svg) return;
        const xml = new XMLSerializer().serializeToString(svg);
        const svg64 = btoa(unescape(encodeURIComponent(xml)));
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = svg.clientWidth || 900;
          canvas.height = svg.clientHeight || height;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          ctx.fillStyle = "#0F172A";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          const a = document.createElement("a");
          a.href = canvas.toDataURL("image/png");
          a.download = `analytics_${view}.png`;
          a.click();
        };
        img.src = `data:image/svg+xml;base64,${svg64}`;
      },
    }));

    const visibleModels = useMemo(
      () => models.filter((m) => visible.has(m.model)),
      [models, visible],
    );

    const activeCount = visibleModels.length;
    const singleModel = activeCount === 1;

    const toggleModel = (model: string) => {
      setVisible((prev) => {
        const next = new Set(prev);
        if (next.has(model)) next.delete(model);
        else next.add(model);
        return next;
      });
    };

    // ── Equity vs B&H ──────────────────────────────────────────────
    const equityData = useMemo(() => {
      if (view !== "equity") return [];
      const series = visibleModels.map((m, i) => ({
        key: `m_${i}`,
        data: toPercent((m.equityCurve ?? []).filter((p) => p.value != null)),
      }));
      const merged = mergeByTime(series);
      if (showBH && buyHoldCurve && buyHoldCurve.length > 0) {
        const bh = toPercent(buyHoldCurve);
        for (const p of bh) {
          if (p.value == null) continue;
          const row = merged.find((r) => r.time === p.time);
          if (row) row.bh = p.value;
        }
      }
      return merged;
    }, [view, visibleModels, showBH, buyHoldCurve]);

    // ── Drawdown ───────────────────────────────────────────────────
    const drawdownData = useMemo(() => {
      if (view !== "drawdown") return [];
      const series = visibleModels.map((m, i) => ({
        key: `m_${i}`,
        data: (m.drawdownCurve ?? []).filter((p) => p.value != null),
      }));
      return mergeByTime(series, (v) => +(v * 100).toFixed(3));
    }, [view, visibleModels]);

    // ── Cumulative P&L ─────────────────────────────────────────────
    const cumPnlData = useMemo(() => {
      if (view !== "cumPnl") return [];
      const series = visibleModels.map((m, i) => ({
        key: `m_${i}`,
        data: cumulativePnlFromTrades(m.trades ?? []),
      }));
      return mergeCumPnl(series);
    }, [view, visibleModels]);

    // ── Rolling Metrics ────────────────────────────────────────────
    const rollingData = useMemo(() => {
      if (view !== "rolling") return [];
      const sharpeSeries = visibleModels.map((m, i) => ({
        key: `m_${i}`,
        data: computeRollingSharpe(m.equityCurve ?? [], 30).map((p) => ({
          time: p.time,
          value: p.sharpe,
        })),
      }));
      const merged = mergeRolling(sharpeSeries);
      if (singleModel && visibleModels[0]?.equityCurve) {
        const ret = computeRollingReturn(visibleModels[0].equityCurve, 30);
        for (const p of ret) {
          const row = merged.find((r) => r.time === p.time);
          if (row) row.rollingReturn = +p.returnPct.toFixed(2);
        }
      }
      return merged;
    }, [view, visibleModels, singleModel]);

    const hasAnyData = models.some(
      (m) => (m.equityCurve?.length ?? 0) > 0 || (m.trades?.length ?? 0) > 0,
    );

    if (!hasAnyData) {
      return (
        <div className="flex flex-col gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-5">
          <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
            Unified Analytics
          </h3>
          <div className="flex items-center justify-center py-12 text-(--color-text-muted)">
            <span className="font-mono text-sm">No chart data available</span>
          </div>
        </div>
      );
    }

    const yFmtEquity = (v: number) => `${v.toFixed(0)}%`;

    return (
      <div className="flex flex-col gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-4">
        {/* ── Header: title + view switches ───────────────────────── */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
            Unified Analytics
          </h3>
          <div className="flex items-center gap-2">
            <SegmentedControl
              segments={[
                { key: "equity", label: "Equity vs B&H" },
                { key: "drawdown", label: "Drawdown" },
                { key: "cumPnl", label: "Cum. P&L" },
                { key: "rolling", label: "Rolling Metrics" },
              ]}
              active={view}
              onChange={(k) => setView(k as View)}
            />
            {view === "equity" && (
              <button
                onClick={() => setShowBH((v) => !v)}
                className="cursor-pointer rounded-md border px-2 py-1 font-mono text-[10px] transition-colors"
                style={{
                  borderColor: showBH ? "var(--color-accent)" : "var(--color-border)",
                  backgroundColor: showBH ? "rgba(0,229,255,0.10)" : "transparent",
                  color: showBH ? "var(--color-accent)" : "var(--color-text-muted)",
                }}
              >
                B&H
              </button>
            )}
          </div>
        </div>

        {/* ── Model legend (clickable toggles) ────────────────────── */}
        {models.length > 1 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
              Models
            </span>
            {models.map((m, i) => {
              const isOn = visible.has(m.model);
              const color = MODEL_COLORS[i % MODEL_COLORS.length];
              return (
                <button
                  key={m.model}
                  onClick={() => toggleModel(m.model)}
                  className="flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-medium tracking-wide transition-all"
                  style={{
                    borderColor: isOn ? color : "var(--color-glass-border)",
                    backgroundColor: isOn ? `${color}14` : "transparent",
                    color: isOn ? "var(--color-text-primary)" : "var(--color-text-muted)",
                    opacity: isOn ? 1 : 0.55,
                  }}
                >
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: isOn ? color : "var(--color-text-dim)" }}
                  />
                  {m.model}
                </button>
              );
            })}
          </div>
        )}

        {/* ── Legend key ──────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center gap-4 text-[10px] text-(--color-text-muted)">
          {view === "equity" && showBH && (
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-4"
                style={{
                  backgroundColor: BH_COLOR,
                  borderTop: "2px dotted " + BH_COLOR,
                  background: "none",
                }}
              />
              Buy &amp; Hold
            </span>
          )}
          {visibleModels.map((m, i) => (
            <span key={m.model} className="flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-4 rounded"
                style={{ backgroundColor: MODEL_COLORS[i % MODEL_COLORS.length] }}
              />
              {m.model}
            </span>
          ))}
        </div>

        {/* ── Chart area ──────────────────────────────────────────── */}
        <div ref={containerRef} className="w-full" style={{ height }}>
          <ResponsiveContainer width="100%" height="100%">
            {view === "equity" ? (
              <LineChart data={equityData} margin={{ top: 5, right: 12, left: 4, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                  tickFormatter={yFmtEquity}
                  width={56}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <ReferenceLine y={0} stroke="var(--color-glass-border)" strokeDasharray="3 3" />
                {visibleModels.map((m, i) => (
                  <Line
                    key={m.model}
                    type="monotone"
                    dataKey={`m_${i}`}
                    name={m.model}
                    stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                    strokeWidth={singleModel ? 2 : 1.5}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
                {showBH && (
                  <Line
                    type="monotone"
                    dataKey="bh"
                    name="Buy & Hold"
                    stroke={BH_COLOR}
                    strokeWidth={1}
                    strokeDasharray="4 4"
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            ) : view === "drawdown" ? (
              singleModel ? (
                <AreaChart data={drawdownData} margin={{ top: 5, right: 12, left: 4, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    interval="preserveStartEnd"
                    minTickGap={50}
                  />
                  <YAxis
                    tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    tickFormatter={(v: number) => `${v}%`}
                    width={56}
                    domain={["dataMin", 0]}
                  />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v) => { const n = typeof v === "number" ? v : 0; return [`${n.toFixed(2)}%`, "Drawdown"]; }} />
                  <Area
                    type="monotone"
                    dataKey="m_0"
                    name={visibleModels[0]?.model ?? "Drawdown"}
                    stroke="var(--color-accent-danger)"
                    fill="rgba(242,54,69,0.22)"
                    strokeWidth={1.5}
                    connectNulls
                    isAnimationActive={false}
                  />
                </AreaChart>
              ) : (
                <LineChart data={drawdownData} margin={{ top: 5, right: 12, left: 4, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    interval="preserveStartEnd"
                    minTickGap={50}
                  />
                  <YAxis
                    tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                    tickFormatter={(v: number) => `${v}%`}
                    width={56}
                    domain={["dataMin", 0]}
                  />
                  <Tooltip contentStyle={tooltipStyle} />
                  {visibleModels.map((m, i) => (
                    <Line
                      key={m.model}
                      type="monotone"
                      dataKey={`m_${i}`}
                      name={m.model}
                      stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                      strokeWidth={1.5}
                      dot={false}
                      connectNulls
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              )
            ) : view === "cumPnl" ? (
              <LineChart data={cumPnlData} margin={{ top: 5, right: 12, left: 4, bottom: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
                <XAxis
                  dataKey="tradeNum"
                  tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                  label={{
                    value: "Trade #",
                    position: "insideBottomRight",
                    offset: -4,
                    style: { fill: "var(--color-text-muted)", fontSize: 10 },
                  }}
                  minTickGap={40}
                />
                <YAxis
                  tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                  width={56}
                />
                <Tooltip contentStyle={tooltipStyle} labelFormatter={(l) => `Trade #${l}`} />
                <ReferenceLine y={0} stroke="var(--color-glass-border)" strokeDasharray="3 3" />
                {visibleModels.map((m, i) => (
                  <Line
                    key={m.model}
                    type="stepAfter"
                    dataKey={`m_${i}`}
                    name={m.model}
                    stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            ) : (
              <LineChart data={rollingData} margin={{ top: 5, right: 12, left: 4, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "var(--color-text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                  interval="preserveStartEnd"
                  minTickGap={50}
                />
                <YAxis
                  yAxisId="sharpe"
                  tick={{ fill: "var(--color-accent-success)", fontSize: 10, fontFamily: "var(--font-mono)" }}
                  width={48}
                  tickFormatter={(v: number) => v.toFixed(1)}
                />
                {singleModel && (
                  <YAxis
                    yAxisId="return"
                    orientation="right"
                    tick={{ fill: ROLLING_RETURN_COLOR, fontSize: 10, fontFamily: "var(--font-mono)" }}
                    tickFormatter={(v: number) => `${v}%`}
                  />
                )}
                <Tooltip contentStyle={tooltipStyle} />
                <ReferenceLine y={0} yAxisId="sharpe" stroke="var(--color-glass-border)" strokeDasharray="3 3" />
                {visibleModels.map((m, i) => (
                  <Line
                    key={m.model}
                    yAxisId="sharpe"
                    type="monotone"
                    dataKey={`m_${i}`}
                    name={`${m.model} · Sharpe`}
                    stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                    strokeWidth={1.5}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                ))}
                {singleModel && (
                  <Line
                    yAxisId="return"
                    type="monotone"
                    dataKey="rollingReturn"
                    name="Rolling Return"
                    stroke={ROLLING_RETURN_COLOR}
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            )}
          </ResponsiveContainer>
        </div>
        {view === "rolling" && (
          <span className="font-mono text-[9px] text-(--color-text-muted)">
            30-bar rolling window · Sharpe (per model)
            {singleModel ? " · Return (dashed, right axis)" : ""}
          </span>
        )}
      </div>
    );
  },
);
