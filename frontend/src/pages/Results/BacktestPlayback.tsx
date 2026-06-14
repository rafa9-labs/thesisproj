import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { X } from "lucide-react";
import {
  createChart,
  type IChartApi,
  CandlestickSeries,
  LineSeries,
  ColorType,
} from "lightweight-charts";
import { PlaybackController } from "@/components/charts/PlaybackController";
import type {
  OHLCBar,
  TradeChartMarker,
  EquityPoint,
  MonthlyResult,
  HpoTrial,
} from "@/api/schemas";

interface BacktestPlaybackProps {
  candles: OHLCBar[];
  trades: TradeChartMarker[];
  equityCurve: EquityPoint[];
  monthlyResults: MonthlyResult[] | null;
  hpoTrials: HpoTrial[] | null;
  pair: string;
  model: string;
  timeframe: string;
  onClose: () => void;
}

function parseMonthToTs(monthStr: string): number | null {
  const parts = monthStr.split("-");
  if (parts.length < 2) return null;
  const year = parseInt(parts[0], 10);
  const month = parseInt(parts[1], 10);
  if (isNaN(year) || isNaN(month)) return null;
  return Math.floor(Date.UTC(year, month - 1, 1) / 1000);
}

export function BacktestPlayback({
  candles,
  trades,
  equityCurve,
  monthlyResults,
  hpoTrials,
  pair,
  model,
  timeframe,
  onClose,
}: BacktestPlaybackProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [currentBar, setCurrentBar] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const totalBars = candles.length;

  const visibleCandles = useMemo(() => candles.slice(0, currentBar + 1), [candles, currentBar]);
  const currentCandle = visibleCandles[visibleCandles.length - 1];
  const currentTimestamp = currentCandle?.t ?? 0;

  const visibleEquity = useMemo(
    () => (equityCurve ? equityCurve.filter((e) => e.time <= currentTimestamp) : []),
    [equityCurve, currentTimestamp],
  );
  const visibleTrades = useMemo(
    () => (trades ? trades.filter((t) => t.entry_time <= currentTimestamp) : []),
    [trades, currentTimestamp],
  );

  useEffect(() => {
    if (!containerRef.current) return;
    if (visibleCandles.length === 0) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0A0D12" },
        textColor: "#80899F",
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1A1F2A" },
        horzLines: { color: "#1A1F2A" },
      },
      width: containerRef.current.clientWidth,
      height: 440,
      crosshair: { mode: 0 },
      timeScale: {
        borderColor: "#2A2E39",
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "#2A2E39",
        scaleMargins: { top: 0.05, bottom: 0.35 },
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
      visibleCandles.map((c) => ({
        time: c.t as number,
        open: c.o,
        high: c.h,
        low: c.l,
        close: c.c,
      })),
    );

    const markers = visibleTrades.map((t) => ({
      time: t.direction === "BUY" ? t.entry_time : t.exit_time,
      position: t.direction === "BUY" ? ("belowBar" as const) : ("aboveBar" as const),
      color: t.direction === "BUY" ? "#26a69a" : "#ef5350",
      shape: t.direction === "BUY" ? ("arrowUp" as const) : ("arrowDown" as const),
      text: `${(t.pnl_pct ?? 0) > 0 ? "+" : ""}${(t.pnl_pct ?? 0).toFixed(1)}%`,
      size: 2,
    }));

    if (markers.length > 0) {
      candleSeries.setMarkers(markers as any);
    }

    if (visibleEquity.length > 1) {
      const equitySeries = chart.addSeries(LineSeries, {
        color: "#42a5f5",
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        priceScaleId: "equity",
      });

      chart.priceScale("equity").applyOptions({
        scaleMargins: { top: 0.7, bottom: 0.05 },
        visible: true,
        borderColor: "#2A2E39",
      });

      equitySeries.setData(
        visibleEquity.map((e) => ({
          time: e.time as number,
          value: e.value,
        })),
      );
    }

    if (monthlyResults && monthlyResults.length > 0) {
      const monthMarkers = monthlyResults
        .map((m) => {
          const ts = parseMonthToTs(m.month);
          if (ts === null) return null;
          const ret = m.return_pct ?? 0;
          return {
            time: ts as number,
            position: "aboveBar" as const,
            color: ret >= 0 ? "#26a69a40" : "#ef535040",
            shape: "circle" as const,
            text: m.month,
            size: 0,
          };
        })
        .filter((m): m is NonNullable<typeof m> => m !== null);

      if (monthMarkers.length > 0) {
        candleSeries.setMarkers([...markers, ...monthMarkers].sort((a, b) => a.time - b.time) as any);
      }
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
  }, [visibleCandles, visibleEquity, visibleTrades, monthlyResults]);

  useEffect(() => {
    if (!isPlaying) return;
    const intervalMs = Math.max(10, Math.round(200 / speed));
    const timer = setInterval(() => {
      setCurrentBar((prev) => {
        if (prev >= totalBars - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);
    return () => clearInterval(timer);
  }, [isPlaying, speed, totalBars]);

  const handlePlay = useCallback(() => {
    if (currentBar >= totalBars - 1) {
      setCurrentBar(0);
    }
    setIsPlaying(true);
  }, [currentBar, totalBars]);

  const handlePause = useCallback(() => setIsPlaying(false), []);
  const handleStepForward = useCallback(() => {
    setIsPlaying(false);
    setCurrentBar((prev) => Math.min(prev + 1, totalBars - 1));
  }, [totalBars]);
  const handleStepBack = useCallback(() => {
    setIsPlaying(false);
    setCurrentBar((prev) => Math.max(prev - 1, 0));
  }, []);
  const handleSeek = useCallback((index: number) => {
    setIsPlaying(false);
    setCurrentBar(index);
  }, []);
  const handleSpeedChange = useCallback((s: number) => setSpeed(s), []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setCurrentBar(0);
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === " ") {
        e.preventDefault();
        if (isPlaying) handlePause();
        else handlePlay();
      }
      if (e.key === "ArrowRight") handleStepForward();
      if (e.key === "ArrowLeft") handleStepBack();
      if (e.key === "Home") handleReset();
      if (e.key === "End") {
        setIsPlaying(false);
        setCurrentBar(totalBars - 1);
      }
    },
    [
      isPlaying,
      handlePlay,
      handlePause,
      handleStepForward,
      handleStepBack,
      handleReset,
      onClose,
      totalBars,
    ],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const currentEquity =
    visibleEquity.length > 0 ? visibleEquity[visibleEquity.length - 1].value : null;
  const equityStart = equityCurve && equityCurve.length > 0 ? equityCurve[0].value : null;
  const equityPnl =
    currentEquity != null && equityStart != null ? currentEquity - equityStart : null;
  const equityPnlPct =
    equityPnl != null && equityStart != null && equityStart !== 0
      ? (equityPnl / equityStart) * 100
      : null;

  const openTrade = trades
    ? trades.find(
        (t) =>
          t.entry_time <= currentTimestamp &&
          (t.exit_time > currentTimestamp || t.exit_time === t.entry_time),
      )
    : null;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-(--color-app)">
      <div className="flex items-center justify-between border-b border-(--color-border-subtle) bg-(--color-surface) px-4 py-2">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
            Backtest Replay
          </h3>
          <span className="font-mono text-[10px] text-(--color-text-muted)">
            {pair} · {model} · {timeframe}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleReset}
            className="rounded border border-(--color-glass-border) px-2 py-1 font-mono text-[10px] font-medium text-(--color-text-muted) uppercase transition hover:bg-[var(--color-glass-hover)]"
          >
            Reset
          </button>
          <button
            onClick={onClose}
            className="rounded p-1.5 text-(--color-text-muted) transition hover:bg-[var(--color-glass-hover)]"
            title="Close (Esc)"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      <div className="flex flex-1 gap-0 overflow-hidden">
        <div className="flex min-w-0 flex-1 flex-col p-4">
          <div
            ref={containerRef}
            className="min-h-[440px] w-full overflow-hidden rounded-sm border border-(--color-glass-border) bg-[#131722]"
          />

          <div className="mt-3">
            <PlaybackController
              currentIndex={currentBar}
              totalBars={totalBars}
              isPlaying={isPlaying}
              speed={speed}
              onPlay={handlePlay}
              onPause={handlePause}
              onStepForward={handleStepForward}
              onStepBack={handleStepBack}
              onSpeedChange={handleSpeedChange}
              onSeek={handleSeek}
            />
          </div>

          {monthlyResults && monthlyResults.length > 0 && (
            <div className="mt-3">
              <div className="mb-1.5 flex items-center gap-1">
                <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  Walk-Forward Months
                </span>
              </div>
              <div className="flex gap-1 overflow-x-auto pb-1">
                {monthlyResults.map((m, i) => {
                  const ret = m.return_pct ?? 0;
                  const isCurrentMonth =
                    currentCandle &&
                    m.month ===
                      monthlyResults.find((mr) => {
                        const start = parseMonthToTs(mr.month);
                        const end = start !== null ? start + 30 * 86400 : null;
                        return (
                          start !== null &&
                          end !== null &&
                          currentCandle.t >= start &&
                          currentCandle.t <= end
                        );
                      })?.month;
                  return (
                    <div
                      key={i}
                      className="min-w-[60px] rounded border px-2 py-1 text-center font-mono text-[9px] whitespace-nowrap transition-all"
                      style={{
                        backgroundColor: ret >= 0 ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)",
                        borderColor: isCurrentMonth
                          ? "var(--color-brand)"
                          : "var(--color-glass-border)",
                        color:
                          ret >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                        boxShadow: isCurrentMonth ? "0 0 8px rgba(0,229,255,0.2)" : "none",
                      }}
                      title={`Win Rate: ${((m.win_rate ?? 0) * 100).toFixed(1)}% | Trades: ${m.trades ?? 0} | Sharpe: ${m.sharpe?.toFixed(2) ?? "—"}`}
                    >
                      <div className="font-medium">{m.month}</div>
                      <div>
                        {ret >= 0 ? "+" : ""}
                        {ret.toFixed(1)}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <div className="flex w-[220px] flex-shrink-0 flex-col gap-4 overflow-y-auto border-l border-(--color-border-subtle) bg-(--color-surface) p-3">
          <div>
            <h4 className="mb-2 text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
              Current Bar
            </h4>
            {currentCandle ? (
              <div className="flex flex-col gap-1">
                <div className="flex justify-between">
                  <span className="text-[10px] text-(--color-text-muted)">Time</span>
                  <span className="font-mono text-[10px] text-(--color-text-secondary)">
                    {new Date((currentCandle.t as number) * 1000).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-(--color-text-muted)">Open</span>
                  <span className="font-mono text-[10px] text-(--color-text-primary)">
                    {currentCandle.o.toFixed(5)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-(--color-text-muted)">High</span>
                  <span className="font-mono text-[10px] text-(--color-text-primary)">
                    {currentCandle.h.toFixed(5)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-(--color-text-muted)">Low</span>
                  <span className="font-mono text-[10px] text-(--color-text-primary)">
                    {currentCandle.l.toFixed(5)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-(--color-text-muted)">Close</span>
                  <span
                    className="font-mono text-[10px] font-medium"
                    style={{
                      color:
                        currentCandle.c >= currentCandle.o
                          ? "var(--color-accent-success)"
                          : "var(--color-accent-danger)",
                    }}
                  >
                    {currentCandle.c.toFixed(5)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-(--color-text-muted)">Volume</span>
                  <span className="font-mono text-[10px] text-(--color-text-secondary)">
                    {(currentCandle.volume ?? 0).toLocaleString()}
                  </span>
                </div>
              </div>
            ) : (
              <span className="text-[10px] text-(--color-text-muted)">No data</span>
            )}
          </div>

          <div>
            <h4 className="mb-2 text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
              Equity
            </h4>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between">
                <span className="text-[10px] text-(--color-text-muted)">Value</span>
                <span className="font-mono text-[11px] font-medium text-(--color-text-primary)">
                  {currentEquity != null ? currentEquity.toFixed(2) : "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[10px] text-(--color-text-muted)">P&L</span>
                <span
                  className="font-mono text-[11px] font-medium"
                  style={{
                    color:
                      equityPnl != null && equityPnl >= 0
                        ? "var(--color-accent-success)"
                        : "var(--color-accent-danger)",
                  }}
                >
                  {equityPnl != null ? `${equityPnl >= 0 ? "+" : ""}${equityPnl.toFixed(2)}` : "—"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[10px] text-(--color-text-muted)">Return</span>
                <span
                  className="font-mono text-[11px] font-medium"
                  style={{
                    color:
                      equityPnlPct != null && equityPnlPct >= 0
                        ? "var(--color-accent-success)"
                        : "var(--color-accent-danger)",
                  }}
                >
                  {equityPnlPct != null
                    ? `${equityPnlPct >= 0 ? "+" : ""}${equityPnlPct.toFixed(2)}%`
                    : "—"}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="mb-2 text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
              Active Trade
            </h4>
            {openTrade ? (
              <div
                className="flex flex-col gap-1 rounded border px-2 py-1.5"
                style={{
                  borderColor:
                    openTrade.direction === "BUY"
                      ? "var(--color-accent-success)"
                      : "var(--color-accent-danger)",
                  backgroundColor:
                    openTrade.direction === "BUY" ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)",
                }}
              >
                <div className="flex justify-between">
                  <span
                    className="font-mono text-[10px] font-semibold"
                    style={{
                      color:
                        openTrade.direction === "BUY"
                          ? "var(--color-accent-success)"
                          : "var(--color-accent-danger)",
                    }}
                  >
                    {openTrade.direction}
                  </span>
                  <span className="font-mono text-[10px] text-(--color-text-secondary)">
                    @ {openTrade.entry_price.toFixed(5)}
                  </span>
                </div>
                {openTrade.pnl_pct != null && (
                  <div className="flex justify-between">
                    <span className="text-[10px] text-(--color-text-muted)">P&L</span>
                    <span
                      className="font-mono text-[10px] font-medium"
                      style={{
                        color:
                          openTrade.pnl_pct >= 0
                            ? "var(--color-accent-success)"
                            : "var(--color-accent-danger)",
                      }}
                    >
                      {openTrade.pnl_pct >= 0 ? "+" : ""}
                      {openTrade.pnl_pct.toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <span className="text-[10px] text-(--color-text-muted)">No active trade</span>
            )}
          </div>

          <div>
            <h4 className="mb-2 text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
              Trades So Far
            </h4>
            <div className="flex justify-between">
              <span className="text-[10px] text-(--color-text-muted)">Completed</span>
              <span className="font-mono text-[11px] font-medium text-(--color-text-primary)">
                {visibleTrades.length}
              </span>
            </div>
            {visibleTrades.length > 0 && (
              <div className="mt-1 flex justify-between">
                <span className="text-[10px] text-(--color-text-muted)">Win Rate</span>
                <span className="font-mono text-[11px] font-medium text-(--color-text-secondary)">
                  {(
                    (visibleTrades.filter((t) => (t.pnl_pct ?? 0) > 0).length /
                      visibleTrades.length) *
                    100
                  ).toFixed(1)}
                  %
                </span>
              </div>
            )}
          </div>

          {hpoTrials && hpoTrials.length > 0 && (
            <div>
              <h4 className="mb-2 text-[10px] font-medium tracking-[0.12em] text-(--color-text-muted) uppercase">
                HPO Trials ({hpoTrials.length})
              </h4>
              <div className="flex max-h-[120px] flex-col gap-0.5 overflow-y-auto">
                {hpoTrials.slice(0, 10).map((t, i) => (
                  <div
                    key={i}
                    className="flex justify-between rounded bg-(--color-glass-hover) px-1.5 py-0.5"
                  >
                    <span className="font-mono text-[9px] text-(--color-text-muted)">
                      T{t.trial_number}
                    </span>
                    <span
                      className="font-mono text-[9px] font-medium"
                      style={{
                        color:
                          t.value >= 0
                            ? "var(--color-accent-success)"
                            : "var(--color-accent-danger)",
                      }}
                    >
                      {t.value.toFixed(3)}
                    </span>
                  </div>
                ))}
                {hpoTrials.length > 10 && (
                  <span className="text-center text-[9px] text-(--color-text-muted)">
                    +{hpoTrials.length - 10} more
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="mt-auto">
            <div className="mb-1 flex items-center gap-1.5">
              <span className="text-[9px] tracking-wider text-(--color-text-muted) uppercase">
                Keyboard
              </span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[9px] text-(--color-text-muted)">
                <kbd className="font-mono text-(--color-text-secondary)">Space</kbd> Play/Pause
              </span>
              <span className="text-[9px] text-(--color-text-muted)">
                <kbd className="font-mono text-(--color-text-secondary)">← →</kbd> Step
              </span>
              <span className="text-[9px] text-(--color-text-muted)">
                <kbd className="font-mono text-(--color-text-secondary)">Esc</kbd> Close
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
