import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Activity, Play, Square, AlertTriangle, TrendingUp, TrendingDown, Minus, Loader2, Box } from "lucide-react";
import { usePairs, useModels, useLivePrices, useDeployLiveSession, useStopLiveSession } from "@/api/queries";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import type { OverlayLine } from "@/components/charts/CandlestickChart";
import { TIMEFRAMES } from "@/lib/constants";
import type { LiveSignalEvent } from "@/api/schemas";

const DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"];
const EQUITY_COLORS = ["#00E5FF", "#A78BFA", "#22C55E", "#F59E0B"];

type SignalDirection = "LONG" | "SHORT" | "FLAT";

interface TradeSignal {
  time: number;
  direction: SignalDirection;
  confidence: number;
  price: number;
}

interface LiveState {
  running: boolean;
  deploying: boolean;
  sessionId: string | null;
  position: SignalDirection;
  equity: number;
  unrealizedPnl: number;
  lastSignal: TradeSignal | null;
  signals: TradeSignal[];
  equityCurve: { time: number; value: number }[];
  error: string | null;
}

const INITIAL_LIVE_STATE: LiveState = {
  running: false,
  deploying: false,
  sessionId: null,
  position: "FLAT",
  equity: 10000,
  unrealizedPnl: 0,
  lastSignal: null,
  signals: [],
  equityCurve: [],
  error: null,
};

function PositionBadge({ position }: { position: SignalDirection }) {
  const isLong = position === "LONG";
  const isShort = position === "SHORT";
  const color = isLong ? "var(--color-accent-success)" : isShort ? "var(--color-accent-danger)" : "var(--color-text-muted)";
  const icon = isLong ? <TrendingUp size={14} /> : isShort ? <TrendingDown size={14} /> : <Minus size={14} />;
  const label = isLong ? "LONG" : isShort ? "SHORT" : "FLAT";
  return (
    <div className="flex items-center gap-1.5 rounded-md border px-3 py-1.5" style={{ borderColor: color, backgroundColor: isLong ? "rgba(34,197,94,0.1)" : isShort ? "rgba(239,68,68,0.1)" : "transparent" }}>
      {icon}
      <span className="text-[11px] font-semibold uppercase" style={{ color, fontFamily: "var(--font-mono)" }}>{label}</span>
    </div>
  );
}

function SignalLog({ signals }: { signals: TradeSignal[] }) {
  const last10 = signals.slice(-10).reverse();
  if (last10.length === 0) {
    return (
      <div className="text-[10px] text-center py-4" style={{ color: "var(--color-text-muted)" }}>
        No signals yet
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-1">
      {last10.map((s, i) => {
        const isLong = s.direction === "LONG";
        const isShort = s.direction === "SHORT";
        const color = isLong ? "var(--color-accent-success)" : isShort ? "var(--color-accent-danger)" : "var(--color-text-muted)";
        const d = new Date(s.time * 1000);
        const ts = `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
        return (
          <div key={i} className="flex items-center justify-between rounded px-2 py-1" style={{ backgroundColor: "var(--color-glass-hover)" }}>
            <span className="text-[10px] font-semibold" style={{ color, fontFamily: "var(--font-mono)" }}>{s.direction}</span>
            <span className="text-[10px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
              {s.confidence.toFixed(0)}%
            </span>
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>{ts}</span>
          </div>
        );
      })}
    </div>
  );
}

export function LiveTradingPage() {
  const navigate = useNavigate();
  const { data: pairs } = usePairs();
  const deployMutation = useDeployLiveSession();
  const stopMutation = useStopLiveSession();

  const availablePairs = useMemo(
    () => (pairs ?? []).map((p) => p.pair?.symbol ?? "").filter((s) => s !== ""),
    [pairs],
  );
  const pairList = availablePairs.length > 0 ? availablePairs : DEFAULT_PAIRS;

  const { data: deployedModels, isLoading: loadingDeployed } = useQuery<Array<{
    id: string;
    model_type: string;
    best_sharpe: number | null;
    best_return: number | null;
    status: string;
    tags: string[];
    created_at: string;
    missing_on_disk?: boolean;
  }>>({
    queryKey: ["deployed-models-for-live"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ models: Array<{ id: string; model_type: string; best_sharpe: number | null; best_return: number | null; status: string; tags: string[]; created_at: string; missing_on_disk?: boolean }> }>("/models/deployed", { params: { status: "active" } });
      return data.models;
    },
    refetchOnMount: true,
  });

  const [selectedPair, setSelectedPair] = useState("EURUSD");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState("M30");
  const [live, setLive] = useState<LiveState>(INITIAL_LIVE_STATE);

  const priceRef = useRef<{ bid: number | null; ask: number | null; mid: number | null }>({ bid: null, ask: null, mid: null });
  const wsRef = useRef<WebSocket | null>(null);

  const { data: priceData } = useLivePrices([selectedPair], 50);

  useEffect(() => {
    if (!priceData?.prices?.[0]) return;
    const p = priceData.prices[0];
    priceRef.current = { bid: p.bid, ask: p.ask, mid: p.mid };
  }, [priceData]);

  useEffect(() => {
    if (!live.sessionId) return;

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.host || "localhost:8001";
    const wsUrl = `${wsProtocol}//${wsHost}/api/v1/live/${live.sessionId}/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg: LiveSignalEvent = JSON.parse(event.data);

        if (msg.event === "signal") {
          setLive((prev) => {
            const signal: TradeSignal = {
              time: msg.time,
              direction: (msg.direction as SignalDirection) || "FLAT",
              confidence: msg.confidence ?? 50,
              price: msg.price ?? 0,
            };
            const newSignals = [...prev.signals, signal];
            const newCurve = msg.equity != null
              ? [...prev.equityCurve, { time: msg.time, value: msg.equity }]
              : prev.equityCurve;

            return {
              ...prev,
              position: (msg.position as SignalDirection) || signal.direction,
              equity: msg.equity ?? prev.equity,
              unrealizedPnl: msg.pnl ?? 0,
              lastSignal: signal,
              signals: newSignals.length > 100 ? newSignals.slice(-100) : newSignals,
              equityCurve: newCurve.length > 500 ? newCurve.slice(-500) : newCurve,
            };
          });
        } else if (msg.event === "stopped") {
          setLive((prev) => ({
            ...prev,
            running: false,
            equity: msg.equity ?? prev.equity,
          }));
          ws.close();
        } else if (msg.event === "error") {
          setLive((prev) => ({ ...prev, error: msg.message ?? "WebSocket error" }));
          ws.close();
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => {
      setLive((prev) => ({ ...prev, error: "WebSocket connection failed" }));
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [live.sessionId]);

  const handleDeploy = useCallback(async () => {
    if (!selectedModelId) return;
    setLive((prev) => ({ ...prev, deploying: true, error: null }));
    try {
      const selected = deployedModels?.find((m) => m.id === selectedModelId);
      const result = await deployMutation.mutateAsync({
        pair: selectedPair,
        model: selected?.model_type ?? "logistic",
        timeframe: timeframe,
        initial_equity: 10000,
        model_id: selectedModelId,
      });
      setLive({
        running: true,
        deploying: false,
        sessionId: result.session_id,
        position: "FLAT",
        equity: 10000,
        unrealizedPnl: 0,
        lastSignal: null,
        signals: [],
        equityCurve: [{ time: Math.floor(Date.now() / 1000), value: 10000 }],
        error: null,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setLive((prev) => ({
        ...prev,
        deploying: false,
        error: detail ?? message ?? "Deploy failed",
      }));
    }
  }, [selectedPair, selectedModelId, timeframe, deployMutation, deployedModels]);

  const handleStop = useCallback(async () => {
    if (!live.sessionId) return;
    try {
      await stopMutation.mutateAsync(live.sessionId);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setLive((prev) => ({ ...prev, running: false }));
    } catch {
      setLive((prev) => ({ ...prev, running: false }));
    }
  }, [live.sessionId, stopMutation]);

  const priceDisplay = priceData?.prices?.[0];
  const midPrice = priceDisplay?.mid;
  const equityOverlay: OverlayLine[] = live.equityCurve.length > 1
    ? [{ data: live.equityCurve, color: EQUITY_COLORS[0], label: "Equity" }]
    : [];

  if (priceData?.source === "key_required") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Live Trading</h2>
        <div className="flex items-center gap-3 rounded-lg border px-4 py-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
          <AlertTriangle size={16} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>{priceData.message}</span>
          <button onClick={() => navigate("/settings")} className="text-[11px] font-medium rounded px-2 py-0.5 transition hover:underline" style={{ color: "var(--color-brand)" }}>Settings</button>
        </div>
      </div>
    );
  }

  if (priceData?.source === "unavailable") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Live Trading</h2>
        <div className="flex items-center gap-3 rounded-lg border px-4 py-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
          <AlertTriangle size={16} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>OANDA API unreachable. Check your connection and API key.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 p-6 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Live Trading</h2>
        <div className="flex items-center gap-2">
          <Activity size={14} style={{ color: live.running ? "var(--color-accent-success)" : "var(--color-text-muted)" }} />
          <span className="text-[10px] font-medium uppercase" style={{ color: live.running ? "var(--color-accent-success)" : "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
            {live.running ? "Live" : live.deploying ? "Deploying..." : "Offline"}
          </span>
        </div>
      </div>

      {live.error && (
        <div className="flex items-center gap-3 rounded-lg border px-4 py-2" style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(239,68,68,0.05)" }}>
          <AlertTriangle size={14} style={{ color: "var(--color-accent-danger)" }} />
          <span className="text-xs" style={{ color: "var(--color-accent-danger)" }}>{live.error}</span>
          <button onClick={() => setLive((p) => ({ ...p, error: null }))} className="text-[10px] underline ml-auto" style={{ color: "var(--color-text-muted)" }}>Dismiss</button>
        </div>
      )}

      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Pair</span>
          <select value={selectedPair} onChange={(e) => setSelectedPair(e.target.value)} disabled={live.running || live.deploying}
            className="rounded-md border px-2.5 py-1 text-xs transition focus:outline-none disabled:opacity-50"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
            {pairList.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Model</span>
          {loadingDeployed ? (
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Loading...</span>
          ) : !deployedModels?.length ? (
            <span className="text-xs flex items-center gap-1" style={{ color: "var(--color-accent-warning)" }}>
              <AlertTriangle size={10} />
              No active models —{" "}
              <button onClick={() => navigate("/models")} className="underline" style={{ color: "var(--color-brand)" }}>
                activate one
              </button>
            </span>
          ) : (
            <select
              value={selectedModelId ?? ""}
              onChange={(e) => setSelectedModelId(e.target.value || null)}
              disabled={live.running || live.deploying}
              className="rounded-md border px-2.5 py-1 text-xs transition focus:outline-none disabled:opacity-50"
              style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
            >
              <option value="">Select...</option>
              {deployedModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.model_type} — SR: {m.best_sharpe != null ? (m.best_sharpe >= 0 ? "+" : "") + m.best_sharpe.toFixed(2) : "—"}
                  {m.tags.length > 0 ? ` [${m.tags.join(",")}]` : ""}
                </option>
              ))}
            </select>
          )}
        </div>

        <div className="flex items-center gap-1.5 ml-2">
          {TIMEFRAMES.map((tf) => (
            <button key={tf.key} onClick={() => setTimeframe(tf.key)} disabled={live.running || live.deploying}
              className="rounded-md border px-2.5 py-0.5 text-[10px] font-medium uppercase transition-all duration-200 disabled:opacity-50"
              style={{
                borderColor: timeframe === tf.key ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: timeframe === tf.key ? "var(--color-brand-glow)" : "transparent",
                color: timeframe === tf.key ? "var(--color-brand)" : "var(--color-text-muted)",
                fontFamily: "var(--font-mono)",
              }}>
              {tf.label}
            </button>
          ))}
        </div>

        <div className="flex-1" />

        <button onClick={live.running ? handleStop : handleDeploy} disabled={live.deploying || !selectedModelId}
          className="flex items-center gap-1.5 rounded-md border px-4 py-1.5 text-[11px] font-semibold uppercase transition-all duration-200 disabled:opacity-50"
          style={{
            borderColor: live.running ? "var(--color-accent-danger)" : "var(--color-accent-success)",
            backgroundColor: live.running ? "rgba(239,68,68,0.1)" : "rgba(34,197,94,0.1)",
            color: live.running ? "var(--color-accent-danger)" : "var(--color-accent-success)",
          }}>
          {live.deploying ? (
            <><Loader2 size={12} className="animate-spin" /> Deploying</>
          ) : live.running ? (
            <><Square size={12} /> Stop</>
          ) : (
            <><Play size={12} /> Deploy</>
          )}
        </button>
      </div>

      {midPrice != null && (
        <div className="flex items-center gap-4">
          <span className="text-lg font-semibold" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
            {midPrice.toFixed(5)}
          </span>
          {priceDisplay?.change_pct != null && (
            <span className="text-[11px] font-medium" style={{
              color: priceDisplay.change_pct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
              fontFamily: "var(--font-mono)",
            }}>
              {priceDisplay.change_pct >= 0 ? "+" : ""}{priceDisplay.change_pct.toFixed(2)}%
            </span>
          )}
        </div>
      )}

      <div className="flex gap-4 flex-1 min-h-0">
        <div className="flex-1 min-w-0">
          <CandlestickChart
            pair={selectedPair}
            timeframe={timeframe}
            limit={300}
            height={520}
            overlayLines={equityOverlay}
            showToolbar={false}
          />
        </div>

        <div className="flex flex-col gap-4 w-[240px] flex-shrink-0">
          <div className="rounded-lg border p-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
            <h4 className="text-[10px] font-medium uppercase tracking-[0.12em] mb-3" style={{ color: "var(--color-text-muted)" }}>Position</h4>
            <div className="flex flex-col gap-2">
              <PositionBadge position={live.position} />
              <div className="flex justify-between">
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Equity</span>
                <span className="text-[11px] font-medium" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                  {live.equity.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Unrealized P&L</span>
                <span className="text-[11px] font-medium" style={{
                  color: live.unrealizedPnl >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                  fontFamily: "var(--font-mono)",
                }}>
                  {live.unrealizedPnl >= 0 ? "+" : ""}{live.unrealizedPnl.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Return</span>
                <span className="text-[11px] font-medium" style={{
                  color: live.equity >= 10000 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                  fontFamily: "var(--font-mono)",
                }}>
                  {((live.equity - 10000) / 10000 * 100).toFixed(2)}%
                </span>
              </div>
              {live.lastSignal && (
                <div className="flex justify-between">
                  <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Confidence</span>
                  <span className="text-[11px] font-medium" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                    {live.lastSignal.confidence.toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="rounded-lg border p-3 flex-1 min-h-0 overflow-y-auto" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
            <h4 className="text-[10px] font-medium uppercase tracking-[0.12em] mb-3" style={{ color: "var(--color-text-muted)" }}>Signal Log</h4>
            <SignalLog signals={live.signals} />
          </div>

          <div className="rounded-lg border p-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
            <h4 className="text-[10px] font-medium uppercase tracking-[0.12em] mb-2" style={{ color: "var(--color-text-muted)" }}>Configuration</h4>
            <div className="flex flex-col gap-1">
              <div className="flex justify-between">
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Pair</span>
                <span className="text-[10px] font-medium" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{selectedPair}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Model</span>
                <span className="text-[10px] font-medium" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{selectedModel}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Timeframe</span>
                <span className="text-[10px] font-medium" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{timeframe}</span>
              </div>
              {live.sessionId && (
                <div className="flex justify-between">
                  <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Session</span>
                  <span className="text-[10px] font-medium" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>{live.sessionId}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}