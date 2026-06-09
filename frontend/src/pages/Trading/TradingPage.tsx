import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Activity, AlertTriangle, ShieldOff, X, ChevronDown, ChevronRight } from "lucide-react";
import {
  usePairs,
  useLivePrices,
  useDeployPaperSession,
  useStopPaperSession,
  useDeployLiveSession,
  useStopLiveSession,
  useEmergencyKillSession,
  useLiveSentiment,
} from "@/api/queries";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import type { OverlayLine } from "@/components/charts/CandlestickChart";
import { TIMEFRAMES } from "@/lib/constants";
import type { PaperSignalEvent, LiveSignalEvent, PaperSummary, LiveSentimentResponse, LiveSentimentArticle } from "@/api/schemas";
import { useSettingsStore } from "@/stores/useSettingsStore";

import { PositionMonitor } from "./PositionMonitor";
import type { SignalDirection } from "./PositionMonitor";
import { TradeHistory } from "./TradeHistory";
import type { TradeRecord } from "./TradeHistory";
import { SessionControls } from "./SessionControls";
import type { TradingMode } from "./SessionControls";
import { RiskPanel } from "./RiskPanel";

const DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"];
const EQUITY_COLORS = ["#00E5FF", "#A78BFA", "#22C55E", "#F59E0B"];

const RISK_DEFAULTS = {
  max_position_pct: 0.25,
  max_daily_trades: 20,
  max_hourly_trades: 5,
  min_confidence: 65,
  max_drawdown_pct: 15,
  max_daily_loss_pct: 5,
  max_consecutive_losses: 5,
  rolling_window_trades: 20,
  min_rolling_sharpe: -1.0,
  min_rolling_win_rate: 0.25,
  max_consecutive_api_errors: 5,
};

interface LiveState {
  running: boolean;
  deploying: boolean;
  sessionId: string | null;
  mode: TradingMode;
  position: SignalDirection;
  equity: number;
  unrealizedPnl: number;
  initialEquity: number;
  trades: TradeRecord[];
  equityCurve: { time: number; value: number }[];
  error: string | null;
  stopResult: PaperSummary | null;
  killed: boolean;
  killReason: string;
  killLevel: string;
  riskBlockedReason: string | null;
}

const INITIAL_LIVE_STATE: LiveState = {
  running: false,
  deploying: false,
  sessionId: null,
  mode: "paper",
  position: "FLAT",
  equity: 10000,
  unrealizedPnl: 0,
  initialEquity: 10000,
  trades: [],
  equityCurve: [],
  error: null,
  stopResult: null,
  killed: false,
  killReason: "",
  killLevel: "",
  riskBlockedReason: null,
};

function StopResultModal({ summary, onClose }: { summary: PaperSummary; onClose: () => void }) {
  const metrics = [
    { label: "Sharpe", value: summary.sharpe?.toFixed(2) ?? "\u2014" },
    { label: "Sortino", value: summary.sortino?.toFixed(2) ?? "\u2014" },
    { label: "Return", value: `${summary.total_return_pct >= 0 ? "+" : ""}${summary.total_return_pct?.toFixed(2)}%` },
    { label: "Max DD", value: `${summary.max_drawdown_pct?.toFixed(2)}%` },
    { label: "Win Rate", value: `${((summary.win_rate ?? 0) * 100).toFixed(0)}%` },
    { label: "Trades", value: summary.total_trades },
    { label: "Profit Factor", value: summary.profit_factor?.toFixed(2) ?? "\u2014" },
    { label: "Avg PnL", value: `${summary.avg_trade_pnl >= 0 ? "+" : ""}${summary.avg_trade_pnl?.toFixed(2)}` },
  ];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: "rgba(0,0,0,0.6)" }}>
      <div className="rounded-sm border p-6 w-[420px] max-h-[80vh] overflow-y-auto shadow-2xl" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-surface)" }}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold uppercase tracking-[0.08em]" style={{ color: "var(--color-text-primary)" }}>Paper Trading Results</h3>
          <button onClick={onClose} aria-label="Close results" className="rounded-md p-1 transition hover:brightness-125" style={{ color: "var(--color-text-muted)" }}><X size={16} /></button>
        </div>
        <div className="grid grid-cols-2 gap-3 mb-4">
          {metrics.map((m) => (
            <div key={m.label} className="rounded-sm border px-3 py-2" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
              <div className="text-[9px] font-medium uppercase tracking-[0.1em] mb-0.5" style={{ color: "var(--color-text-muted)" }}>{m.label}</div>
              <div className="text-base font-semibold" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{String(m.value)}</div>
            </div>
          ))}
        </div>
        <div className="rounded-sm border px-3 py-2 mb-4" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
          <div className="flex justify-between">
            <span className="text-[9px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Final Equity</span>
            <span className="text-sm font-semibold" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{summary.final_equity?.toFixed(2) ?? "\u2014"}</span>
          </div>
        </div>
        <button onClick={onClose} className="w-full rounded-md border px-4 py-2 text-xs font-semibold uppercase transition hover:brightness-110" style={{ borderColor: "var(--color-glass-border)", color: "var(--color-text-secondary)", backgroundColor: "var(--color-glass)" }}>Close</button>
      </div>
    </div>
  );
}

export default function TradingPage() {
  const navigate = useNavigate();
  const { data: pairs } = usePairs();
  const deployPaper = useDeployPaperSession();
  const stopPaper = useStopPaperSession();
  const deployLive = useDeployLiveSession();
  const stopLive = useStopLiveSession();
  const emergencyKill = useEmergencyKillSession();

  const availablePairs = useMemo(
    () => (pairs ?? []).map((p) => p.pair?.symbol ?? "").filter((s) => s !== ""),
    [pairs],
  );
  const pairList = availablePairs.length > 0 ? availablePairs : DEFAULT_PAIRS;

  const { data: deployedModels, isLoading: loadingDeployed } = useQuery<Array<{
    id: string; model_type: string; best_sharpe: number | null; best_return: number | null;
    status: string; tags: string[]; created_at: string; missing_on_disk?: boolean;
  }>>({
    queryKey: ["deployed-models-for-live"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ models: Array<{
        id: string; model_type: string; best_sharpe: number | null; best_return: number | null;
        status: string; tags: string[]; created_at: string; missing_on_disk?: boolean;
      }> }>("/models/deployed", { params: { status: "active" } });
      return data.models;
    },
    refetchOnMount: true,
  });

  const [selectedPair, setSelectedPair] = useState("EURUSD");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState("M30");
  const [positionSizing, setPositionSizing] = useState("fixed");
  const [initialEquity, setInitialEquity] = useState(10000);
  const [tradingMode, setTradingMode] = useState<TradingMode>("paper");
  const [showRiskConfig, setShowRiskConfig] = useState(false);
  const [riskConfig, setRiskConfig] = useState(RISK_DEFAULTS);
  const [confirmLive, setConfirmLive] = useState(false);
  const [live, setLive] = useState<LiveState>(INITIAL_LIVE_STATE);
  const newsBlendEnabled = useSettingsStore((s) => s.liveNewsBlendEnabled);
  const newsBlendWeight = useSettingsStore((s) => s.liveNewsBlendWeight);
  const setField = useSettingsStore((s) => s.setField);
  const { data: sentiment } = useLiveSentiment(selectedPair);

  const wsRef = useRef<WebSocket | null>(null);
  const { data: priceData } = useLivePrices([selectedPair], 50);
  const wsPrefix = tradingMode === "paper" ? "trading/paper" : "trading/live";

  useEffect(() => {
    if (!live.sessionId) return;
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.host || "localhost:8001";
    const wsUrl = `${wsProtocol}//${wsHost}/api/v1/${wsPrefix}/${live.sessionId}/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const raw = JSON.parse(event.data);
        if (tradingMode === "paper") handlePaperWSEvent(raw as PaperSignalEvent);
        else handleLiveWSEvent(raw as LiveSignalEvent);
      } catch { /* ignore */ }
    };

    ws.onerror = () => setLive((prev) => ({ ...prev, error: "WebSocket connection failed" }));
    ws.onclose = () => { wsRef.current = null; };
    return () => { ws.close(); wsRef.current = null; };
  }, [live.sessionId, wsPrefix, tradingMode]);

  function paperTradeFromEvent(e: PaperSignalEvent): TradeRecord {
    return {
      trade_id: e.trade_id ?? "", direction: (e.direction as SignalDirection) || "FLAT",
      size: e.size ?? 0, entry_price: e.entry_price ?? 0, exit_price: e.exit_price ?? null,
      pnl: e.pnl ?? 0, exit_reason: e.exit_reason ?? "", time: e.time ?? Math.floor(Date.now() / 1000),
    };
  }

  function handlePaperSubEvents(subs: PaperSignalEvent[]) {
    for (const sub of subs) {
      if (sub.event === "trade_closed") {
        setLive((prev) => ({ ...prev, trades: [...prev.trades, paperTradeFromEvent(sub)].slice(-200), position: "FLAT" }));
      } else if (sub.event === "trade_opened") {
        setLive((prev) => ({ ...prev, position: (sub.direction as SignalDirection) || prev.position }));
      }
    }
  }

  function handlePaperWSEvent(msg: PaperSignalEvent) {
    if (msg.event === "signal" || msg.event === "hold") {
      setLive((prev) => {
        const eq = msg.equity ?? prev.equity;
        const now = msg.time ?? Math.floor(Date.now() / 1000);
        return { ...prev, position: (msg.position as SignalDirection) || prev.position, equity: eq, unrealizedPnl: msg.unrealized_pnl ?? prev.unrealizedPnl, equityCurve: [...prev.equityCurve, { time: now, value: eq }].slice(-500), stopResult: null };
      });
      if (msg.sub_events) handlePaperSubEvents(msg.sub_events);
    } else if (msg.event === "trade_closed") {
      setLive((prev) => ({ ...prev, trades: [...prev.trades, paperTradeFromEvent(msg)].slice(-200), position: "FLAT" }));
    } else if (msg.event === "stopped") {
      setLive((prev) => ({ ...prev, running: false }));
    }
  }

  function handleLiveWSEvent(msg: LiveSignalEvent) {
    if (msg.event === "signal" || msg.event === "hold") {
      setLive((prev) => {
        const eq = msg.equity ?? prev.equity;
        const now = msg.time ?? Math.floor(Date.now() / 1000);
        return { ...prev, position: (msg.position as SignalDirection) || prev.position, equity: eq, unrealizedPnl: msg.unrealized_pnl ?? prev.unrealizedPnl, equityCurve: [...prev.equityCurve, { time: now, value: eq }].slice(-500) };
      });
      if (msg.sub_events) for (const sub of msg.sub_events) handleLiveWSEvent(sub);
    } else if (msg.event === "risk_blocked") {
      setLive((prev) => ({ ...prev, riskBlockedReason: msg.reason ?? "risk_blocked" }));
      setTimeout(() => setLive((p) => ({ ...p, riskBlockedReason: null })), 3000);
    } else if (msg.event === "order_placed") {
      const newTrade: TradeRecord = {
        trade_id: msg.oanda_order_id ?? `${Date.now()}`,
        direction: (msg.direction as SignalDirection) || "FLAT",
        size: msg.units ?? 0, entry_price: msg.price ?? 0, exit_price: null, pnl: 0,
        exit_reason: "", time: msg.time ?? Math.floor(Date.now() / 1000),
        oanda_order_id: msg.oanda_order_id,
      };
      setLive((prev) => ({ ...prev, trades: [...prev.trades, newTrade].slice(-200) }));
    } else if (msg.event === "trade_closed") {
      const newTrade: TradeRecord = {
        trade_id: `close_${Date.now()}`, direction: (msg.direction as SignalDirection) || "FLAT",
        size: msg.units ?? 0, entry_price: 0, exit_price: msg.price ?? null, pnl: msg.pnl ?? 0,
        exit_reason: msg.exit_reason ?? "signal", time: msg.time ?? Math.floor(Date.now() / 1000),
      };
      setLive((prev) => ({ ...prev, trades: [...prev.trades, newTrade].slice(-200), position: "FLAT" }));
    } else if (msg.event === "kill") {
      setLive((prev) => ({ ...prev, running: false, killed: true, killReason: msg.reason ?? "kill_signal", killLevel: msg.level ?? "" }));
    } else if (msg.event === "stopped") {
      setLive((prev) => ({ ...prev, running: false }));
    }
  }

  const handleDeploy = useCallback(async () => {
    if (!selectedModelId) return;
    if (tradingMode === "live" && !confirmLive) { setConfirmLive(true); return; }
    setConfirmLive(false);
    setLive((prev) => ({ ...prev, deploying: true, error: null, killed: false, killReason: "", killLevel: "" }));
    try {
      const selected = deployedModels?.find((m) => m.id === selectedModelId);
      const basePayload = {
        pair: selectedPair, model_id: selectedModelId, model_type: selected?.model_type ?? "logistic",
        timeframe, initial_equity: initialEquity, position_sizing: positionSizing, sizing_config: {},
        live_news_blend_enabled: newsBlendEnabled,
        live_news_blend_weight: newsBlendWeight,
      };
      let result: { session_id: string };
      if (tradingMode === "paper") {
        result = await deployPaper.mutateAsync(basePayload);
      } else {
        result = await deployLive.mutateAsync({
          ...basePayload, mode: "demo",
          risk_config: riskConfig as unknown as Record<string, unknown>,
        });
      }
      setLive({
        running: true, deploying: false, mode: tradingMode,
        sessionId: result.session_id, position: "FLAT", equity: initialEquity,
        unrealizedPnl: 0, initialEquity, trades: [],
        equityCurve: [{ time: Math.floor(Date.now() / 1000), value: initialEquity }],
        error: null, stopResult: null, killed: false, killReason: "", killLevel: "",
        riskBlockedReason: null,
      });
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setLive((prev) => ({ ...prev, deploying: false, error: detail ?? String(err) }));
    }
  }, [selectedPair, selectedModelId, timeframe, initialEquity, positionSizing, tradingMode, confirmLive, riskConfig, deployPaper, deployLive, deployedModels, newsBlendEnabled, newsBlendWeight]);

  const handleStop = useCallback(async () => {
    if (!live.sessionId) return;
    try {
      if (tradingMode === "paper") {
        const r = await stopPaper.mutateAsync(live.sessionId);
        if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
        setLive((prev) => ({ ...prev, running: false, stopResult: (r as unknown as { summary: PaperSummary }).summary }));
      } else {
        const r = await stopLive.mutateAsync(live.sessionId);
        if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
        setLive((prev) => ({ ...prev, running: false, killed: r.killed }));
      }
    } catch { setLive((prev) => ({ ...prev, running: false })); }
  }, [live.sessionId, stopPaper, stopLive, tradingMode]);

  const handleEmergency = useCallback(async () => {
    if (!live.sessionId) return;
    try {
      await emergencyKill.mutateAsync(live.sessionId);
      if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
      setLive((prev) => ({ ...prev, running: false, killed: true, killReason: "emergency_kill_button", killLevel: "K1" }));
    } catch { /* ignore */ }
  }, [live.sessionId, emergencyKill]);

  const priceDisplay = priceData?.prices?.[0];
  const midPrice = priceDisplay?.mid;
  const equityOverlay: OverlayLine[] = live.equityCurve.length > 1
    ? [{ data: live.equityCurve, color: EQUITY_COLORS[0], label: "Equity" }] : [];

  const isRunning = live.running;
  const hasModel = selectedModelId != null;

  if (priceData?.source === "key_required") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Trading</h2>
        <div className="flex items-center gap-3 rounded-sm border px-4 py-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
          <AlertTriangle size={16} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>{priceData?.message}</span>
          <button onClick={() => navigate("/settings")} className="text-[11px] font-medium rounded px-2 py-0.5 transition hover:underline" style={{ color: "var(--color-brand)" }}>Settings</button>
        </div>
      </div>
    );
  }
  if (priceData?.source === "unavailable") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Trading</h2>
        <div className="flex items-center gap-3 rounded-sm border px-4 py-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
          <AlertTriangle size={16} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>OANDA API unreachable</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 h-full">
      {live.stopResult && tradingMode === "paper" && (
        <StopResultModal summary={live.stopResult} onClose={() => setLive((p) => ({ ...p, stopResult: null }))} />
      )}
      {confirmLive && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ backgroundColor: "rgba(0,0,0,0.6)" }}>
          <div className="rounded-sm border p-6 w-[380px] shadow-2xl" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-surface)" }}>
            <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--color-accent-danger)" }}>Confirm Live Trading</h3>
            <p className="text-xs mb-4" style={{ color: "var(--color-text-secondary)" }}>This will submit real orders to your OANDA account. Positions will be opened and closed by the model automatically.</p>
            <div className="flex gap-3">
              <button onClick={() => setConfirmLive(false)} className="flex-1 rounded-md border px-4 py-2 text-xs font-semibold uppercase" style={{ borderColor: "var(--color-glass-border)", color: "var(--color-text-secondary)", backgroundColor: "var(--color-glass)" }}>Cancel</button>
              <button onClick={handleDeploy} className="flex-1 rounded-md px-4 py-2 text-xs font-semibold uppercase" style={{ backgroundColor: "var(--color-accent-danger)", color: "var(--color-text-inverse)" }}>Confirm & Deploy</button>
            </div>
          </div>
        </div>
      )}

      {/* Title + status */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Trading</h2>
        <div className="flex items-center gap-2">
          <Activity size={14} style={{ color: isRunning ? "var(--color-accent-success)" : "var(--color-text-muted)" }} />
          <span className="text-[10px] font-medium uppercase" style={{ color: isRunning ? "var(--color-accent-success)" : "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
            {isRunning ? "Live" : live.deploying ? "Deploying..." : live.killed ? "KILLED" : "Offline"}
          </span>
        </div>
      </div>

      {/* Alerts */}
      {live.killed && (
        <div className="flex items-center gap-3 rounded-sm border px-4 py-2" style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(239,68,68,0.08)" }}>
          <ShieldOff size={14} style={{ color: "var(--color-accent-danger)" }} />
          <span className="text-xs font-semibold" style={{ color: "var(--color-accent-danger)" }}>Session killed — {live.killReason}</span>
          <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>Level: {live.killLevel}</span>
        </div>
      )}
      {live.riskBlockedReason && (
        <div className="flex items-center gap-2 rounded-sm border px-4 py-2" style={{ borderColor: "var(--color-accent-warning)", backgroundColor: "rgba(245,158,11,0.06)" }}>
          <AlertTriangle size={12} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-xs" style={{ color: "var(--color-accent-warning)", fontFamily: "var(--font-mono)" }}>Risk blocked: {live.riskBlockedReason}</span>
        </div>
      )}
      {live.error && (
        <div className="flex items-center gap-3 rounded-sm border px-4 py-2" style={{ borderColor: "var(--color-accent-danger)", backgroundColor: "rgba(239,68,68,0.05)" }}>
          <AlertTriangle size={14} style={{ color: "var(--color-accent-danger)" }} />
          <span className="text-xs" style={{ color: "var(--color-accent-danger)" }}>{live.error}</span>
          <button onClick={() => setLive((p) => ({ ...p, error: null }))} className="text-[10px] underline ml-auto" style={{ color: "var(--color-text-muted)" }}>Dismiss</button>
        </div>
      )}

      {/* Pair / Model / Timeframe */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Pair</span>
          <select value={selectedPair} onChange={(e) => setSelectedPair(e.target.value)} disabled={isRunning || live.deploying} aria-label="Select trading pair"
            className="rounded-md border px-2.5 py-1 text-xs transition focus:outline-none disabled:opacity-50"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
            {pairList.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Model</span>
          {loadingDeployed ? <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Loading...</span> : !deployedModels?.length ? (
            <span className="text-xs flex items-center gap-1" style={{ color: "var(--color-accent-warning)" }}>
              <AlertTriangle size={10} />No active models — <button onClick={() => navigate("/models")} className="underline" style={{ color: "var(--color-brand)" }}>activate one</button>
            </span>
          ) : (
            <select value={selectedModelId ?? ""} onChange={(e) => setSelectedModelId(e.target.value || null)} disabled={isRunning || live.deploying} aria-label="Select model"
              className="rounded-md border px-2.5 py-1 text-xs transition focus:outline-none disabled:opacity-50"
              style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
              <option value="">Select...</option>
              {deployedModels.map((m) => (
                <option key={m.id} value={m.id}>{m.model_type} — SR: {m.best_sharpe != null ? (m.best_sharpe >= 0 ? "+" : "") + m.best_sharpe.toFixed(2) : "\u2014"}{m.tags.length > 0 ? ` [${m.tags.join(",")}]` : ""}</option>
              ))}
            </select>
          )}
        </div>
        <div className="flex items-center gap-1.5 ml-2">
          {TIMEFRAMES.map((tf) => (
            <button key={tf.key} onClick={() => setTimeframe(tf.key)} disabled={isRunning || live.deploying}
              className="rounded-md border px-2.5 py-0.5 text-[10px] font-medium uppercase transition-all duration-200 disabled:opacity-50"
              style={{
                borderColor: timeframe === tf.key ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: timeframe === tf.key ? "var(--color-brand-glow)" : "transparent",
                color: timeframe === tf.key ? "var(--color-brand)" : "var(--color-text-muted)", fontFamily: "var(--font-mono)",
              }}>{tf.label}</button>
          ))}
        </div>
      </div>

      {/* Session controls (mode, sizing, equity, deploy/stop/kill) */}
      <SessionControls
        mode={tradingMode}
        isRunning={isRunning}
        isDeploying={live.deploying}
        isKilled={live.killed}
        hasModel={hasModel}
        positionSizing={positionSizing}
        initialEquity={initialEquity}
        onModeChange={setTradingMode}
        onSizingChange={setPositionSizing}
        onEquityChange={setInitialEquity}
        onDeploy={handleDeploy}
        onStop={handleStop}
        onEmergency={handleEmergency}
      />

      {/* Risk config (live mode only, pre-deploy) */}
      {tradingMode === "live" && !isRunning && (
        <RiskPanel
          show={showRiskConfig}
          config={riskConfig}
          onChange={(update) => setRiskConfig((prev) => ({ ...prev, ...update }))}
          onToggle={() => setShowRiskConfig((v) => !v)}
        />
      )}

      {/* Price line */}
      {midPrice != null && (
        <div className="flex items-center gap-4">
          <span className="text-lg font-semibold" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{midPrice.toFixed(5)}</span>
          {priceDisplay?.change_pct != null && (
            <span className="text-[11px] font-medium" style={{ color: priceDisplay.change_pct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>{priceDisplay.change_pct >= 0 ? "+" : ""}{priceDisplay.change_pct.toFixed(2)}%</span>
          )}
          {priceDisplay?.bid != null && (
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>B: {priceDisplay.bid.toFixed(5)} / A: {priceDisplay.ask.toFixed(5)}</span>
          )}
          {tradingMode === "live" && <span className="text-[10px] font-semibold uppercase" style={{ color: "var(--color-accent-danger)" }}>[OANDA]</span>}
        </div>
      )}

      {/* Main content: chart + sidebar panels */}
      <div className="flex gap-4 flex-1 min-h-0">
        <div className="flex-1 min-w-0">
          <CandlestickChart pair={selectedPair} timeframe={timeframe} limit={300} height={520} overlayLines={equityOverlay} showToolbar={false} />
        </div>

        <div className="flex flex-col gap-4 w-[280px] flex-shrink-0">
          <PositionMonitor
            position={live.position}
            equity={live.equity}
            unrealizedPnl={live.unrealizedPnl}
            initialEquity={live.initialEquity}
            tradeCount={live.trades.length}
          />

          <TradeHistory trades={live.trades} />

          {/* News Sentiment sidebar — toggle + weight only */}
          <div className="rounded-sm border p-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
            <h4 className="text-[10px] font-medium uppercase tracking-[0.12em] mb-2" style={{ color: "var(--color-text-muted)" }}>News Sentiment</h4>
            <label className="flex items-center gap-2 cursor-pointer mb-2">
              <input type="checkbox" checked={newsBlendEnabled} onChange={(e) => setField("liveNewsBlendEnabled", e.target.checked)} disabled={isRunning || live.deploying}
                className="rounded" style={{ accentColor: "var(--color-brand)" }} />
              <span className="text-[10px]" style={{ color: newsBlendEnabled ? "var(--color-text-primary)" : "var(--color-text-muted)" }}>Blend into signals</span>
            </label>
            {newsBlendEnabled && (
              <div className="flex flex-col gap-1">
                <div className="flex justify-between">
                  <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>Influence</span>
                  <span className="text-[9px] tabular-nums" style={{ color: "var(--color-brand)", fontFamily: "var(--font-mono)" }}>{(newsBlendWeight * 100).toFixed(0)}%</span>
                </div>
                <input type="range" min="0" max="0.30" step="0.01" value={newsBlendWeight}
                  onChange={(e) => setField("liveNewsBlendWeight", parseFloat(e.target.value))}
                  disabled={isRunning || live.deploying}
                  className="w-full h-1 rounded-full appearance-none cursor-pointer disabled:opacity-50"
                  style={{ accentColor: "var(--color-brand)", background: "var(--color-glass-hover)" }} />
              </div>
            )}
            {sentiment?.pairs?.[selectedPair] && (
              <div className="mt-2 rounded-sm px-2 py-1 flex items-center justify-between" style={{ backgroundColor: "var(--color-glass-hover)" }}>
                <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>Current</span>
                <span className="text-[10px] font-semibold tabular-nums" style={{
                  color: sentiment.pairs[selectedPair].blended_sentiment >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                  fontFamily: "var(--font-mono)",
                }}>
                  {sentiment.pairs[selectedPair].blended_sentiment > 0 ? "+" : ""}
                  {sentiment.pairs[selectedPair].blended_sentiment.toFixed(2)}
                </span>
              </div>
            )}
          </div>

          <div className="rounded-sm border p-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
            <h4 className="text-[10px] font-medium uppercase tracking-[0.12em] mb-2" style={{ color: "var(--color-text-muted)" }}>Configuration</h4>
            <div className="flex flex-col gap-1">
              <Row label="Mode" value={tradingMode === "paper" ? "Paper" : "Live (Demo)"} />
              <Row label="Pair" value={selectedPair} />
              <Row label="Timeframe" value={timeframe} />
              {live.sessionId && <Row label="Session" value={live.sessionId} muted />}
              {newsBlendEnabled && <Row label="News Blend" value={`${(newsBlendWeight * 100).toFixed(0)}%`} muted />}
            </div>
          </div>
        </div>
      </div>

      {/* ── Full-width News Sentiment Panel (below chart) ── */}
      <NewsSentimentPanel pair={selectedPair} sentiment={sentiment} />
    </div>
  );
}

function Row({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className="flex justify-between">
      <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span className="text-[10px] font-medium" style={{ color: muted ? "var(--color-text-muted)" : "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{value}</span>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
// News Sentiment Panel
// ════════════════════════════════════════════════════════════════════

function NewsSentimentPanel({ pair, sentiment }: { pair: string; sentiment?: LiveSentimentResponse | null }) {
  const [expandedTiers, setExpandedTiers] = useState<Record<string, boolean>>({ exact: true });
  const [panelOpen, setPanelOpen] = useState(true);

  if (!sentiment) return null;

  const pairData = sentiment.pairs?.[pair];
  const articles = (sentiment.top_articles ?? []) as LiveSentimentArticle[];
  const tierCounts = sentiment.article_count_by_tier ?? pairData?.article_count_by_tier;
  const tierLabels: Record<string, string> = {
    exact: `${pair} articles`,
    partial: "Related currency",
    other: "Other / untagged",
  };

  const tiers = [
    { key: "exact", label: tierLabels.exact, filter: (a: LiveSentimentArticle) => a.relevance_tier === 1 },
    { key: "partial", label: tierLabels.partial, filter: (a: LiveSentimentArticle) => a.relevance_tier === 2 },
    { key: "other", label: tierLabels.other, filter: (a: LiveSentimentArticle) => (a.relevance_tier ?? 0) === 0 },
  ];

  const toggle = (key: string) => setExpandedTiers((p) => ({ ...p, [key]: !p[key] }));

  return (
    <div className="rounded-sm border" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
      <button onClick={() => setPanelOpen((v) => !v)} className="flex items-center gap-2 w-full px-4 py-2 text-left">
        {panelOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <h4 className="text-[10px] font-medium uppercase tracking-[0.12em]" style={{ color: "var(--color-text-muted)" }}>News Sentiment</h4>
        <span className="text-[10px] tabular-nums" style={{ color: "var(--color-text-dim)", fontFamily: "var(--font-mono)" }}>
          {articles.length} articles
        </span>
        <span className="text-[8px] uppercase ml-auto" style={{ color: "var(--color-text-dim)" }}>{sentiment.backend ?? "vader"}</span>
      </button>

      {panelOpen && (
        <div className="px-4 pb-3">
          {/* Sentiment score + contributions */}
          {pairData && (
            <div className="flex flex-wrap items-center gap-x-6 gap-y-1 mb-3">
              <div className="flex items-center gap-2">
                <span className="text-[9px] uppercase tracking-[0.08em]" style={{ color: "var(--color-text-muted)" }}>Blended</span>
                <span className="text-sm font-semibold tabular-nums" style={{
                  color: pairData.blended_sentiment >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
                  fontFamily: "var(--font-mono)",
                }}>
                  {pairData.blended_sentiment > 0 ? "+" : ""}{pairData.blended_sentiment.toFixed(3)}
                </span>
              </div>
              {pairData.vader_contribution != null && pairData.llm_contribution != null && (
                <div className="flex items-center gap-3 text-[10px]" style={{ color: "var(--color-text-dim)", fontFamily: "var(--font-mono)" }}>
                  <span>VADER: {pairData.vader_contribution > 0 ? "+" : ""}{pairData.vader_contribution.toFixed(2)}</span>
                  <span>LLM: {pairData.llm_contribution > 0 ? "+" : ""}{pairData.llm_contribution.toFixed(2)}</span>
                </div>
              )}
              <div className="flex items-center gap-3 text-[9px]" style={{ color: "var(--color-text-dim)" }}>
                <span>VADER score: {pairData.vader_sentiment > 0 ? "+" : ""}{pairData.vader_sentiment.toFixed(3)}</span>
                <span>magnitude: {pairData.vader_magnitude.toFixed(2)}</span>
                {pairData.llm_sentiment != null && (
                  <span>LLM dir: {pairData.llm_sentiment > 0 ? "+" : ""}{pairData.llm_sentiment.toFixed(2)}</span>
                )}
              </div>
            </div>
          )}

          {/* Tier counts + articles */}
          {articles.length > 0 ? (
            <div className="flex flex-col gap-0 overflow-y-auto max-h-[420px]">
              {tiers.map(({ key, label, filter }) => {
                const tierArticles = articles.filter(filter);
                if (tierArticles.length === 0) return null;
                const count = tierCounts?.[key as keyof typeof tierCounts] ?? tierArticles.length;
                const isExpanded = expandedTiers[key] ?? (key === "exact");
                return (
                  <div key={key}>
                    <button onClick={() => toggle(key)} className="flex items-center gap-1.5 w-full text-left py-1.5 border-b" style={{ borderColor: "var(--color-glass-border)" }}>
                      {isExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                      <span className="text-[9px] uppercase tracking-[0.06em] font-medium" style={{ color: "var(--color-text-muted)" }}>{label}</span>
                      <span className="text-[8px] tabular-nums" style={{ color: "var(--color-text-dim)", fontFamily: "var(--font-mono)" }}>({count})</span>
                    </button>
                    {isExpanded && tierArticles.map((a, i) => (
                      <div key={`${a.title}-${i}`} className="flex items-center gap-2 px-3 py-1.5 border-b" style={{ borderColor: "var(--color-glass-border)" }}>
                        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: a.sentiment_score >= 0.05 ? "var(--color-accent-success)" : a.sentiment_score <= -0.05 ? "var(--color-accent-danger)" : "var(--color-text-dim)" }} />
                        <span className="text-[9px] truncate flex-1" style={{ color: "var(--color-text-secondary)" }}>{a.title}</span>
                        <span className="text-[8px] tabular-nums flex-shrink-0" style={{ color: "var(--color-text-dim)", fontFamily: "var(--font-mono)" }}>
                          {formatTimeAgo(a.timestamp)}
                        </span>
                        <span className="text-[9px] font-semibold tabular-nums w-[42px] text-right flex-shrink-0" style={{
                          color: a.sentiment_score >= 0.05 ? "var(--color-accent-success)" : a.sentiment_score <= -0.05 ? "var(--color-accent-danger)" : "var(--color-text-dim)",
                          fontFamily: "var(--font-mono)",
                        }}>
                          {a.sentiment_score > 0 ? "+" : ""}{a.sentiment_score.toFixed(2)}
                        </span>
                        <span className="text-[7px] uppercase w-[48px] text-right flex-shrink-0" style={{ color: "var(--color-text-dim)" }}>{a.source}</span>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          ) : (
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>No articles available. RSS feeds will populate over time.</span>
          )}
        </div>
      )}
    </div>
  );
}

function formatTimeAgo(ts: string): string {
  try {
    const d = new Date(ts);
    const now = Date.now();
    const diff = now - d.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h`;
    return `${Math.floor(hrs / 24)}d`;
  } catch {
    return "";
  }
}
