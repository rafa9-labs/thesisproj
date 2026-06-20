import { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { AlertTriangle, ShieldOff, X, ChevronDown, ChevronUp } from "lucide-react";
import { useAppStore } from "@/stores/useAppStore";
import {
  usePairs,
  useLivePrices,
  useDeployPaperSession,
  useStopPaperSession,
  useDeployLiveSession,
  useStopLiveSession,
  useEmergencyKillSession,
  useSavedCommittees,
} from "@/api/queries";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import type { OverlayLine } from "@/components/charts/CandlestickChart";
import type { PaperSignalEvent, LiveSignalEvent, PaperSummary, CandleBar } from "@/api/schemas";

import type { TradingMode } from "./SessionControls";
import type { SignalDirection } from "./PositionMonitor";
import type { TradeRecord } from "./TradeHistory";
import { MarketBar } from "./MarketBar";
import { OrderPanel, type RiskConfig } from "./OrderPanel";
import { TradeTable } from "./TradeTable";

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
  isCommittee: boolean;
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

export interface ChartMarker {
  time: number;
  position: "aboveBar" | "belowBar" | "inBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle";
  text: string;
  size?: number;
}

const INITIAL_LIVE_STATE: LiveState = {
  running: false,
  deploying: false,
  sessionId: null,
  mode: "paper",
  isCommittee: false,
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
    {
      label: "Return",
      value: `${summary.total_return_pct >= 0 ? "+" : ""}${summary.total_return_pct?.toFixed(2)}%`,
    },
    { label: "Max DD", value: `${summary.max_drawdown_pct?.toFixed(2)}%` },
    { label: "Win Rate", value: `${((summary.win_rate ?? 0) * 100).toFixed(0)}%` },
    { label: "Trades", value: summary.total_trades },
    { label: "Profit Factor", value: summary.profit_factor?.toFixed(2) ?? "\u2014" },
    {
      label: "Avg PnL",
      value: `${summary.avg_trade_pnl >= 0 ? "+" : ""}${summary.avg_trade_pnl?.toFixed(2)}`,
    },
  ];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.6]">
      <div className="max-h-[80vh] w-[420px] overflow-y-auto rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-6 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
            Paper Trading Results
          </h3>
          <button
            onClick={onClose}
            aria-label="Close results"
            className="rounded-md p-1 text-(--color-text-muted) transition hover:brightness-125"
          >
            <X size={16} />
          </button>
        </div>
        <div className="mb-4 grid grid-cols-2 gap-3">
          {metrics.map((m) => (
            <div
              key={m.label}
              className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-3 py-2"
            >
              <div className="mb-0.5 text-[9px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                {m.label}
              </div>
              <div className="font-mono text-base font-semibold text-(--color-text-primary)">
                {String(m.value)}
              </div>
            </div>
          ))}
        </div>
        <div className="mb-4 rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-3 py-2">
          <div className="flex justify-between">
            <span className="text-[9px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
              Final Equity
            </span>
            <span className="font-mono text-sm font-semibold text-(--color-text-primary)">
              {summary.final_equity?.toFixed(2) ?? "\u2014"}
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="w-full rounded-md border border-(--color-glass-border) bg-(--color-glass) px-4 py-2 text-xs font-semibold text-(--color-text-secondary) uppercase transition hover:brightness-110"
        >
          Close
        </button>
      </div>
    </div>
  );
}

export default function TradingPage() {
  const navigate = useNavigate();
  const { data: pairs } = usePairs();
  const demoMode = useAppStore((s) => s.demoMode);
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

  const { data: deployedModels, isLoading: loadingDeployed } = useQuery<
    Array<{
      id: string;
      model_type: string;
      best_sharpe: number | null;
      best_return: number | null;
      status: string;
      tags: string[];
      created_at: string;
      missing_on_disk?: boolean;
    }>
  >({
    queryKey: ["deployed-models-for-live"],
    queryFn: async () => {
      const { data } = await apiClient.get<{
        models: Array<{
          id: string;
          model_type: string;
          best_sharpe: number | null;
          best_return: number | null;
          status: string;
          tags: string[];
          created_at: string;
          missing_on_disk?: boolean;
        }>;
      }>("/models/deployed");
      return data.models;
    },
    refetchOnMount: true,
  });

  const [selectedPair, setSelectedPair] = useState("EURUSD");
  const [deployType, setDeployType] = useState<"model" | "committee">("model");
  const [selectedCommittee, setSelectedCommittee] = useState<string>("");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState("M30");
  const [searchParams] = useSearchParams();

  // Pre-select model from URL param (e.g. /trading?modelId=abc123)
  useEffect(() => {
    const mid = searchParams.get("modelId");
    if (mid && deployedModels?.some((m) => m.id === mid)) {
      setSelectedModelId(mid);
    }
  }, [searchParams, deployedModels]);
  const [positionSizing, setPositionSizing] = useState("fixed");
  const [initialEquity, setInitialEquity] = useState(10000);
  const [tradingMode, setTradingMode] = useState<TradingMode>("paper");
  const [riskConfig, setRiskConfig] = useState(RISK_DEFAULTS);
  const [confirmLive, setConfirmLive] = useState(false);
  const [live, setLive] = useState<LiveState>(INITIAL_LIVE_STATE);
  const [liveCandle, setLiveCandle] = useState<CandleBar | null>(null);
  const [chartMarkers, setChartMarkers] = useState<ChartMarker[]>([]);

  const { data: committeeData } = useSavedCommittees();
  const committeeList = committeeData?.committees?.map((c) => c.name) ?? [];

  useEffect(() => {
    if (committeeList.length === 0 && deployType === "committee") {
      setDeployType("model");
      setSelectedCommittee("");
    }
  }, [committeeList, deployType]);

  const recoverRef = useRef(false);
  useEffect(() => {
    if (recoverRef.current) return;
    recoverRef.current = true;
    const recover = async () => {
      try {
        const [liveRes, paperRes] = await Promise.all([
          apiClient.get("/trading/live/sessions").catch(() => ({ data: [] })),
          apiClient.get("/trading/paper/sessions").catch(() => ({ data: [] })),
        ]);
        const liveSessions: Array<Record<string, unknown>> = Array.isArray(liveRes.data)
          ? liveRes.data
          : [];
        const paperSessions: Array<Record<string, unknown>> = Array.isArray(paperRes.data)
          ? paperRes.data
          : [];
        const allSessions = [
          ...liveSessions.map((s: Record<string, unknown>) => ({ ...s, _isCommittee: s.model_type === "committee" })),
          ...paperSessions.map((s: Record<string, unknown>) => ({ ...s, mode: "paper", _isCommittee: false })),
        ];
        if (allSessions.length === 0) return;
        const running = allSessions.find(
          (s: Record<string, unknown>) => s.status === "running",
        );
        if (running) {
          const initEq = Number(running.initial_equity) || 10000;
          setInitialEquity(initEq);
          setLive((prev) => ({
            ...prev,
            running: true,
            sessionId: running.session_id as string,
            mode: (running.mode as TradingMode) || "paper",
            isCommittee: !!running._isCommittee,
            equity: initEq,
            initialEquity: initEq,
          }));
        }
      } catch {
        /* ignore recovery errors */
      }
    };
    recover();
  }, []);

  const [chartFraction, setChartFraction] = useState(0.7);
  const [tableMinimized, setTableMinimized] = useState(false);
  const splitDragRef = useRef({ active: false, startY: 0, startFrac: 0.7 });
  const leftColRef = useRef<HTMLDivElement>(null);

  const onSplitMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      splitDragRef.current = { active: true, startY: e.clientY, startFrac: chartFraction };
    },
    [chartFraction],
  );

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = splitDragRef.current;
      if (!d.active || !leftColRef.current) return;
      const rect = leftColRef.current.getBoundingClientRect();
      const total = rect.height;
      if (total <= 0) return;
      const dy = e.clientY - d.startY;
      const newFrac = Math.max(0.25, Math.min(0.85, d.startFrac + dy / total));
      setChartFraction(newFrac);
    };
    const onUp = () => {
      splitDragRef.current.active = false;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }, []);

  const wsRef = useRef<WebSocket | null>(null);
  const { data: priceData } = useLivePrices([selectedPair], 50, !demoMode);
  const wsPrefix = live.isCommittee
    ? "trading/live"
    : tradingMode === "paper"
      ? "trading/paper"
      : "trading/live";

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
      } catch {
        /* ignore */
      }
    };

    ws.onerror = () => setLive((prev) => ({ ...prev, error: "WebSocket connection failed" }));
    ws.onclose = () => {
      wsRef.current = null;
    };
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [live.sessionId, wsPrefix, tradingMode]);

  const signalMarker = (time: number, direction?: string, confidence?: number): ChartMarker => ({
    time,
    position: direction === "SHORT" ? "aboveBar" : "belowBar",
    color: direction === "LONG" ? "#22c55e" : direction === "SHORT" ? "#ef4444" : "#6b7280",
    shape: direction === "LONG" ? "arrowUp" : direction === "SHORT" ? "arrowDown" : "circle",
    text: direction === "FLAT" ? "FLAT" : `${Math.round(confidence ?? 0)}%`,
  });

  const tradeEntryMarker = (time: number, _price: number, direction?: string): ChartMarker => ({
    time,
    position: direction === "SHORT" ? "aboveBar" : "belowBar",
    color: direction === "LONG" ? "#16a34a" : "#dc2626",
    shape: direction === "LONG" ? "arrowUp" : "arrowDown",
    text: "ENTRY",
    size: 2,
  });

  const tradeExitMarker = (time: number, _price: number, pnl: number, direction?: string): ChartMarker => ({
    time,
    position: "inBar",
    color: pnl >= 0 ? "#22c55e" : "#ef4444",
    shape: "circle",
    text: pnl >= 0 ? `+${pnl.toFixed(1)}` : pnl.toFixed(1),
    size: 2,
  });

  function paperTradeFromEvent(e: PaperSignalEvent): TradeRecord {
    return {
      trade_id: e.trade_id ?? "",
      direction: (e.direction as SignalDirection) || "FLAT",
      size: e.size ?? 0,
      entry_price: e.entry_price ?? 0,
      exit_price: e.exit_price ?? null,
      pnl: e.pnl ?? 0,
      exit_reason: e.exit_reason ?? "",
      time: e.time ?? Math.floor(Date.now() / 1000),
    };
  }

  function handlePaperSubEvents(subs: PaperSignalEvent[]) {
    for (const sub of subs) {
      if (sub.event === "trade_closed") {
        setLive((prev) => ({
          ...prev,
          trades: [...prev.trades, paperTradeFromEvent(sub)].slice(-200),
          position: "FLAT",
        }));
        if (sub.exit_price && sub.time) {
          setChartMarkers((prev) =>
            [...prev, tradeExitMarker(sub.time, sub.exit_price, sub.pnl ?? 0, sub.direction)].slice(-200),
          );
        }
      } else if (sub.event === "trade_opened") {
        setLive((prev) => ({
          ...prev,
          position: (sub.direction as SignalDirection) || prev.position,
        }));
        if (sub.entry_price && sub.time) {
          setChartMarkers((prev) =>
            [...prev, tradeEntryMarker(sub.time, sub.entry_price, sub.direction)].slice(-200),
          );
        }
      }
    }
  }

  function handlePaperWSEvent(msg: PaperSignalEvent) {
    if (msg.candle) setLiveCandle(msg.candle);

    if (msg.event === "signal" || msg.event === "hold") {
      setLive((prev) => {
        const eq = msg.equity ?? prev.equity;
        const now = msg.time ?? Math.floor(Date.now() / 1000);
        return {
          ...prev,
          position: (msg.position as SignalDirection) || prev.position,
          equity: eq,
          unrealizedPnl: msg.unrealized_pnl ?? prev.unrealizedPnl,
          equityCurve: [...prev.equityCurve, { time: now, value: eq }].slice(-500),
          stopResult: null,
        };
      });
      if (msg.time) {
        setChartMarkers((prev) =>
          [...prev, signalMarker(msg.time, msg.direction, msg.confidence)].slice(-500),
        );
      }
      if (msg.sub_events) handlePaperSubEvents(msg.sub_events);
    } else if (msg.event === "trade_closed") {
      setLive((prev) => ({
        ...prev,
        trades: [...prev.trades, paperTradeFromEvent(msg)].slice(-200),
        position: "FLAT",
      }));
      if (msg.exit_price && msg.time) {
        setChartMarkers((prev) =>
          [...prev, tradeExitMarker(msg.time, msg.exit_price, msg.pnl ?? 0, msg.direction)].slice(-200),
        );
      }
    } else if (msg.event === "stopped") {
      setLive((prev) => ({ ...prev, running: false }));
    }
  }

  function handleLiveWSEvent(msg: LiveSignalEvent) {
    if (msg.candle) setLiveCandle(msg.candle);

    if (msg.event === "signal" || msg.event === "hold") {
      setLive((prev) => {
        const eq = msg.equity ?? prev.equity;
        const now = msg.time ?? Math.floor(Date.now() / 1000);
        return {
          ...prev,
          position: (msg.position as SignalDirection) || prev.position,
          equity: eq,
          unrealizedPnl: msg.unrealized_pnl ?? prev.unrealizedPnl,
          equityCurve: [...prev.equityCurve, { time: now, value: eq }].slice(-500),
        };
      });
      if (msg.time) {
        setChartMarkers((prev) =>
          [...prev, signalMarker(msg.time, msg.direction, msg.confidence)].slice(-500),
        );
      }
      if (msg.sub_events) for (const sub of msg.sub_events) handleLiveWSEvent(sub);
    } else if (msg.event === "risk_blocked") {
      setLive((prev) => ({ ...prev, riskBlockedReason: msg.reason ?? "risk_blocked" }));
      setTimeout(() => setLive((p) => ({ ...p, riskBlockedReason: null })), 3000);
    } else if (msg.event === "order_placed") {
      const newTrade: TradeRecord = {
        trade_id: msg.oanda_order_id ?? `${Date.now()}`,
        direction: (msg.direction as SignalDirection) || "FLAT",
        size: msg.units ?? 0,
        entry_price: msg.price ?? 0,
        exit_price: null,
        pnl: 0,
        exit_reason: "",
        time: msg.time ?? Math.floor(Date.now() / 1000),
        oanda_order_id: msg.oanda_order_id,
      };
      setLive((prev) => ({ ...prev, trades: [...prev.trades, newTrade].slice(-200) }));
      if (msg.price && msg.time) {
        setChartMarkers((prev) =>
          [...prev, tradeEntryMarker(msg.time, msg.price, msg.direction)].slice(-200),
        );
      }
    } else if (msg.event === "trade_closed") {
      const newTrade: TradeRecord = {
        trade_id: `close_${Date.now()}`,
        direction: (msg.direction as SignalDirection) || "FLAT",
        size: msg.units ?? 0,
        entry_price: 0,
        exit_price: msg.price ?? null,
        pnl: msg.pnl ?? 0,
        exit_reason: msg.exit_reason ?? "signal",
        time: msg.time ?? Math.floor(Date.now() / 1000),
      };
      setLive((prev) => ({
        ...prev,
        trades: [...prev.trades, newTrade].slice(-200),
        position: "FLAT",
      }));
      if (msg.price && msg.time) {
        setChartMarkers((prev) =>
          [...prev, tradeExitMarker(msg.time, msg.price, msg.pnl ?? 0, msg.direction)].slice(-200),
        );
      }
    } else if (msg.event === "kill") {
      setLive((prev) => ({
        ...prev,
        running: false,
        killed: true,
        killReason: msg.reason ?? "kill_signal",
        killLevel: msg.level ?? "",
      }));
    } else if (msg.event === "stopped") {
      setLive((prev) => ({ ...prev, running: false }));
    }
  }

  const handleDeploy = useCallback(async () => {
    const committee = selectedCommittee
      ? committeeData?.committees?.find((c) => c.name === selectedCommittee)
      : null;
    const isCommitteeDeploy = !!committee;

    if (!isCommitteeDeploy && !selectedModelId) return;
    if (tradingMode === "live" && !confirmLive) {
      setConfirmLive(true);
      return;
    }
    setConfirmLive(false);
    setLive((prev) => ({
      ...prev,
      deploying: true,
      error: null,
      killed: false,
      killReason: "",
      killLevel: "",
    }));
    try {
      let result: { session_id: string; equity?: number };
      if (isCommitteeDeploy) {
        const res = await apiClient.post("/trading/live/committee/start", {
          pair: selectedPair,
          timeframe,
          initial_equity: initialEquity,
          confidence_threshold: 0.55,
          mode: tradingMode,
          full_cycle_job_id: committee!.full_cycle_job_id,
        });
        result = res.data;
      } else {
        const selected = deployedModels?.find((m) => m.id === selectedModelId);
        const basePayload = {
          pair: selectedPair,
          model_id: selectedModelId,
          model_type: selected?.model_type ?? "logistic",
          timeframe,
          initial_equity: initialEquity,
          position_sizing: positionSizing,
          sizing_config: {},
          live_news_blend_enabled: false,
          live_news_blend_weight: 0,
        };
        if (tradingMode === "paper") {
          result = await deployPaper.mutateAsync(basePayload);
        } else {
          result = await deployLive.mutateAsync({
            ...basePayload,
            mode: "demo",
            risk_config: riskConfig as unknown as Record<string, unknown>,
          });
        }
      }
      const engineEquity = result.equity ?? initialEquity;
      setLive({
        running: true,
        deploying: false,
        mode: tradingMode,
        isCommittee: isCommitteeDeploy,
        sessionId: result.session_id,
        position: "FLAT",
        equity: engineEquity,
        unrealizedPnl: 0,
        initialEquity: engineEquity,
        trades: [],
        equityCurve: [{ time: Math.floor(Date.now() / 1000), value: engineEquity }],
        error: null,
        stopResult: null,
        killed: false,
        killReason: "",
        killLevel: "",
        riskBlockedReason: null,
      });
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setLive((prev) => ({ ...prev, deploying: false, error: detail ?? String(err) }));
    }
  }, [
    selectedPair,
    selectedModelId,
    selectedCommittee,
    committeeData,
    timeframe,
    initialEquity,
    positionSizing,
    tradingMode,
    confirmLive,
    riskConfig,
    deployPaper,
    deployLive,
    deployedModels,
  ]);

  const handleStop = useCallback(async () => {
    if (!live.sessionId) return;
    try {
      if (live.isCommittee || tradingMode !== "paper") {
        const r = await stopLive.mutateAsync(live.sessionId);
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
        setLive((prev) => ({ ...prev, running: false, killed: r.killed }));
      } else {
        const r = await stopPaper.mutateAsync(live.sessionId);
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
        setLive((prev) => ({
          ...prev,
          running: false,
          stopResult: (r as unknown as { summary: PaperSummary }).summary,
        }));
      }
    } catch {
      setLive((prev) => ({ ...prev, running: false }));
    }
  }, [live.sessionId, live.isCommittee, stopPaper, stopLive, tradingMode]);

  const handleEmergency = useCallback(async () => {
    if (!live.sessionId) return;
    try {
      await emergencyKill.mutateAsync(live.sessionId);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setLive((prev) => ({
        ...prev,
        running: false,
        killed: true,
        killReason: "emergency_kill_button",
        killLevel: "K1",
      }));
    } catch {
      /* ignore */
    }
  }, [live.sessionId, emergencyKill]);

  const priceDisplay = priceData?.prices?.[0];
  const midPrice = priceDisplay?.mid;
  const changePct = priceDisplay?.change_pct;
  const equityOverlay: OverlayLine[] =
    live.equityCurve.length > 1
      ? [{ data: live.equityCurve, color: EQUITY_COLORS[0], label: "Equity" }]
      : [];

  const isRunning = live.running;
  const hasModel =
    selectedModelId != null || (selectedCommittee != null && selectedCommittee !== "");
  const isCommitteeSelected = selectedCommittee != null && selectedCommittee !== "";

  if (priceData?.source === "key_required") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold text-(--color-text-primary)">Trading</h2>
        <div className="flex items-center gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-4 py-3">
          <AlertTriangle size={16} className="text-(--color-accent-warning)" />
          <span className="text-xs text-(--color-text-secondary)">{priceData?.message}</span>
          <button
            onClick={() => navigate("/settings")}
            className="rounded px-2 py-0.5 text-[11px] font-medium text-(--color-brand) transition hover:underline"
          >
            Settings
          </button>
        </div>
      </div>
    );
  }
  if (priceData?.source === "unavailable") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold text-(--color-text-primary)">Trading</h2>
        <div className="flex items-center gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-4 py-3">
          <AlertTriangle size={16} className="text-(--color-accent-warning)" />
          <span className="text-xs text-(--color-text-secondary)">OANDA API unreachable</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {live.stopResult && tradingMode === "paper" && (
        <StopResultModal
          summary={live.stopResult}
          onClose={() => setLive((p) => ({ ...p, stopResult: null }))}
        />
      )}
      {confirmLive && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.6]">
          <div className="w-[380px] rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-6 shadow-2xl">
            <h3 className="mb-2 text-sm font-semibold text-(--color-accent-danger)">
              Confirm Live Trading
            </h3>
            <p className="mb-4 text-xs text-(--color-text-secondary)">
              This will submit real orders to your OANDA account. Positions will be opened and
              closed by the model automatically.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setConfirmLive(false)}
                className="flex-1 rounded-md border border-(--color-glass-border) bg-(--color-glass) px-4 py-2 text-xs font-semibold text-(--color-text-secondary) uppercase"
              >
                Cancel
              </button>
              <button
                onClick={handleDeploy}
                className="flex-1 rounded-md bg-(--color-accent-danger) px-4 py-2 text-xs font-semibold text-(--color-text-inverse) uppercase"
              >
                Confirm & Deploy
              </button>
            </div>
          </div>
        </div>
      )}

      {live.killed && (
        <div className="flex shrink-0 items-center gap-3 border-b border-(--color-accent-danger) bg-red-500/[0.08] px-4 py-2">
          <ShieldOff size={14} className="text-(--color-accent-danger)" />
          <span className="text-xs font-semibold text-(--color-accent-danger)">
            Session killed — {live.killReason}
          </span>
          <span className="font-mono text-[10px] text-(--color-text-muted)">
            Level: {live.killLevel}
          </span>
        </div>
      )}
      {live.riskBlockedReason && (
        <div className="flex shrink-0 items-center gap-2 border-b border-(--color-accent-warning) bg-amber-500/[0.06] px-4 py-2">
          <AlertTriangle size={12} className="text-(--color-accent-warning)" />
          <span className="font-mono text-xs text-(--color-accent-warning)">
            Risk blocked: {live.riskBlockedReason}
          </span>
        </div>
      )}
      {live.error && (
        <div className="flex shrink-0 items-center gap-3 border-b border-(--color-accent-danger) bg-red-500/[0.05] px-4 py-2">
          <AlertTriangle size={14} className="text-(--color-accent-danger)" />
          <span className="text-xs text-(--color-accent-danger)">{live.error}</span>
          <button
            onClick={() => setLive((p) => ({ ...p, error: null }))}
            className="ml-auto text-[10px] text-(--color-text-muted) underline"
          >
            Dismiss
          </button>
        </div>
      )}

      <MarketBar
        selectedPair={selectedPair}
        pairList={pairList}
        onPairChange={setSelectedPair}
        deployType={deployType}
        onDeployTypeChange={(type) => {
          setDeployType(type);
          if (type === "model") setSelectedCommittee("");
          else setSelectedModelId(null);
        }}
        selectedCommittee={selectedCommittee}
        committeeList={committeeList}
        onCommitteeChange={(c) => {
          setSelectedCommittee(c);
          if (c) setSelectedModelId(null);
        }}
        selectedModelId={selectedModelId}
        deployedModels={deployedModels}
        loadingDeployed={loadingDeployed}
        onModelChange={(id) => {
          setSelectedModelId(id);
          if (id) setSelectedCommittee("");
        }}
        timeframe={timeframe}
        onTimeframeChange={setTimeframe}
        midPrice={midPrice}
        changePct={changePct}
        tradingMode={tradingMode}
        isRunning={isRunning}
        deploying={live.deploying}
      />

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div ref={leftColRef} className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <div
            className="min-h-[180px] sm:min-h-[200px] md:min-h-[250px] overflow-hidden"
            style={
              tableMinimized ? { flex: 1 } : { flex: `${Math.round(chartFraction * 100)} 1 0px` }
            }
          >
            <CandlestickChart
              pair={selectedPair}
              timeframe={timeframe}
              limit={300}
              overlayLines={equityOverlay}
              showToolbar={false}
              liveCandle={liveCandle}
              chartMarkers={chartMarkers}
              livePrice={!isRunning ? midPrice : null}
            />
          </div>

          <div className="relative flex h-1 flex-shrink-0 items-center justify-center py-1">
            <div
              className="absolute inset-0 cursor-row-resize bg-(--color-glass-border) transition-colors hover:bg-(--color-brand)"
              onMouseDown={onSplitMouseDown}
            />
            <button
              onClick={() => setTableMinimized((v) => !v)}
              className="relative z-10 flex h-5 w-5 items-center justify-center rounded-full border border-(--color-glass-border) bg-(--color-surface) text-(--color-text-muted) transition hover:border-(--color-brand) hover:text-(--color-brand)"
              aria-label={tableMinimized ? "Expand trade table" : "Minimize trade table"}
            >
              {tableMinimized ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
            </button>
          </div>

          <TradeTable
            trades={live.trades}
            selectedPair={selectedPair}
            className={tableMinimized ? "shrink-0" : "min-h-0 flex-1"}
            style={
              tableMinimized
                ? { minHeight: 150 }
                : { flex: `${Math.round((1 - chartFraction) * 100)} 1 0px`, minHeight: 150 }
            }
          />
        </div>

        <OrderPanel
          equity={live.equity}
          initialEquity={live.initialEquity}
          unrealizedPnl={live.unrealizedPnl}
          position={live.position}
          tradeCount={live.trades.length}
          tradingMode={tradingMode}
          isRunning={isRunning}
          deploying={live.deploying}
          killed={live.killed}
          hasModel={hasModel}
          isCommittee={isCommitteeSelected}
          positionSizing={positionSizing}
          initialEquityInput={initialEquity}
          riskConfig={riskConfig as unknown as RiskConfig}
          onModeChange={setTradingMode}
          onSizingChange={setPositionSizing}
          onEquityChange={setInitialEquity}
          onDeploy={handleDeploy}
          onStop={handleStop}
          onEmergency={handleEmergency}
          onChangeRisk={(update) => setRiskConfig((prev) => ({ ...prev, ...update }))}
        />
      </div>
    </div>
  );
}
