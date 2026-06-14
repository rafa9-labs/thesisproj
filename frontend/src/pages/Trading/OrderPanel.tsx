import { Loader2, ShieldOff, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { TradingMode } from "./SessionControls";
import type { SignalDirection } from "./PositionMonitor";

interface OrderPanelProps {
  equity: number;
  initialEquity: number;
  unrealizedPnl: number;
  position: SignalDirection;
  tradeCount: number;
  tradingMode: TradingMode;
  isRunning: boolean;
  deploying: boolean;
  killed: boolean;
  hasModel: boolean;
  positionSizing: string;
  initialEquityInput: number;
  riskConfig: RiskConfig;
  onModeChange: (mode: TradingMode) => void;
  onSizingChange: (s: string) => void;
  onEquityChange: (e: number) => void;
  onDeploy: () => void;
  onStop: () => void;
  onEmergency: () => void;
  onChangeRisk: (update: Partial<RiskConfig>) => void;
}

export interface RiskConfig {
  max_drawdown_pct: number;
  max_daily_loss_pct: number;
  max_consecutive_losses: number;
  min_confidence: number;
  max_position_pct: number;
  max_daily_trades: number;
}

const SIZING_OPTIONS = [
  { value: "fixed", label: "Fixed" },
  { value: "fixed_fractional", label: "Fractional" },
  { value: "kelly", label: "Kelly Criterion" },
  { value: "atr", label: "ATR Volatility" },
  { value: "vol_target", label: "Vol Target" },
];

const EXECUTION_OPTIONS = [
  { value: "passive", label: "Passive" },
  { value: "smart", label: "Smart Routing" },
  { value: "aggressive", label: "Aggressive" },
];

export function OrderPanel({
  equity,
  initialEquity,
  unrealizedPnl,
  position,
  tradeCount,
  tradingMode,
  isRunning,
  deploying,
  killed,
  hasModel,
  positionSizing,
  initialEquityInput,
  riskConfig,
  onModeChange,
  onSizingChange,
  onEquityChange,
  onDeploy,
  onStop,
  onEmergency,
  onChangeRisk,
}: OrderPanelProps) {
  const disabled = isRunning || deploying;
  const returnPct = initialEquity > 0 ? ((equity - initialEquity) / initialEquity) * 100 : 0;

  return (
    <div className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto border-t border-l-0 border-(--color-border-subtle) px-3 py-3 lg:w-[320px] lg:border-t-0 lg:border-l">
      {/* ── Account Equity ── */}
      <div className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-3">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
            Account Equity
          </span>
          {tradingMode === "live" && (
            <span className="rounded border border-(--color-accent-danger) px-1.5 py-0.5 text-[8px] font-bold text-(--color-accent-danger) uppercase">
              LIVE
            </span>
          )}
        </div>
        <div className="font-mono text-xl font-semibold text-(--color-text-primary) tabular-nums">
          ${equity.toFixed(2)}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 rounded-sm bg-(--color-glass-hover) px-2 py-1.5">
            <span className="text-[8px] text-(--color-text-muted) uppercase">
              Unrealized P&amp;L
            </span>
            <div
              className="font-mono text-xs font-semibold tabular-nums"
              style={{
                color:
                  unrealizedPnl >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
              }}
            >
              {unrealizedPnl >= 0 ? "+" : ""}
              {unrealizedPnl.toFixed(2)}
            </div>
          </div>
          <div className="flex-1 rounded-sm bg-(--color-glass-hover) px-2 py-1.5">
            <span className="text-[8px] text-(--color-text-muted) uppercase">Return</span>
            <div
              className="font-mono text-xs font-semibold tabular-nums"
              style={{
                color:
                  returnPct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
              }}
            >
              {returnPct >= 0 ? "+" : ""}
              {returnPct.toFixed(2)}%
            </div>
          </div>
        </div>

        <div className="mt-2 flex items-center rounded border border-(--color-glass-border) p-0.5">
          <button
            onClick={() => onModeChange("paper")}
            disabled={disabled}
            className="flex-1 rounded py-0.5 text-[10px] font-semibold uppercase transition disabled:opacity-50"
            style={{
              backgroundColor: tradingMode === "paper" ? "var(--color-brand)" : "transparent",
              color:
                tradingMode === "paper" ? "var(--color-text-inverse)" : "var(--color-text-muted)",
            }}
          >
            Paper
          </button>
          <button
            onClick={() => onModeChange("live")}
            disabled={disabled}
            className="flex-1 rounded py-0.5 text-[10px] font-semibold uppercase transition disabled:opacity-50"
            style={{
              backgroundColor:
                tradingMode === "live" ? "var(--color-accent-danger)" : "transparent",
              color:
                tradingMode === "live" ? "var(--color-text-inverse)" : "var(--color-text-muted)",
            }}
          >
            Live
          </button>
        </div>
      </div>

      {/* ── Algo Deployment ── */}
      <div className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-3">
        <h4 className="mb-3 text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
          Algo Deployment
        </h4>

        <div className="mb-2.5 flex flex-col gap-1">
          <span className="text-[9px] text-(--color-text-muted) uppercase">Risk Strategy</span>
          <select
            value={positionSizing}
            onChange={(e) => onSizingChange(e.target.value)}
            disabled={disabled}
            className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1.5 font-mono text-xs text-(--color-text-primary) outline-none disabled:opacity-50"
          >
            {SIZING_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="mb-2.5 flex flex-col gap-1">
          <span className="text-[9px] text-(--color-text-muted) uppercase">Max Drawdown Limit</span>
          <div className="flex items-center gap-1">
            <input
              type="number"
              value={riskConfig.max_drawdown_pct}
              onChange={(e) => onChangeRisk({ max_drawdown_pct: Number(e.target.value) })}
              disabled={disabled}
              min={5}
              max={30}
              step={1}
              className="flex-1 rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1.5 font-mono text-xs text-(--color-text-primary) outline-none disabled:opacity-50"
            />
            <span className="font-mono text-[10px] text-(--color-text-muted)">%</span>
          </div>
        </div>

        <div className="mb-2.5 flex flex-col gap-1">
          <span className="text-[9px] text-(--color-text-muted) uppercase">Execution Logic</span>
          <select
            disabled={disabled}
            className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1.5 font-mono text-xs text-(--color-text-primary) outline-none disabled:opacity-50"
          >
            {EXECUTION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="mb-3 flex flex-col gap-1">
          <span className="text-[9px] text-(--color-text-muted) uppercase">Initial Equity</span>
          <input
            type="number"
            value={initialEquityInput}
            onChange={(e) => onEquityChange(Number(e.target.value) || 10000)}
            disabled={disabled}
            min={100}
            max={1000000}
            className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1.5 font-mono text-xs text-(--color-text-primary) outline-none disabled:opacity-50"
          />
        </div>

        {isRunning && (
          <button
            onClick={onEmergency}
            className="mb-2 flex w-full animate-pulse items-center justify-center gap-1.5 rounded border px-3 py-2 text-[10px] font-bold uppercase"
            style={{
              borderColor: "var(--color-accent-danger)",
              backgroundColor: "rgba(242,54,69,0.12)",
              color: "var(--color-accent-danger)",
            }}
          >
            <ShieldOff size={12} />
            Kill Switch / Halt
          </button>
        )}

        <button
          onClick={isRunning ? onStop : onDeploy}
          disabled={deploying || !hasModel || killed}
          className="flex w-full items-center justify-center gap-1.5 rounded border px-4 py-2.5 text-[11px] font-bold tracking-[0.06em] uppercase transition-all duration-200 disabled:opacity-50"
          style={{
            borderColor: isRunning ? "var(--color-accent-danger)" : "var(--color-brand)",
            backgroundColor: isRunning ? "rgba(239,68,68,0.08)" : "rgba(0,229,255,0.1)",
            color: isRunning ? "var(--color-accent-danger)" : "var(--color-brand)",
          }}
        >
          {deploying ? (
            <>
              <Loader2 size={14} className="animate-spin" />
              Deploying...
            </>
          ) : isRunning ? (
            <>
              <ShieldOff size={14} />
              Stop
            </>
          ) : (
            "Deploy Committee"
          )}
        </button>
      </div>

      {/* ── Current Position ── */}
      <div className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-3">
        <h4 className="mb-2 text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
          Current Position
        </h4>

        <div className="mb-2 flex items-center gap-2">
          <PositionBadge position={position} />
        </div>

        <div className="flex flex-col gap-1">
          <Row label="Equity" value={equity.toFixed(2)} />
          <Row
            label="P&amp;L"
            value={`${unrealizedPnl >= 0 ? "+" : ""}${unrealizedPnl.toFixed(2)}`}
            color={
              unrealizedPnl >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)"
            }
          />
          <Row
            label="Return"
            value={`${returnPct >= 0 ? "+" : ""}${returnPct.toFixed(2)}%`}
            color={returnPct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)"}
          />
          <Row label="Trades" value={String(tradeCount)} />
        </div>
      </div>
    </div>
  );
}

function PositionBadge({ position }: { position: SignalDirection }) {
  const isLong = position === "LONG";
  const isShort = position === "SHORT";
  const color = isLong
    ? "var(--color-accent-success)"
    : isShort
      ? "var(--color-accent-danger)"
      : "var(--color-text-muted)";
  const icon = isLong ? (
    <TrendingUp size={14} />
  ) : isShort ? (
    <TrendingDown size={14} />
  ) : (
    <Minus size={14} />
  );
  const label = isLong ? "LONG" : isShort ? "SHORT" : "FLAT";
  return (
    <div
      className="flex items-center gap-1.5 rounded border px-3 py-1.5"
      style={{
        borderColor: color,
        backgroundColor: isLong
          ? "rgba(34,197,94,0.1)"
          : isShort
            ? "rgba(239,68,68,0.1)"
            : "transparent",
      }}
    >
      {icon}
      <span className="font-mono text-[11px] font-semibold uppercase">{label}</span>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-[9px] text-(--color-text-muted)">{label}</span>
      <span
        className="font-mono text-[10px] font-medium tabular-nums"
        style={{ color: color ?? "var(--color-text-primary)" }}
      >
        {value}
      </span>
    </div>
  );
}
