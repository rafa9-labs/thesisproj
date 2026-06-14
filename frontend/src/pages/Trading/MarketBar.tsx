import { TIMEFRAMES } from "@/lib/constants";
import type { TradingMode } from "./SessionControls";

interface MarketBarProps {
  selectedPair: string;
  pairList: string[];
  onPairChange: (pair: string) => void;
  selectedCommittee: string;
  committeeList: string[];
  onCommitteeChange: (c: string) => void;
  selectedModelId: string | null;
  deployedModels:
    | Array<{
        id: string;
        model_type: string;
        best_sharpe: number | null;
        tags: string[];
      }>
    | undefined;
  loadingDeployed: boolean;
  onModelChange: (id: string | null) => void;
  timeframe: string;
  onTimeframeChange: (tf: string) => void;
  midPrice: number | undefined;
  changePct: number | undefined;
  tradingMode: TradingMode;
  isRunning: boolean;
  deploying: boolean;
}

export function MarketBar({
  selectedPair,
  pairList,
  onPairChange,
  selectedCommittee,
  committeeList,
  onCommitteeChange,
  selectedModelId,
  deployedModels,
  loadingDeployed,
  onModelChange,
  timeframe,
  onTimeframeChange,
  midPrice,
  changePct,
  tradingMode,
  isRunning,
  deploying,
}: MarketBarProps) {
  const disabled = isRunning || deploying;

  const statusLabel = isRunning ? "LIVE EXECUTION" : deploying ? "DEPLOYING..." : "PAPER";

  const isLive = isRunning && tradingMode === "live";

  return (
    <div className="flex h-16 shrink-0 items-center border-b border-(--color-border-subtle) bg-(--color-app) px-0">
      {/* Zone 1: Configuration */}
      <div className="flex items-center gap-3 border-r border-(--color-glass-border) px-4">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
            Pair
          </span>
          <select
            value={selectedPair}
            onChange={(e) => onPairChange(e.target.value)}
            disabled={disabled}
            aria-label="Select trading pair"
            className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1 font-mono text-xs text-(--color-text-primary) transition focus:outline-none disabled:opacity-50"
          >
            {pairList.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
            Com
          </span>
          <select
            value={selectedCommittee}
            onChange={(e) => onCommitteeChange(e.target.value)}
            disabled={disabled}
            aria-label="Select committee"
            className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1 font-mono text-xs text-(--color-text-primary) transition focus:outline-none disabled:opacity-50"
          >
            {committeeList.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
            Model
          </span>
          {loadingDeployed ? (
            <span className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1 font-mono text-xs text-(--color-text-muted)">
              Loading...
            </span>
          ) : !deployedModels?.length ? (
            <div className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1 font-mono text-xs text-(--color-accent-warning)">
              No Active Models
            </div>
          ) : (
            <select
              value={selectedModelId ?? ""}
              onChange={(e) => onModelChange(e.target.value || null)}
              disabled={disabled}
              aria-label="Select model"
              className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1 font-mono text-xs text-(--color-text-primary) transition focus:outline-none disabled:opacity-50"
            >
              <option value="">Select...</option>
              {deployedModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.model_type} {m.tags.length > 0 ? `[${m.tags.join(",")}]` : ""}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {/* Zone 2: Market Data */}
      <div className="flex items-center gap-2 border-r border-(--color-glass-border) px-4">
        {midPrice != null ? (
          <span className="font-mono text-lg font-semibold text-(--color-text-primary) tabular-nums">
            {midPrice.toFixed(5)}
          </span>
        ) : (
          <span className="font-mono text-sm text-(--color-text-muted)">--.-----</span>
        )}
        {changePct != null && (
          <span
            className="font-mono text-[11px] font-semibold tabular-nums"
            style={{
              color: changePct >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)",
            }}
          >
            {changePct >= 0 ? "+" : ""}
            {changePct.toFixed(2)}%
          </span>
        )}
      </div>

      {/* Zone 3: Environment Status */}
      <div className="flex items-center border-r border-(--color-glass-border) px-4">
        {isRunning ? (
          <span
            className="flex items-center gap-2 rounded-full border px-3 py-1 text-[10px] font-bold tracking-[0.12em] uppercase"
            style={{
              borderColor: isLive ? "rgba(8,153,129,0.25)" : "rgba(0,229,255,0.25)",
              backgroundColor: isLive ? "rgba(8,153,129,0.08)" : "rgba(0,229,255,0.08)",
              color: isLive ? "var(--color-accent-success)" : "var(--color-brand)",
            }}
          >
            <span
              className="inline-block h-2 w-2 animate-pulse rounded-full"
              style={{
                backgroundColor: isLive ? "var(--color-accent-success)" : "var(--color-brand)",
              }}
            />
            {statusLabel}
          </span>
        ) : (
          <span className="rounded border border-(--color-glass-border) px-3 py-1 text-[10px] font-semibold tracking-[0.1em] text-(--color-text-muted) uppercase">
            {statusLabel}
          </span>
        )}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Zone 4: Timeframes */}
      <div className="flex items-center gap-1 px-4">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.key}
            onClick={() => onTimeframeChange(tf.key)}
            disabled={disabled}
            className="rounded border px-2 py-0.5 font-mono text-[10px] font-medium uppercase transition-all duration-200 disabled:opacity-50"
            style={{
              borderColor:
                timeframe === tf.key ? "var(--color-brand)" : "var(--color-glass-border)",
              backgroundColor: timeframe === tf.key ? "var(--color-brand-glow)" : "transparent",
              color: timeframe === tf.key ? "var(--color-brand)" : "var(--color-text-muted)",
            }}
          >
            {tf.label}
          </button>
        ))}
      </div>
    </div>
  );
}
