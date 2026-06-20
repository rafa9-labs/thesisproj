import { TIMEFRAMES } from "@/lib/constants";
import type { TradingMode } from "./SessionControls";

interface MarketBarProps {
  selectedPair: string;
  pairList: string[];
  onPairChange: (pair: string) => void;
  deployType: "model" | "committee";
  onDeployTypeChange: (type: "model" | "committee") => void;
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
  deployType,
  onDeployTypeChange,
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
  const locked = isRunning || deploying;
  const hasCommittees = committeeList.length > 0;
  const hasModels = (deployedModels?.length ?? 0) > 0;

  const statusLabel = isRunning ? "LIVE EXECUTION" : deploying ? "DEPLOYING..." : "PAPER";

  const isLive = isRunning && tradingMode === "live";

  return (
    <div className="flex min-h-16 shrink-0 flex-wrap items-center gap-y-1.5 border-b border-(--color-border-subtle) bg-(--color-app) py-1.5">
      {/* Zone 1: Configuration */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-r border-(--color-glass-border) px-3 sm:px-4">
        <div className="flex items-center gap-1.5">
          <span className="hidden text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase sm:inline">
            Pair
          </span>
          <select
            value={selectedPair}
            onChange={(e) => onPairChange(e.target.value)}
            disabled={locked}
            aria-label="Select trading pair"
            className="rounded border border-(--color-glass-border) bg-(--color-glass) px-1.5 py-1 font-mono text-[10px] text-(--color-text-primary) transition focus:outline-none disabled:opacity-50 sm:px-2 sm:text-xs"
          >
            {pairList.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        {/* Deploy type toggle */}
        {hasCommittees && (
        <div className="flex items-center rounded border border-(--color-glass-border) bg-(--color-glass) p-0.5">
          <button
            onClick={() => onDeployTypeChange("model")}
            disabled={deploying}
            className={`rounded px-3 py-1 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all disabled:opacity-50 ${
              deployType === "model"
                ? "bg-cyan-500/20 text-cyan-400"
                : "text-(--color-text-muted) hover:text-(--color-text-secondary)"
            }`}
          >
            Model
          </button>
          <button
            onClick={() => onDeployTypeChange("committee")}
            disabled={deploying}
            className={`rounded px-3 py-1 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all disabled:opacity-50 ${
              deployType === "committee"
                ? "bg-amber-500/20 text-amber-400"
                : "text-(--color-text-muted) hover:text-(--color-text-secondary)"
            }`}
          >
            Committee
          </button>
        </div>
        )}

        {deployType === "committee" ? (
          <div className="flex items-center gap-1.5">
            <span className="hidden text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase sm:inline">
              Com
            </span>
            <select
              value={selectedCommittee}
              onChange={(e) => onCommitteeChange(e.target.value)}
              disabled={locked}
              aria-label="Select committee"
              className="rounded border border-(--color-glass-border) bg-(--color-glass) px-1.5 py-1 font-mono text-[10px] text-(--color-text-primary) transition focus:outline-none disabled:opacity-50 sm:px-2 sm:text-xs"
            >
              <option value="">Select...</option>
              {committeeList.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <span className="hidden text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase sm:inline">
              Model
            </span>
            {loadingDeployed ? (
              <span className="rounded border border-(--color-glass-border) bg-(--color-glass) px-1.5 py-1 font-mono text-[10px] text-(--color-text-muted) sm:px-2 sm:text-xs">
                Loading...
              </span>
            ) : !deployedModels?.length ? (
              <div className="rounded border border-(--color-glass-border) bg-(--color-glass) px-1.5 py-1 font-mono text-[10px] text-(--color-accent-warning) sm:px-2 sm:text-xs">
                No Active Models
              </div>
            ) : (
              <select
                value={selectedModelId ?? ""}
                onChange={(e) => onModelChange(e.target.value || null)}
                disabled={locked}
                aria-label="Select model"
                className="rounded border border-(--color-glass-border) bg-(--color-glass) px-1.5 py-1 font-mono text-[10px] text-(--color-text-primary) transition focus:outline-none disabled:opacity-50 sm:px-2 sm:text-xs"
              >
                <option value="">Select...</option>
                {(deployedModels ?? []).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.model_type} {m.tags.length > 0 ? `[${m.tags.join(",")}]` : ""}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}
      </div>

      {/* Zone 2: Market Data */}
      <div className="flex items-center gap-1.5 border-r border-(--color-glass-border) px-3 sm:gap-2 sm:px-4">
        {midPrice != null ? (
          <span className="font-mono text-sm font-semibold text-(--color-text-primary) tabular-nums sm:text-lg">
            {midPrice.toFixed(5)}
          </span>
        ) : (
          <span className="font-mono text-xs text-(--color-text-muted) sm:text-sm">--.-----</span>
        )}
        {changePct != null && (
          <span
            className="font-mono text-[10px] font-semibold tabular-nums sm:text-[11px]"
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
      <div className="flex items-center border-r border-(--color-glass-border) px-2 sm:px-4">
        {isRunning ? (
          <span
            className="flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[8px] font-bold tracking-[0.12em] uppercase sm:gap-2 sm:px-3 sm:py-1 sm:text-[10px]"
            style={{
              borderColor: isLive ? "rgba(8,153,129,0.25)" : "rgba(0,229,255,0.25)",
              backgroundColor: isLive ? "rgba(8,153,129,0.08)" : "rgba(0,229,255,0.08)",
              color: isLive ? "var(--color-accent-success)" : "var(--color-brand)",
            }}
          >
            <span
              className="inline-block h-1.5 w-1.5 animate-pulse rounded-full sm:h-2 sm:w-2"
              style={{
                backgroundColor: isLive ? "var(--color-accent-success)" : "var(--color-brand)",
              }}
            />
            {statusLabel}
          </span>
        ) : (
          <span className="rounded border border-(--color-glass-border) px-2 py-0.5 text-[8px] font-semibold tracking-[0.1em] text-(--color-text-muted) uppercase sm:px-3 sm:py-1 sm:text-[10px]">
            {statusLabel}
          </span>
        )}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Zone 4: Timeframes */}
      <div className="flex items-center gap-0.5 px-2 sm:gap-1 sm:px-4">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.key}
            onClick={() => onTimeframeChange(tf.key)}
            disabled={locked}
            className="rounded border px-1.5 py-0.5 font-mono text-[9px] font-medium uppercase transition-all duration-200 disabled:opacity-50 sm:px-2 sm:text-[10px]"
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
