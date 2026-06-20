import { useLivePrices } from "@/api/queries";
import { useNavigate } from "react-router-dom";
import { useDashboardStore } from "@/stores/useDashboardStore";
import { useAppStore } from "@/stores/useAppStore";
import { Settings } from "lucide-react";
import type { LivePrice } from "@/api/schemas";
import { cn } from "@/lib/utils";

function Sparkline({ points }: { points: { t: number; v: number }[] }) {
  if (!points || points.length < 2) {
    return (
      <div className="flex h-[26px] w-full items-center justify-center border border-dashed border-(--color-glass-border) rounded-sm bg-(--color-glass-hover)">
        <span className="text-[8px] text-(--color-text-dim)" title="Insufficient price history">
          No history
        </span>
      </div>
    );
  }

  const values = points.map((p) => p.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const width = 120;
  const height = 26;
  const isUp = values[values.length - 1] >= values[0];
  const color = isUp ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className="flex-shrink-0">
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={1.2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PriceCard({
  price,
  isActive,
  onSelect,
}: {
  price: LivePrice;
  isActive: boolean;
  onSelect: () => void;
}) {
  const isUp = (price.change_pct ?? 0) >= 0;
  const changeColor = isUp ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const arrow = isUp ? "▲" : "▼";

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={isActive}
      aria-label={`Select ${price.symbol}`}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      className={cn(
        "min-w-0 flex-1 rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-3 py-2.5 backdrop-blur-[12px] transition-colors duration-150",
        isActive
          ? "border-l-[3px] border-l-(--color-selected-border) shadow-[inset_0_0_16px_rgba(16,185,129,0.06)]"
          : "cursor-pointer hover:bg-(--color-glass-hover) focus:outline-none focus:ring-1 focus:ring-(--color-brand)",
      )}
    >
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-mono text-[12px] font-semibold text-(--color-text-primary)">
          {price.symbol}
        </span>
        {price.change_pct != null && (
          <span className="font-mono text-[10px] font-medium tabular-nums" style={{ color: changeColor }}>
            {arrow} {price.change_pct > 0 ? "+" : ""}
            {price.change_pct.toFixed(2)}%
          </span>
        )}
      </div>

      <div className="mb-2 flex items-baseline gap-2">
        <div className="flex flex-col">
          <span className="text-[8px] leading-none text-(--color-text-muted)">BID</span>
          <span className="font-mono text-base leading-none font-bold text-(--color-accent-danger) tabular-nums">
            {price.bid != null ? price.bid.toFixed(5) : "—"}
          </span>
        </div>
        <div className="flex flex-col">
          <span className="text-[8px] leading-none text-(--color-text-muted)">ASK</span>
          <span className="font-mono text-base leading-none font-bold text-(--color-accent-success) tabular-nums">
            {price.ask != null ? price.ask.toFixed(5) : "—"}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-1 text-[8px] text-(--color-text-muted) font-mono tabular-nums">
        <span>S: {price.spread_pips != null ? `${price.spread_pips}p` : "—"}</span>
        <span>M: {price.mid != null ? price.mid.toFixed(5) : "—"}</span>
      </div>

      {price.sparkline && (
        <div className="mt-1.5">
          <Sparkline points={price.sparkline} />
        </div>
      )}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="min-w-0 flex-1 animate-pulse rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-3 py-2.5">
      <div className="mb-2 h-3 w-14 rounded bg-(--color-glass-hover)" />
      <div className="mb-2 flex gap-2">
        <div className="h-8 w-16 rounded bg-(--color-glass-hover)" />
        <div className="h-8 w-16 rounded bg-(--color-glass-hover)" />
      </div>
      <div className="mb-1.5 h-3 w-24 rounded bg-(--color-glass-hover)" />
      <div className="h-[26px] w-full rounded bg-(--color-glass-hover)" />
    </div>
  );
}

export function PriceTicker({ pairs, activePair }: { pairs: string[]; activePair: string }) {
  const navigate = useNavigate();
  const setActivePair = useDashboardStore((s) => s.setActivePair);
  const demoMode = useAppStore((s) => s.demoMode);
  const displayPairs = (pairs ?? []).slice(0, 3);
  const { data, isLoading } = useLivePrices(displayPairs, 50, !demoMode);

  if (!pairs || pairs.length === 0) return null;

  if (data?.source === "key_required") {
    return (
      <div className="flex items-center gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-4 py-3 backdrop-blur-[12px]">
        <Settings size={16} className="text-(--color-text-muted)" />
        <span className="text-xs text-(--color-text-secondary)">{data.message}</span>
        <button
          onClick={() => navigate("/settings")}
          className="rounded px-2 py-0.5 text-[11px] font-medium text-(--color-brand) transition hover:underline"
        >
          Settings
        </button>
      </div>
    );
  }

  if (data?.source === "unavailable") {
    return (
      <div className="flex items-center gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-glass) px-4 py-3">
        <span className="text-[11px] text-(--color-accent-warning)">
          OANDA API unreachable. Check your connection.
        </span>
      </div>
    );
  }

  return (
    <div className="flex gap-2">
      {isLoading
        ? displayPairs.map((p) => <SkeletonCard key={p} />)
        : (
            data?.prices ??
            displayPairs.map((s) => ({
              symbol: s,
              bid: null,
              ask: null,
              mid: null,
              spread_pips: null,
              change_pct: null,
              timestamp: "",
              sparkline: [],
            }))
          ).map((price) => (
            <PriceCard
              key={price.symbol}
              price={price}
              isActive={price.symbol === activePair}
              onSelect={() => setActivePair(price.symbol)}
            />
          ))}
    </div>
  );
}
