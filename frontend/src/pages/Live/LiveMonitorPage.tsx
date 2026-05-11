import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle } from "lucide-react";
import { useLivePrices } from "@/api/queries";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { TIMEFRAMES } from "@/lib/constants";
import type { LivePrice } from "@/api/schemas";

const DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"];

function Sparkline({ points }: { points: { t: number; v: number }[] }) {
  if (!points || points.length < 2) return <div className="h-6 w-full rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />;
  const values = points.map((p) => p.v);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const w = 100;
  const h = 24;
  const isUp = values[values.length - 1] >= values[0];
  const color = isUp ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  const path = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 4) - 2;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={w} height={h} className="flex-shrink-0">
      <path d={path} fill="none" stroke={color} strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PriceHeader({ price }: { price: LivePrice }) {
  const isUp = (price.change_pct ?? 0) >= 0;
  const arrow = isUp ? "▲" : "▼";
  const changeColor = isUp ? "var(--color-accent-success)" : "var(--color-accent-danger)";
  return (
    <div className="flex flex-col gap-1.5 mb-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
          {price.symbol}
        </span>
        {price.change_pct != null && (
          <span className="text-[10px] font-medium" style={{ color: changeColor, fontFamily: "var(--font-mono)" }}>
            {arrow} {price.change_pct > 0 ? "+" : ""}{price.change_pct.toFixed(2)}%
          </span>
        )}
      </div>
      <div className="flex items-center gap-4">
        <div>
          <span className="text-[9px] block" style={{ color: "var(--color-text-muted)" }}>BID</span>
          <span className="text-[12px] font-medium" style={{ color: "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>
            {price.bid != null ? price.bid.toFixed(5) : "—"}
          </span>
        </div>
        <div>
          <span className="text-[9px] block" style={{ color: "var(--color-text-muted)" }}>ASK</span>
          <span className="text-[12px] font-medium" style={{ color: "var(--color-accent-success)", fontFamily: "var(--font-mono)" }}>
            {price.ask != null ? price.ask.toFixed(5) : "—"}
          </span>
        </div>
        <div className="flex-1" />
        <div className="text-right">
          <span className="text-[9px] block" style={{ color: "var(--color-text-muted)" }}>SPREAD</span>
          <span className="text-[11px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
            {price.spread_pips != null ? `${price.spread_pips} pips` : "—"}
          </span>
        </div>
      </div>
    </div>
  );
}

function SkeletonColumn() {
  return (
    <div className="flex flex-col gap-2 animate-pulse"
      style={{ backgroundColor: "var(--color-glass)", border: "1px solid var(--color-glass-border)", borderRadius: 10, padding: 12 }}>
      <div className="h-3 w-14 rounded mb-1" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      <div className="flex justify-between mb-1">
        <div className="h-6 w-16 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
        <div className="h-6 w-16 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      </div>
      <div className="h-3 w-20 rounded mb-2" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      <div className="h-[220px] rounded-lg" style={{ backgroundColor: "var(--color-glass-hover)" }} />
    </div>
  );
}

export function LiveMonitorPage() {
  const navigate = useNavigate();
  const [timeframe, setTimeframe] = useState("M30");

  const { data, isLoading } = useLivePrices(DEFAULT_PAIRS, 30);

  if (data?.source === "key_required") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Live Monitor</h2>
        <div className="flex items-center gap-3 rounded-lg border px-4 py-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
          <AlertTriangle size={16} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>{data.message}</span>
          <button onClick={() => navigate("/settings")} className="text-[11px] font-medium rounded px-2 py-0.5 transition hover:underline" style={{ color: "var(--color-brand)" }}>Settings</button>
        </div>
      </div>
    );
  }

  if (data?.source === "unavailable") {
    return (
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Live Monitor</h2>
        <div className="flex items-center gap-3 rounded-lg border px-4 py-3" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
          <AlertTriangle size={16} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>OANDA API unreachable. Check your connection and API key in Settings.</span>
        </div>
      </div>
    );
  }

  const prices = data?.prices ?? DEFAULT_PAIRS.map((s) =>
    ({ symbol: s, bid: null, ask: null, mid: null, spread_pips: null, change_pct: null, timestamp: "", sparkline: [] }));

  return (
    <div className="flex flex-col gap-4 p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Live Monitor</h2>
        <div className="flex items-center gap-1.5">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.key}
              onClick={() => setTimeframe(tf.key)}
              className="rounded-md border px-2.5 py-0.5 text-[10px] font-medium uppercase transition-all duration-200"
              style={{
                borderColor: timeframe === tf.key ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: timeframe === tf.key ? "var(--color-brand-glow)" : "transparent",
                color: timeframe === tf.key ? "var(--color-brand)" : "var(--color-text-muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {isLoading
          ? DEFAULT_PAIRS.map((p) => <SkeletonColumn key={p} />)
          : prices.map((price) => (
              <div key={price.symbol}
                className="flex flex-col gap-2"
                style={{ backgroundColor: "var(--color-glass)", border: "1px solid var(--color-glass-border)", borderRadius: 10, backdropFilter: "blur(12px)", padding: 12 }}>
                <PriceHeader price={price} />
                <Sparkline points={price.sparkline} />
                <CandlestickChart pair={price.symbol} timeframe={timeframe} limit={100} height={220} showVolume={false} showToolbar={false} />
              </div>
            ))}
      </div>
    </div>
  );
}