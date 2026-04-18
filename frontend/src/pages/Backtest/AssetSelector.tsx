import { usePairs } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

export function AssetSelector() {
  const { data: pairs, isLoading } = usePairs();
  const pair = useBacktestStore((s) => s.pair);
  const timeframe = useBacktestStore((s) => s.timeframe);
  const setField = useBacktestStore((s) => s.setField);

  const selected = pairs?.find((p) => p.pair.symbol === pair);
  const tfData = selected?.timeframes.find((t) => t.timeframe === timeframe);

  return (
    <div
      className="rounded-lg border p-4"
      style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
    >
      <h3
        className="mb-3 text-xs font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Asset Selection
      </h3>

      {isLoading ? (
        <div className="h-8 animate-skeleton rounded" style={{ backgroundColor: "var(--color-elevated)" }} />
      ) : (
        <>
          <div className="flex gap-4">
            <div className="flex flex-1 flex-col gap-1.5">
              <label className="text-xs" style={{ color: "var(--color-text-muted)" }}>
                Pair
              </label>
              <select
                value={pair}
                onChange={(e) => setField("pair", e.target.value)}
                className="rounded border px-3 py-2 text-sm"
                style={{
                  backgroundColor: "var(--color-elevated)",
                  borderColor: "var(--color-border)",
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {pairs?.map((p) => (
                  <option key={p.pair.symbol} value={p.pair.symbol}>
                    {p.pair.symbol}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-1 flex-col gap-1.5">
              <label className="text-xs" style={{ color: "var(--color-text-muted)" }}>
                Timeframe
              </label>
              <select
                value={timeframe}
                onChange={(e) => setField("timeframe", e.target.value)}
                className="rounded border px-3 py-2 text-sm"
                style={{
                  backgroundColor: "var(--color-elevated)",
                  borderColor: "var(--color-border)",
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              >
                {selected?.timeframes.map((t) => (
                  <option key={t.timeframe} value={t.timeframe}>
                    {t.timeframe}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {tfData && (
            <p
              className="mt-2 text-xs"
              style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
            >
              {tfData.rows.toLocaleString()} rows | {tfData.start_date?.slice(0, 10)} →{" "}
              {tfData.end_date?.slice(0, 10)} | OANDA
            </p>
          )}
        </>
      )}
    </div>
  );
}
