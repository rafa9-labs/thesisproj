import { useEffect } from "react";
import { usePairs } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

export function AssetSelector() {
  const { data: pairs, isLoading } = usePairs();
  const pair = useBacktestStore((s) => s.pair);
  const timeframe = useBacktestStore((s) => s.timeframe);
  const startDate = useBacktestStore((s) => s.startDate);
  const endDate = useBacktestStore((s) => s.endDate);
  const setField = useBacktestStore((s) => s.setField);

  const selected = pairs?.find((p) => p.pair.symbol === pair);
  const tfData = selected?.timeframes.find((t) => t.timeframe === timeframe);

  const dataMin = tfData?.start_date?.slice(0, 10) ?? "";
  const dataMax = tfData?.end_date?.slice(0, 10) ?? "";

  useEffect(() => {
    if (!dataMin && !dataMax) return;
    if (startDate && dataMin && startDate < dataMin) setField("startDate", "");
    if (endDate && dataMax && endDate > dataMax) setField("endDate", "");
  }, [dataMin, dataMax, startDate, endDate, setField]);

  const handleStartChange = (val: string) => {
    if (dataMin && val < dataMin) return;
    if (dataMax && val > dataMax) return;
    setField("startDate", val);
  };

  const handleEndChange = (val: string) => {
    if (dataMin && val < dataMin) return;
    if (dataMax && val > dataMax) return;
    setField("endDate", val);
  };

  const inputStyle: React.CSSProperties = {
    backgroundColor: "var(--color-elevated)",
    borderColor: "var(--color-border)",
    color: "var(--color-text-primary)",
    fontFamily: "var(--font-mono)",
  };

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
                style={inputStyle}
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
                style={inputStyle}
              >
                {selected?.timeframes.map((t) => (
                  <option key={t.timeframe} value={t.timeframe}>
                    {t.timeframe}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-1 flex-col gap-1.5">
              <label className="text-xs" style={{ color: "var(--color-text-muted)" }}>
                Start Date
              </label>
              <input
                type="date"
                value={startDate}
                min={dataMin}
                max={endDate || dataMax}
                onChange={(e) => handleStartChange(e.target.value)}
                className="rounded border px-3 py-2 text-sm"
                style={inputStyle}
              />
            </div>

            <div className="flex flex-1 flex-col gap-1.5">
              <label className="text-xs" style={{ color: "var(--color-text-muted)" }}>
                End Date
              </label>
              <input
                type="date"
                value={endDate}
                min={startDate || dataMin}
                max={dataMax}
                onChange={(e) => handleEndChange(e.target.value)}
                className="rounded border px-3 py-2 text-sm"
                style={inputStyle}
              />
            </div>
          </div>

          <div className="mt-2 flex items-center gap-4">
            {tfData && (
              <p
                className="text-xs"
                style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
              >
                {tfData.rows.toLocaleString()} rows | {dataMin} → {dataMax} | OANDA
              </p>
            )}
            <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              {startDate || endDate ? "Custom range" : "Full range"}
            </p>
          </div>
        </>
      )}
    </div>
  );
}
