import { useEffect } from "react";
import { CheckCircle, AlertTriangle } from "lucide-react";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import { useDataStatus } from "@/api/queries";
import { Panel, PanelHeader } from "@/components/shared/Panel";

const PAIR_OPTIONS = [
  { value: "EURUSD", label: "EURUSD" },
  { value: "GBPUSD", label: "GBPUSD" },
  { value: "USDJPY", label: "USDJPY" },
  { value: "AUDUSD", label: "AUDUSD" },
  { value: "USDCAD", label: "USDCAD" },
];

const TF_OPTIONS = ["H1", "M30", "H4"];

export function AssetTimeTab() {
  const store = useFullCycleStore();
  const pair = store.pair;
  const timeframe = store.timeframe;
  const startDate = store.startDate;
  const endDate = store.endDate;
  const setPair = store.setPair;
  const setTimeframe = store.setTimeframe;
  const setStartDate = store.setStartDate;
  const setEndDate = store.setEndDate;

  const { data: dataStatus, isLoading: dsLoading } = useDataStatus(pair);

  const isReady = dataStatus?.ready;
  const isMissing = dataStatus && !isReady;
  const tfStatus = dataStatus?.timeframes?.[timeframe] ?? dataStatus?.timeframes?.["H1"];
  const dataMin = tfStatus?.start?.slice(0, 10) ?? "";
  const dataMax = tfStatus?.end?.slice(0, 10) ?? "";
  const barsAvailable = tfStatus?.bars ?? 0;

  useEffect(() => {
    if (!dataMin && !dataMax) return;
    if (startDate && dataMin && startDate < dataMin) setStartDate("");
    if (endDate && dataMax && endDate > dataMax) setEndDate("");
  }, [dataMin, dataMax, startDate, endDate, setStartDate, setEndDate]);

  const handleStartChange = (val: string) => {
    if (dataMin && val < dataMin) return;
    if (dataMax && val > dataMax) return;
    setStartDate(val);
  };

  const handleEndChange = (val: string) => {
    if (dataMin && val < dataMin) return;
    if (dataMax && val > dataMax) return;
    setEndDate(val);
  };

  const INPUT_STYLE: React.CSSProperties = {
    backgroundColor: "var(--color-glass)",
    borderColor: "var(--color-glass-border)",
    color: "var(--color-text-primary)",
  };

  const LABEL_STYLE: React.CSSProperties = {
    color: "var(--color-text-muted)",
  };

  return (
    <Panel>
      <PanelHeader
        title="Configuration"
        subtitle="Select pair, timeframe, and optional date range for the committee pipeline."
      />

      <div className="flex flex-col gap-2">
        <div className="flex flex-1 flex-col gap-1.5">
          <label
            className="text-[11px] font-medium tracking-[0.1em] uppercase"
            style={LABEL_STYLE}
          >
            Pair
          </label>
          <select
            value={pair}
            onChange={(e) => setPair(e.target.value)}
            className="rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
            style={INPUT_STYLE}
          >
            {PAIR_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-1 flex-col gap-1.5">
          <label
            className="text-[11px] font-medium tracking-[0.1em] uppercase"
            style={LABEL_STYLE}
          >
            Timeframe (base)
          </label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
            style={INPUT_STYLE}
          >
            {TF_OPTIONS.map((tf) => (
              <option key={tf} value={tf}>
                {tf}
              </option>
            ))}
          </select>
        </div>

        <div className="flex gap-3">
          <div className="flex flex-1 flex-col gap-1.5">
            <label
              className="text-[11px] font-medium tracking-[0.1em] uppercase"
              style={LABEL_STYLE}
            >
              Start Date
            </label>
            <input
              type="date"
              value={startDate}
              min={dataMin}
              max={endDate || dataMax}
              onChange={(e) => handleStartChange(e.target.value)}
              className="rounded border px-3 py-2 font-mono text-sm transition-all duration-200 focus:outline-none"
              style={INPUT_STYLE}
            />
          </div>

          <div className="flex flex-1 flex-col gap-1.5">
            <label
              className="text-[11px] font-medium tracking-[0.1em] uppercase"
              style={LABEL_STYLE}
            >
              End Date
            </label>
            <input
              type="date"
              value={endDate}
              min={startDate || dataMin}
              max={dataMax}
              onChange={(e) => handleEndChange(e.target.value)}
              className="rounded border px-3 py-2 font-mono text-sm transition-all duration-200 focus:outline-none"
              style={INPUT_STYLE}
            />
          </div>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-3">
          {dsLoading ? (
            <span className="text-[11px] text-(--color-text-muted)">Checking data...</span>
          ) : isReady ? (
            <div
              className="flex items-center gap-1.5 rounded-full border px-2.5 py-1"
              style={{
                borderColor: "rgba(34,197,94,0.3)",
                backgroundColor: "rgba(34,197,94,0.06)",
              }}
            >
              <CheckCircle size={11} className="text-(--color-accent-success)" />
              <span className="text-[10px] font-medium tracking-[0.05em] text-(--color-accent-success) uppercase">
                Ready
              </span>
              <span className="font-mono text-[10px] text-(--color-text-secondary)">
                {dataMin}
              </span>
              <span className="text-[10px] text-(--color-text-muted)">→</span>
              <span className="font-mono text-[10px] text-(--color-text-secondary)">
                {dataMax}
              </span>
              <span className="font-mono text-[10px] text-(--color-text-muted)">
                {barsAvailable.toLocaleString()} bars
              </span>
            </div>
          ) : isMissing ? (
            <div
              className="flex items-center gap-1.5 rounded-full border px-2.5 py-1"
              style={{
                borderColor: "rgba(245,158,11,0.3)",
                backgroundColor: "rgba(245,158,11,0.06)",
              }}
            >
              <AlertTriangle size={11} className="text-(--color-accent-warning)" />
              <span className="text-[10px] font-medium tracking-[0.05em] text-(--color-accent-warning) uppercase">
                Data Missing
              </span>
              {dataMin && (
                <>
                  <span className="font-mono text-[10px] text-(--color-text-secondary)">
                    {dataMin}
                  </span>
                  <span className="text-[10px] text-(--color-text-muted)">→</span>
                  <span className="font-mono text-[10px] text-(--color-text-secondary)">
                    {dataMax}
                  </span>
                </>
              )}
              <span className="text-[10px] text-(--color-text-muted)">
                Missing: {dataStatus?.missing?.join(", ") || "unknown"}
              </span>
            </div>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}
