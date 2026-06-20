import { useEffect, useState } from "react";
import { Download, CheckCircle, AlertTriangle } from "lucide-react";
import { TooltipLabel } from "@/components/shared/TooltipLabel";
import {
  usePairs,
  useDataStatus,
  useDownloadData,
  useDownloadJobStatus,
  useDefinePair,
} from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { Panel, PanelHeader } from "@/components/shared/Panel";

export function AssetSelector() {
  const { data: pairs, isLoading } = usePairs();
  const pair = useBacktestStore((s) => s.pair);
  const timeframe = useBacktestStore((s) => s.timeframe);
  const startDate = useBacktestStore((s) => s.startDate);
  const endDate = useBacktestStore((s) => s.endDate);
  const setField = useBacktestStore((s) => s.setField);

  const { data: dataStatus, isLoading: dsLoading } = useDataStatus(pair);
  const downloadData = useDownloadData();
  const definePair = useDefinePair();

  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);
  const dlCompleted = downloadJobId != null;
  const { data: dlJobStatus } = useDownloadJobStatus(dlCompleted ? downloadJobId : null);

  const [showDefineModal, setShowDefineModal] = useState(false);
  const [defineSymbol, setDefineSymbol] = useState("");
  const [definePip, setDefinePip] = useState("0.0001");
  const [defineDecimals] = useState("4");

  const handleDownload = async () => {
    try {
      const result = await downloadData.mutateAsync({ pair });
      setDownloadJobId(result.job_id);
    } catch {
      /* ignore */
    }
  };

  const handleDefine = async () => {
    try {
      const pipVal = parseFloat(definePip);
      const decPlaces = parseInt(defineDecimals, 10);
      if (isNaN(pipVal) || pipVal <= 0 || isNaN(decPlaces) || decPlaces < 0) return;
      const result = await definePair.mutateAsync({
        symbol: defineSymbol.toUpperCase(),
        pip_value: pipVal,
        decimal_places: decPlaces,
      });
      setField("pair", result.symbol);
      setShowDefineModal(false);
      setDefineSymbol("");
    } catch {
      /* ignore */
    }
  };

  const isKnown = pairs?.some((p) => p.pair.symbol === pair);
  const pairListPairs = pairs?.map((p) => p.pair.symbol) ?? [];

  // data-status helpers
  const isReady = dataStatus?.ready;
  const isMissing = dataStatus && !isReady;
  const isDownloading =
    downloadJobId &&
    dlJobStatus &&
    (dlJobStatus.status === "pending" || dlJobStatus.status === "running");
  const dlFailed = dlJobStatus?.status === "failed";
  const m30Data = dataStatus?.timeframes?.["M30"];
  const hasSomeData = m30Data?.available;

  const dataMin = m30Data?.start?.slice(0, 10) ?? "";
  const dataMax = m30Data?.end?.slice(0, 10) ?? "";
  const barCount = m30Data?.bars ?? 0;

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
      <PanelHeader title="Configuration" subtitle="Select pair, timeframe, and optional date range for the backtest." />

      {isLoading ? (
        <div className="h-8 animate-skeleton rounded bg-(--color-glass-hover)" />
      ) : (
        <>
          <div className="flex flex-col gap-2">
            <div className="flex flex-1 flex-col gap-1.5">
              <label
                className="text-[11px] font-medium tracking-[0.1em] uppercase"
                style={LABEL_STYLE}
              >
                Pair
              </label>
              <select
                value={isKnown ? pair : "__custom__"}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === "__custom__") {
                    setShowDefineModal(true);
                  } else {
                    setField("pair", val);
                  }
                }}
                className="rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
                style={INPUT_STYLE}
              >
                {pairListPairs.map((sym) => (
                  <option key={sym} value={sym}>
                    {sym}
                  </option>
                ))}
                <option value="__custom__" className="text-(--color-brand)">
                  + Define New Pair
                </option>
              </select>
              {!isKnown && (
                <span className="mt-0.5 font-mono text-[10px] text-(--color-brand)">
                  Custom: {pair}
                </span>
              )}
            </div>

            <div className="flex flex-1 flex-col gap-1.5">
              <TooltipLabel
                label="Timeframe (base)"
                tooltip="Backtester uses M30+H1+H4 for multi-timeframe features. Selecting your base timeframe determines which primary data series feeds the model."
              />
              <select
                value={timeframe}
                onChange={(e) => setField("timeframe", e.target.value)}
                className="rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
                style={INPUT_STYLE}
              >
                {["H1", "M30", "H4"].map((tf) => (
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
          </div>

          {/* Data status indicator */}
          <div className="mt-2 flex flex-wrap items-center gap-3">
            {dsLoading ? (
              <span className="text-[11px] text-(--color-text-muted)">Checking data…</span>
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
                <span className="text-[10px] text-(--color-text-muted)">&rarr;</span>
                <span className="font-mono text-[10px] text-(--color-text-secondary)">
                  {dataMax}
                </span>
                <span className="font-mono text-[10px] text-(--color-text-muted)">
                  {barCount.toLocaleString()} bars
                </span>
              </div>
            ) : hasSomeData ? (
              <div
                className="flex items-center gap-1.5 rounded-full border px-2.5 py-1"
                style={{
                  borderColor: "rgba(245,158,11,0.3)",
                  backgroundColor: "rgba(245,158,11,0.06)",
                }}
              >
                <AlertTriangle size={11} className="text-(--color-accent-warning)" />
                <span className="text-[10px] font-medium tracking-[0.05em] text-(--color-accent-warning) uppercase">
                  Partial
                </span>
                <span className="font-mono text-[10px] text-(--color-text-secondary)">
                  {dataMin}
                </span>
                <span className="text-[10px] text-(--color-text-muted)">&rarr;</span>
                <span className="font-mono text-[10px] text-(--color-text-secondary)">
                  {dataMax}
                </span>
                <span className="text-[10px] text-(--color-text-muted)">
                  Missing: {dataStatus?.missing?.join(", ")}
                </span>
              </div>
            ) : null}

            {isDownloading ? (
              <span className="text-[10px] text-(--color-brand)">
                Downloading {pair}…{" "}
                {dlJobStatus?.progress
                  ? `${Math.round(((dlJobStatus.progress as Record<string, number>).completed_work / (dlJobStatus.progress as Record<string, number>).total_work) * 100)}%`
                  : ""}
              </span>
            ) : null}

            {dlFailed && (
              <span className="text-[10px] text-(--color-accent-danger)">Download failed</span>
            )}
          </div>

          {/* Download + Import actions */}
          <div className="mt-3 flex items-center gap-3">
            {isMissing && !isDownloading && !dsLoading && (
              <button
                onClick={handleDownload}
                disabled={downloadData.isPending}
                className="flex items-center gap-1.5 rounded-md border border-(--color-brand) px-3 py-1.5 text-[11px] font-medium tracking-wider text-(--color-brand) uppercase transition-all"
                style={{
                  backgroundColor: "rgba(0,229,255,0.08)",
                  cursor: downloadData.isPending ? "not-allowed" : "pointer",
                  opacity: downloadData.isPending ? 0.6 : 1,
                }}
              >
                <Download size={12} strokeWidth={2} />
                Download History {downloadData.isPending ? "…" : `(${pair})`}
              </button>
            )}
          </div>
        </>
      )}

      {/* Define New Pair Modal */}
      {showDefineModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.7]"
          onClick={() => setShowDefineModal(false)}
        >
          <div
            className="flex w-[400px] flex-col gap-4 rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-sm font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
              Define New Pair
            </h3>
            <p className="text-[11px] text-(--color-text-muted)">
              Enter the pair code and its pip value to register it.
            </p>

            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
                Symbol
              </label>
              <input
                type="text"
                value={defineSymbol}
                onChange={(e) => setDefineSymbol(e.target.value)}
                placeholder="NZDUSD"
                maxLength={6}
                className="rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all focus:outline-none"
                style={INPUT_STYLE}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] tracking-[0.06em] text-(--color-text-muted) uppercase">
                Pip Value
              </label>
              <input
                type="text"
                value={definePip}
                onChange={(e) => setDefinePip(e.target.value)}
                placeholder="0.0001"
                className="rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all focus:outline-none"
                style={INPUT_STYLE}
              />
              <span className="text-[9px] text-(--color-text-muted)">
                0.0001 for most pairs, 0.01 for JPY pairs
              </span>
            </div>

            <div className="flex justify-end gap-3 border-t border-(--color-glass-border) pt-4">
              <button
                onClick={() => setShowDefineModal(false)}
                className="rounded-md border border-(--color-glass-border) px-4 py-1.5 text-[11px] font-semibold text-(--color-text-muted) uppercase"
              >
                Cancel
              </button>
              <button
                onClick={handleDefine}
                disabled={!defineSymbol || defineSymbol.length !== 6 || definePair.isPending}
                className="rounded-md bg-(--color-brand) px-5 py-1.5 text-[11px] font-semibold text-(--color-text-inverse) uppercase transition-all"
                style={{
                  letterSpacing: "0.05em",
                  opacity: !defineSymbol || defineSymbol.length !== 6 ? 0.5 : 1,
                  cursor: !defineSymbol || defineSymbol.length !== 6 ? "not-allowed" : "pointer",
                }}
              >
                {definePair.isPending ? "Registering…" : "Register"}
              </button>
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
