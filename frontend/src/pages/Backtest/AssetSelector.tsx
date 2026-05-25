import { useEffect, useState } from "react";
import { Download, CheckCircle, AlertTriangle } from "lucide-react";
import { usePairs, useDataStatus, useDownloadData, useDownloadJobStatus, useDefinePair } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

export function AssetSelector() {
  const { data: pairs, isLoading } = usePairs();
  const pair       = useBacktestStore((s) => s.pair);
  const timeframe  = useBacktestStore((s) => s.timeframe);
  const startDate  = useBacktestStore((s) => s.startDate);
  const endDate    = useBacktestStore((s) => s.endDate);
  const setField   = useBacktestStore((s) => s.setField);

  const { data: dataStatus, isLoading: dsLoading } = useDataStatus(pair);
  const downloadData = useDownloadData();
  const definePair   = useDefinePair();

  const [downloadJobId, setDownloadJobId]   = useState<string | null>(null);
  const dlCompleted = downloadJobId != null;
  const { data: dlJobStatus } = useDownloadJobStatus(dlCompleted ? downloadJobId : null);

  const [showDefineModal, setShowDefineModal] = useState(false);
  const [defineSymbol, setDefineSymbol]       = useState("");
  const [definePip, setDefinePip]             = useState("0.0001");
  const [defineDecimals]                      = useState("4");

  const handleDownload = async () => {
    try {
      const result = await downloadData.mutateAsync({ pair });
      setDownloadJobId(result.job_id);
    } catch { /* ignore */ }
  };

  const handleDefine = async () => {
    try {
      const pipVal    = parseFloat(definePip);
      const decPlaces = parseInt(defineDecimals, 10);
      if (isNaN(pipVal) || pipVal <= 0 || isNaN(decPlaces) || decPlaces < 0) return;
      const result = await definePair.mutateAsync({
        symbol:          defineSymbol.toUpperCase(),
        pip_value:       pipVal,
        decimal_places:  decPlaces,
      });
      setField("pair", result.symbol);
      setShowDefineModal(false);
      setDefineSymbol("");
    } catch { /* ignore */ }
  };

  const isKnown       = pairs?.some((p) => p.pair.symbol === pair);
  const pairListPairs = pairs?.map((p) => p.pair.symbol) ?? [];

  const isReady       = dataStatus?.ready;
  const isMissing     = dataStatus && !isReady;
  const isDownloading = downloadJobId && dlJobStatus && (dlJobStatus.status === "pending" || dlJobStatus.status === "running");
  const dlFailed      = dlJobStatus?.status === "failed";
  const m30Data       = dataStatus?.timeframes?.["M30"];
  const hasSomeData   = m30Data?.available;

  const dataMin  = m30Data?.start?.slice(0, 10) ?? "";
  const dataMax  = m30Data?.end?.slice(0, 10) ?? "";
  const barCount = m30Data?.bars ?? 0;

  useEffect(() => {
    if (!dataMin && !dataMax) return;
    if (startDate && dataMin && startDate < dataMin) setField("startDate", "");
    if (endDate && dataMax && endDate > dataMax)     setField("endDate", "");
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

  const selectStyle: React.CSSProperties = {
    backgroundColor: "#1E222D",
    border: "1px solid #2A2E39",
    color: "#D1D4DC",
    fontFamily: "var(--font-mono)",
    fontSize: 11,
    padding: "4px 8px",
    borderRadius: 4,
    outline: "none",
    height: 28,
    cursor: "pointer",
  };

  const inputStyle: React.CSSProperties = {
    backgroundColor: "#1E222D",
    border: "1px solid #2A2E39",
    color: "#D1D4DC",
    fontFamily: "var(--font-mono)",
    fontSize: 11,
    padding: "4px 8px",
    borderRadius: 4,
    outline: "none",
    height: 28,
  };

  const labelStyle: React.CSSProperties = {
    color: "#787B86",
    fontSize: 9,
    fontWeight: 600,
    textTransform: "uppercase" as const,
    letterSpacing: "0.1em",
    display: "block",
    marginBottom: 3,
  };

  return (
    <div
      className="flex flex-col gap-2"
      style={{
        backgroundColor: "#1A1D27",
        border: "1px solid #2A2E39",
        borderRadius: 6,
        padding: "10px 14px",
      }}
    >
      {/* Section label */}
      <div className="flex items-center gap-2 mb-1">
        <div style={{ width: 2, height: 10, backgroundColor: "#22D3EE", borderRadius: 1 }} />
        <span
          className="text-[9px] font-semibold uppercase tracking-[0.14em]"
          style={{ color: "#787B86" }}
        >
          Asset Selection
        </span>
      </div>

      {isLoading ? (
        <div
          className="animate-pulse rounded"
          style={{ height: 28, backgroundColor: "#2A2E39" }}
        />
      ) : (
        <>
          {/* Single inline row */}
          <div className="flex items-end gap-4 flex-wrap">
            {/* Pair */}
            <div>
              <label style={labelStyle}>Pair</label>
              <select
                value={isKnown ? pair : "__custom__"}
                onChange={(e) => {
                  const val = e.target.value;
                  if (val === "__custom__") setShowDefineModal(true);
                  else setField("pair", val);
                }}
                style={selectStyle}
              >
                {pairListPairs.map((sym) => (
                  <option key={sym} value={sym}>{sym}</option>
                ))}
                <option value="__custom__" style={{ color: "#22D3EE" }}>+ Define New</option>
              </select>
            </div>

            {/* Timeframe */}
            <div>
              <label style={labelStyle}>Timeframe</label>
              <select
                value={timeframe}
                onChange={(e) => setField("timeframe", e.target.value)}
                style={selectStyle}
              >
                {["H1", "M30", "H4"].map((tf) => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            </div>

            {/* Start Date */}
            <div>
              <label style={labelStyle}>Start Date</label>
              <input
                type="date"
                value={startDate}
                min={dataMin}
                max={endDate || dataMax}
                onChange={(e) => handleStartChange(e.target.value)}
                style={inputStyle}
              />
            </div>

            {/* End Date */}
            <div>
              <label style={labelStyle}>End Date</label>
              <input
                type="date"
                value={endDate}
                min={startDate || dataMin}
                max={dataMax}
                onChange={(e) => handleEndChange(e.target.value)}
                style={inputStyle}
              />
            </div>

            {/* Data status badge — inline */}
            {!dsLoading && (
              <div className="flex items-center" style={{ marginBottom: 1 }}>
                {isReady ? (
                  <div
                    className="flex items-center gap-1.5 rounded"
                    style={{
                      border: "1px solid rgba(34,197,94,0.25)",
                      backgroundColor: "rgba(34,197,94,0.06)",
                      padding: "3px 8px",
                      height: 28,
                    }}
                  >
                    <CheckCircle size={10} style={{ color: "#22C55E", flexShrink: 0 }} />
                    <span
                      className="text-[9px] font-semibold uppercase tracking-wider"
                      style={{ color: "#22C55E" }}
                    >
                      Ready
                    </span>
                    <span
                      className="text-[9px] tabular-nums"
                      style={{ color: "#787B86", fontFamily: "var(--font-mono)" }}
                    >
                      {dataMin} — {dataMax}
                    </span>
                    <span
                      className="text-[9px] tabular-nums"
                      style={{ color: "#4A5568", fontFamily: "var(--font-mono)" }}
                    >
                      {barCount.toLocaleString()} bars
                    </span>
                  </div>
                ) : hasSomeData ? (
                  <div
                    className="flex items-center gap-1.5 rounded"
                    style={{
                      border: "1px solid rgba(245,158,11,0.25)",
                      backgroundColor: "rgba(245,158,11,0.06)",
                      padding: "3px 8px",
                      height: 28,
                    }}
                  >
                    <AlertTriangle size={10} style={{ color: "#F59E0B", flexShrink: 0 }} />
                    <span
                      className="text-[9px] font-semibold uppercase tracking-wider"
                      style={{ color: "#F59E0B" }}
                    >
                      Partial
                    </span>
                    <span
                      className="text-[9px]"
                      style={{ color: "#787B86" }}
                    >
                      Missing: {dataStatus?.missing?.join(", ")}
                    </span>
                  </div>
                ) : null}
              </div>
            )}

            {/* Download button — inline when missing */}
            {isMissing && !isDownloading && !dsLoading && (
              <button
                onClick={handleDownload}
                disabled={downloadData.isPending}
                className="flex items-center gap-1.5 rounded text-[10px] font-semibold uppercase tracking-wider transition-colors"
                style={{
                  border: "1px solid #22D3EE",
                  backgroundColor: "rgba(34,211,238,0.06)",
                  color: "#22D3EE",
                  padding: "4px 10px",
                  height: 28,
                  cursor: downloadData.isPending ? "not-allowed" : "pointer",
                  opacity: downloadData.isPending ? 0.6 : 1,
                }}
              >
                <Download size={11} strokeWidth={2} />
                {downloadData.isPending ? "Downloading…" : `Download (${pair})`}
              </button>
            )}

            {isDownloading && (
              <span
                className="text-[10px] tabular-nums"
                style={{ color: "#22D3EE", fontFamily: "var(--font-mono)", marginBottom: 1 }}
              >
                Downloading {pair}…
              </span>
            )}
            {dlFailed && (
              <span
                className="text-[10px]"
                style={{ color: "#EF4444", marginBottom: 1 }}
              >
                Download failed
              </span>
            )}
          </div>
        </>
      )}

      {/* Define New Pair Modal */}
      {showDefineModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ backgroundColor: "rgba(0,0,0,0.75)" }}
          onClick={() => setShowDefineModal(false)}
        >
          <div
            className="flex flex-col gap-4 rounded-lg border"
            style={{
              width: 380,
              backgroundColor: "#1E222D",
              borderColor: "#2A2E39",
              padding: "20px 24px",
              boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3
              className="text-sm font-semibold uppercase tracking-[0.08em]"
              style={{ color: "#D1D4DC" }}
            >
              Define New Pair
            </h3>
            <p className="text-[11px]" style={{ color: "#787B86" }}>
              Enter the pair code and its pip value to register it.
            </p>

            <div className="flex flex-col gap-1.5">
              <label className="text-[9px] uppercase tracking-[0.1em] font-semibold" style={{ color: "#787B86" }}>Symbol</label>
              <input
                type="text"
                value={defineSymbol}
                onChange={(e) => setDefineSymbol(e.target.value)}
                placeholder="NZDUSD"
                maxLength={6}
                className="rounded text-xs focus:outline-none"
                style={{ ...inputStyle, height: 32, fontSize: 12 }}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-[9px] uppercase tracking-[0.1em] font-semibold" style={{ color: "#787B86" }}>Pip Value</label>
              <input
                type="text"
                value={definePip}
                onChange={(e) => setDefinePip(e.target.value)}
                placeholder="0.0001"
                className="rounded text-xs focus:outline-none"
                style={{ ...inputStyle, height: 32, fontSize: 12 }}
              />
              <span className="text-[9px]" style={{ color: "#4A5568" }}>
                0.0001 for most pairs · 0.01 for JPY pairs
              </span>
            </div>

            <div
              className="flex gap-3 justify-end pt-3"
              style={{ borderTop: "1px solid #2A2E39" }}
            >
              <button
                onClick={() => setShowDefineModal(false)}
                className="rounded text-[10px] font-semibold uppercase tracking-wider"
                style={{
                  border: "1px solid #2A2E39",
                  color: "#787B86",
                  padding: "5px 14px",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleDefine}
                disabled={!defineSymbol || defineSymbol.length !== 6 || definePair.isPending}
                className="rounded text-[10px] font-semibold uppercase tracking-wider transition-opacity"
                style={{
                  backgroundColor: "#22D3EE",
                  color: "#0A0D12",
                  padding: "5px 16px",
                  cursor: (!defineSymbol || defineSymbol.length !== 6) ? "not-allowed" : "pointer",
                  opacity: (!defineSymbol || defineSymbol.length !== 6) ? 0.45 : 1,
                }}
              >
                {definePair.isPending ? "Registering…" : "Register"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
