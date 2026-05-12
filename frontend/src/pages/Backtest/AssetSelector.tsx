import { useEffect, useRef, useState } from "react";
import { Upload, Download, CheckCircle, XCircle, AlertTriangle } from "lucide-react";
import { usePairs, useUploadCsv, useDataStatus, useDownloadData, useDownloadJobStatus, useDefinePair } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

export function AssetSelector() {
  const { data: pairs, isLoading } = usePairs();
  const pair = useBacktestStore((s) => s.pair);
  const timeframe = useBacktestStore((s) => s.timeframe);
  const startDate = useBacktestStore((s) => s.startDate);
  const endDate = useBacktestStore((s) => s.endDate);
  const setField = useBacktestStore((s) => s.setField);
  const uploadCsv = useUploadCsv();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "success" | "error">("idle");

  const { data: dataStatus, isLoading: dsLoading } = useDataStatus(pair);
  const downloadData = useDownloadData();
  const definePair = useDefinePair();

  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);
  const { data: dlJobStatus } = useDownloadJobStatus(downloadJobId);

  const [showDefineModal, setShowDefineModal] = useState(false);
  const [defineSymbol, setDefineSymbol] = useState("");
  const [definePip, setDefinePip] = useState("0.0001");
  const [defineDecimals, setDefineDecimals] = useState("4");

  useEffect(() => {
    if (dlJobStatus?.status === "completed") {
      setDownloadJobId(null);
    }
  }, [dlJobStatus?.status]);

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadStatus("uploading");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("pair", pair);
    formData.append("timeframe", timeframe);
    try {
      await uploadCsv.mutateAsync(formData);
      setUploadStatus("success");
      setTimeout(() => setUploadStatus("idle"), 3000);
    } catch {
      setUploadStatus("error");
      setTimeout(() => setUploadStatus("idle"), 3000);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDownload = async () => {
    try {
      const result = await downloadData.mutateAsync({ pair });
      setDownloadJobId(result.job_id);
    } catch {}
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
    } catch {}
  };

  const isKnown = pairs?.some((p) => p.pair.symbol === pair);
  const pairListPairs = pairs?.map((p) => p.pair.symbol) ?? [];

  // data-status helpers
  const isReady = dataStatus?.ready;
  const isMissing = dataStatus && !isReady;
  const isDownloading = downloadJobId && dlJobStatus && (dlJobStatus.status === "pending" || dlJobStatus.status === "running");
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

  const inputStyle: React.CSSProperties = {
    backgroundColor: "var(--color-glass)",
    borderColor: "var(--color-glass-border)",
    color: "var(--color-text-primary)",
    fontFamily: "var(--font-mono)",
    backdropFilter: "blur(8px)",
  };

  const labelStyle: React.CSSProperties = {
    color: "var(--color-text-muted)",
    fontSize: "11px",
    fontWeight: 500,
    textTransform: "uppercase",
    letterSpacing: "0.1em",
  };

  return (
    <div
      className="flex flex-col gap-6 rounded-xl border p-6"
      style={{
        backgroundColor: "var(--color-glass)",
        borderColor: "var(--color-glass-border)",
        backdropFilter: "blur(12px)",
      }}
    >
      <div className="flex items-center gap-2 mb-1">
        <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: "var(--color-brand)" }} />
        <h3
          className="text-[11px] font-medium uppercase tracking-[0.12em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Asset Selection
        </h3>
      </div>
      <p className="mb-2 text-[11px] font-light leading-relaxed max-w-[720px]" style={{ color: "var(--color-text-muted)" }}>
        Choose the currency pair, timeframe, and date range for the backtest. The model trains on historical OHLCV data from the selected period.
      </p>

      {isLoading ? (
        <div className="h-8 animate-skeleton rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      ) : (
        <>
          <div className="flex gap-4">
            <div className="flex flex-1 flex-col gap-1.5">
              <label style={labelStyle}>Pair</label>
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
                className="rounded border px-3 py-2 text-sm transition-all duration-200 focus:outline-none"
                style={inputStyle}
              >
                {pairListPairs.map((sym) => (
                  <option key={sym} value={sym}>
                    {sym}
                  </option>
                ))}
                <option value="__custom__" style={{ color: "var(--color-brand)" }}>
                  + Define New Pair
                </option>
              </select>
              {!isKnown && (
                <span className="text-[10px] mt-0.5" style={{ color: "var(--color-brand)", fontFamily: "var(--font-mono)" }}>
                  Custom: {pair}
                </span>
              )}
            </div>

            <div className="flex flex-1 flex-col gap-1.5">
              <label style={labelStyle}>Timeframe (base)</label>
              <select
                value={timeframe}
                onChange={(e) => setField("timeframe", e.target.value)}
                className="rounded border px-3 py-2 text-sm transition-all duration-200 focus:outline-none"
                style={inputStyle}
              >
                {["H1", "M30", "H4"].map((tf) => (
                  <option key={tf} value={tf}>
                    {tf}
                  </option>
                ))}
              </select>
              <span className="text-[9px] mt-0.5" style={{ color: "var(--color-text-muted)" }}>
                Backtester uses M30+H1+H4 for MTF features
              </span>
            </div>

            <div className="flex flex-1 flex-col gap-1.5">
              <label style={labelStyle}>Start Date</label>
              <input
                type="date"
                value={startDate}
                min={dataMin}
                max={endDate || dataMax}
                onChange={(e) => handleStartChange(e.target.value)}
                className="rounded border px-3 py-2 text-sm transition-all duration-200 focus:outline-none"
                style={inputStyle}
              />
            </div>

            <div className="flex flex-1 flex-col gap-1.5">
              <label style={labelStyle}>End Date</label>
              <input
                type="date"
                value={endDate}
                min={startDate || dataMin}
                max={dataMax}
                onChange={(e) => handleEndChange(e.target.value)}
                className="rounded border px-3 py-2 text-sm transition-all duration-200 focus:outline-none"
                style={inputStyle}
              />
            </div>
          </div>

          {/* Data status indicator */}
          <div className="mt-2 flex flex-wrap items-center gap-3">
            {dsLoading ? (
              <span className="text-[11px]" style={{ color: "var(--color-text-muted)" }}>Checking data…</span>
            ) : isReady ? (
              <div
                className="flex items-center gap-1.5 rounded-full border px-2.5 py-1"
                style={{ borderColor: "rgba(34,197,94,0.3)", backgroundColor: "rgba(34,197,94,0.06)" }}
              >
                <CheckCircle size={11} style={{ color: "var(--color-accent-success)" }} />
                <span className="text-[10px] font-medium uppercase tracking-[0.05em]" style={{ color: "var(--color-accent-success)" }}>
                  Ready
                </span>
                <span className="text-[10px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                  {dataMin}
                </span>
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>&rarr;</span>
                <span className="text-[10px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                  {dataMax}
                </span>
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                  {barCount.toLocaleString()} bars
                </span>
              </div>
            ) : hasSomeData ? (
              <div
                className="flex items-center gap-1.5 rounded-full border px-2.5 py-1"
                style={{ borderColor: "rgba(245,158,11,0.3)", backgroundColor: "rgba(245,158,11,0.06)" }}
              >
                <AlertTriangle size={11} style={{ color: "var(--color-accent-warning)" }} />
                <span className="text-[10px] font-medium uppercase tracking-[0.05em]" style={{ color: "var(--color-accent-warning)" }}>
                  Partial
                </span>
                <span className="text-[10px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                  {dataMin}
                </span>
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>&rarr;</span>
                <span className="text-[10px]" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                  {dataMax}
                </span>
                <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                  Missing: {dataStatus?.missing?.join(", ")}
                </span>
              </div>
            ) : null}

            {isDownloading ? (
              <span className="text-[10px]" style={{ color: "var(--color-brand)" }}>
                Downloading {pair}… {dlJobStatus?.progress ? `${Math.round((dlJobStatus.progress as Record<string,number>).completed_work / (dlJobStatus.progress as Record<string,number>).total_work * 100)}%` : ""}
              </span>
            ) : null}

            {dlFailed && (
              <span className="text-[10px]" style={{ color: "var(--color-accent-danger)" }}>Download failed</span>
            )}
          </div>

          {/* Download + Import actions */}
          <div className="mt-3 flex items-center gap-3">
            {isMissing && !isDownloading && !dsLoading && (
              <button
                onClick={handleDownload}
                disabled={downloadData.isPending}
                className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-medium uppercase tracking-wider transition-all"
                style={{
                  borderColor: "var(--color-brand)",
                  backgroundColor: "rgba(0,229,255,0.08)",
                  color: "var(--color-brand)",
                  cursor: downloadData.isPending ? "not-allowed" : "pointer",
                  opacity: downloadData.isPending ? 0.6 : 1,
                }}
              >
                <Download size={12} strokeWidth={2} />
                Download History {downloadData.isPending ? "…" : `(${pair})`}
              </button>
            )}

            <button
              onClick={handleImportClick}
              disabled={uploadStatus === "uploading"}
              className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-medium uppercase tracking-wider transition-all"
              style={{
                borderColor: "var(--color-glass-border)",
                backgroundColor: "var(--color-glass-hover)",
                color: "var(--color-text-secondary)",
                cursor: uploadStatus === "uploading" ? "not-allowed" : "pointer",
                opacity: uploadStatus === "uploading" ? 0.6 : 1,
              }}
            >
              <Upload size={12} strokeWidth={2} />
              Import CSV
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileChange}
            />
            {uploadStatus === "uploading" && (
              <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>Uploading…</span>
            )}
            {uploadStatus === "success" && (
              <span className="text-[10px] font-medium" style={{ color: "var(--color-accent-success)" }}>Imported</span>
            )}
            {uploadStatus === "error" && (
              <span className="text-[10px] font-medium" style={{ color: "var(--color-accent-danger)" }}>Upload failed</span>
            )}
          </div>
        </>
      )}

      {/* Define New Pair Modal */}
      {showDefineModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ backgroundColor: "rgba(0,0,0,0.7)" }}
          onClick={() => setShowDefineModal(false)}
        >
          <div
            className="flex w-[400px] flex-col gap-4 rounded-lg border p-6"
            style={{
              backgroundColor: "var(--color-surface)",
              borderColor: "var(--color-glass-border)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-sm font-semibold uppercase tracking-[0.08em]" style={{ color: "var(--color-text-primary)" }}>
              Define New Pair
            </h3>
            <p className="text-[11px]" style={{ color: "var(--color-text-muted)" }}>
              Enter the pair code and its pip value to register it.
            </p>

            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>Symbol</label>
              <input
                type="text"
                value={defineSymbol}
                onChange={(e) => setDefineSymbol(e.target.value)}
                placeholder="NZDUSD"
                maxLength={6}
                className="rounded border px-3 py-2 text-sm transition-all focus:outline-none"
                style={inputStyle}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>Pip Value</label>
              <input
                type="text"
                value={definePip}
                onChange={(e) => setDefinePip(e.target.value)}
                placeholder="0.0001"
                className="rounded border px-3 py-2 text-sm transition-all focus:outline-none"
                style={inputStyle}
              />
              <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>0.0001 for most pairs, 0.01 for JPY pairs</span>
            </div>

            <div className="flex gap-3 justify-end border-t pt-4" style={{ borderColor: "var(--color-glass-border)" }}>
              <button
                onClick={() => setShowDefineModal(false)}
                className="rounded-md border px-4 py-1.5 text-[11px] font-semibold uppercase"
                style={{ borderColor: "var(--color-glass-border)", color: "var(--color-text-muted)" }}
              >
                Cancel
              </button>
              <button
                onClick={handleDefine}
                disabled={!defineSymbol || defineSymbol.length !== 6 || definePair.isPending}
                className="rounded-md px-5 py-1.5 text-[11px] font-semibold uppercase transition-all"
                style={{
                  backgroundColor: "var(--color-brand)",
                  color: "var(--color-text-inverse)",
                  letterSpacing: "0.05em",
                  opacity: (!defineSymbol || defineSymbol.length !== 6) ? 0.5 : 1,
                  cursor: (!defineSymbol || defineSymbol.length !== 6) ? "not-allowed" : "pointer",
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
