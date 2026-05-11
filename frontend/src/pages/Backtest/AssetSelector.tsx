import { useEffect, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { usePairs, useUploadCsv } from "@/api/queries";
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
                value={pair}
                onChange={(e) => setField("pair", e.target.value)}
                className="rounded border px-3 py-2 text-sm transition-all duration-200 focus:outline-none"
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
              <label style={labelStyle}>Timeframe</label>
              <select
                value={timeframe}
                onChange={(e) => setField("timeframe", e.target.value)}
                className="rounded border px-3 py-2 text-sm transition-all duration-200 focus:outline-none"
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

          <div className="mt-1 flex items-center gap-4">
            {tfData && (
              <p
                className="text-[11px] font-light"
                style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
              >
                {tfData.rows.toLocaleString()} rows | {dataMin} → {dataMax} | OANDA
              </p>
            )}
            <p className="text-[11px] font-light" style={{ color: "var(--color-text-muted)" }}>
              {startDate || endDate ? "Custom range" : "Full range"}
            </p>
          </div>

          <div className="mt-3 flex items-center gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileChange}
            />
            <button
              onClick={handleImportClick}
              disabled={uploadStatus === "uploading"}
              className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-medium uppercase tracking-wider transition-all hover:border-[var(--color-border-active)]"
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
    </div>
  );
}
