import { useState } from "react";
import {
  X,
  ChevronLeft,
  Upload,
  Key,
  Zap,
  Database,
  CheckCircle,
  AlertCircle,
  FileText,
} from "lucide-react";
import { useDemoSeed, useStoreApiKey, useUploadCsv } from "@/api/queries";
import apiClient from "@/api/client";

type WizardStep = "choice" | "demo-info" | "csv-upload" | "oanda-key" | "oanda-download";

interface DataSourceModalProps {
  isOpen: boolean;
  onBack: () => void;
  onStart: (mode: string, value?: string) => void;
}

const DOWNLOAD_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"];
const DOWNLOAD_TFS = ["M30", "H1", "H4"];

export function DataSourceModal({ isOpen, onBack, onStart }: DataSourceModalProps) {
  const [step, setStep] = useState<WizardStep>("choice");
  const [oandaKey, setOandaKey] = useState("");
  const [accountId, setAccountId] = useState("");
  const [csvFiles, setCsvFiles] = useState<File[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedPairs, setSelectedPairs] = useState(() => new Set(DOWNLOAD_PAIRS.slice(0, 3)));
  const [selectedTfs, setSelectedTfs] = useState(() => new Set(["M30", "H1"]));

  const demoSeed = useDemoSeed();
  const storeApiKey = useStoreApiKey();
  const uploadCsv = useUploadCsv();

  if (!isOpen) return null;

  /* ── Handlers ────────────────────────────────────────────── */

  const goBack = () => {
    if (step !== "choice") {
      if (step === "oanda-download") {
        setStep("oanda-key");
      } else if (step === "oanda-key") {
        setStep("choice");
      } else {
        setStep("choice");
      }
      setError("");
    } else {
      onBack();
    }
  };

  const handleStartDemo = async () => {
    setLoading(true);
    setError("");
    try {
      await demoSeed.mutateAsync({
        pairs: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
        timeframes: ["M30", "H1", "H4"],
      });
      onStart("demo");
    } catch (err) {
      setError("Failed to load demo data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleConnectOanda = () => {
    if (!oandaKey.trim()) {
      setError("Please enter your OANDA API key");
      return;
    }
    setLoading(true);
    setError("");
    try {
      storeApiKey.mutate({ name: "oanda", value: oandaKey.trim() });
      if (accountId.trim()) {
        storeApiKey.mutate({ name: "oanda_account_id", value: accountId.trim() });
      }
      setStep("oanda-download");
    } catch {
      setError("Failed to store API key. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadAndStart = async () => {
    setLoading(true);
    setError("");
    const pairs = Array.from(selectedPairs);
    const tfs = Array.from(selectedTfs);
    try {
      const results = [];
      for (const pair of pairs) {
        for (const tf of tfs) {
          await apiClient.post("/data/download", { pair, years: 5 });
          results.push(`${pair} ${tf}`);
        }
      }
      onStart("oanda");
    } catch (err) {
      setError("Download failed. Check your connection and try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    const valid = files.filter((f) => f.name.endsWith(".csv"));
    if (valid.length === 0) {
      setError("Please select valid CSV files");
      return;
    }
    setCsvFiles((prev) => [...prev, ...valid]);
    setError("");
  };

  const handleImportCsv = async () => {
    if (csvFiles.length === 0) {
      setError("Please select at least one CSV file");
      return;
    }
    setLoading(true);
    setError("");
    try {
      for (const file of csvFiles) {
        await uploadCsv.mutateAsync(file);
      }
      onStart("csv");
    } catch (err) {
      setError("Failed to import CSV files. Check format and try again.");
    } finally {
      setLoading(false);
    }
  };

  const removeCsvFile = (idx: number) => {
    setCsvFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const togglePair = (p: string) => {
    const next = new Set(selectedPairs);
    if (next.has(p)) next.delete(p);
    else next.add(p);
    setSelectedPairs(next);
  };

  const toggleTf = (tf: string) => {
    const next = new Set(selectedTfs);
    if (next.has(tf)) next.delete(tf);
    else next.add(tf);
    setSelectedTfs(next);
  };

  /* ── Shared styles ───────────────────────────────────────── */

  const primaryBtn = (disabled = false) => ({
    width: "100%" as const,
    padding: "12px 16px",
    borderRadius: "8px",
    border: "none",
    fontWeight: 600,
    fontSize: "13px",
    display: "flex" as const,
    alignItems: "center" as const,
    justifyContent: "center" as const,
    gap: "8px",
    backgroundColor: disabled ? "var(--color-elevated)" : "var(--color-brand)",
    color: disabled ? "var(--color-text-muted)" : "var(--color-text-inverse)",
    cursor: (disabled ? "not-allowed" : "pointer") as "not-allowed" | "pointer",
    opacity: loading ? 0.7 : 1,
    boxShadow: disabled ? "none" : "0 0 12px rgba(0, 229, 255, 0.4)",
    transition: "all 0.15s",
  });

  const BACK_BTN_STYLE: React.CSSProperties = {
    border: "1px solid #1E2A3A",
    backgroundColor: "var(--color-input-bg)",
    color: "var(--color-text-muted)",
  };

  const BACK_BTN_CLASS = "p-2 rounded-lg cursor-pointer transition duration-150";

  const INPUT_CLASS = "w-full px-3 py-2.5 rounded-lg font-mono text-[13px] outline-none";

  const inputStyle: React.CSSProperties = {
    backgroundColor: "var(--color-input-bg)",
    color: "var(--color-text-primary)",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.7] backdrop-blur-[2px]">
      <div className="relative w-full max-w-[520px] overflow-hidden rounded-sm border border-[#1E2A3A] bg-(--color-app) shadow-2xl">
        {/* cyan top accent */}
        <div
          className="h-[3px] w-full"
          style={{
            background: "linear-gradient(90deg, var(--color-brand) 0%, rgba(0,229,255,0.15) 100%)",
          }}
        />

        <div className="flex flex-col gap-5 px-8 pt-7 pb-8">
          {/* ── Header with back button ── */}
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <h1 className="text-lg font-bold tracking-[0.5px] text-(--color-text-primary)">
                {step === "choice" && "Connect Your Data"}
                {step === "demo-info" && "Demo Mode"}
                {step === "csv-upload" && "Upload CSV Data"}
                {step === "oanda-key" && "Enter OANDA API Key"}
                {step === "oanda-download" && "Download Market Data"}
              </h1>
              <p className="mt-1 text-xs text-(--color-text-muted)">
                {step === "choice" && "Select how you want to get started"}
                {step === "demo-info" && "Get started instantly with pre-loaded market data"}
                {step === "csv-upload" && "Import your own historical data files"}
                {step === "oanda-key" && "Connect to OANDA for live and historical data"}
                {step === "oanda-download" && "Choose which data to fetch from OANDA"}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {step !== "choice" && (
                <button onClick={goBack} className={BACK_BTN_CLASS} style={BACK_BTN_STYLE}>
                  <ChevronLeft size={16} />
                </button>
              )}
              <button onClick={onBack} className={BACK_BTN_CLASS} style={BACK_BTN_STYLE}>
                <X size={16} />
              </button>
            </div>
          </div>

          {/* ── ERROR ── */}
          {error && (
            <div className="flex items-center gap-2 rounded-sm border border-red-500/[0.2] bg-red-500/[0.1] px-3 py-2 text-xs text-(--color-event-high)">
              <AlertCircle size={12} />
              {error}
            </div>
          )}

          {/* ================================================================ */}
          {/* STEP: CHOICE                                                    */}
          {/* ================================================================ */}
          {step === "choice" && (
            <div className="flex flex-col gap-3">
              {/* Demo */}
              <button
                onClick={() => setStep("demo-info")}
                className="flex cursor-pointer items-center gap-4 rounded-sm border border-(--color-card-border) bg-(--color-input-bg) p-4 text-left transition-all hover:border-(--color-brand) hover:bg-[rgba(0,229,255,0.03)]"
              >
                <div className="bg-cyan/[0.12] rounded-sm p-2.5">
                  <Database size={18} className="text-(--color-brand)" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-(--color-text-primary)">Demo Mode</p>
                  <p className="text-xs text-(--color-text-muted)">Pre-loaded sample market data</p>
                </div>
                <div className="h-3 w-3 rounded-full bg-(--color-brand)" />
              </button>

              {/* OANDA */}
              <button
                onClick={() => setStep("oanda-key")}
                className="flex cursor-pointer items-center gap-4 rounded-sm border border-(--color-card-border) bg-(--color-input-bg) p-4 text-left transition-all hover:border-[#2962FF] hover:bg-[rgba(41,98,255,0.03)]"
              >
                <div className="rounded-sm bg-[rgba(41,98,255,0.12)] p-2.5">
                  <Key size={18} className="text-[#2962FF]" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-(--color-text-primary)">OANDA API</p>
                  <p className="text-xs text-(--color-text-muted)">Live market data connection</p>
                </div>
                <div className="h-3 w-3 rounded-full bg-[#2962FF]" />
              </button>

              {/* CSV */}
              <button
                onClick={() => setStep("csv-upload")}
                className="flex cursor-pointer items-center gap-4 rounded-sm border border-(--color-card-border) bg-(--color-input-bg) p-4 text-left transition-all hover:border-(--color-brand) hover:bg-[rgba(0,229,255,0.03)]"
              >
                <div className="bg-cyan/[0.12] rounded-sm p-2.5">
                  <Upload size={18} className="text-(--color-brand)" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-(--color-text-primary)">CSV Upload</p>
                  <p className="text-xs text-(--color-text-muted)">Backtest with historical data</p>
                </div>
                <div className="h-3 w-3 rounded-full bg-(--color-brand)" />
              </button>
            </div>
          )}

          {/* ================================================================ */}
          {/* STEP: DEMO INFO                                                 */}
          {/* ================================================================ */}
          {step === "demo-info" && (
            <div className="flex flex-col gap-5">
              <div className="bg-cyan/[0.06] flex flex-col gap-3 rounded-sm border-l-[3px] border-l-(--color-brand) p-5">
                <p className="text-sm font-semibold text-(--color-text-primary)">
                  What you get with Demo Mode
                </p>
                <div className="flex flex-col gap-2">
                  {[
                    "Pre-loaded data for 5 major pairs (M30, H1, H4, 2016–2026)",
                    "1,000,000+ historical candles — charts work immediately",
                    "Full backtest pipeline with all model types",
                    "Paper trading pre-enabled on the Trading page",
                    "No API key required — start in seconds",
                  ].map((item) => (
                    <div key={item} className="flex items-start gap-2">
                      <CheckCircle size={13} className="mt-0.5 shrink-0 text-(--color-brand)" />
                      <span className="text-xs leading-normal text-white/[0.80]">{item}</span>
                    </div>
                  ))}
                </div>
              </div>

              <button onClick={handleStartDemo} disabled={loading} style={primaryBtn(loading)}>
                <Zap size={16} />
                {loading ? "Loading Demo Data..." : "Start with Demo Data"}
              </button>
            </div>
          )}

          {/* ================================================================ */}
          {/* STEP: CSV UPLOAD                                                */}
          {/* ================================================================ */}
          {step === "csv-upload" && (
            <div className="flex flex-col gap-4">
              {/* Drop zone */}
              <label
                className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-sm border-2 border-dashed px-4 py-8 transition-all hover:border-(--color-brand) hover:bg-[rgba(0,229,255,0.05)] ${csvFiles.length > 0 ? "border-(--color-brand) bg-[rgba(0,229,255,0.03)]" : "border-(--color-card-border) bg-(--color-input-bg)"}`}
              >
                <Upload
                  size={24}
                  style={{
                    color: csvFiles.length > 0 ? "var(--color-brand)" : "var(--color-text-muted)",
                  }}
                />
                <p
                  className="text-sm font-semibold"
                  style={{
                    color:
                      csvFiles.length > 0 ? "var(--color-text-primary)" : "var(--color-text-muted)",
                  }}
                >
                  {csvFiles.length > 0
                    ? `${csvFiles.length} file(s) selected`
                    : "Click to select CSV files"}
                </p>
                <p className="text-xs text-(--color-text-dim)">OHLC data with timestamps (.csv)</p>
                <input
                  type="file"
                  accept=".csv"
                  multiple
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </label>

              {/* File list */}
              {csvFiles.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  {csvFiles.map((f, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 rounded-sm bg-(--color-input-bg) px-3 py-2"
                    >
                      <FileText size={12} className="text-(--color-text-muted)" />
                      <span className="flex-1 truncate font-mono text-xs text-(--color-text-primary)">
                        {f.name}
                      </span>
                      <span className="font-mono text-[10px] text-(--color-text-muted) tabular-nums">
                        {(f.size / 1024).toFixed(0)} KB
                      </span>
                      <button
                        onClick={() => removeCsvFile(i)}
                        className="cursor-pointer text-[10px] text-(--color-event-high)"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <button
                onClick={handleImportCsv}
                disabled={loading || csvFiles.length === 0}
                style={primaryBtn(loading || csvFiles.length === 0)}
              >
                <Zap size={16} />
                {loading
                  ? "Importing..."
                  : `Import & Start${csvFiles.length > 0 ? ` (${csvFiles.length} file${csvFiles.length > 1 ? "s" : ""})` : ""}`}
              </button>
            </div>
          )}

          {/* ================================================================ */}
          {/* STEP: OANDA KEY                                                 */}
          {/* ================================================================ */}
          {step === "oanda-key" && (
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold tracking-wider text-(--color-text-muted) uppercase">
                  OANDA API Key
                </label>
                <input
                  type="password"
                  value={oandaKey}
                  onChange={(e) => {
                    setOandaKey(e.target.value);
                    setError("");
                  }}
                  placeholder="Paste your API key here"
                  onBlur={(e) => {
                    e.currentTarget.style.borderColor = error
                      ? "var(--color-event-high)"
                      : "var(--color-card-border)";
                  }}
                  className={`${INPUT_CLASS} ${error ? "border-(--color-event-high)" : "border-(--color-card-border)"} focus:border-(--color-brand)`}
                  style={inputStyle}
                />
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold tracking-wider text-(--color-text-muted) uppercase">
                  Account ID <span className="font-normal text-(--color-text-dim)">(optional)</span>
                </label>
                <input
                  type="text"
                  value={accountId}
                  onChange={(e) => setAccountId(e.target.value)}
                  placeholder="e.g. 101-001-1234567-001"
                  className={`${INPUT_CLASS} border-(--color-card-border) focus:border-(--color-brand)`}
                  style={inputStyle}
                />
              </div>

              <button onClick={handleConnectOanda} disabled={loading} style={primaryBtn(loading)}>
                <Key size={16} />
                {loading ? "Connecting..." : "Connect"}
              </button>
            </div>
          )}

          {/* ================================================================ */}
          {/* STEP: OANDA DOWNLOAD                                            */}
          {/* ================================================================ */}
          {step === "oanda-download" && (
            <div className="flex flex-col gap-5">
              <div className="bg-cyan/[0.06] flex items-center gap-2 rounded-sm border-l-[3px] border-l-(--color-brand) p-3 text-xs text-white/[0.80]">
                <CheckCircle size={13} className="text-(--color-brand)" />
                Connected to OANDA — select data to download
              </div>

              {/* Pairs */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold tracking-wider text-(--color-text-muted) uppercase">
                  Pairs
                </label>
                <div className="flex flex-wrap gap-2">
                  {DOWNLOAD_PAIRS.map((p) => (
                    <button
                      key={p}
                      onClick={() => togglePair(p)}
                      className="cursor-pointer rounded-[6px] px-[14px] py-[6px] font-mono text-[12px] transition duration-150"
                      style={{
                        border: `1px solid ${selectedPairs.has(p) ? "var(--color-brand)" : "var(--color-card-border)"}`,
                        backgroundColor: selectedPairs.has(p)
                          ? "rgba(0, 229, 255, 0.1)"
                          : "transparent",
                        color: selectedPairs.has(p)
                          ? "var(--color-brand)"
                          : "var(--color-text-muted)",
                      }}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              {/* Timeframes */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold tracking-wider text-(--color-text-muted) uppercase">
                  Timeframes
                </label>
                <div className="flex gap-2">
                  {DOWNLOAD_TFS.map((tf) => (
                    <button
                      key={tf}
                      onClick={() => toggleTf(tf)}
                      className="cursor-pointer rounded-[6px] px-[14px] py-[6px] font-mono text-[12px] transition duration-150"
                      style={{
                        border: `1px solid ${selectedTfs.has(tf) ? "var(--color-brand)" : "var(--color-card-border)"}`,
                        backgroundColor: selectedTfs.has(tf)
                          ? "rgba(0, 229, 255, 0.1)"
                          : "transparent",
                        color: selectedTfs.has(tf)
                          ? "var(--color-brand)"
                          : "var(--color-text-muted)",
                      }}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>

              {/* Summary */}
              <div className="text-xs text-(--color-text-dim)">
                {Array.from(selectedPairs).length * Array.from(selectedTfs).length} download(s)
                queued &middot; 5 years each
              </div>

              <button
                onClick={handleDownloadAndStart}
                disabled={loading || selectedPairs.size === 0 || selectedTfs.size === 0}
                style={primaryBtn(loading || selectedPairs.size === 0 || selectedTfs.size === 0)}
              >
                <Zap size={16} />
                {loading ? "Downloading..." : "Download & Start"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
