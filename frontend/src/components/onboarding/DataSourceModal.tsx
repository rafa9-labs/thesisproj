import { useState } from "react";
import { ChevronLeft, Upload, Key, Zap, Database, CheckCircle, AlertCircle, FileText } from "lucide-react";
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
      if (step === "oanda-download") { setStep("oanda-key"); }
      else if (step === "oanda-key") { setStep("choice"); }
      else { setStep("choice"); }
      setError("");
    } else {
      onBack();
    }
  };

  const handleStartDemo = async () => {
    setLoading(true);
    setError("");
    try {
      await demoSeed.mutateAsync({ pairs: ["EURUSD"], timeframes: ["M30", "H1", "H4"] });
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
    if (next.has(p)) next.delete(p); else next.add(p);
    setSelectedPairs(next);
  };

  const toggleTf = (tf: string) => {
    const next = new Set(selectedTfs);
    if (next.has(tf)) next.delete(tf); else next.add(tf);
    setSelectedTfs(next);
  };

  /* ── Shared styles ───────────────────────────────────────── */

  const cardBtn = (active = false) => ({
    display: "flex" as const,
    alignItems: "center" as const,
    gap: "16px",
    padding: "16px",
    borderRadius: "8px",
    border: `1px solid ${active ? "#00E5FF" : "#1E2A3A"}`,
    backgroundColor: active ? "rgba(0, 229, 255, 0.05)" : "#0F1825",
    cursor: "pointer" as const,
    transition: "all 0.15s",
  });

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
    backgroundColor: disabled ? "#2A3A50" : "#00E5FF",
    color: disabled ? "#787B86" : "#050608",
    cursor: (disabled ? "not-allowed" : "pointer") as "not-allowed" | "pointer",
    opacity: loading ? 0.7 : 1,
    boxShadow: disabled ? "none" : "0 0 12px rgba(0, 229, 255, 0.4)",
    transition: "all 0.15s",
  });

  const backBtn = {
    padding: "8px",
    borderRadius: "8px",
    border: "1px solid #1E2A3A",
    backgroundColor: "#0F1825",
    color: "#787B86",
    cursor: "pointer" as const,
    transition: "all 0.15s",
  };

  const inputStyle = (hasError = false) => ({
    width: "100%" as const,
    padding: "10px 12px",
    borderRadius: "8px",
    border: `1px solid ${hasError ? "#ef4444" : "#1E2A3A"}`,
    backgroundColor: "#0F1825",
    color: "#E8ECF1",
    fontFamily: "var(--font-mono, monospace)" as const,
    fontSize: "13px",
    outline: "none" as const,
  });

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.7)", backdropFilter: "blur(2px)" }}
    >
      <div
        className="relative w-full max-w-[520px] rounded-xl overflow-hidden shadow-2xl"
        style={{ backgroundColor: "#0B111E", border: "1px solid #1E2A3A" }}
      >
        {/* cyan top accent */}
        <div
          className="h-[3px] w-full"
          style={{ background: "linear-gradient(90deg, #00E5FF 0%, rgba(0,229,255,0.15) 100%)" }}
        />

        <div className="px-8 pt-7 pb-8 flex flex-col gap-5">
          {/* ── Header with back button ── */}
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <h1 className="text-lg font-bold" style={{ color: "#E8ECF1", letterSpacing: "0.5px" }}>
                {step === "choice" && "Connect Your Data"}
                {step === "demo-info" && "Demo Mode"}
                {step === "csv-upload" && "Upload CSV Data"}
                {step === "oanda-key" && "Enter OANDA API Key"}
                {step === "oanda-download" && "Download Market Data"}
              </h1>
              <p className="text-xs mt-1" style={{ color: "#787B86" }}>
                {step === "choice" && "Select how you want to get started"}
                {step === "demo-info" && "Get started instantly with pre-loaded market data"}
                {step === "csv-upload" && "Import your own historical data files"}
                {step === "oanda-key" && "Connect to OANDA for live and historical data"}
                {step === "oanda-download" && "Choose which data to fetch from OANDA"}
              </p>
            </div>
            {step !== "choice" && (
              <button onClick={goBack} style={backBtn}>
                <ChevronLeft size={16} />
              </button>
            )}
          </div>

          {/* ── ERROR ── */}
          {error && (
            <div
              className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
              style={{ backgroundColor: "rgba(239, 68, 68, 0.1)", color: "#ef4444", border: "1px solid rgba(239, 68, 68, 0.2)" }}
            >
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
                className="flex items-center gap-4 p-4 rounded-lg border transition-all text-left"
                style={{ backgroundColor: "#0F1825", borderColor: "#1E2A3A", cursor: "pointer" }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#00E5FF"; e.currentTarget.style.backgroundColor = "rgba(0, 229, 255, 0.03)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#1E2A3A"; e.currentTarget.style.backgroundColor = "#0F1825"; }}
              >
                <div className="p-2.5 rounded-lg" style={{ backgroundColor: "rgba(0, 229, 255, 0.12)" }}>
                  <Database size={18} style={{ color: "#00E5FF" }} />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold" style={{ color: "#E8ECF1" }}>Demo Mode</p>
                  <p className="text-xs" style={{ color: "#787B86" }}>Pre-loaded sample market data</p>
                </div>
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: "#00E5FF" }} />
              </button>

              {/* OANDA */}
              <button
                onClick={() => setStep("oanda-key")}
                className="flex items-center gap-4 p-4 rounded-lg border transition-all text-left"
                style={{ backgroundColor: "#0F1825", borderColor: "#1E2A3A", cursor: "pointer" }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#2962FF"; e.currentTarget.style.backgroundColor = "rgba(41, 98, 255, 0.03)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#1E2A3A"; e.currentTarget.style.backgroundColor = "#0F1825"; }}
              >
                <div className="p-2.5 rounded-lg" style={{ backgroundColor: "rgba(41, 98, 255, 0.12)" }}>
                  <Key size={18} style={{ color: "#2962FF" }} />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold" style={{ color: "#E8ECF1" }}>OANDA API</p>
                  <p className="text-xs" style={{ color: "#787B86" }}>Live market data connection</p>
                </div>
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: "#2962FF" }} />
              </button>

              {/* CSV */}
              <button
                onClick={() => setStep("csv-upload")}
                className="flex items-center gap-4 p-4 rounded-lg border transition-all text-left"
                style={{ backgroundColor: "#0F1825", borderColor: "#1E2A3A", cursor: "pointer" }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#00E5FF"; e.currentTarget.style.backgroundColor = "rgba(0, 229, 255, 0.03)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#1E2A3A"; e.currentTarget.style.backgroundColor = "#0F1825"; }}
              >
                <div className="p-2.5 rounded-lg" style={{ backgroundColor: "rgba(0, 229, 255, 0.12)" }}>
                  <Upload size={18} style={{ color: "#00E5FF" }} />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold" style={{ color: "#E8ECF1" }}>CSV Upload</p>
                  <p className="text-xs" style={{ color: "#787B86" }}>Backtest with historical data</p>
                </div>
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: "#00E5FF" }} />
              </button>
            </div>
          )}

          {/* ================================================================ */}
          {/* STEP: DEMO INFO                                                 */}
          {/* ================================================================ */}
          {step === "demo-info" && (
            <div className="flex flex-col gap-5">
              <div
                className="rounded-lg p-5 flex flex-col gap-3"
                style={{ backgroundColor: "rgba(0, 229, 255, 0.06)", borderLeft: "3px solid #00E5FF" }}
              >
                <p className="text-sm font-semibold" style={{ color: "#E8ECF1" }}>
                  What you get with Demo Mode
                </p>
                <div className="flex flex-col gap-2">
                  {[
                    "Pre-loaded EUR/USD data (M30, H1, H4, 2016–2026)",
                    "124,000+ historical candles — charts work immediately",
                    "Full backtest pipeline with all model types",
                    "Paper trading pre-enabled on the Trading page",
                    "No API key required — start in seconds",
                  ].map((item) => (
                    <div key={item} className="flex items-start gap-2">
                      <CheckCircle size={13} style={{ color: "#00E5FF", marginTop: 2, flexShrink: 0 }} />
                      <span className="text-xs" style={{ color: "#C8CCD0", lineHeight: 1.5 }}>{item}</span>
                    </div>
                  ))}
                </div>
              </div>

              <button
                onClick={handleStartDemo}
                disabled={loading}
                style={primaryBtn(loading)}
              >
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
                className="px-4 py-8 rounded-lg border-2 border-dashed flex flex-col items-center justify-center gap-2 cursor-pointer transition-all"
                style={{
                  borderColor: csvFiles.length > 0 ? "#00E5FF" : "#1E2A3A",
                  backgroundColor: csvFiles.length > 0 ? "rgba(0, 229, 255, 0.03)" : "#0F1825",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#00E5FF"; e.currentTarget.style.backgroundColor = "rgba(0, 229, 255, 0.05)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = csvFiles.length > 0 ? "#00E5FF" : "#1E2A3A"; e.currentTarget.style.backgroundColor = csvFiles.length > 0 ? "rgba(0, 229, 255, 0.03)" : "#0F1825"; }}
              >
                <Upload size={24} style={{ color: csvFiles.length > 0 ? "#00E5FF" : "#787B86" }} />
                <p className="text-sm font-semibold" style={{ color: csvFiles.length > 0 ? "#E8ECF1" : "#787B86" }}>
                  {csvFiles.length > 0 ? `${csvFiles.length} file(s) selected` : "Click to select CSV files"}
                </p>
                <p className="text-xs" style={{ color: "#4E5870" }}>
                  OHLC data with timestamps (.csv)
                </p>
                <input type="file" accept=".csv" multiple onChange={handleFileSelect} className="hidden" />
              </label>

              {/* File list */}
              {csvFiles.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  {csvFiles.map((f, i) => (
                    <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{ backgroundColor: "#0F1825" }}>
                      <FileText size={12} style={{ color: "#787B86" }} />
                      <span className="flex-1 text-xs truncate" style={{ color: "#E8ECF1", fontFamily: "var(--font-mono)" }}>
                        {f.name}
                      </span>
                      <span className="text-[10px] tabular-nums" style={{ color: "#787B86", fontFamily: "var(--font-mono)" }}>
                        {(f.size / 1024).toFixed(0)} KB
                      </span>
                      <button onClick={() => removeCsvFile(i)} className="text-[10px]" style={{ color: "#ef4444", cursor: "pointer" }}>
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
                {loading ? "Importing..." : `Import & Start${csvFiles.length > 0 ? ` (${csvFiles.length} file${csvFiles.length > 1 ? "s" : ""})` : ""}`}
              </button>
            </div>
          )}

          {/* ================================================================ */}
          {/* STEP: OANDA KEY                                                 */}
          {/* ================================================================ */}
          {step === "oanda-key" && (
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: "#787B86" }}>
                  OANDA API Key
                </label>
                <input
                  type="password"
                  value={oandaKey}
                  onChange={(e) => { setOandaKey(e.target.value); setError(""); }}
                  placeholder="Paste your API key here"
                  onFocus={(e) => { e.currentTarget.style.borderColor = "#00E5FF"; }}
                  onBlur={(e) => { e.currentTarget.style.borderColor = error ? "#ef4444" : "#1E2A3A"; }}
                  style={inputStyle(!!error)}
                />
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: "#787B86" }}>
                  Account ID <span style={{ color: "#4E5870", fontWeight: 400 }}>(optional)</span>
                </label>
                <input
                  type="text"
                  value={accountId}
                  onChange={(e) => setAccountId(e.target.value)}
                  placeholder="e.g. 101-001-1234567-001"
                  onFocus={(e) => { e.currentTarget.style.borderColor = "#00E5FF"; }}
                  onBlur={(e) => { e.currentTarget.style.borderColor = "#1E2A3A"; }}
                  style={inputStyle()}
                />
              </div>

              <button
                onClick={handleConnectOanda}
                disabled={loading}
                style={primaryBtn(loading)}
              >
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
              <div
                className="rounded-lg p-3 text-xs flex items-center gap-2"
                style={{ backgroundColor: "rgba(0, 229, 255, 0.06)", borderLeft: "3px solid #00E5FF", color: "#C8CCD0" }}
              >
                <CheckCircle size={13} style={{ color: "#00E5FF" }} />
                Connected to OANDA — select data to download
              </div>

              {/* Pairs */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: "#787B86" }}>
                  Pairs
                </label>
                <div className="flex flex-wrap gap-2">
                  {DOWNLOAD_PAIRS.map((p) => (
                    <button
                      key={p}
                      onClick={() => togglePair(p)}
                      style={{
                        padding: "6px 14px",
                        borderRadius: "6px",
                        border: `1px solid ${selectedPairs.has(p) ? "#00E5FF" : "#1E2A3A"}`,
                        backgroundColor: selectedPairs.has(p) ? "rgba(0, 229, 255, 0.1)" : "transparent",
                        color: selectedPairs.has(p) ? "#00E5FF" : "#787B86",
                        fontSize: "12px",
                        fontFamily: "var(--font-mono)",
                        cursor: "pointer",
                        transition: "all 0.15s",
                      }}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              {/* Timeframes */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: "#787B86" }}>
                  Timeframes
                </label>
                <div className="flex gap-2">
                  {DOWNLOAD_TFS.map((tf) => (
                    <button
                      key={tf}
                      onClick={() => toggleTf(tf)}
                      style={{
                        padding: "6px 14px",
                        borderRadius: "6px",
                        border: `1px solid ${selectedTfs.has(tf) ? "#00E5FF" : "#1E2A3A"}`,
                        backgroundColor: selectedTfs.has(tf) ? "rgba(0, 229, 255, 0.1)" : "transparent",
                        color: selectedTfs.has(tf) ? "#00E5FF" : "#787B86",
                        fontSize: "12px",
                        fontFamily: "var(--font-mono)",
                        cursor: "pointer",
                        transition: "all 0.15s",
                      }}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>

              {/* Summary */}
              <div className="text-xs" style={{ color: "#4E5870" }}>
                {Array.from(selectedPairs).length * Array.from(selectedTfs).length} download(s) queued &middot; 5 years each
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
