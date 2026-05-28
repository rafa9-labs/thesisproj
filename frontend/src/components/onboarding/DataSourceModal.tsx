import { useState } from "react";
import { ChevronLeft, Upload, Key, Zap, Database } from "lucide-react";

type DataSourceMode = "choice" | "oanda" | "csv" | "demo";

interface DataSourceModalProps {
  isOpen: boolean;
  onBack: () => void;
  onStart: (mode: string, value?: string) => void;
}

export function DataSourceModal({ isOpen, onBack, onStart }: DataSourceModalProps) {
  const [step, setStep] = useState<DataSourceMode>("choice");
  const [oandaKey, setOandaKey] = useState("");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleStartDemo = () => {
    setLoading(true);
    setTimeout(() => {
      onStart("demo");
      setLoading(false);
    }, 500);
  };

  const handleStartOanda = () => {
    if (!oandaKey.trim()) {
      setError("Please enter your OANDA API key");
      return;
    }
    setLoading(true);
    setTimeout(() => {
      onStart("oanda", oandaKey);
      setLoading(false);
    }, 500);
  };

  const handleStartCsv = () => {
    if (!csvFile) {
      setError("Please select a CSV file");
      return;
    }
    setLoading(true);
    setTimeout(() => {
      onStart("csv", csvFile.name);
      setLoading(false);
    }, 500);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith(".csv")) {
      setCsvFile(file);
      setError("");
    } else {
      setError("Please select a valid CSV file");
    }
  };

  const goBack = () => {
    if (step !== "choice") {
      setStep("choice");
      setError("");
      setOandaKey("");
      setCsvFile(null);
    } else {
      onBack();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.7)", backdropFilter: "blur(2px)" }}
    >
      <div
        className="relative w-full max-w-[480px] rounded-xl overflow-hidden shadow-2xl"
        style={{
          backgroundColor: "#0B111E",
          border: "1px solid #1E2A3A",
        }}
      >
        {/* ── cyan top accent line ── */}
        <div
          className="h-[3px] w-full"
          style={{ background: "linear-gradient(90deg, #00E5FF 0%, rgba(0,229,255,0.15) 100%)" }}
        />

        <div className="px-8 pt-7 pb-8 flex flex-col gap-5">
          {/* ── Header ── */}
          <div className="flex items-center justify-between">
            <div className="flex-1">
              {step === "choice" ? (
              <h1
                className="text-lg font-bold"
                style={{ color: "#E8ECF1", letterSpacing: "0.5px" }}
              >
                Connect Your Data
              </h1>
            ) : step === "demo" ? (
              <h1
                className="text-lg font-bold"
                style={{ color: "#E8ECF1", letterSpacing: "0.5px" }}
              >
                Demo Mode
              </h1>
            ) : step === "oanda" ? (
              <h1
                className="text-lg font-bold"
                style={{ color: "#E8ECF1", letterSpacing: "0.5px" }}
              >
                Enter OANDA API Key
              </h1>
            ) : (
              <h1
                className="text-lg font-bold"
                style={{ color: "#E8ECF1", letterSpacing: "0.5px" }}
              >
                Upload CSV Data
                </h1>
              )}
              <p
                className="text-xs mt-1"
                style={{ color: "#787B86" }}
              >
                {step === "choice"
                  ? "Select your data source to begin backtesting"
                  : step === "demo"
                    ? "Get started instantly with pre-loaded market data"
                    : step === "oanda"
                      ? "Provide your API key for live market data"
                      : "Select a CSV file with your historical data"}
              </p>
            </div>
            {step !== "choice" && (
              <button
                onClick={goBack}
                className="p-2 rounded-lg transition-all"
                style={{
                  backgroundColor: "#0F1825",
                  border: "1px solid #1E2A3A",
                  color: "#787B86",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "#00E5FF";
                  e.currentTarget.style.color = "#00E5FF";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#1E2A3A";
                  e.currentTarget.style.color = "#787B86";
                }}
              >
                <ChevronLeft size={16} />
              </button>
            )}
          </div>

          {/* ── CHOICE STEP ── */}
          {step === "choice" && (
            <div className="flex flex-col gap-3">
              {/* Demo Option */}
              <button
                onClick={() => {
                  setError("");
                  handleStartDemo();
                }}
                disabled={loading}
                className="flex items-center gap-4 p-4 rounded-lg border transition-all text-left"
                style={{
                  backgroundColor: "#0F1825",
                  borderColor: "#1E2A3A",
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.7 : 1,
                }}
                onMouseEnter={(e) => {
                  if (!loading) {
                    e.currentTarget.style.borderColor = "#00E5FF";
                    e.currentTarget.style.backgroundColor = "rgba(0, 229, 255, 0.03)";
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#1E2A3A";
                  e.currentTarget.style.backgroundColor = "#0F1825";
                }}
              >
                <div
                  className="p-2.5 rounded-lg"
                  style={{ backgroundColor: "rgba(0, 229, 255, 0.12)" }}
                >
                  <Database size={18} style={{ color: "#00E5FF" }} />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold" style={{ color: "#E8ECF1" }}>
                    Demo Mode
                  </p>
                  <p className="text-xs" style={{ color: "#787B86" }}>
                    Pre-loaded sample market data
                  </p>
                </div>
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: "#00E5FF" }}
                />
              </button>

              {/* Live OANDA Option */}
              <button
                onClick={() => {
                  setStep("oanda");
                  setError("");
                }}
                className="flex items-center gap-4 p-4 rounded-lg border transition-all text-left"
                style={{
                  backgroundColor: "#0F1825",
                  borderColor: "#1E2A3A",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "#00E5FF";
                  e.currentTarget.style.backgroundColor = "rgba(0, 229, 255, 0.03)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#1E2A3A";
                  e.currentTarget.style.backgroundColor = "#0F1825";
                }}
              >
                <div
                  className="p-2.5 rounded-lg"
                  style={{ backgroundColor: "rgba(41, 98, 255, 0.12)" }}
                >
                  <Key size={18} style={{ color: "#2962FF" }} />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold" style={{ color: "#E8ECF1" }}>
                    OANDA API
                  </p>
                  <p className="text-xs" style={{ color: "#787B86" }}>
                    Live market data connection
                  </p>
                </div>
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: "#2962FF" }}
                />
              </button>

              {/* CSV Upload Option */}
              <button
                onClick={() => {
                  setStep("csv");
                  setError("");
                }}
                className="flex items-center gap-4 p-4 rounded-lg border transition-all text-left"
                style={{
                  backgroundColor: "#0F1825",
                  borderColor: "#1E2A3A",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "#00E5FF";
                  e.currentTarget.style.backgroundColor = "rgba(0, 229, 255, 0.03)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#1E2A3A";
                  e.currentTarget.style.backgroundColor = "#0F1825";
                }}
              >
                <div
                  className="p-2.5 rounded-lg"
                  style={{ backgroundColor: "rgba(0, 229, 255, 0.12)" }}
                >
                  <Upload size={18} style={{ color: "#00E5FF" }} />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold" style={{ color: "#E8ECF1" }}>
                    CSV Upload
                  </p>
                  <p className="text-xs" style={{ color: "#787B86" }}>
                    Backtest with historical data
                  </p>
                </div>
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: "#00E5FF" }}
                />
              </button>
            </div>
          )}

          {/* ── OANDA STEP ── */}
          {step === "oanda" && (
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <label
                  className="text-xs font-semibold"
                  style={{ color: "#787B86", textTransform: "uppercase", letterSpacing: "0.5px" }}
                >
                  OANDA API KEY
                </label>
                <input
                  type="password"
                  value={oandaKey}
                  onChange={(e) => {
                    setOandaKey(e.target.value);
                    setError("");
                  }}
                  placeholder="Paste your API key here"
                  className="px-3 py-2.5 rounded-lg border text-sm transition-all"
                  style={{
                    borderColor: error ? "#ef4444" : "#1E2A3A",
                    backgroundColor: "#0F1825",
                    color: "#E8ECF1",
                    fontFamily: "var(--font-mono, monospace)",
                    fontSize: "13px",
                  }}
                  onKeyDown={(e) => e.key === "Enter" && handleStartOanda()}
                  onFocus={(e) => {
                    if (!error) e.currentTarget.style.borderColor = "#00E5FF";
                  }}
                  onBlur={(e) => {
                    if (!error) e.currentTarget.style.borderColor = "#1E2A3A";
                  }}
                />
              </div>

              {error && (
                <p className="text-xs" style={{ color: "#ef4444" }}>
                  {error}
                </p>
              )}

              <button
                onClick={handleStartOanda}
                disabled={loading}
                className="px-4 py-3 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-2"
                style={{
                  backgroundColor: "#00E5FF",
                  color: "#050608",
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.7 : 1,
                  boxShadow: loading ? "none" : "0 0 12px rgba(0, 229, 255, 0.4)",
                }}
              >
                <Zap size={16} />
                {loading ? "Connecting..." : "Start Backtesting"}
              </button>
            </div>
          )}

          {/* ── CSV STEP ── */}
          {step === "csv" && (
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <label
                  className="text-xs font-semibold"
                  style={{ color: "#787B86", textTransform: "uppercase", letterSpacing: "0.5px" }}
                >
                  SELECT CSV FILE
                </label>
                <label
                  className="px-4 py-6 rounded-lg border-2 border-dashed transition-all flex flex-col items-center justify-center gap-2 cursor-pointer"
                  style={{
                    borderColor: csvFile ? "#00E5FF" : "#1E2A3A",
                    backgroundColor: csvFile ? "rgba(0, 229, 255, 0.03)" : "#0F1825",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "#00E5FF";
                    e.currentTarget.style.backgroundColor = "rgba(0, 229, 255, 0.05)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = csvFile ? "#00E5FF" : "#1E2A3A";
                    e.currentTarget.style.backgroundColor = csvFile ? "rgba(0, 229, 255, 0.03)" : "#0F1825";
                  }}
                >
                  <Upload
                    size={24}
                    style={{ color: csvFile ? "#00E5FF" : "#787B86" }}
                  />
                  <div className="text-center">
                    <p
                      className="text-sm font-semibold"
                      style={{ color: csvFile ? "#E8ECF1" : "#787B86" }}
                    >
                      {csvFile ? csvFile.name : "Click to select CSV file"}
                    </p>
                    <p className="text-xs" style={{ color: "#4E5870" }}>
                      OHLC data with timestamps
                    </p>
                  </div>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
              </div>

              {error && (
                <p className="text-xs" style={{ color: "#ef4444" }}>
                  {error}
                </p>
              )}

              <button
                onClick={handleStartCsv}
                disabled={loading || !csvFile}
                className="px-4 py-3 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-2"
                style={{
                  backgroundColor: csvFile ? "#00E5FF" : "#2A3A50",
                  color: csvFile ? "#050608" : "#787B86",
                  cursor: csvFile && !loading ? "pointer" : "not-allowed",
                  opacity: loading ? 0.7 : 1,
                  boxShadow: csvFile && !loading ? "0 0 12px rgba(0, 229, 255, 0.4)" : "none",
                }}
              >
                <Zap size={16} />
                {loading ? "Loading..." : "Start Backtesting"}
              </button>
            </div>
          )}

          {/* ── DEMO MODE STEP (No inputs, just confirmation) ── */}
          {step === "demo" && (
            <div className="flex flex-col gap-4">
              <div className="rounded-lg p-4" style={{ backgroundColor: "rgba(0, 229, 255, 0.08)", borderLeft: "3px solid #00E5FF" }}>
                <p className="text-sm" style={{ color: "#E8ECF1", lineHeight: "1.5" }}>
                  You&apos;re about to start backtesting with pre-loaded sample market data. This includes historical OHLC prices and volume data to help you explore the platform&apos;s capabilities.
                </p>
              </div>

              <button
                onClick={() => {
                  setError("");
                  handleStartDemo();
                }}
                disabled={loading}
                className="px-4 py-3 rounded-lg font-semibold text-sm transition-all flex items-center justify-center gap-2"
                style={{
                  backgroundColor: "#00E5FF",
                  color: "#050608",
                  cursor: loading ? "not-allowed" : "pointer",
                  opacity: loading ? 0.7 : 1,
                  boxShadow: loading ? "none" : "0 0 12px rgba(0, 229, 255, 0.4)",
                }}
              >
                <Zap size={16} />
                {loading ? "Loading..." : "Start with Demo Data"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
