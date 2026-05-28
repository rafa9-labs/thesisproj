import { useState } from "react";
import { Wifi, FolderOpen, FlaskConical, Eye, EyeOff, ChevronRight, ArrowLeft } from "lucide-react";

type DataMode = "LIVE_API" | "LOCAL_CSV" | "DEMO";

interface DataSourceModalProps {
  isOpen: boolean;
  onBack: () => void;
  onStart: (mode: DataMode, value?: string) => void;
}

const TABS: { id: DataMode; label: string; icon: React.ReactNode }[] = [
  { id: "LIVE_API",   label: "Live API",      icon: <Wifi size={14} /> },
  { id: "LOCAL_CSV",  label: "Local CSV",     icon: <FolderOpen size={14} /> },
  { id: "DEMO",       label: "Demo Sandbox",  icon: <FlaskConical size={14} /> },
];

export function DataSourceModal({ isOpen, onBack, onStart }: DataSourceModalProps) {
  const [activeTab, setActiveTab]   = useState<DataMode>("LIVE_API");
  const [apiKey,    setApiKey]      = useState("");
  const [csvPath,   setCsvPath]     = useState("");
  const [showKey,   setShowKey]     = useState(false);

  if (!isOpen) return null;

  /* ── readiness gate ── */
  const canStart =
    (activeTab === "LIVE_API"  && apiKey.trim().length > 0) ||
    (activeTab === "LOCAL_CSV" && csvPath.trim().length > 0) ||
     activeTab === "DEMO";

  const handleStart = () => {
    if (!canStart) return;
    const value =
      activeTab === "LIVE_API"  ? apiKey.trim()  :
      activeTab === "LOCAL_CSV" ? csvPath.trim() :
      undefined;
    onStart(activeTab, value);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(5,6,8,0.82)", backdropFilter: "blur(4px)" }}
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

        <div className="px-7 pt-6 pb-7 flex flex-col gap-5">
          {/* ── header ── */}
          <div className="flex flex-col gap-1">
            <h2
              className="text-[18px] font-bold tracking-tight"
              style={{ color: "#E8ECF1", letterSpacing: "-0.01em" }}
            >
              Connect Your Data
            </h2>
            <p className="text-[12px] leading-relaxed" style={{ color: "#4E5870" }}>
              A data source is required to initialize the terminal workspace.
            </p>
          </div>

          {/* ── tab row ── */}
          <div
            className="flex rounded-lg overflow-hidden"
            style={{ border: "1px solid #1E2A3A", backgroundColor: "#07090F" }}
          >
            {TABS.map((tab) => {
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 text-[11px] font-semibold uppercase tracking-[0.08em] transition-all"
                  style={{
                    backgroundColor: active ? "#00E5FF" : "transparent",
                    color:           active ? "#07090F" : "#4E5870",
                    borderRight:     tab.id !== "DEMO" ? "1px solid #1E2A3A" : "none",
                    cursor:          "pointer",
                  }}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* ── tab body ── */}
          <div className="flex flex-col gap-4 min-h-[120px]">
            {activeTab === "LIVE_API" && (
              <div className="flex flex-col gap-3">
                <p className="text-[11px]" style={{ color: "#4E5870" }}>
                  Optional: provide your OANDA API key for live data. You can skip this and add it later in Settings.
                </p>

                <div className="flex flex-col gap-1.5">
                  <label
                    className="text-[10px] font-bold uppercase tracking-[0.1em]"
                    style={{ color: "#00E5FF" }}
                  >
                    OANDA API KEY (OPTIONAL)
                  </label>
                  <div className="relative">
                    <input
                      type={showKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="Enter your OANDA API key..."
                      className="w-full rounded-md px-3 py-2.5 text-[12px] pr-9"
                      style={{
                        backgroundColor: "#07090F",
                        border: "1px solid #1E2A3A",
                        color: "#E8ECF1",
                        fontFamily: "var(--font-mono)",
                        outline: "none",
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = "rgba(0,229,255,0.35)";
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = "#1E2A3A";
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey((v) => !v)}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2"
                      style={{ color: "#4E5870", cursor: "pointer", background: "none", border: "none" }}
                    >
                      {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  </div>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label
                    className="text-[10px] font-bold uppercase tracking-[0.1em]"
                    style={{ color: "#00E5FF" }}
                  >
                    DATA DIRECTORY (OPTIONAL)
                  </label>
                  <input
                    type="text"
                    value={csvPath}
                    onChange={(e) => setCsvPath(e.target.value)}
                    placeholder="Path to CSV data files (leave empty for default)"
                    className="w-full rounded-md px-3 py-2.5 text-[12px]"
                    style={{
                      backgroundColor: "#07090F",
                      border: "1px solid #1E2A3A",
                      color: "#E8ECF1",
                      fontFamily: "var(--font-mono)",
                      outline: "none",
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = "rgba(0,229,255,0.35)";
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = "#1E2A3A";
                    }}
                  />
                </div>
              </div>
            )}

            {activeTab === "LOCAL_CSV" && (
              <div className="flex flex-col gap-3">
                <p className="text-[11px]" style={{ color: "#4E5870" }}>
                  Point the terminal to a local folder containing OHLCV CSV files. Headers must include{" "}
                  <span style={{ color: "#00E5FF", fontFamily: "var(--font-mono)" }}>
                    datetime, open, high, low, close, volume
                  </span>.
                </p>
                <div className="flex flex-col gap-1.5">
                  <label
                    className="text-[10px] font-bold uppercase tracking-[0.1em]"
                    style={{ color: "#00E5FF" }}
                  >
                    DATA DIRECTORY
                  </label>
                  <div className="relative">
                    <FolderOpen
                      size={13}
                      className="absolute left-3 top-1/2 -translate-y-1/2"
                      style={{ color: "#4E5870" }}
                    />
                    <input
                      type="text"
                      value={csvPath}
                      onChange={(e) => setCsvPath(e.target.value)}
                      placeholder="/home/user/trading_data/"
                      className="w-full rounded-md pl-8 pr-3 py-2.5 text-[12px]"
                      style={{
                        backgroundColor: "#07090F",
                        border: "1px solid #1E2A3A",
                        color: "#E8ECF1",
                        fontFamily: "var(--font-mono)",
                        outline: "none",
                      }}
                      onFocus={(e) => {
                        e.currentTarget.style.borderColor = "rgba(0,229,255,0.35)";
                      }}
                      onBlur={(e) => {
                        e.currentTarget.style.borderColor = "#1E2A3A";
                      }}
                    />
                  </div>
                </div>
              </div>
            )}

            {activeTab === "DEMO" && (
              <div
                className="flex flex-col gap-3 rounded-lg p-4"
                style={{ backgroundColor: "#07090F", border: "1px solid #1E2A3A" }}
              >
                <div className="flex items-center gap-2">
                  <FlaskConical size={16} style={{ color: "#00E5FF" }} />
                  <span
                    className="text-[12px] font-bold uppercase tracking-[0.06em]"
                    style={{ color: "#00E5FF" }}
                  >
                    Demo Sandbox
                  </span>
                </div>
                <p className="text-[12px] leading-relaxed" style={{ color: "#7A8494" }}>
                  Load 30 days of historical{" "}
                  <span style={{ color: "#E8ECF1", fontFamily: "var(--font-mono)" }}>BTC/USD</span>{" "}
                  and{" "}
                  <span style={{ color: "#E8ECF1", fontFamily: "var(--font-mono)" }}>EUR/USD</span>{" "}
                  dummy data to test the interface instantly. No API key or files required.
                </p>
                <div
                  className="flex items-center gap-2 text-[11px]"
                  style={{ color: "#089981" }}
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-current" />
                  Ready to start immediately
                </div>
              </div>
            )}
          </div>

          {/* ── footer ── */}
          <div className="flex flex-col gap-3 pt-1">
            <div className="flex items-center gap-3">
              <button
                onClick={onBack}
                className="flex items-center gap-1.5 px-4 py-2.5 rounded-md text-[11px] font-semibold uppercase tracking-[0.08em] transition-all"
                style={{
                  backgroundColor: "transparent",
                  border: "1px solid #1E2A3A",
                  color: "#4E5870",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "#2A3A50";
                  e.currentTarget.style.color = "#7A8494";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#1E2A3A";
                  e.currentTarget.style.color = "#4E5870";
                }}
              >
                <ArrowLeft size={13} />
                Back
              </button>

              <button
                onClick={handleStart}
                disabled={!canStart}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-md text-[11px] font-bold uppercase tracking-[0.1em] transition-all"
                style={{
                  backgroundColor: canStart ? "#00E5FF" : "#0E1520",
                  color:           canStart ? "#07090F"  : "#2A3A50",
                  border:          canStart ? "none"     : "1px solid #1E2A3A",
                  cursor:          canStart ? "pointer"  : "not-allowed",
                  boxShadow:       canStart ? "0 0 16px rgba(0,229,255,0.3)" : "none",
                  transition:      "all 200ms ease",
                }}
              >
                Start Backtesting
                <ChevronRight size={14} />
              </button>
            </div>


          </div>
        </div>
      </div>
    </div>
  );
}
