import { useState } from "react";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { useStoreApiKey } from "@/api/queries";

const PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"] as const;
const TIMEFRAMES = ["M30", "H1", "H4"] as const;

type Step = "pair" | "apikey" | "done";

export function WelcomePage({ onComplete }: { onComplete: () => void }) {
  const { setField } = useSettingsStore();
  const setBacktestField = useBacktestStore((s) => s.setField);
  const storeApiKey = useStoreApiKey();

  const [step, setStep] = useState<Step>("pair");
  const [pair, setPair] = useState("EURUSD");
  const [timeframe, setTimeframe] = useState("H1");
  const [apiKey, setApiKey] = useState("");
  const [dataPath, setDataPath] = useState("");

  const handleFinish = () => {
    setField("oandaApiKey", apiKey || null);
    if (apiKey) {
      storeApiKey.mutate({ name: "oanda", value: apiKey });
    }
    if (dataPath) setField("dataDir", dataPath);
    setBacktestField("pair", pair);
    setBacktestField("timeframe", timeframe);
    localStorage.setItem("kodaquant-welcome-done", "1");
    onComplete();
  };

  return (
    <div
      className="flex min-h-screen items-center justify-center"
      style={{ backgroundColor: "var(--color-app)" }}
    >
      <div
        className="w-full max-w-lg rounded-xl border p-8"
        style={{ backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)" }}
      >
        {step === "pair" && (
          <>
            <StepIndicator current={1} total={2} />
            <h1
              className="mt-4 text-lg font-semibold"
              style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
            >
              Choose Your Market
            </h1>
            <p className="mt-1 text-sm" style={{ color: "var(--color-text-muted)" }}>
              Select a currency pair and timeframe to get started. You can change this later.
            </p>

            <div className="mt-6">
              <label
                className="mb-1.5 block text-xs font-medium uppercase tracking-wider"
                style={{ color: "var(--color-text-secondary)" }}
              >
                Currency Pair
              </label>
              <div className="flex flex-wrap gap-2">
                {PAIRS.map((p) => (
                  <button
                    key={p}
                    onClick={() => setPair(p)}
                    className="rounded-md border px-4 py-2 text-xs font-semibold transition-all"
                    style={{
                      borderColor: pair === p ? "var(--color-accent)" : "var(--color-border)",
                      backgroundColor: pair === p ? "rgba(0,229,255,0.1)" : "transparent",
                      color: pair === p ? "var(--color-accent)" : "var(--color-text-secondary)",
                      cursor: "pointer",
                    }}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-4">
              <label
                className="mb-1.5 block text-xs font-medium uppercase tracking-wider"
                style={{ color: "var(--color-text-secondary)" }}
              >
                Timeframe
              </label>
              <div className="flex gap-2">
                {TIMEFRAMES.map((tf) => (
                  <button
                    key={tf}
                    onClick={() => setTimeframe(tf)}
                    className="rounded-md border px-5 py-2 text-xs font-semibold transition-all"
                    style={{
                      borderColor: timeframe === tf ? "var(--color-accent)" : "var(--color-border)",
                      backgroundColor: timeframe === tf ? "rgba(0,229,255,0.1)" : "transparent",
                      color: timeframe === tf ? "var(--color-accent)" : "var(--color-text-secondary)",
                      cursor: "pointer",
                    }}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={() => setStep("apikey")}
              className="mt-8 w-full rounded-md py-2.5 text-xs font-semibold uppercase tracking-wider transition-all hover:brightness-110"
              style={{
                background: "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)",
                color: "var(--color-text-inverse)",
                cursor: "pointer",
              }}
            >
              Continue
            </button>
          </>
        )}

        {step === "apikey" && (
          <>
            <StepIndicator current={2} total={2} />
            <h1
              className="mt-4 text-lg font-semibold"
              style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
            >
              Connect Your Data
            </h1>
            <p className="mt-1 text-sm" style={{ color: "var(--color-text-muted)" }}>
              Optional: provide your OANDA API key for live data. You can skip this and add it later in Settings.
            </p>

            <div className="mt-6">
              <label
                className="mb-1.5 block text-xs font-medium uppercase tracking-wider"
                style={{ color: "var(--color-text-secondary)" }}
              >
                OANDA API Key (optional)
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your OANDA API key..."
                className="w-full rounded-md border px-3 py-2.5 text-xs outline-none transition-colors"
                style={{
                  backgroundColor: "var(--color-elevated)",
                  borderColor: "var(--color-border)",
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </div>

            <div className="mt-4">
              <label
                className="mb-1.5 block text-xs font-medium uppercase tracking-wider"
                style={{ color: "var(--color-text-secondary)" }}
              >
                Data Directory (optional)
              </label>
              <input
                type="text"
                value={dataPath}
                onChange={(e) => setDataPath(e.target.value)}
                placeholder="Path to CSV data files (leave empty for default)"
                className="w-full rounded-md border px-3 py-2.5 text-xs outline-none transition-colors"
                style={{
                  backgroundColor: "var(--color-elevated)",
                  borderColor: "var(--color-border)",
                  color: "var(--color-text-primary)",
                  fontFamily: "var(--font-mono)",
                }}
              />
            </div>

            <div className="mt-6 flex gap-3">
              <button
                onClick={() => setStep("pair")}
                className="flex-1 rounded-md border py-2.5 text-xs font-semibold uppercase tracking-wider transition-colors"
                style={{
                  borderColor: "var(--color-border)",
                  color: "var(--color-text-secondary)",
                  cursor: "pointer",
                  backgroundColor: "transparent",
                }}
              >
                Back
              </button>
              <button
                onClick={handleFinish}
                className="flex-[2] rounded-md py-2.5 text-xs font-semibold uppercase tracking-wider transition-all hover:brightness-110"
                style={{
                  background: "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)",
                  color: "var(--color-text-inverse)",
                  cursor: "pointer",
                }}
              >
                Start Backtesting
              </button>
            </div>

            <button
              onClick={handleFinish}
              className="mt-3 w-full py-2 text-xs"
              style={{ color: "var(--color-text-muted)", cursor: "pointer" }}
            >
              Skip for now
            </button>
          </>
        )}
      </div>
    </div>
  );
}

function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex gap-2">
      {Array.from({ length: total }, (_, i) => (
        <div
          key={i}
          className="h-1 flex-1 rounded-full transition-all"
          style={{
            backgroundColor: i < current ? "var(--color-accent)" : "var(--color-elevated)",
          }}
        />
      ))}
    </div>
  );
}