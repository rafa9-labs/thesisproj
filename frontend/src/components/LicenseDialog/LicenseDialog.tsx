import { useState } from "react";
import { Key, Clock, X, ExternalLink } from "lucide-react";

interface LicenseDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onActivated: () => void;
  trialDaysLeft?: number;
}

export function LicenseDialog({ isOpen, onClose, onActivated, trialDaysLeft }: LicenseDialogProps) {
  const [licenseKey, setLicenseKey] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"activate" | "trial">("activate");
  const hasTrial = trialDaysLeft !== undefined && trialDaysLeft > 0;

  if (!isOpen) return null;

  const handleActivate = async () => {
    if (!licenseKey.trim()) {
      setError("Please enter a license key");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
      const res = await fetch(`${apiBase}/license/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ license_key: licenseKey.trim() }),
      });
      const data = await res.json();
      if (data.success) {
        onActivated();
      } else {
        setError(data.detail || data.error || "Activation failed");
      }
    } catch {
      setError("Could not connect to license server. Check your internet connection.");
    } finally {
      setLoading(false);
    }
  };

  const handleStartTrial = async () => {
    setLoading(true);
    setError("");
    try {
      const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
      const res = await fetch(`${apiBase}/license/trial`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json();
      if (data.success) {
        onActivated();
      } else {
        setError(data.error || "Trial could not be started");
      }
    } catch {
      setError("Could not connect to server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.6)" }}
    >
      <div className="relative w-full max-w-md rounded-sm border border-(--color-border) bg-(--color-surface) p-6 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-(--color-text-muted)"
          className="cursor-pointer"
        >
          <X size={18} />
        </button>

        <div className="mb-4 flex items-center gap-3">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-sm"
            style={{ backgroundColor: "rgba(41,98,255,0.1)" }}
          >
            <Key size={20} className="text-(--color-accent)" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-(--color-text-primary)">
              Activate KodaQuant
            </h2>
            <p className="text-xs text-(--color-text-muted)">
              Enter your license key or start a free trial
            </p>
          </div>
        </div>

        <div className="mb-4 flex gap-2">
          <button
            onClick={() => setMode("activate")}
            className="flex-1 rounded-md px-3 py-2 text-xs font-medium transition-colors"
            style={{
              backgroundColor:
                mode === "activate" ? "var(--color-accent)" : "var(--color-elevated)",
              color: mode === "activate" ? "#fff" : "var(--color-text-secondary)",
              cursor: "pointer",
            }}
          >
            License Key
          </button>
          <button
            onClick={() => setMode("trial")}
            className="flex-1 rounded-md px-3 py-2 text-xs font-medium transition-colors"
            style={{
              backgroundColor: mode === "trial" ? "var(--color-accent)" : "var(--color-elevated)",
              color: mode === "trial" ? "#fff" : "var(--color-text-secondary)",
              cursor: "pointer",
            }}
          >
            Free Trial
          </button>
        </div>

        {mode === "activate" ? (
          <div className="flex flex-col gap-3">
            <input
              type="text"
              value={licenseKey}
              onChange={(e) => setLicenseKey(e.target.value)}
              placeholder="XXXX-XXXX-XXXX-XXXX"
              className="rounded-md border border-(--color-border) bg-(--color-app) px-3 py-2 font-mono text-sm text-(--color-text-primary)"
              onKeyDown={(e) => e.key === "Enter" && handleActivate()}
            />
            <button
              onClick={handleActivate}
              disabled={loading}
              className="rounded-md px-3 py-2 text-sm font-semibold transition-colors"
              style={{
                backgroundColor: loading ? "var(--color-elevated)" : "var(--color-accent)",
                color: "#fff",
                cursor: loading ? "not-allowed" : "pointer",
                opacity: loading ? 0.6 : 1,
              }}
            >
              {loading ? "Activating..." : "Activate License"}
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="rounded-md border border-(--color-border) bg-(--color-app) px-4 py-3">
              <div className="mb-2 flex items-center gap-2">
                <Clock size={16} className="text-(--color-accent)" />
                <span className="text-sm font-medium text-(--color-text-primary)">
                  14-Day Free Trial
                </span>
              </div>
              <p className="text-xs text-(--color-text-muted)">
                Full access to all models, execution types, HPO, and news features. No credit card
                required.
              </p>
            </div>
            <button
              onClick={handleStartTrial}
              disabled={loading || hasTrial}
              className="rounded-md px-3 py-2 text-sm font-semibold transition-colors"
              style={{
                backgroundColor:
                  loading || hasTrial ? "var(--color-elevated)" : "var(--color-accent)",
                color: "#fff",
                cursor: loading || hasTrial ? "not-allowed" : "pointer",
                opacity: loading || hasTrial ? 0.6 : 1,
              }}
            >
              {hasTrial
                ? `Trial active (${trialDaysLeft} days left)`
                : loading
                  ? "Starting..."
                  : "Start Free Trial"}
            </button>
          </div>
        )}

        {error && (
          <p className="mt-3 text-xs" style={{ color: "var(--color-accent-error, #ef4444)" }}>
            {error}
          </p>
        )}

        <div className="mt-4 border-t border-(--color-border) pt-3">
          <div className="flex items-center justify-between">
            <span className="text-xs text-(--color-text-muted)">
              Free tier: 3 models + basic execution
            </span>
            <a
              href="https://kodaquant.com/pricing"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-(--color-accent)"
            >
              Buy license <ExternalLink size={10} />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
