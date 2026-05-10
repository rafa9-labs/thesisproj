import { useState, useEffect, useRef } from "react";
import {
  Settings as SettingsIcon,
  Cpu,
  Database,
  Key,
  Info,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Check,
  X,
  AlertTriangle,
} from "lucide-react";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { useConfig, useSaveConfig, useStoreApiKey } from "@/api/queries";

interface SectionProps {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function Section({ icon, title, children, defaultOpen = false }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div
      className="rounded-lg border transition-all duration-200 hover:border-[var(--color-border-active)]"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "var(--color-glass)",
        backdropFilter: "blur(12px)",
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-3 px-5 py-3.5 text-left transition-colors duration-200"
        style={{ cursor: "pointer" }}
      >
        <span style={{ color: "var(--color-text-muted)" }}>{icon}</span>
        <span className="flex-1 text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
          {title}
        </span>
        {open ? (
          <ChevronDown size={16} strokeWidth={1.5} style={{ color: "var(--color-text-muted)" }} />
        ) : (
          <ChevronRight size={16} strokeWidth={1.5} style={{ color: "var(--color-text-muted)" }} />
        )}
      </button>
      {open && (
        <div
          className="border-t px-5 py-4"
          style={{ borderColor: "var(--color-glass-border)" }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2.5">
      <span className="text-[11px] font-light tracking-wide" style={{ color: "var(--color-text-secondary)" }}>
        {label}
      </span>
      <div className="flex items-center gap-2">{children}</div>
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      className="relative h-5 w-9 rounded-full transition-all duration-200"
      style={{
        backgroundColor: value ? "var(--color-brand)" : "var(--color-glass-border)",
        cursor: "pointer",
        boxShadow: value ? "0 0 8px rgba(0,229,255,0.25)" : "none",
      }}
    >
      <span
        className="absolute top-0.5 h-4 w-4 rounded-full transition-transform duration-200"
        style={{
          backgroundColor: value ? "var(--color-text-inverse)" : "var(--color-text-muted)",
          left: value ? 18 : 2,
        }}
      />
    </button>
  );
}

interface LicenseInfo {
  plan: string;
  licensed: boolean;
  trial_active: boolean;
  trial_days_left: number;
  needs_activation: boolean;
  license_key: string;
  expires_at: string;
  machine_id: string;
}

function LicenseSection() {
  const [license, setLicense] = useState<LicenseInfo | null>(null);
  const [inputKey, setInputKey] = useState("");
  const [activating, setActivating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
    fetch(`${apiBase}/license/status`)
      .then((r) => r.json())
      .then((data) => setLicense(data as LicenseInfo))
      .catch(() => {});
  }, []);

  const handleActivate = async () => {
    if (!inputKey.trim()) return;
    setActivating(true);
    setError("");
    try {
      const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
      const res = await fetch(`${apiBase}/license/activate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ license_key: inputKey.trim() }),
      });
      const data = await res.json();
      if (data.success) {
        const statusRes = await fetch(`${apiBase}/license/status`);
        setLicense(await statusRes.json());
        setInputKey("");
      } else {
        setError(data.detail || data.error || "Activation failed");
      }
    } catch {
      setError("Connection failed");
    } finally {
      setActivating(false);
    }
  };

  const handleDeactivate = async () => {
    try {
      const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
      await fetch(`${apiBase}/license/deactivate`, { method: "POST" });
      const statusRes = await fetch(`${apiBase}/license/status`);
      setLicense(await statusRes.json());
    } catch {}
  };

  if (!license) {
    return (
      <div className="py-3 text-xs" style={{ color: "var(--color-text-muted)" }}>
        Checking license status...
      </div>
    );
  }

  const planLabel: Record<string, string> = {
    free: "Free",
    trial: "Trial",
    pro: "Pro",
    team: "Team",
  };
  const planColor: Record<string, string> = {
    free: "var(--color-text-muted)",
    trial: "#f59e0b",
    pro: "var(--color-accent)",
    team: "#10b981",
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="rounded-md border px-3 py-1.5 text-xs font-semibold"
            style={{
              borderColor: planColor[license.plan] || "var(--color-border)",
              backgroundColor: `${planColor[license.plan] || "var(--color-accent)"}15`,
              color: planColor[license.plan] || "var(--color-text-primary)",
            }}
          >
            {planLabel[license.plan] || license.plan}
          </span>
          {license.trial_active && (
            <span className="text-xs" style={{ color: "#f59e0b" }}>
              {license.trial_days_left} days left
            </span>
          )}
        </div>
        {license.licensed && (
          <button
            onClick={handleDeactivate}
            className="rounded-md border px-2 py-1 text-xs"
            style={{
              borderColor: "var(--color-border)",
              color: "var(--color-text-muted)",
              cursor: "pointer",
            }}
          >
            Deactivate
          </button>
        )}
      </div>

      {license.needs_activation && !license.trial_active && (
        <div className="flex flex-col gap-2">
          <input
            type="text"
            value={inputKey}
            onChange={(e) => setInputKey(e.target.value)}
            placeholder="XXXX-XXXX-XXXX-XXXX"
            className="rounded-md border px-3 py-1.5 text-xs"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-app)",
              color: "var(--color-text-primary)",
              fontFamily: "var(--font-mono)",
            }}
            onKeyDown={(e) => e.key === "Enter" && handleActivate()}
          />
          <div className="flex gap-2">
            <button
              onClick={handleActivate}
              disabled={activating}
              className="rounded-md px-3 py-1.5 text-xs font-semibold"
              style={{
                backgroundColor: "var(--color-accent)",
                color: "#fff",
                cursor: activating ? "not-allowed" : "pointer",
                opacity: activating ? 0.6 : 1,
              }}
            >
              {activating ? "Activating..." : "Activate"}
            </button>
          </div>
          {error && (
            <p className="text-xs" style={{ color: "#ef4444" }}>{error}</p>
          )}
        </div>
      )}

      {license.machine_id && (
        <FieldRow label="Machine ID">
          <span className="text-xs font-mono" style={{ color: "var(--color-text-muted)" }}>
            {license.machine_id}
          </span>
        </FieldRow>
      )}

      <div className="text-xs" style={{ color: "var(--color-text-muted)" }}>
        {license.plan === "free"
          ? "Free: 3 models (logistic, XGBoost, RF) + fixed lot sizing"
          : license.plan === "trial"
            ? "Full access during trial period"
            : "All features unlocked"}
      </div>
    </div>
  );
}

export function SettingsPage() {
  const store = useSettingsStore();
  const { data: remoteConfig } = useConfig();
  const saveConfig = useSaveConfig();
  const storeApiKey = useStoreApiKey();
  const [apiKeySaved, setApiKeySaved] = useState(false);
  const synced = useRef(false);

  useEffect(() => {
    if (remoteConfig && !synced.current) {
      synced.current = true;
      if (remoteConfig.threadBudget != null) store.setField("threadBudget", remoteConfig.threadBudget as number);
      if (remoteConfig.mixedPrecision != null) store.setField("mixedPrecision", remoteConfig.mixedPrecision as boolean);
      if (remoteConfig.verboseMode != null) store.setField("verboseMode", remoteConfig.verboseMode as boolean);
      if (remoteConfig.apiUrl != null) store.setField("apiUrl", remoteConfig.apiUrl as string);
    }
  }, [remoteConfig]);

  const syncToBackend = (key: string, value: unknown) => {
    saveConfig.mutate({ [key]: value, ...store });
  };

  const handleOandaBlur = () => {
    const key = store.oandaApiKey;
    if (key) {
      storeApiKey.mutate({ name: "oanda", value: key }, {
        onSuccess: () => {
          setApiKeySaved(true);
          setTimeout(() => setApiKeySaved(false), 2000);
        },
      });
    }
  };

  return (
    <div className="flex flex-col gap-5">

      <Section icon={<SettingsIcon size={18} />} title="General" defaultOpen>
        <FieldRow label="Verbose Mode (Apprentice)">
          <Toggle
            value={store.verboseMode}
            onChange={(v) => store.setField("verboseMode", v)}
          />
        </FieldRow>
        <FieldRow label="API URL">
          <input
            type="text"
            value={store.apiUrl}
            onChange={(e) => store.setField("apiUrl", e.target.value)}
            className="rounded-md border px-3 py-1.5 text-xs transition-all duration-200 focus:outline-none"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "var(--color-glass)",
              color: "var(--color-text-primary)",
              fontFamily: "var(--font-mono)",
              width: 240,
              backdropFilter: "blur(8px)",
            }}
          />
        </FieldRow>
        <FieldRow label="Theme">
          <span
            className="rounded-md border px-3 py-1.5 text-xs"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "var(--color-glass)",
              color: "var(--color-text-primary)",
              backdropFilter: "blur(8px)",
            }}
          >
            Dark (only)
          </span>
        </FieldRow>
      </Section>

      <Section icon={<Cpu size={18} />} title="GPU & Compute">
        <FieldRow label="Thread Budget">
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={1}
              max={16}
              value={store.threadBudget}
              onChange={(e) => store.setField("threadBudget", Number(e.target.value))}
              className="w-32"
              style={{
                accentColor: "var(--color-brand)",
                background: `linear-gradient(to right, var(--color-brand) 0%, var(--color-brand) ${((store.threadBudget - 1) / 15) * 100}%, var(--color-glass-border) ${((store.threadBudget - 1) / 15) * 100}%, var(--color-glass-border) 100%)`,
              }}
            />
            <span
              className="text-xs font-medium"
              style={{ color: "var(--color-brand)", fontFamily: "var(--font-mono)", minWidth: 24 }}
            >
              {store.threadBudget}
            </span>
          </div>
        </FieldRow>
        <FieldRow label="Mixed Precision (FP16)">
          <Toggle
            value={store.mixedPrecision}
            onChange={(v) => store.setField("mixedPrecision", v)}
          />
        </FieldRow>
        <FieldRow label="GPU Status">
          <span
            className="text-xs"
            style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
          >
            Detected at startup — see terminal panel
          </span>
        </FieldRow>
      </Section>

      <Section icon={<Database size={18} />} title="Data Sources">
        <FieldRow label="OANDA API Key">
          <div className="flex items-center gap-2">
            <input
              type="password"
              value={store.oandaApiKey ?? ""}
              onChange={(e) => store.setField("oandaApiKey", e.target.value || null)}
              onBlur={handleOandaBlur}
              placeholder="Enter API key…"
              className="rounded-md border px-3 py-1.5 text-xs transition-all duration-200 focus:outline-none"
              style={{
                borderColor: "var(--color-glass-border)",
                backgroundColor: "var(--color-glass)",
                color: "var(--color-text-primary)",
                fontFamily: "var(--font-mono)",
                width: 240,
                backdropFilter: "blur(8px)",
              }}
            />
            {apiKeySaved && (
              <span className="text-[10px] font-medium" style={{ color: "var(--color-accent-success)" }}>
                Saved
              </span>
            )}
          </div>
        </FieldRow>
        <FieldRow label="Data Directory">
          <span
            className="text-xs"
            style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
          >
            Configured via environment variable
          </span>
        </FieldRow>
      </Section>

      <Section icon={<Key size={18} />} title="License">
        <LicenseSection />
      </Section>

      <Section icon={<SettingsIcon size={18} />} title="Pipeline Configuration">
        <div className="flex flex-col gap-2">
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            Advanced pipeline parameters are configured per-backtest on the Backtest page.
            Default constants are defined in <code style={{ fontFamily: "var(--font-mono)" }}>config.py</code>.
          </span>
          <button
            onClick={() => {
              store.setField("verboseMode", false);
              store.setField("threadBudget", 4);
              store.setField("mixedPrecision", true);
              store.setField("apiUrl", "http://localhost:8000");
              syncToBackend("reset", true);
            }}
            className="mt-2 self-start rounded-md border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.08em] transition-all duration-200 hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "var(--color-glass-hover)",
              color: "var(--color-text-secondary)",
              cursor: "pointer",
            }}
          >
            Reset to Defaults
          </button>
        </div>
      </Section>

      <Section icon={<Info size={18} />} title="About">
        <div className="flex flex-col gap-2">
          <FieldRow label="Version">
            <span style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
              v1.0.0-dev
            </span>
          </FieldRow>
          <FieldRow label="Pipeline">
            <span style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
              Forex ML Backtester
            </span>
          </FieldRow>
          <FieldRow label="Models">
            <span style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)", fontSize: 12 }}>
              10 registered
            </span>
          </FieldRow>
          <div className="flex items-center gap-2 pt-2">
            <ExternalLink size={12} style={{ color: "var(--color-text-muted)" }} />
            <a
              href="https://github.com/rafa9-labs/thesisproj"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs"
              style={{ color: "var(--color-accent)" }}
            >
              GitHub Repository
            </a>
          </div>
        </div>
      </Section>
    </div>
  );
}
