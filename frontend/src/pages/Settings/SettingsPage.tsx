import { useState, useEffect, useRef } from "react";
import {
  Settings as SettingsIcon,
  Cpu,
  Database,
  Key,
  Info,
  ExternalLink,
  Check,
} from "lucide-react";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { useConfig, useSaveConfig, useStoreApiKey, useStoreKv } from "@/api/queries";

/* ─────────────────────── shared primitives ─────────────────────── */

const cardStyle: React.CSSProperties = {
  borderColor: "var(--color-glass-border)",
  backgroundColor: "rgba(255,255,255,0.02)",
};

function SectionCard({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="rounded-sm border flex flex-col gap-5 p-6"
      style={cardStyle}
    >
      <div className="flex items-center gap-2.5">
        <div
          className="h-4 w-[2px] rounded-full flex-shrink-0"
          style={{ backgroundColor: "var(--color-brand)" }}
        />
        <span style={{ color: "var(--color-brand)", display: "flex" }}>{icon}</span>
        <h3
          className="text-[11px] font-medium uppercase tracking-[0.12em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          {title}
        </h3>
      </div>
      {children}
    </div>
  );
}

function FieldRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2 border-t" style={{ borderColor: "var(--color-glass-border)" }}>
      <div className="flex flex-col gap-0.5 min-w-0">
        <span
          className="text-[12px] font-light"
          style={{ color: "var(--color-text-primary)" }}
        >
          {label}
        </span>
        {hint && (
          <span
            className="text-[10px]"
            style={{ color: "var(--color-text-muted)" }}
          >
            {hint}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">{children}</div>
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      aria-checked={value}
      role="switch"
      className="relative h-5 w-9 rounded-full transition-all duration-200 flex-shrink-0"
      style={{
        backgroundColor: value ? "var(--color-brand)" : "var(--color-glass-border)",
        cursor: "pointer",
        boxShadow: value ? "0 0 8px rgba(0,229,255,0.25)" : "none",
      }}
    >
      <span
        className="absolute top-0.5 h-4 w-4 rounded-full transition-all duration-200"
        style={{
          backgroundColor: value ? "var(--color-text-inverse)" : "var(--color-text-muted)",
          left: value ? 18 : 2,
        }}
      />
    </button>
  );
}

function TextInput({
  value,
  onChange,
  onBlur,
  placeholder,
  type = "text",
  width = 220,
  mono = true,
}: {
  value: string;
  onChange: (v: string) => void;
  onBlur?: () => void;
  placeholder?: string;
  type?: string;
  width?: number;
  mono?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onBlur}
      placeholder={placeholder}
      className="rounded-md border px-3 py-1.5 text-xs transition-colors duration-200 focus:outline-none"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "var(--color-elevated)",
        color: "var(--color-text-primary)",
        fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
        width,
      }}
    />
  );
}

function StaticPill({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="rounded-md border px-3 py-1.5 text-xs"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "var(--color-elevated)",
        color: "var(--color-text-muted)",
        fontFamily: "var(--font-mono)",
      }}
    >
      {children}
    </span>
  );
}

function SavedBadge({ show }: { show: boolean }) {
  if (!show) return null;
  return (
    <span
      className="flex items-center gap-1 text-[10px] font-medium"
      style={{ color: "var(--color-accent-success)" }}
    >
      <Check size={10} strokeWidth={2.5} />
      Saved
    </span>
  );
}

/* ─────────────────────── license sub-section ─────────────────────── */

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
    } catch { /* ignore */ }
  };

  if (!license) {
    return (
      <p className="text-[11px]" style={{ color: "var(--color-text-muted)" }}>
        Checking license status…
      </p>
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
    trial: "var(--color-accent-warning)",
    pro: "var(--color-brand)",
    team: "var(--color-accent-success)",
  };
  const color = planColor[license.plan] ?? "var(--color-text-muted)";

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span
            className="rounded-md border px-3 py-1 text-[11px] font-semibold uppercase tracking-wider"
            style={{
              borderColor: color,
              backgroundColor: `color-mix(in srgb, ${color} 9%, transparent)`,
              color,
            }}
          >
            {planLabel[license.plan] || license.plan}
          </span>
          {license.trial_active && (
            <span className="text-[11px]" style={{ color: "var(--color-accent-warning)" }}>
              {license.trial_days_left} days remaining
            </span>
          )}
        </div>
        {license.licensed && (
          <button
            onClick={handleDeactivate}
            className="rounded-md border px-2.5 py-1 text-[10px] uppercase tracking-wider transition-colors duration-150 hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "var(--color-glass-border)",
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
          <div className="flex items-center gap-2">
            <TextInput
              value={inputKey}
              onChange={setInputKey}
              placeholder="XXXX-XXXX-XXXX-XXXX"
              width={240}
            />
            <button
              onClick={handleActivate}
              disabled={activating}
              className="rounded-md px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition-all duration-200"
              style={{
                backgroundColor: "var(--color-brand)",
                color: "var(--color-text-inverse)",
                cursor: activating ? "not-allowed" : "pointer",
                opacity: activating ? 0.6 : 1,
                boxShadow: "0 0 10px rgba(0,229,255,0.2)",
              }}
            >
              {activating ? "Activating…" : "Activate"}
            </button>
          </div>
          {error && (
            <p className="text-[10px]" style={{ color: "var(--color-accent-danger)" }}>
              {error}
            </p>
          )}
        </div>
      )}

      {license.machine_id && (
        <FieldRow label="Machine ID">
          <span style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)", fontSize: 11 }}>
            {license.machine_id}
          </span>
        </FieldRow>
      )}

      <p className="text-[11px]" style={{ color: "var(--color-text-muted)" }}>
        {license.plan === "free"
          ? "Free tier: 3 models (Logistic, XGBoost, RF) + fixed lot sizing"
          : license.plan === "trial"
            ? "Full feature access during trial period"
            : "All features unlocked"}
      </p>
    </div>
  );
}

/* ─────────────────────── main page ─────────────────────── */

export function SettingsPage() {
  const store = useSettingsStore();
  const { data: remoteConfig } = useConfig();
  const saveConfig = useSaveConfig();
  const storeApiKey = useStoreApiKey();
  const storeKv = useStoreKv();
  const [apiKeySaved, setApiKeySaved] = useState(false);
  const [accountIdSaved, setAccountIdSaved] = useState(false);
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

  const handleAccountIdBlur = () => {
    const acc = store.oandaAccountId;
    if (acc) {
      storeKv.mutate({ key: "oanda_account_id", value: acc }, {
        onSuccess: () => {
          setAccountIdSaved(true);
          setTimeout(() => setAccountIdSaved(false), 2000);
        },
      });
    }
  };

  const threadPct = ((store.threadBudget - 1) / 15) * 100;

  return (
    <div className="flex flex-col gap-5 max-w-4xl">

      {/* ── Row 1: General + GPU & Compute ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        <SectionCard icon={<SettingsIcon size={14} strokeWidth={1.5} />} title="General">
          <FieldRow label="Verbose Mode" hint="Outputs detailed logs during pipeline runs">
            <Toggle
              value={store.verboseMode}
              onChange={(v) => {
                store.setField("verboseMode", v);
                syncToBackend("verboseMode", v);
              }}
            />
          </FieldRow>
          <FieldRow label="API URL" hint="Backend service endpoint">
            <TextInput
              value={store.apiUrl}
              onChange={(v) => store.setField("apiUrl", v)}
              onBlur={() => syncToBackend("apiUrl", store.apiUrl)}
              width={200}
            />
          </FieldRow>
          <FieldRow label="Theme">
            <StaticPill>Dark (only)</StaticPill>
          </FieldRow>
        </SectionCard>

        <SectionCard icon={<Cpu size={14} strokeWidth={1.5} />} title="GPU & Compute">
          <FieldRow label="Thread Budget" hint="Parallel workers for training and evaluation">
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={1}
                max={16}
                value={store.threadBudget}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  store.setField("threadBudget", v);
                  syncToBackend("threadBudget", v);
                }}
                className="w-28"
                style={{
                  accentColor: "var(--color-brand)",
                  background: `linear-gradient(to right, var(--color-brand) 0%, var(--color-brand) ${threadPct}%, var(--color-glass-border) ${threadPct}%, var(--color-glass-border) 100%)`,
                }}
              />
              <span
                className="text-xs font-semibold w-5 text-right"
                style={{ color: "var(--color-brand)", fontFamily: "var(--font-mono)" }}
              >
                {store.threadBudget}
              </span>
            </div>
          </FieldRow>
          <FieldRow label="Mixed Precision (FP16)" hint="Reduces VRAM usage, may affect accuracy">
            <Toggle
              value={store.mixedPrecision}
              onChange={(v) => {
                store.setField("mixedPrecision", v);
                syncToBackend("mixedPrecision", v);
              }}
            />
          </FieldRow>
          <FieldRow label="GPU Status">
            <StaticPill>Detected at startup</StaticPill>
          </FieldRow>
        </SectionCard>
      </div>

      {/* ── Row 2: Data Sources ── */}
      <SectionCard icon={<Database size={14} strokeWidth={1.5} />} title="Data Sources">
        <FieldRow label="OANDA API Key" hint="Used for live price feeds and order routing">
          <TextInput
            value={store.oandaApiKey ?? ""}
            onChange={(v) => store.setField("oandaApiKey", v || null)}
            onBlur={handleOandaBlur}
            placeholder="Enter API key…"
            type="password"
            width={240}
          />
          <SavedBadge show={apiKeySaved} />
        </FieldRow>
        <FieldRow label="OANDA Account ID" hint="Your brokerage account identifier">
          <TextInput
            value={store.oandaAccountId ?? ""}
            onChange={(v) => store.setField("oandaAccountId", v || null)}
            onBlur={handleAccountIdBlur}
            placeholder="Enter account ID…"
            width={240}
          />
          <SavedBadge show={accountIdSaved} />
        </FieldRow>
        <FieldRow label="Data Directory" hint="Override via KODA_DATA_DIR environment variable">
          <StaticPill>Configured via env</StaticPill>
        </FieldRow>
      </SectionCard>

      {/* ── Row 3: License + Pipeline ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        <SectionCard icon={<Key size={14} strokeWidth={1.5} />} title="License">
          <LicenseSection />
        </SectionCard>

        <SectionCard icon={<SettingsIcon size={14} strokeWidth={1.5} />} title="Pipeline Configuration">
          <p
            className="text-[11px] leading-relaxed"
            style={{ color: "var(--color-text-muted)" }}
          >
            Advanced pipeline parameters are configured per-backtest on the Backtest page.
            Global defaults are defined in{" "}
            <code style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
              config.py
            </code>
            .
          </p>
          <div className="pt-1">
            <button
              onClick={() => {
                store.setField("verboseMode", false);
                store.setField("threadBudget", 4);
                store.setField("mixedPrecision", true);
                store.setField("apiUrl", "http://localhost:8000");
                syncToBackend("reset", true);
              }}
              className="rounded-md border px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.08em] transition-all duration-200 hover:border-[var(--color-border-active)]"
              style={{
                borderColor: "var(--color-glass-border)",
                backgroundColor: "var(--color-elevated)",
                color: "var(--color-text-secondary)",
                cursor: "pointer",
              }}
            >
              Reset to Defaults
            </button>
          </div>
        </SectionCard>
      </div>

      {/* ── Row 4: About ── */}
      <SectionCard icon={<Info size={14} strokeWidth={1.5} />} title="About">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Version", value: "v1.0.0-dev" },
            { label: "Pipeline", value: "Forex ML" },
            { label: "Models", value: "10 registered" },
            { label: "Build", value: "rafa9-labs" },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded-sm border p-3 flex flex-col gap-1"
              style={{
                borderColor: "var(--color-glass-border)",
                backgroundColor: "var(--color-elevated)",
              }}
            >
              <span
                className="text-[9px] uppercase tracking-[0.1em]"
                style={{ color: "var(--color-text-muted)" }}
              >
                {label}
              </span>
              <span
                className="text-[12px] font-medium"
                style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
              >
                {value}
              </span>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-1.5 pt-1">
          <ExternalLink size={11} style={{ color: "var(--color-text-muted)" }} />
          <a
            href="https://github.com/rafa9-labs/thesisproj"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] transition-colors duration-150 hover:underline"
            style={{ color: "var(--color-brand)" }}
          >
            github.com/rafa9-labs/thesisproj
          </a>
        </div>
      </SectionCard>

    </div>
  );
}
