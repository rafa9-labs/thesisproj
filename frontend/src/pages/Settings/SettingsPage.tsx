import { useState, useEffect, useRef } from "react";
import {
  Settings as SettingsIcon,
  Cpu,
  Database,
  Key,
  Info,
  ExternalLink,
  GitBranch,
} from "lucide-react";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { useConfig, useSaveConfig, useStoreApiKey, useStoreKv } from "@/api/queries";

// ─── Primitives ────────────────────────────────────────────────────────────────

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className="relative flex-shrink-0 rounded-full transition-colors"
      style={{
        width: 28,
        height: 14,
        backgroundColor: value ? "#1D4ED833" : "#1F2937",
        border: `1px solid ${value ? "#3B82F6" : "#374151"}`,
      }}
      aria-pressed={value}
    >
      <span
        className="absolute rounded-full transition-all"
        style={{
          width: 8,
          height: 8,
          top: 2,
          left: value ? 15 : 3,
          backgroundColor: value ? "#3B82F6" : "#4B5563",
        }}
      />
    </button>
  );
}

function NumericInput({
  value,
  min,
  max,
  onChange,
  width = 56,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  width?: number;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (v >= min && v <= max) onChange(v);
      }}
      className="rounded border text-right focus:outline-none"
      style={{
        width,
        height: 26,
        padding: "0 6px",
        backgroundColor: "#131722",
        borderColor: "#2A2E39",
        color: "#D1D4DC",
        fontSize: 11,
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
      }}
    />
  );
}

function TextInput({
  value,
  onChange,
  onBlur,
  placeholder,
  type = "text",
  width = 220,
}: {
  value: string;
  onChange: (v: string) => void;
  onBlur?: () => void;
  placeholder?: string;
  type?: string;
  width?: number;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onBlur}
      placeholder={placeholder}
      className="rounded border focus:outline-none"
      style={{
        width,
        height: 26,
        padding: "0 8px",
        backgroundColor: "#131722",
        borderColor: "#2A2E39",
        color: "#D1D4DC",
        fontSize: 11,
        fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
      }}
    />
  );
}

// ─── Layout primitives ─────────────────────────────────────────────────────────

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
    <div
      className="flex items-center justify-between"
      style={{ minHeight: 34, borderBottom: "1px solid #2A2E3940", padding: "0 0" }}
    >
      <div className="flex flex-col" style={{ gap: 1 }}>
        <span style={{ fontSize: 11, color: "#787B86", letterSpacing: "0.03em" }}>{label}</span>
        {hint && <span style={{ fontSize: 10, color: "#4B5563" }}>{hint}</span>}
      </div>
      <div className="flex items-center gap-2">{children}</div>
    </div>
  );
}

function SectionPanel({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="flex flex-col"
      style={{
        backgroundColor: "#1E222D",
        border: "1px solid #2A2E39",
        borderRadius: 4,
        padding: "10px 14px",
        gap: 0,
      }}
    >
      {children}
    </div>
  );
}

function SectionLabel({ label }: { label: string }) {
  return (
    <div
      className="flex items-center gap-2"
      style={{ marginBottom: 8 }}
    >
      <span
        style={{
          fontSize: 9,
          letterSpacing: "0.1em",
          color: "#4B5563",
          textTransform: "uppercase",
          fontWeight: 600,
        }}
      >
        {label}
      </span>
      <div style={{ flex: 1, height: 1, backgroundColor: "#2A2E39" }} />
    </div>
  );
}

// ─── Nav items ─────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { key: "general", label: "General", icon: <SettingsIcon size={13} strokeWidth={1.5} /> },
  { key: "gpu", label: "GPU & Compute", icon: <Cpu size={13} strokeWidth={1.5} /> },
  { key: "datasources", label: "Data Sources", icon: <Database size={13} strokeWidth={1.5} /> },
  { key: "license", label: "License", icon: <Key size={13} strokeWidth={1.5} /> },
  { key: "pipeline", label: "Pipeline Configuration", icon: <GitBranch size={13} strokeWidth={1.5} /> },
  { key: "about", label: "About", icon: <Info size={13} strokeWidth={1.5} /> },
];

// ─── License sub-section ───────────────────────────────────────────────────────

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

function LicenseContent() {
  const [license, setLicense] = useState<LicenseInfo | null>(null);
  const [inputKey, setInputKey] = useState("");
  const [activating, setActivating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
    fetch(`${apiBase}/license/status`)
      .then((r) => r.json())
      .then((d) => setLicense(d as LicenseInfo))
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
        const r2 = await fetch(`${apiBase}/license/status`);
        setLicense(await r2.json());
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

  if (!license) {
    return <span style={{ fontSize: 11, color: "#4B5563" }}>Checking license status…</span>;
  }

  const planColor: Record<string, string> = {
    free: "#4B5563",
    trial: "#F59E0B",
    pro: "#3B82F6",
    team: "#10B981",
  };

  return (
    <div className="flex flex-col gap-3">
      <FieldRow label="Plan">
        <span
          style={{
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            color: planColor[license.plan] ?? "#D1D4DC",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          {license.plan}
          {license.trial_active && ` · ${license.trial_days_left}d left`}
        </span>
      </FieldRow>
      {license.machine_id && (
        <FieldRow label="Machine ID">
          <span style={{ fontSize: 10, fontFamily: "var(--font-mono)", color: "#4B5563" }}>
            {license.machine_id}
          </span>
        </FieldRow>
      )}
      {license.needs_activation && !license.trial_active && (
        <div className="flex flex-col gap-2 pt-1">
          <div className="flex items-center gap-2">
            <TextInput
              value={inputKey}
              onChange={setInputKey}
              placeholder="XXXX-XXXX-XXXX-XXXX"
              width={200}
            />
            <button
              type="button"
              onClick={handleActivate}
              disabled={activating}
              style={{
                height: 26,
                padding: "0 12px",
                fontSize: 10,
                letterSpacing: "0.06em",
                backgroundColor: "#1D4ED818",
                border: "1px solid #3B82F655",
                color: "#60A5FA",
                borderRadius: 3,
                cursor: activating ? "not-allowed" : "pointer",
                opacity: activating ? 0.6 : 1,
                fontFamily: "inherit",
              }}
            >
              {activating ? "ACTIVATING…" : "ACTIVATE"}
            </button>
          </div>
          {error && <span style={{ fontSize: 10, color: "#F23645" }}>{error}</span>}
        </div>
      )}
      {license.licensed && (
        <button
          type="button"
          onClick={async () => {
            const apiBase = import.meta.env.VITE_API_URL ?? "/api/v1";
            await fetch(`${apiBase}/license/deactivate`, { method: "POST" }).catch(() => {});
            const r = await fetch(`${apiBase}/license/status`).catch(() => null);
            if (r) setLicense(await r.json());
          }}
          style={{
            alignSelf: "flex-start",
            height: 24,
            padding: "0 10px",
            fontSize: 10,
            letterSpacing: "0.06em",
            backgroundColor: "transparent",
            border: "1px solid #2A2E39",
            color: "#4B5563",
            borderRadius: 3,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          DEACTIVATE
        </button>
      )}
    </div>
  );
}

// ─── Content panels per section ────────────────────────────────────────────────

function GeneralContent({ store, syncToBackend }: { store: ReturnType<typeof useSettingsStore>; syncToBackend: (k: string, v: unknown) => void }) {
  return (
    <SectionPanel>
      <SectionLabel label="Application" />
      <FieldRow label="Verbose Mode (Apprentice)" hint="Show extended explanations in tooltips">
        <Toggle
          value={store.verboseMode}
          onChange={(v) => { store.setField("verboseMode", v); syncToBackend("verboseMode", v); }}
        />
      </FieldRow>
      <FieldRow label="Theme">
        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "#4B5563" }}>Dark (only)</span>
      </FieldRow>
      <FieldRow label="API URL">
        <TextInput
          value={store.apiUrl}
          onChange={(v) => store.setField("apiUrl", v)}
          onBlur={() => syncToBackend("apiUrl", store.apiUrl)}
          width={200}
        />
      </FieldRow>
    </SectionPanel>
  );
}

function GpuContent({ store, syncToBackend }: { store: ReturnType<typeof useSettingsStore>; syncToBackend: (k: string, v: unknown) => void }) {
  return (
    <SectionPanel>
      <SectionLabel label="Compute Resources" />
      <FieldRow label="Thread Budget" hint="CPU threads allocated to training workers (1–16)">
        <NumericInput
          value={store.threadBudget}
          min={1}
          max={16}
          onChange={(v) => { store.setField("threadBudget", v); syncToBackend("threadBudget", v); }}
        />
      </FieldRow>
      <FieldRow label="Mixed Precision (FP16)" hint="Use half-precision tensors on CUDA devices">
        <Toggle
          value={store.mixedPrecision}
          onChange={(v) => { store.setField("mixedPrecision", v); syncToBackend("mixedPrecision", v); }}
        />
      </FieldRow>
      <FieldRow label="GPU Status">
        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "#4B5563" }}>
          Detected at startup — see terminal
        </span>
      </FieldRow>
    </SectionPanel>
  );
}

function DataSourcesContent({
  store,
  onOandaBlur,
  onAccountBlur,
  apiKeySaved,
  accountIdSaved,
}: {
  store: ReturnType<typeof useSettingsStore>;
  onOandaBlur: () => void;
  onAccountBlur: () => void;
  apiKeySaved: boolean;
  accountIdSaved: boolean;
}) {
  return (
    <SectionPanel>
      <SectionLabel label="OANDA" />
      <FieldRow label="API Key">
        <div className="flex items-center gap-2">
          <TextInput
            value={store.oandaApiKey ?? ""}
            onChange={(v) => store.setField("oandaApiKey", v || null)}
            onBlur={onOandaBlur}
            placeholder="Enter API key…"
            type="password"
            width={220}
          />
          {apiKeySaved && (
            <span style={{ fontSize: 10, color: "#089981", fontFamily: "var(--font-mono)" }}>SAVED</span>
          )}
        </div>
      </FieldRow>
      <FieldRow label="Account ID">
        <div className="flex items-center gap-2">
          <TextInput
            value={store.oandaAccountId ?? ""}
            onChange={(v) => store.setField("oandaAccountId", v || null)}
            onBlur={onAccountBlur}
            placeholder="e.g. 001-001-12345678-001"
            width={220}
          />
          {accountIdSaved && (
            <span style={{ fontSize: 10, color: "#089981", fontFamily: "var(--font-mono)" }}>SAVED</span>
          )}
        </div>
      </FieldRow>
      <FieldRow label="Data Directory">
        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "#4B5563" }}>
          Set via DATA_DIR env var
        </span>
      </FieldRow>
    </SectionPanel>
  );
}

function PipelineContent({ store, syncToBackend }: { store: ReturnType<typeof useSettingsStore>; syncToBackend: (k: string, v: unknown) => void }) {
  return (
    <SectionPanel>
      <SectionLabel label="Pipeline Defaults" />
      <FieldRow label="Configuration">
        <span style={{ fontSize: 11, color: "#4B5563" }}>
          Advanced parameters are set per-backtest.{" "}
          <code style={{ fontFamily: "var(--font-mono)", color: "#6B7280" }}>config.py</code> holds global defaults.
        </span>
      </FieldRow>
      <div style={{ paddingTop: 8 }}>
        <button
          type="button"
          onClick={() => {
            store.setField("verboseMode", false);
            store.setField("threadBudget", 4);
            store.setField("mixedPrecision", true);
            store.setField("apiUrl", "http://localhost:8000");
            syncToBackend("reset", true);
          }}
          style={{
            height: 26,
            padding: "0 12px",
            fontSize: 10,
            letterSpacing: "0.06em",
            backgroundColor: "transparent",
            border: "1px solid #2A2E39",
            color: "#787B86",
            borderRadius: 3,
            cursor: "pointer",
            fontFamily: "inherit",
            textTransform: "uppercase",
          }}
        >
          Reset to Defaults
        </button>
      </div>
    </SectionPanel>
  );
}

function AboutContent() {
  return (
    <SectionPanel>
      <SectionLabel label="Application Info" />
      <FieldRow label="Version">
        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "#D1D4DC" }}>v1.0.0-dev</span>
      </FieldRow>
      <FieldRow label="Pipeline">
        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "#D1D4DC" }}>Forex ML Backtester</span>
      </FieldRow>
      <FieldRow label="Models Registered">
        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "#D1D4DC" }}>10</span>
      </FieldRow>
      <FieldRow label="Repository">
        <a
          href="https://github.com/rafa9-labs/thesisproj"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1"
          style={{ fontSize: 11, color: "#3B82F6", fontFamily: "var(--font-mono)" }}
        >
          rafa9-labs/thesisproj
          <ExternalLink size={10} strokeWidth={1.5} />
        </a>
      </FieldRow>
    </SectionPanel>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export function SettingsPage() {
  const store = useSettingsStore();
  const { data: remoteConfig } = useConfig();
  const saveConfig = useSaveConfig();
  const storeApiKey = useStoreApiKey();
  const storeKv = useStoreKv();
  const [activeSection, setActiveSection] = useState("general");
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
    if (store.oandaApiKey) {
      storeApiKey.mutate({ name: "oanda", value: store.oandaApiKey }, {
        onSuccess: () => { setApiKeySaved(true); setTimeout(() => setApiKeySaved(false), 2000); },
      });
    }
  };

  const handleAccountIdBlur = () => {
    if (store.oandaAccountId) {
      storeKv.mutate({ key: "oanda_account_id", value: store.oandaAccountId }, {
        onSuccess: () => { setAccountIdSaved(true); setTimeout(() => setAccountIdSaved(false), 2000); },
      });
    }
  };

  const renderContent = () => {
    switch (activeSection) {
      case "general": return <GeneralContent store={store} syncToBackend={syncToBackend} />;
      case "gpu": return <GpuContent store={store} syncToBackend={syncToBackend} />;
      case "datasources": return (
        <DataSourcesContent
          store={store}
          onOandaBlur={handleOandaBlur}
          onAccountBlur={handleAccountIdBlur}
          apiKeySaved={apiKeySaved}
          accountIdSaved={accountIdSaved}
        />
      );
      case "license": return <SectionPanel><SectionLabel label="License" /><LicenseContent /></SectionPanel>;
      case "pipeline": return <PipelineContent store={store} syncToBackend={syncToBackend} />;
      case "about": return <AboutContent />;
      default: return null;
    }
  };

  return (
    <div className="flex gap-0" style={{ minHeight: 0 }}>
      {/* Left nav sidebar */}
      <div
        style={{
          width: 200,
          flexShrink: 0,
          borderRight: "1px solid #2A2E39",
          paddingRight: 0,
        }}
      >
        {NAV_ITEMS.map((item) => {
          const active = activeSection === item.key;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => setActiveSection(item.key)}
              className="flex items-center gap-2.5 w-full transition-colors"
              style={{
                height: 36,
                padding: "0 14px",
                backgroundColor: active ? "#1E222D" : "transparent",
                borderLeft: `2px solid ${active ? "#3B82F6" : "transparent"}`,
                color: active ? "#D1D4DC" : "#787B86",
                fontSize: 11,
                cursor: "pointer",
                textAlign: "left",
                letterSpacing: "0.02em",
              }}
            >
              <span style={{ color: active ? "#3B82F6" : "#4B5563", flexShrink: 0 }}>
                {item.icon}
              </span>
              {item.label}
            </button>
          );
        })}
      </div>

      {/* Right content */}
      <div style={{ flex: 1, paddingLeft: 20, paddingTop: 2 }}>
        {renderContent()}
      </div>
    </div>
  );
}
