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
} from "lucide-react";
import { useSettingsStore } from "@/stores/useSettingsStore";
import { useConfig, useSaveConfig } from "@/api/queries";

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
      className="rounded-lg border"
      style={{ borderColor: "var(--color-border)", backgroundColor: "var(--color-surface)" }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors"
        style={{ cursor: "pointer" }}
      >
        <span style={{ color: "var(--color-text-muted)" }}>{icon}</span>
        <span className="flex-1 text-sm font-medium" style={{ color: "var(--color-text-primary)" }}>
          {title}
        </span>
        {open ? (
          <ChevronDown size={16} style={{ color: "var(--color-text-muted)" }} />
        ) : (
          <ChevronRight size={16} style={{ color: "var(--color-text-muted)" }} />
        )}
      </button>
      {open && (
        <div
          className="border-t px-4 py-4"
          style={{ borderColor: "var(--color-border)" }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 py-2">
      <span className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
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
      className="relative h-5 w-9 rounded-full transition-colors"
      style={{
        backgroundColor: value ? "var(--color-accent-success)" : "var(--color-elevated)",
        cursor: "pointer",
      }}
    >
      <span
        className="absolute top-0.5 h-4 w-4 rounded-full transition-transform"
        style={{
          backgroundColor: "var(--color-text-primary)",
          left: value ? 18 : 2,
        }}
      />
    </button>
  );
}

export function SettingsPage() {
  const store = useSettingsStore();
  const { data: remoteConfig } = useConfig();
  const saveConfig = useSaveConfig();
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

  return (
    <div className="flex flex-col gap-4">

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
            className="rounded-md border px-3 py-1.5 text-xs"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-app)",
              color: "var(--color-text-primary)",
              fontFamily: "var(--font-mono)",
              width: 240,
            }}
          />
        </FieldRow>
        <FieldRow label="Theme">
          <span
            className="rounded-md border px-3 py-1.5 text-xs"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-elevated)",
              color: "var(--color-text-primary)",
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
              style={{ accentColor: "var(--color-primary)" }}
            />
            <span
              className="text-xs"
              style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", minWidth: 24 }}
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
          <input
            type="password"
            value={store.oandaApiKey ?? ""}
            onChange={(e) => store.setField("oandaApiKey", e.target.value || null)}
            placeholder="Enter API key…"
            className="rounded-md border px-3 py-1.5 text-xs"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-app)",
              color: "var(--color-text-primary)",
              fontFamily: "var(--font-mono)",
              width: 240,
            }}
          />
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
        <div className="flex flex-col items-center gap-3 py-4">
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            License activation will be available in the commercial release.
          </span>
          <span
            className="rounded-md border px-3 py-1.5 text-xs"
            style={{
              borderColor: "var(--color-accent)",
              backgroundColor: "rgba(41,98,255,0.1)",
              color: "var(--color-accent)",
            }}
          >
            Developer Build — Unlimited
          </span>
        </div>
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
            className="mt-2 self-start rounded-md border px-3 py-1.5 text-xs font-bold uppercase"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-elevated)",
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
