import { useState, useEffect, useRef } from "react";
import {
  Settings as SettingsIcon,
  Cpu,
  Database,
  Key,
  Info,
  ExternalLink,
  Check,
  Lock,
  Unlock,
  Layers,
  AlertTriangle,
  Server,
  Search,
  Trash2,
} from "lucide-react";
import { useSettingsStore } from "@/stores/useSettingsStore";
import {
  useConfig,
  useSaveConfig,
  useStoreApiKey,
  useStoreKv,
  useHardware,
  useLicenseStatus,
  useActivateLicense,
  useDeactivateLicense,
  useExecutionSettings,
  useSaveExecutionSettings,
  useVastSettings,
  useSaveVastSettings,
  useStoreVastApiKey,
  useVastOffers,
  useLaunchVastInstance,
  useVastInstances,
  useDestroyVastInstance,
} from "@/api/queries";
import { GpuStatusCard } from "@/components/GpuStatusCard";
import { DataManager } from "@/components/DataManager";

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
    <div className="flex flex-col rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-6">
      <div className="mb-6 flex items-center gap-2.5 border-b border-(--color-glass-border) pb-4">
        <span className="flex text-(--color-brand)">{icon}</span>
        <h3 className="text-[11px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
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
    <div className="flex flex-col justify-between gap-2 border-t border-(--color-glass-border) py-3 sm:flex-row sm:items-center sm:gap-4">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-[12px] font-medium text-(--color-text-primary)">{label}</span>
        {hint && <span className="text-[10px] text-(--color-text-muted)">{hint}</span>}
      </div>
      <div className="flex shrink-0 items-center gap-2">{children}</div>
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!value)}
      aria-checked={value}
      role="switch"
      className="relative h-5 w-9 flex-shrink-0 cursor-pointer rounded-full transition-all duration-200"
      style={{
        backgroundColor: value ? "var(--color-brand)" : "var(--color-glass-border)",
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
  mono = true,
}: {
  value: string;
  onChange: (v: string) => void;
  onBlur?: () => void;
  placeholder?: string;
  type?: string;
  mono?: boolean;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onBlur={onBlur}
      placeholder={placeholder}
      className="w-full rounded-md border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 text-xs text-(--color-text-primary) transition-colors duration-200 focus:outline-none sm:w-64"
      style={{ fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)" }}
    />
  );
}

function StaticPill({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-md border border-(--color-glass-border) bg-(--color-glass-hover) px-3 py-1.5 font-mono text-xs text-(--color-text-secondary)">
      {children}
    </span>
  );
}

function SavedBadge({ show }: { show: boolean }) {
  if (!show) return null;
  return (
    <span className="flex items-center gap-1 text-[10px] font-medium text-(--color-accent-success)">
      <Check size={10} strokeWidth={2.5} />
      Saved
    </span>
  );
}

const VAST_GPU_CLASSES = [
  "H100",
  "A100",
  "RTX 6000 ADA",
  "L40S",
  "RTX 4090",
  "A6000",
  "RTX 4080",
  "RTX 3090 Ti",
  "RTX 3090",
  "A5000",
  "RTX 4070",
  "RTX 3080",
  "RTX 3070",
];

function VastSection() {
  const { data: vastSettings } = useVastSettings();
  const saveVast = useSaveVastSettings();
  const storeKey = useStoreVastApiKey();
  const launch = useLaunchVastInstance();
  const destroy = useDestroyVastInstance();
  const { data: instances } = useVastInstances();

  const [apiKey, setApiKey] = useState("");
  const [keySaved, setKeySaved] = useState(false);
  const [vastSaved, setVastSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offerFilters, setOfferFilters] = useState<{
    gpu_class?: string;
    min_vram_gb?: number;
    max_dph?: number;
  } | null>(null);
  const offersQuery = useVastOffers(offerFilters);

  const [form, setForm] = useState({
    vast_enabled: false,
    vast_min_gpu_class: "RTX 3090",
    vast_min_vram_gb: 16,
    vast_max_dph: 0.5,
    vast_disk_gb: 60,
  });
  const [formReady, setFormReady] = useState(false);

  useEffect(() => {
    if (vastSettings && !formReady) {
      setForm({
        vast_enabled: vastSettings.vast_enabled,
        vast_min_gpu_class: vastSettings.vast_min_gpu_class,
        vast_min_vram_gb: vastSettings.vast_min_vram_gb,
        vast_max_dph: vastSettings.vast_max_dph,
        vast_disk_gb: vastSettings.vast_disk_gb,
      });
      setFormReady(true);
    }
  }, [vastSettings, formReady]);

  const set = (patch: Partial<typeof form>) => setForm((f) => ({ ...f, ...patch }));

  const handleSaveKey = () => {
    if (!apiKey.trim()) return;
    storeKey.mutate(apiKey.trim(), {
      onSuccess: () => {
        setKeySaved(true);
        setApiKey("");
        setError(null);
        setTimeout(() => setKeySaved(false), 3000);
      },
      onError: (e) => setError(e.message),
    });
  };

  const handleSaveVast = () => {
    setError(null);
    saveVast.mutate(form, {
      onSuccess: () => {
        setVastSaved(true);
        setTimeout(() => setVastSaved(false), 3000);
      },
      onError: (e) => setError(e.message),
    });
  };

  const handleSearchOffers = () => {
    setError(null);
    setOfferFilters({
      gpu_class: form.vast_min_gpu_class,
      min_vram_gb: form.vast_min_vram_gb,
      max_dph: form.vast_max_dph,
    });
  };

  const handleRent = (askId?: number) => {
    setError(null);
    launch.mutate(
      askId != null
        ? { ask_id: askId, disk_gb: form.vast_disk_gb }
        : { gpu_class: form.vast_min_gpu_class, disk_gb: form.vast_disk_gb },
      {
        onSuccess: () => setOfferFilters(null),
        onError: (e) => setError(e.message),
      },
    );
  };

  return (
    <SectionCard icon={<Server size={14} strokeWidth={1.5} />} title="GPU Rental (Vast.ai)">
      {!vastSettings?.has_api_key && (
        <div className="mb-4 rounded-md border border-(--color-accent-warning) bg-(color-mix(in srgb, var(--color-accent-warning) 8%, transparent)) px-3 py-2">
          <p className="text-[10px] leading-relaxed text-(--color-accent-warning)">
            No Vast.ai API key configured. Add your key to search offers and rent GPUs.
          </p>
        </div>
      )}

      <FieldRow label="Vast.ai API Key" hint="Stored encrypted; used to search offers and rent instances">
        <div className="flex items-center gap-3">
          <TextInput
            value={apiKey}
            onChange={setApiKey}
            placeholder="Enter Vast.ai API key…"
            type="password"
          />
          <button
            onClick={handleSaveKey}
            disabled={storeKey.isPending || !apiKey.trim()}
            className="rounded-md border border-(--color-brand) bg-(--color-brand-glow) px-4 py-1.5 text-[10px] font-semibold tracking-[0.08em] text-(--color-brand) uppercase transition hover:brightness-110 disabled:opacity-40"
          >
            {storeKey.isPending ? "Saving…" : "Save Key"}
          </button>
          <SavedBadge show={keySaved} />
        </div>
      </FieldRow>

      <FieldRow label="Enable Rented GPU" hint="Offload GPU model training to a rented Vast.ai instance">
        <Toggle value={form.vast_enabled} onChange={(v) => set({ vast_enabled: v })} />
      </FieldRow>

      <FieldRow label="Minimum GPU Class" hint="Offers below this class are filtered out">
        <select
          value={form.vast_min_gpu_class}
          onChange={(e) => set({ vast_min_gpu_class: e.target.value })}
          className="w-full rounded-md border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 text-xs text-(--color-text-primary) sm:w-64"
        >
          {VAST_GPU_CLASSES.map((g) => (
            <option key={g} value={g}>
              {g} or better
            </option>
          ))}
        </select>
      </FieldRow>

      <FieldRow label="Minimum VRAM (GB)">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={8}
            max={48}
            step={4}
            value={form.vast_min_vram_gb}
            onChange={(e) => set({ vast_min_vram_gb: Number(e.target.value) })}
            className="w-28"
            style={{ accentColor: "var(--color-brand)" }}
          />
          <span className="w-10 text-right font-mono text-xs font-semibold text-(--color-brand)">
            {form.vast_min_vram_gb} GB
          </span>
        </div>
      </FieldRow>

      <FieldRow label="Max Hourly Price ($)">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0.05}
            max={2}
            step={0.05}
            value={form.vast_max_dph}
            onChange={(e) => set({ vast_max_dph: Number(e.target.value) })}
            className="w-28"
            style={{ accentColor: "var(--color-brand)" }}
          />
          <span className="w-10 text-right font-mono text-xs font-semibold text-(--color-brand)">
            ${form.vast_max_dph.toFixed(2)}
          </span>
        </div>
      </FieldRow>

      <FieldRow label="Instance Disk (GB)" hint="Disk size for the rented instance">
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={20}
            max={200}
            step={10}
            value={form.vast_disk_gb}
            onChange={(e) => set({ vast_disk_gb: Number(e.target.value) })}
            className="w-28"
            style={{ accentColor: "var(--color-brand)" }}
          />
          <span className="w-10 text-right font-mono text-xs font-semibold text-(--color-brand)">
            {form.vast_disk_gb} GB
          </span>
        </div>
      </FieldRow>

      <div className="mt-4 flex items-center justify-between border-t border-(--color-glass-border) pt-4">
        {error && (
          <span className="text-[10px] text-(--color-accent-danger)">{error}</span>
        )}
        <div className="ml-auto flex items-center gap-3">
          {vastSaved && (
            <span className="flex items-center gap-1 text-[10px] font-medium text-(--color-brand)">
              <Check size={11} />
              Saved
            </span>
          )}
          <button
            onClick={handleSaveVast}
            disabled={saveVast.isPending}
            className="rounded-md border border-(--color-brand) bg-(--color-brand-glow) px-4 py-1.5 text-[10px] font-semibold tracking-[0.08em] text-(--color-brand) uppercase transition hover:brightness-110"
          >
            {saveVast.isPending ? "Saving…" : "Save Rental Settings"}
          </button>
        </div>
      </div>

      <div className="mt-6 border-t border-(--color-glass-border) pt-5">
        <div className="flex items-center justify-between">
          <h4 className="text-[10px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
            Available Offers
          </h4>
          <button
            onClick={handleSearchOffers}
            disabled={offersQuery.isFetching || !vastSettings?.has_api_key}
            className="flex items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 text-[10px] font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase transition hover:border-(--color-brand)"
          >
            <Search size={11} />
            {offersQuery.isFetching ? "Searching…" : "Search Offers"}
          </button>
        </div>

        {offersQuery.data && offersQuery.data.length === 0 && (
          <p className="pt-3 text-[10px] text-(--color-text-muted)">
            No offers match the current filters.
          </p>
        )}

        {offersQuery.data && offersQuery.data.length > 0 && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-[10px]">
              <thead>
                <tr className="text-[9px] tracking-[0.1em] text-(--color-text-muted) uppercase">
                  <th className="pb-2 pr-3 font-medium">GPU</th>
                  <th className="pb-2 pr-3 font-medium">VRAM</th>
                  <th className="pb-2 pr-3 font-medium">$/hr</th>
                  <th className="pb-2 pr-3 font-medium">Reliability</th>
                  <th className="pb-2 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {offersQuery.data.slice(0, 6).map((o) => (
                  <tr key={o.ask_id} className="border-t border-(--color-glass-border)">
                    <td className="py-2 pr-3 font-mono text-(--color-text-primary)">
                      {o.gpu_name}
                    </td>
                    <td className="py-2 pr-3 font-mono text-(--color-text-secondary)">
                      {o.gpu_ram_gb} GB
                    </td>
                    <td className="py-2 pr-3 font-mono text-(--color-text-secondary)">
                      ${o.dph_total.toFixed(2)}
                    </td>
                    <td className="py-2 pr-3 font-mono text-(--color-text-secondary)">
                      {o.reliability != null ? `${(o.reliability * 100).toFixed(0)}%` : "—"}
                    </td>
                    <td className="py-2 text-right">
                      <button
                        onClick={() => handleRent(o.ask_id)}
                        disabled={launch.isPending}
                        className="rounded-md border border-(--color-brand) bg-(--color-brand-glow) px-3 py-1 text-[9px] font-semibold tracking-[0.08em] text-(--color-brand) uppercase transition hover:brightness-110"
                      >
                        Rent
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button
              onClick={() => handleRent()}
              disabled={launch.isPending}
              className="mt-3 rounded-md border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 text-[10px] font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase transition hover:border-(--color-brand)"
            >
              {launch.isPending ? "Launching…" : "Rent Cheapest Match"}
            </button>
          </div>
        )}
      </div>

      <div className="mt-6 border-t border-(--color-glass-border) pt-5">
        <h4 className="text-[10px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
          Active Instances
        </h4>
        {(!instances || instances.length === 0) && (
          <p className="pt-3 text-[10px] text-(--color-text-muted)">
            No active Vast.ai instances.
          </p>
        )}
        {instances && instances.length > 0 && (
          <div className="mt-3 space-y-2">
            {instances.map((inst) => (
              <div
                key={inst.id}
                className="flex items-center justify-between rounded-md border border-(--color-glass-border) bg-(--color-glass-hover) px-3 py-2"
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-[11px] text-(--color-text-primary)">
                    #{inst.id}
                  </span>
                  <span className="font-mono text-[10px] text-(--color-text-secondary)">
                    {inst.gpu_name}
                  </span>
                  <span className="rounded-full border border-(--color-brand) bg-(--color-brand-glow) px-2 py-0.5 text-[9px] font-medium tracking-[0.08em] text-(--color-brand) uppercase">
                    {inst.actual_status}
                  </span>
                  {inst.remote_api_url && (
                    <a
                      href={inst.remote_api_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-[10px] text-(--color-brand) hover:underline"
                    >
                      <ExternalLink size={10} />
                      Remote API
                    </a>
                  )}
                </div>
                <button
                  onClick={() => {
                    setError(null);
                    destroy.mutate(inst.id, {
                      onError: (e) => setError(e.message),
                    });
                  }}
                  disabled={destroy.isPending}
                  className="flex items-center gap-1 rounded-md border border-(--color-accent-danger) px-3 py-1 text-[9px] font-semibold tracking-[0.08em] text-(--color-accent-danger) uppercase transition hover:brightness-110"
                >
                  <Trash2 size={10} />
                  Destroy
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </SectionCard>
  );
}

/* ─────────────────────── license sub-section ─────────────────────── */

function LicenseSection() {
  const { data: license, isLoading } = useLicenseStatus();
  const activateLicense = useActivateLicense();
  const deactivateLicense = useDeactivateLicense();
  const [inputKey, setInputKey] = useState("");
  const [error, setError] = useState("");

  const handleActivate = () => {
    if (!inputKey.trim()) return;
    setError("");
    activateLicense.mutate(inputKey.trim(), {
      onSuccess: () => {
        setInputKey("");
      },
      onError: (err: Error) => {
        setError(err.message || "Activation failed");
      },
    });
  };

  const handleDeactivate = () => {
    deactivateLicense.mutate();
  };

  if (isLoading) {
    return <p className="text-[11px] text-(--color-text-muted)">Checking license status…</p>;
  }

  if (!license) {
    return <p className="text-[11px] text-(--color-accent-danger)">Could not load license status</p>;
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
            className="rounded-md border px-3 py-1 text-[11px] font-semibold tracking-wider uppercase"
            style={{
              borderColor: color,
              backgroundColor: `color-mix(in srgb, ${color} 9%, transparent)`,
              color,
            }}
          >
            {planLabel[license.plan] || license.plan}
          </span>
          {license.trial_active && (
            <span className="text-[11px] text-(--color-accent-warning)">
              {license.trial_days_left} days remaining
            </span>
          )}
        </div>
        {license.licensed && (
          <button
            onClick={handleDeactivate}
            disabled={deactivateLicense.isPending}
            className="cursor-pointer rounded-md border border-(--color-glass-border) px-2.5 py-1 text-[10px] tracking-wider text-(--color-text-muted) uppercase transition-colors duration-150 hover:border-[var(--color-border-active)] disabled:opacity-50"
          >
            {deactivateLicense.isPending ? "Deactivating…" : "Deactivate"}
          </button>
        )}
      </div>

      {license.needs_activation && !license.trial_active && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <TextInput value={inputKey} onChange={setInputKey} placeholder="XXXX-XXXX-XXXX-XXXX" />
            <button
              onClick={handleActivate}
              disabled={activateLicense.isPending}
              className="flex shrink-0 items-center rounded-md bg-(--color-brand) px-3 py-1.5 text-xs font-semibold tracking-wider text-(--color-text-inverse) uppercase shadow-[0_0_10px_rgba(0,229,255,0.2)] transition-all duration-200"
              style={{
                cursor: activateLicense.isPending ? "not-allowed" : "pointer",
                opacity: activateLicense.isPending ? 0.6 : 1,
              }}
            >
              {activateLicense.isPending ? "Activating…" : "Activate"}
            </button>
          </div>
          {error && <p className="text-[10px] text-(--color-accent-danger)">{error}</p>}
        </div>
      )}

      {license.machine_id && (
        <FieldRow label="Machine ID">
          <span className="font-mono text-[11px] text-(--color-text-muted)">
            {license.machine_id}
          </span>
        </FieldRow>
      )}

      <p className="text-[11px] text-(--color-text-muted)">
        {license.plan === "free"
          ? "Free tier: 3 models (Logistic, XGBoost, RF) + fixed lot sizing"
          : license.plan === "trial"
            ? "Full feature access during trial period"
            : "All features unlocked"}
      </p>

      {/* Model availability */}
      {license.available_models.length > 0 && (
        <div className="border-t border-(--color-glass-border) pt-3">
          <span className="text-[10px] text-(--color-text-muted) uppercase tracking-wider">
            Available Models ({license.available_models.length})
          </span>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {license.available_models.map((m) => (
              <span
                key={m}
                className="flex items-center gap-1 rounded bg-(color-mix(in srgb, var(--color-accent-success) 10%, transparent)) border border-(--color-accent-success) px-1.5 py-0.5 text-[9px] font-mono text-(--color-accent-success)"
              >
                <Unlock size={8} />
                {m}
              </span>
            ))}
          </div>
        </div>
      )}

      {license.locked_models.length > 0 && (
        <div className="border-t border-(--color-glass-border) pt-3">
          <span className="text-[10px] text-(--color-text-muted) uppercase tracking-wider">
            Locked Models ({license.locked_models.length})
          </span>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {license.locked_models.map((m) => (
              <span
                key={m}
                className="flex items-center gap-1 rounded bg-(--color-glass-hover) border border-(--color-glass-border) px-1.5 py-0.5 text-[9px] font-mono text-(--color-text-muted) line-through"
              >
                <Lock size={8} />
                {m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─────────────────────── main page ─────────────────────── */

export function SettingsPage() {
  const store = useSettingsStore();
  const { data: remoteConfig } = useConfig();
  const { data: hw } = useHardware();
  const { data: remoteExec } = useExecutionSettings();
  const saveConfig = useSaveConfig();
  const saveExec = useSaveExecutionSettings();
  const storeApiKey = useStoreApiKey();
  const storeKv = useStoreKv();
  const [apiKeySaved, setApiKeySaved] = useState(false);
  const [accountIdSaved, setAccountIdSaved] = useState(false);
  const [execSaved, setExecSaved] = useState(false);
  const synced = useRef(false);
  const execSynced = useRef(false);

  useEffect(() => {
    if (remoteConfig && !synced.current) {
      synced.current = true;
      if (remoteConfig.threadBudget != null)
        store.setField("threadBudget", remoteConfig.threadBudget as number);
      if (remoteConfig.mixedPrecision != null)
        store.setField("mixedPrecision", remoteConfig.mixedPrecision as boolean);
      if (remoteConfig.verboseMode != null)
        store.setField("verboseMode", remoteConfig.verboseMode as boolean);
      if (remoteConfig.apiUrl != null) store.setField("apiUrl", remoteConfig.apiUrl as string);
      if (remoteConfig.ramLimit != null) store.setField("ramLimit", remoteConfig.ramLimit as number);
    }
  }, [remoteConfig]);

  useEffect(() => {
    if (remoteExec && !execSynced.current) {
      execSynced.current = true;
      if (remoteExec.max_concurrent_backtests != null)
        store.setField("maxConcurrentBacktests", remoteExec.max_concurrent_backtests);
      if (remoteExec.gpu_enabled != null)
        store.setField("gpuEnabled", remoteExec.gpu_enabled);
      if (remoteExec.max_concurrent_gpu != null)
        store.setField("maxConcurrentGpu", remoteExec.max_concurrent_gpu);
    }
  }, [remoteExec]);

  const syncToBackend = (key: string, value: unknown) => {
    saveConfig.mutate({ [key]: value, ...store });
  };

  const handleSaveExec = () => {
    saveExec.mutate(
      {
        max_concurrent_backtests: store.maxConcurrentBacktests,
        gpu_enabled: store.gpuEnabled,
        max_concurrent_gpu: store.maxConcurrentGpu,
      },
      {
        onSuccess: () => {
          setExecSaved(true);
          setTimeout(() => setExecSaved(false), 3000);
        },
      },
    );
  };

  const handleOandaBlur = () => {
    const key = store.oandaApiKey;
    if (key) {
      storeApiKey.mutate(
        { name: "oanda", value: key },
        {
          onSuccess: () => {
            setApiKeySaved(true);
            setTimeout(() => setApiKeySaved(false), 2000);
          },
        },
      );
    }
  };

  const handleAccountIdBlur = () => {
    const acc = store.oandaAccountId;
    if (acc) {
      storeKv.mutate(
        { key: "oanda_account_id", value: acc },
        {
          onSuccess: () => {
            setAccountIdSaved(true);
            setTimeout(() => setAccountIdSaved(false), 2000);
          },
        },
      );
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
      {/* ── Row 1: General + GPU & Compute ── */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <SectionCard icon={<SettingsIcon size={14} strokeWidth={1.5} />} title="General">
          {import.meta.env.DEV && (
            <FieldRow label="Verbose Mode" hint="Outputs detailed logs during pipeline runs">
              <Toggle
                value={store.verboseMode}
                onChange={(v) => {
                  store.setField("verboseMode", v);
                  syncToBackend("verboseMode", v);
                }}
              />
            </FieldRow>
          )}
          {import.meta.env.DEV && (
            <FieldRow label="API URL" hint="Backend service endpoint">
              <TextInput
                value={store.apiUrl}
                onChange={(v) => store.setField("apiUrl", v)}
                onBlur={() => syncToBackend("apiUrl", store.apiUrl)}
              />
            </FieldRow>
          )}
          <FieldRow label="Theme">
            <StaticPill>Dark (only)</StaticPill>
          </FieldRow>
        </SectionCard>

        <SectionCard icon={<Cpu size={14} strokeWidth={1.5} />} title="GPU & Compute">
          <FieldRow
            label="Thread Budget"
            hint={hw ? `Recommended: ${hw.budget.blas_threads} for your ${hw.cpu.physical_cores}-core CPU` : "Parallel workers for training and evaluation"}
          >
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
                }}
              />
              <span className="w-5 text-right font-mono text-xs font-semibold text-(--color-brand)">
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
            <GpuStatusCard />
          </FieldRow>
          <FieldRow
            label="RAM Limit"
            hint={hw ? `System RAM: ${hw.cpu.ram_total_gb} GB (recommended: ${hw.budget.ram_limit_gb} GB)` : "Maximum RAM usage for pipeline operations"}
          >
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={2}
                max={128}
                step={1}
                value={store.ramLimit}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  store.setField("ramLimit", v);
                  syncToBackend("ramLimit", v);
                }}
                className="w-28"
                style={{
                  accentColor: "var(--color-brand)",
                }}
              />
              <span className="w-10 text-right font-mono text-xs font-semibold text-(--color-brand)">
                {store.ramLimit} GB
              </span>
            </div>
          </FieldRow>
        </SectionCard>
      </div>

      {/* ── Row 1.5: Execution Engine ── */}
      <SectionCard icon={<Layers size={14} strokeWidth={1.5} />} title="Execution Engine">
        <FieldRow
          label="Max CPU Backtests"
          hint="Maximum number of simultaneous CPU-only backtests (1-8)"
        >
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={1}
              max={8}
              value={store.maxConcurrentBacktests}
              onChange={(e) => store.setField("maxConcurrentBacktests", Number(e.target.value))}
              className="w-28"
              style={{ accentColor: "var(--color-brand)" }}
            />
            <span className="w-5 text-right font-mono text-xs font-semibold text-(--color-brand)">
              {store.maxConcurrentBacktests}
            </span>
          </div>
        </FieldRow>

        <FieldRow
          label="Enable GPU Models"
          hint="Run LSTM, CNN, and Transformer on GPU if available"
        >
          <Toggle
            value={store.gpuEnabled}
            onChange={(v) => store.setField("gpuEnabled", v)}
          />
        </FieldRow>

        <FieldRow
          label="Max GPU Backtests"
          hint={store.gpuEnabled ? "Maximum GPU models running simultaneously (1-2)" : "GPU models disabled — turn on Enable GPU Models above"}
        >
          <div className={`flex items-center gap-3 ${!store.gpuEnabled ? "opacity-40 pointer-events-none" : ""}`}>
            <input
              type="range"
              min={1}
              max={2}
              value={store.maxConcurrentGpu}
              onChange={(e) => store.setField("maxConcurrentGpu", Number(e.target.value))}
              disabled={!store.gpuEnabled}
              className="w-28"
              style={{ accentColor: "var(--color-brand)" }}
            />
            <span className={`w-5 text-right font-mono text-xs font-semibold ${store.gpuEnabled ? "text-(--color-brand)" : "text-(--color-text-dim)"}`}>
              {store.gpuEnabled ? store.maxConcurrentGpu : "—"}
            </span>
          </div>
        </FieldRow>

        <div className="mt-4 flex items-center justify-between border-t border-(--color-glass-border) pt-4">
          <div className="flex items-center gap-2">
            <AlertTriangle size={12} className="text-(--color-accent-warning)" />
            <span className="text-[10px] text-(--color-accent-warning)">
              Changes will take effect upon application restart
            </span>
          </div>
          <div className="flex items-center gap-3">
            {execSaved && (
              <span className="flex items-center gap-1 text-[10px] font-medium text-(--color-brand)">
                <Check size={11} />
                Saved
              </span>
            )}
            <button
              onClick={handleSaveExec}
              disabled={saveExec.isPending}
              className="rounded-md border border-(--color-brand) bg-(--color-brand-glow) px-4 py-1.5 text-[10px] font-semibold tracking-[0.08em] text-(--color-brand) uppercase transition hover:brightness-110"
            >
              {saveExec.isPending ? "Saving..." : "Save Execution Settings"}
            </button>
          </div>
        </div>
      </SectionCard>

      {/* ── Row 2: Data Sources ── */}
      <SectionCard icon={<Database size={14} strokeWidth={1.5} />} title="Data Sources">
        <div className="rounded-md border border-(--color-accent-warning) bg-(color-mix(in srgb, var(--color-accent-warning) 8%, transparent)) px-3 py-2 mb-2">
          <p className="text-[10px] leading-relaxed text-(--color-accent-warning)">
            Live data requires your own OANDA API key and compliance with{" "}
            <a
              href="https://legal.oanda.com"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-(--color-brand)"
            >
              OANDA's API License Agreement
            </a>
            . You must not redistribute OANDA data to third parties.
          </p>
        </div>
        <FieldRow label="OANDA API Key" hint="Used for live price feeds and order routing">
          <TextInput
            value={store.oandaApiKey ?? ""}
            onChange={(v) => store.setField("oandaApiKey", v || null)}
            onBlur={handleOandaBlur}
            placeholder="Enter API key…"
            type="password"
          />
          <SavedBadge show={apiKeySaved} />
        </FieldRow>
        <FieldRow label="OANDA Account ID" hint="Your brokerage account identifier">
          <TextInput
            value={store.oandaAccountId ?? ""}
            onChange={(v) => store.setField("oandaAccountId", v || null)}
            onBlur={handleAccountIdBlur}
            placeholder="Enter account ID…"
          />
          <SavedBadge show={accountIdSaved} />
        </FieldRow>
        <DataManager />
      </SectionCard>

      {/* ── Row 3: License + Pipeline ── */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <SectionCard icon={<Key size={14} strokeWidth={1.5} />} title="License">
          <LicenseSection />
        </SectionCard>

        <SectionCard
          icon={<SettingsIcon size={14} strokeWidth={1.5} />}
          title="Pipeline Configuration"
        >
          <p className="text-[11px] leading-relaxed text-(--color-text-muted)">
            Advanced pipeline parameters are configured per-backtest on the Backtest page. Global
            defaults are defined in{" "}
            <code className="font-mono text-(--color-text-secondary)">config.py</code>.
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
              className="cursor-pointer rounded-md border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 text-[11px] font-medium tracking-[0.08em] text-(--color-text-secondary) uppercase transition-all duration-200 hover:border-[var(--color-border-active)]"
            >
              Reset to Defaults
            </button>
          </div>
        </SectionCard>
      </div>

      {/* ── Row 4: GPU Rental ── */}
      <VastSection />

      {/* ── Row 5: About ── */}
      <SectionCard icon={<Info size={14} strokeWidth={1.5} />} title="About">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {[
            { label: "Version", value: import.meta.env.DEV ? "v1.0.0-dev" : "v1.0.0" },
            { label: "Pipeline", value: "Forex ML" },
            { label: "Models", value: "10 registered" },
            { label: "Build", value: "rafa9-labs" },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="flex flex-col gap-1 rounded-lg border border-(--color-glass-border) bg-(--color-glass-hover) p-3"
            >
              <span className="text-[9px] tracking-[0.1em] text-(--color-text-muted) uppercase">
                {label}
              </span>
              <span className="font-mono text-[12px] font-medium text-(--color-text-primary)">
                {value}
              </span>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-1.5 pt-4">
          <ExternalLink size={11} className="text-(--color-text-muted)" />
          <a
            href="https://github.com/rafa9-labs/thesisproj"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-(--color-brand) transition-colors duration-150 hover:underline"
          >
            github.com/rafa9-labs/thesisproj
          </a>
        </div>
      </SectionCard>
    </div>
  );
}
