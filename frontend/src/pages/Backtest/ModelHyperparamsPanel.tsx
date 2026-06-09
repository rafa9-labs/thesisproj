import { useMemo } from "react";
import { useModelHyperparams } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { ParamSlider } from "@/components/shared/ParamSlider";
import { ParamSelect } from "@/components/shared/ParamSelect";
import type { HyperparamSpec, HyperparamRange, HyperparamFixed } from "@/api/schemas";

const MODEL_COLORS: Record<string, string> = {
  classical: "var(--color-accent-classical, #22d3ee)",
  deep: "#a78bfa",
  rl: "#f59e0b",
  ensemble: "#ec4899",
};

function formatParamLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function getStoreKey(model: string, param: string): string {
  return `${model}__${param}`;
}

function HyperparamField({
  model,
  param,
  spec,
  value,
  onChange,
}: {
  model: string;
  param: string;
  spec: HyperparamSpec;
  value: number | string | undefined;
  onChange: (v: number | string) => void;
}) {
  if (spec.type === "fixed") {
    return (
      <div className="flex flex-col gap-1.5">
        <span className="text-[11px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>
          {formatParamLabel(param)}
        </span>
        <span className="text-xs font-mono" style={{ color: "var(--color-text-muted)" }}>
          {String(spec.value)} <span className="text-[9px]">(fixed)</span>
        </span>
      </div>
    );
  }

  if (spec.type === "choice") {
    const options = spec.values.map((v) => ({
      value: String(v),
      label: String(v) === "None" ? "None (unlimited)" : String(v),
    }));
    return (
      <ParamSelect
        label={formatParamLabel(param)}
        value={value != null ? String(value) : String(spec.default ?? spec.values[0])}
        options={options}
        description={`Choose ${formatParamLabel(param).toLowerCase()} for ${model}.`}
        onChange={(v) => {
          const numVal = Number(v);
          onChange(isNaN(numVal) || v === "None" ? v : numVal);
        }}
      />
    );
  }

  const range = spec as HyperparamRange;
  const numVal = typeof value === "number" ? value : (range.default ?? range.low);
  const step = range.step ?? (range.log_scale
    ? Math.max(1e-6, (range.high - range.low) / 200)
    : range.type === "int_range" ? 1 : Math.max(0.001, (range.high - range.low) / 200));

  return (
    <ParamSlider
      label={formatParamLabel(param)}
      value={numVal}
      min={range.low}
      max={range.high}
      step={step}
      description={`${formatParamLabel(param).toLowerCase()} ${range.log_scale ? "(log scale)" : ""}. Range: [${range.low}, ${range.high}]`}
      onChange={onChange}
    />
  );
}

export function ModelHyperparamsPanel() {
  const { data: hyperparams, isLoading } = useModelHyperparams();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const store = useBacktestStore.getState();

  const relevantModels = useMemo(() => {
    if (!hyperparams) return [];
    return hyperparams.filter((m) => selectedModels.includes(m.model));
  }, [hyperparams, selectedModels]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-4">
        <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Loading hyperparameters...</span>
      </div>
    );
  }

  if (relevantModels.length === 0) {
    return null;
  }

  const tunableModels = relevantModels.filter((m) => m.tunable);
  const nonTunableModels = relevantModels.filter((m) => !m.tunable);
  const setField = store.setField;

  return (
    <div className="flex flex-col gap-4">
      {tunableModels.map((m) => {
        const accent = MODEL_COLORS[m.category] ?? "var(--color-brand)";
        const tunableParams = Object.entries(m.params).filter(([, s]) => s.type !== "fixed");
        const fixedParams = Object.entries(m.params).filter(([, s]) => s.type === "fixed");

        return (
          <div
            key={m.model}
            className="rounded-sm border p-6"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "rgba(255,255,255,0.02)" }}
          >
            <div className="flex items-center gap-2 mb-1">
              <div className="h-3 w-[2px] rounded-full" style={{ backgroundColor: accent }} />
              <h4
                className="text-[11px] font-medium uppercase tracking-[0.12em]"
                style={{ color: accent }}
              >
                {m.display_name} — Hyperparameters
              </h4>
            </div>
            <p
              className="text-[11px] font-light leading-relaxed max-w-[720px] mb-5"
              style={{ color: "var(--color-text-muted)" }}
            >
              Fine-tune {m.display_name}. Overrides take priority over HPO search — set a value to fix it, or leave default to let the optimizer explore.
            </p>

            {tunableParams.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-6 mb-4">
                {tunableParams.map(([param, spec]) => {
                  const storeKey = getStoreKey(m.model, param);
                  const currentValue = (store as Record<string, unknown>)[storeKey];
                  return (
                    <HyperparamField
                      key={param}
                      model={m.model}
                      param={param}
                      spec={spec}
                      value={currentValue as number | string | undefined}
                      onChange={(v) => setField(storeKey as keyof typeof store, v as never)}
                    />
                  );
                })}
              </div>
            )}

            {fixedParams.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-4">
                {fixedParams.map(([param, spec]) => (
                  <HyperparamField
                    key={param}
                    model={m.model}
                    param={param}
                    spec={spec}
                    value={(spec as HyperparamFixed).value as number | string}
                    onChange={() => {}}
                  />
                ))}
              </div>
            )}
          </div>
        );
      })}

      {nonTunableModels.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {nonTunableModels.map((m) => (
            <span key={m.model} className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
              {m.display_name}: no tunable hyperparameters (uses built-in defaults)
            </span>
          ))}
        </div>
      )}
    </div>
  );
}