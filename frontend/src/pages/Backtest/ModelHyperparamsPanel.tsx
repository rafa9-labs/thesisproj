import { useMemo, useEffect } from "react";
import { useModelHyperparams } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { Section } from "@/components/shared/Panel";
import { Tier1Slider } from "@/components/shared/Tier1Slider";
import { Tier2Dropdown } from "@/components/shared/Tier2Dropdown";
import { Tier3Accordion } from "@/components/shared/Tier3Accordion";
import type { Tier3Param } from "@/components/shared/Tier3Accordion";
import type { HyperparamSpec, HyperparamRange, HyperparamChoice, HyperparamFixed, ModelHyperparams } from "@/api/schemas";

const MODEL_COLORS: Record<string, string> = {
  classical: "var(--color-accent-classical, #22d3ee)",
  deep: "#a78bfa",
  rl: "#f59e0b",
  ensemble: "#ec4899",
};

function getStoreKey(model: string, param: string): string {
  return `${model}__${param}`;
}

function initParamDefaults(
  models: ModelHyperparams[],
  store: ReturnType<typeof useBacktestStore.getState>,
) {
  let changed = false;
  const updates: Record<string, unknown> = {};

  for (const m of models) {
    for (const [paramKey, spec] of Object.entries(m.params)) {
      if (spec.tier === 3) continue;
      const storeKey = getStoreKey(m.model, paramKey);
      if ((store as Record<string, unknown>)[storeKey] !== undefined) continue;

      if (spec.type === "choice") {
        updates[storeKey] = spec.default ?? (spec as HyperparamChoice).values[0];
      } else if (spec.type !== "fixed") {
        updates[storeKey] = (spec as HyperparamRange).default ?? (spec as HyperparamRange).low;
      }
      changed = true;
    }
  }

  if (changed) {
    for (const [key, value] of Object.entries(updates)) {
      store.setField(key as keyof typeof store, value as never);
    }
  }
}

export function ModelHyperparamsPanel() {
  const { data: hyperparams, isLoading } = useModelHyperparams();
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const store = useBacktestStore.getState();

  const relevantModels = useMemo(() => {
    if (!hyperparams) return [];
    return hyperparams.filter((m) => selectedModels.includes(m.model));
  }, [hyperparams, selectedModels]);

  useEffect(() => {
    if (hyperparams && hyperparams.length > 0) {
      initParamDefaults(hyperparams, useBacktestStore.getState());
    }
  }, [hyperparams]);

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 py-4">
        <span className="text-xs text-(--color-text-muted)">Loading hyperparameters...</span>
      </div>
    );
  }

  if (relevantModels.length === 0) {
    return null;
  }

  return (
    <div className="flex min-w-0 flex-col gap-4">
      {relevantModels.map((m) => {
        const accent = MODEL_COLORS[m.category] ?? "var(--color-brand)";

        const tier1 = Object.entries(m.params).filter(([, s]) => s.tier === 1);
        const tier2 = Object.entries(m.params).filter(([, s]) => s.tier === 2);
        const tier3 = Object.entries(m.params).filter(([, s]) => s.tier === 3);

        const tier3Params: Tier3Param[] = tier3.map(([, spec]) => {
          const fixed = spec as HyperparamFixed;
          return {
            displayName: spec.display_name,
            value: String(fixed.value ?? ""),
            description: spec.description ?? "",
          };
        });

        const hasTier1 = tier1.length > 0;
        const hasTier2 = tier2.length > 0;
        const hasTier3 = tier3.length > 0;

        return (
          <Section
            key={m.model}
            title={`${m.display_name}`}
            accent={accent}
          >
            {hasTier1 && (
              <div className="mb-3 min-w-0">
                <h5 className="mb-2 text-[10px] font-bold tracking-[0.12em] text-(--color-text-muted)/70 uppercase">
                  Core Alpha Drivers
                </h5>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                  {tier1.map(([param, spec]) => (
                    <Tier1Field
                      key={param}
                      model={m.model}
                      param={param}
                      spec={spec}
                      store={store}
                    />
                  ))}
                </div>
              </div>
            )}

            {hasTier2 && (
              <div className="mb-3 min-w-0">
                <h5 className="mb-2 text-[10px] font-bold tracking-[0.12em] text-(--color-text-muted)/70 uppercase">
                  Structural Settings
                </h5>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                  {tier2.map(([param, spec]) => (
                    <Tier2Field
                      key={param}
                      model={m.model}
                      param={param}
                      spec={spec}
                      store={store}
                    />
                  ))}
                </div>
              </div>
            )}

            {hasTier3 && (
              <Tier3Accordion params={tier3Params} />
            )}

            {!hasTier1 && !hasTier2 && (
              <div className="text-[10px] text-(--color-text-muted)">
                All parameters are fixed at safe defaults. See Advanced Parameters for details.
              </div>
            )}
          </Section>
        );
      })}
    </div>
  );
}

function Tier1Field({
  model, param, spec, store,
}: {
  model: string;
  param: string;
  spec: HyperparamSpec;
  store: ReturnType<typeof useBacktestStore.getState>;
}) {
  const range = spec as HyperparamRange;
  const storeKey = getStoreKey(model, param);
  const minKey = `${storeKey}__min`;
  const maxKey = `${storeKey}__max`;
  const s = store as Record<string, unknown>;

  const currentValue = (s[storeKey] as number) ?? range.default ?? range.low;
  const hpoMin = s[minKey] as number | undefined;
  const hpoMax = s[maxKey] as number | undefined;

  const step =
    range.step ??
    (range.log_scale ? Math.max(1e-6, (range.high - range.low) / 200) : 1);

  return (
    <Tier1Slider
      label={spec.display_name || param}
      description={spec.description ?? ""}
      value={currentValue}
      min={range.low}
      max={range.high}
      step={step}
      hpoMin={hpoMin}
      hpoMax={hpoMax}
      onChange={(v) => store.setField(storeKey as keyof typeof store, v as never)}
      onHpoToggle={(enabled, minVal, maxVal) => {
        if (enabled) {
          store.setField(minKey as keyof typeof store, minVal as never);
          store.setField(maxKey as keyof typeof store, maxVal as never);
        } else {
          store.setField(minKey as keyof typeof store, undefined as never);
          store.setField(maxKey as keyof typeof store, undefined as never);
        }
      }}
    />
  );
}

function Tier2Field({
  model, param, spec, store,
}: {
  model: string;
  param: string;
  spec: HyperparamSpec;
  store: ReturnType<typeof useBacktestStore.getState>;
}) {
  const choice = spec as HyperparamChoice;
  const storeKey = getStoreKey(model, param);
  const s = store as Record<string, unknown>;

  const currentValue = String(s[storeKey] ?? choice.default ?? choice.values[0] ?? "");
  const options = choice.values.map((v) => String(v));

  return (
    <Tier2Dropdown
      label={spec.display_name || param}
      description={spec.description ?? ""}
      value={currentValue}
      options={options}
      onChange={(v) => {
        const numVal = Number(v);
        store.setField(storeKey as keyof typeof store, (isNaN(numVal) || v === "None" ? v : numVal) as never);
      }}
    />
  );
}
