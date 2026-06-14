import { useState, useEffect } from "react";
import { useCommitteeConfig, useSaveCommitteeConfig } from "@/api/queries";
import type { CommitteeConfigSchema, RegimeAssignmentSchema } from "@/api/schemas";

const AVAILABLE_MODELS = [
  "logistic",
  "svm",
  "random_forest",
  "decision_tree",
  "xgboost",
  "lightgbm",
  "catboost",
  "cnn",
  "lstm",
  "transformer",
  "gru",
  "gru_lstm",
  "dqn",
  "ensemble_adaptive_regime",
  "meta_ensemble",
  "stacking_ensemble",
];

export function CommitteeConfigPanel() {
  const { data: config, isLoading } = useCommitteeConfig();
  const saveMutation = useSaveCommitteeConfig();
  const [editing, setEditing] = useState<CommitteeConfigSchema | null>(null);

  useEffect(() => {
    if (config && !editing) setEditing(structuredClone(config));
  }, [config]);

  if (isLoading || !editing) {
    return <div className="text-xs text-(--color-text-muted)">Loading config...</div>;
  }

  const regimeNames = Object.keys(editing.regimes);

  function addModel(regime: string) {
    const copy = structuredClone(editing!);
    const a = copy.regimes[regime];
    a.models.push(AVAILABLE_MODELS.filter((m) => !a.models.includes(m))[0] ?? "logistic");
    a.weights.push(0.0);
    setEditing(recomputeWeights(copy));
  }

  function removeModel(regime: string, idx: number) {
    const copy = structuredClone(editing!);
    const a = copy.regimes[regime];
    if (a.models.length <= 1) return;
    a.models.splice(idx, 1);
    a.weights.splice(idx, 1);
    setEditing(recomputeWeights(copy));
  }

  function changeWeight(regime: string, idx: number, value: number) {
    const copy = structuredClone(editing!);
    const a = copy.regimes[regime];
    a.weights[idx] = value;
    setEditing(recomputeWeights(copy));
  }

  function recomputeWeights(cfg: CommitteeConfigSchema): CommitteeConfigSchema {
    for (const r of Object.values(cfg.regimes)) {
      const w = r.weights.map((v) => Math.max(0, isNaN(v) ? 0 : v));
      const total = w.reduce((a, b) => a + b, 0) || 1;
      r.weights = w.map((v) => parseFloat((v / total).toFixed(3)));
    }
    return cfg;
  }

  function handleSave() {
    if (editing) {
      const payload = structuredClone(editing);
      if (config?.model_params) payload.model_params = config.model_params;
      saveMutation.mutate(payload);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[13px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          Committee Configuration
        </span>
        <button
          onClick={handleSave}
          disabled={saveMutation.isPending}
          className="cursor-pointer rounded border-none bg-(--color-brand) px-[16px] py-[6px] text-[11px] font-semibold tracking-[0.06em] text-(--color-text-inverse) uppercase"
          style={{ opacity: saveMutation.isPending ? 0.5 : 1 }}
        >
          {saveMutation.isPending ? "Saving..." : "Save Config"}
        </button>
      </div>

      {/* Regime Cards */}
      <div
        className="grid gap-4"
        style={{
          gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
        }}
      >
        {regimeNames.map((regimeName) => {
          const assignment = editing.regimes[regimeName];
          return (
            <div
              key={regimeName}
              className="rounded border border-(--color-glass-border) bg-(--color-surface) p-4"
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs font-semibold tracking-[0.06em] text-(--color-brand) uppercase">
                  {regimeName.replace(/_/g, " ")}
                </span>
                <button
                  onClick={() => addModel(regimeName)}
                  className="cursor-pointer rounded border border-(--color-glass-border) bg-transparent px-2 py-0.5 text-[10px] text-(--color-text-secondary)"
                >
                  + Add Model
                </button>
              </div>

              {assignment.models.map((model, idx) => (
                <div
                  key={`${model}-${idx}`}
                  className="flex items-center gap-2 py-[6px]"
                  style={{
                    borderBottom:
                      idx < assignment.models.length - 1
                        ? "1px solid var(--color-glass-border)"
                        : "none",
                  }}
                >
                  <span className="w-[120px] shrink-0 font-mono text-[11px] text-(--color-text-primary)">
                    {model}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(assignment.weights[idx] * 100)}
                    onChange={(e) => changeWeight(regimeName, idx, Number(e.target.value) / 100)}
                    className="h-1 flex-1"
                    style={{ accentColor: "var(--color-brand)" }}
                  />
                  <span className="w-10 text-right font-mono text-[11px] text-(--color-text-secondary)">
                    {(assignment.weights[idx] * 100).toFixed(0)}%
                  </span>
                  <button
                    onClick={() => removeModel(regimeName, idx)}
                    disabled={assignment.models.length <= 1}
                    className="cursor-pointer border-none bg-transparent px-1 text-sm text-(--color-accent-danger)"
                    style={{ opacity: assignment.models.length <= 1 ? 0.3 : 1 }}
                  >
                    ×
                  </button>
                </div>
              ))}

              {assignment.models.length === 0 && (
                <div className="py-2 text-[11px] text-(--color-text-dim)">
                  No models assigned. Click "+ Add Model".
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Fallback Card */}
      <div className="rounded border border-(--color-glass-border) bg-(--color-surface) p-4">
        <span className="text-xs font-semibold tracking-[0.06em] text-(--color-accent-warning) uppercase">
          Fallback
        </span>
        <div className="mt-2">
          {editing.fallback.models.map((model, idx) => (
            <span
              key={`fb-${model}-${idx}`}
              className="font-mono text-[11px] text-(--color-text-secondary)"
            >
              {model}
              {idx < editing.fallback.models.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      </div>

      {saveMutation.isSuccess && (
        <div className="text-[11px] tracking-[0.06em] text-(--color-accent-success)">
          Config saved successfully.
        </div>
      )}
    </div>
  );
}
