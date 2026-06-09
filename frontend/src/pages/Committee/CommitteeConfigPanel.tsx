import { useState, useEffect } from "react";
import { useCommitteeConfig, useSaveCommitteeConfig } from "@/api/queries";
import type { CommitteeConfigSchema, RegimeAssignmentSchema } from "@/api/schemas";

const AVAILABLE_MODELS = [
  "logistic", "svm", "random_forest", "decision_tree",
  "xgboost", "lightgbm", "catboost",
  "cnn", "lstm", "transformer", "gru", "gru_lstm",
  "dqn", "ensemble_adaptive_regime",
  "meta_ensemble", "stacking_ensemble",
];

export function CommitteeConfigPanel() {
  const { data: config, isLoading } = useCommitteeConfig();
  const saveMutation = useSaveCommitteeConfig();
  const [editing, setEditing] = useState<CommitteeConfigSchema | null>(null);

  useEffect(() => {
    if (config && !editing) setEditing(structuredClone(config));
  }, [config]);

  if (isLoading || !editing) {
    return (
      <div style={{ color: "var(--color-text-muted)", fontSize: 12 }}>
        Loading config...
      </div>
    );
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
    if (editing) saveMutation.mutate(editing);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span
          style={{
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "var(--color-text-primary)",
          }}
        >
          Committee Configuration
        </span>
        <button
          onClick={handleSave}
          disabled={saveMutation.isPending}
          style={{
            background: "var(--color-brand)",
            color: "var(--color-text-inverse)",
            border: "none",
            borderRadius: 4,
            padding: "6px 16px",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            cursor: "pointer",
            opacity: saveMutation.isPending ? 0.5 : 1,
          }}
        >
          {saveMutation.isPending ? "Saving..." : "Save Config"}
        </button>
      </div>

      {/* Regime Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
          gap: 16,
        }}
      >
        {regimeNames.map((regimeName) => {
          const assignment = editing.regimes[regimeName];
          return (
            <div
              key={regimeName}
              style={{
                background: "var(--color-surface)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 4,
                padding: 16,
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 12,
                }}
              >
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    color: "var(--color-brand)",
                  }}
                >
                  {regimeName.replace(/_/g, " ")}
                </span>
                <button
                  onClick={() => addModel(regimeName)}
                  style={{
                    background: "transparent",
                    border: "1px solid var(--color-glass-border)",
                    borderRadius: 4,
                    color: "var(--color-text-secondary)",
                    padding: "2px 8px",
                    fontSize: 10,
                    cursor: "pointer",
                  }}
                >
                  + Add Model
                </button>
              </div>

              {assignment.models.map((model, idx) => (
                <div
                  key={`${model}-${idx}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "6px 0",
                    borderBottom:
                      idx < assignment.models.length - 1
                        ? "1px solid var(--color-glass-border)"
                        : "none",
                  }}
                >
                  <span
                    style={{
                      width: 120,
                      fontSize: 11,
                      fontFamily: "var(--font-mono)",
                      color: "var(--color-text-primary)",
                      flexShrink: 0,
                    }}
                  >
                    {model}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(assignment.weights[idx] * 100)}
                    onChange={(e) =>
                      changeWeight(
                        regimeName,
                        idx,
                        Number(e.target.value) / 100,
                      )
                    }
                    style={{ flex: 1, height: 4, accentColor: "var(--color-brand)" }}
                  />
                  <span
                    style={{
                      width: 40,
                      fontSize: 11,
                      fontFamily: "var(--font-mono)",
                      color: "var(--color-text-secondary)",
                      textAlign: "right",
                    }}
                  >
                    {(assignment.weights[idx] * 100).toFixed(0)}%
                  </span>
                  <button
                    onClick={() => removeModel(regimeName, idx)}
                    disabled={assignment.models.length <= 1}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--color-accent-danger)",
                      fontSize: 14,
                      cursor: "pointer",
                      opacity: assignment.models.length <= 1 ? 0.3 : 1,
                      padding: "0 4px",
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}

              {assignment.models.length === 0 && (
                <div
                  style={{ color: "var(--color-text-dim)", fontSize: 11, padding: "8px 0" }}
                >
                  No models assigned. Click "+ Add Model".
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Fallback Card */}
      <div
        style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-glass-border)",
          borderRadius: 4,
          padding: 16,
        }}
      >
        <span
          style={{
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
            color: "var(--color-accent-warning)",
          }}
        >
          Fallback
        </span>
        <div style={{ marginTop: 8 }}>
          {editing.fallback.models.map((model, idx) => (
            <span
              key={`fb-${model}-${idx}`}
              style={{
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--color-text-secondary)",
              }}
            >
              {model}
              {idx < editing.fallback.models.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      </div>

      {saveMutation.isSuccess && (
        <div
          style={{
            fontSize: 11,
            color: "var(--color-accent-success)",
            letterSpacing: "0.06em",
          }}
        >
          Config saved successfully.
        </div>
      )}
    </div>
  );
}
