/* eslint-disable react-refresh/only-export-components */
import { useState, useMemo } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { modelDescriptions } from "@/lib/tokens";
import { ChevronDown, ChevronRight, AlertTriangle, Lightbulb } from "lucide-react";

export function ParameterGuide() {
  const [open, setOpen] = useState(false);
  const s = useBacktestStore();
  const selectedModels = s.selectedModels;

  const guides = useMemo(() => buildGuides(selectedModels, s), [selectedModels, s.lags]);

  if (selectedModels.length === 0) return null;

  return (
    <div
      className="rounded-lg border overflow-hidden"
      style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full px-3 py-2 text-left"
        style={{ cursor: "pointer", color: "var(--color-text-secondary)" }}
      >
        <div className="flex items-center gap-1.5">
          <Lightbulb size={12} style={{ color: "var(--color-accent-warning)" }} />
          <span className="text-[10px] font-semibold uppercase tracking-[0.06em]">
            Parameter Guide
          </span>
        </div>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>

      {open && <GuideContent guides={guides} />}
    </div>
  );
}

/** Always-visible inline version for the Hyperparameters tab. */
export function ParameterGuideInline() {
  const s = useBacktestStore();
  const selectedModels = s.selectedModels;
  const guides = useMemo(() => buildGuides(selectedModels, s), [selectedModels, s.lags]);

  if (guides.length === 0) return null;

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2">
        <Lightbulb size={12} style={{ color: "var(--color-accent-warning)" }} />
        <span className="text-[10px] font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-secondary)" }}>
          Parameter Tips
        </span>
      </div>
      <GuideContent guides={guides} />
    </div>
  );
}

function buildGuides(selectedModels: string[], s: Record<string, unknown>) {
  return selectedModels
    .map((modelKey) => {
      const name = (modelDescriptions as Record<string, { name: string }>)[modelKey]?.name ?? modelKey;
      return { modelKey, name, ...getDynamicWarnings(modelKey, s) };
    })
    .filter((g) => g.warnings.length > 0 || g.tips.length > 0);
}

function GuideContent({ guides }: { guides: { modelKey: string; name: string; warnings: string[]; tips: string[] }[] }) {
  return (
    <div className="flex flex-col gap-2 px-3 pb-3 pt-1">
      {guides.map((g) => (
        <div
          key={g.modelKey}
          className="rounded p-2"
          style={{ backgroundColor: "var(--color-elevated)" }}
        >
          <span className="text-[10px] font-semibold uppercase tracking-[0.05em]" style={{ color: "var(--color-brand)" }}>
            {g.name}
          </span>
          <div className="mt-1.5 flex flex-col gap-1.5">
            {g.warnings.map((w, i) => (
              <div key={`w-${i}`} className="flex items-start gap-1.5">
                <AlertTriangle size={10} style={{ color: "var(--color-accent-warning)", marginTop: 1, flexShrink: 0 }} />
                <span className="text-[9px] leading-relaxed" style={{ color: "var(--color-accent-warning)" }}>
                  {w}
                </span>
              </div>
            ))}
            {g.tips.map((t, i) => (
              <div key={`t-${i}`} className="flex items-start gap-1.5">
                <Lightbulb size={10} style={{ color: "var(--color-text-muted)", marginTop: 1, flexShrink: 0 }} />
                <span className="text-[9px] leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
                  {t}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
      <span className="text-[8px]" style={{ color: "var(--color-text-muted)" }}>
        Based on your current settings and known structural constraints.
      </span>
    </div>
  );
}

export function getDynamicWarnings(modelKey: string, s: Record<string, unknown>) {
  const warnings: string[] = [];
  const tips: string[] = [];
  const lags = (s.lags as number) ?? 14;

  switch (modelKey) {
    case "lstm": {
      const units = (s.lstm__units as number) ?? 64;
      const ratio = units / lags;
      if (ratio > 5) {
        warnings.push(`units (${units}) / lags (${lags}) = ${ratio.toFixed(1)} — ratio is very high. At >5, the model likely memorizes noise. Try units in range ${Math.round(lags * 1.5)}–${Math.round(lags * 2.5)} (ratio 1.5–2.5).`);
      } else if (ratio < 1) {
        warnings.push(`units (${units}) / lags (${lags}) = ${ratio.toFixed(1)} — very low. The model may not have enough capacity to learn temporal patterns. Try raising units to at least ${Math.round(lags * 1.5)}.`);
      }
      tips.push(`Start with ${Math.round(lags * 1.5)}–${Math.round(lags * 2.5)} units for ${lags} lags. Increase lags before increasing units.`);
      tips.push("Learning rate: 1e-4 to 5e-3 (log scale). Use early stopping (patience=6).");
      break;
    }

    case "cnn": {
      const kernelSize = (s.cnn__kernel_size as number) ?? 3;
      const layers = (s.cnn__num_layers as number) ?? 3;
      const rf = kernelSize * layers;
      const note = rf < lags ? `Only the last ${rf} of ${lags} lags matter — earlier lags are wasted.` : `Receptive field (${rf}) covers all ${lags} lags.`;
      warnings.push(`CNN receptive field: ${layers} layers × kernel ${kernelSize} = ${rf} bars. ${note}`);
      tips.push("Filters: 32/64/96. More filters = more pattern detectors.");
      tips.push("Learning rate: 1e-4 to 5e-3 (log scale).");
      break;
    }

    case "transformer": {
      tips.push("d_model must be divisible by num_heads. Ratio d_model/num_heads ≥ 8.");
      tips.push("Attention weights are NOT feature importance — trust gradient-based rankings.");
      tips.push("d_model: 32/64/128. Dropout: 0.1–0.4 (higher for smaller datasets).");
      break;
    }

    case "xgboost": {
      const lr = (s.xgboost__learning_rate as number) ?? 0.1;
      tips.push(`Current learning rate: ${lr}. Lower LR (0.01–0.05) paired with more estimators (400–800) often generalizes better on FX data.`);
      tips.push("max_depth 3–8. Shallower trees generalize better on financial data.");
      tips.push("subsample 0.6–1.0 helps prevent overfitting.");
      break;
    }

    case "logistic": {
      const C = (s.logitC as number) ?? 1.0;
      const solver = (s.logitSolver as string) ?? "lbfgs";
      const penalty = (s.logitPenalty as string) ?? "l2";
      tips.push(`C = ${C}. Range 0.01–100 (log scale). Higher C = less regularization.`);
      tips.push(`Solver: ${solver} with penalty: ${penalty}. ${solver === "lbfgs" && penalty !== "l2" ? "⚠ lbfgs only supports l2." : ""}`);
      tips.push("Check Training Diagnostics for VIF > 10 collinearity warnings after the run.");
      break;
    }

    case "svm": {
      tips.push("Gamma is now categorical [0.0001, 0.001, 0.01, 0.1, 0.5]. No more gamma=10 that memorizes noise.");
      tips.push("RBF kernel is fixed — best for non-linear FX patterns.");
      break;
    }

    case "random_forest": {
      tips.push("max_depth is always bounded (no None option). Trees are depth-limited by design.");
      tips.push("n_estimators: 300–1000. More trees = smoother predictions.");
      tips.push("min_samples_leaf: 1–10. Higher = less overfitting.");
      break;
    }

    case "decision_tree": {
      warnings.push("Single tree is highly prone to overfitting. Use for feature analysis, not production.");
      tips.push("Compare its feature importance with XGBoost — agreement suggests robust features.");
      break;
    }

    case "dqn": {
      warnings.push("Very slow per trial (~2.5 min). Use Minimal preset for quick tests.");
      tips.push("GPU strongly recommended. Start with fewer episodes, then scale up.");
      break;
    }

    case "ensemble_adaptive_regime": {
      tips.push("Delegates to RF for regime detection + XGBoost for prediction. Feature importance from sub-model.");
      tips.push("Best for mixed-regime markets (sideways + trending + volatile).");
      break;
    }

    case "ensemble_cnn_lstm_xgboost": {
      warnings.push("Very slow per trial (~3 min). GPU strongly recommended.");
      tips.push("Feature importance extracted from XGBoost sub-model via TreeSHAP.");
      tips.push("Validate component models individually before using the ensemble.");
      break;
    }
  }

  return { warnings, tips };
}
