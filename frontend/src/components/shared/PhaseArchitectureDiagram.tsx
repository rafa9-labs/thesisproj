import { useState, useCallback, useRef, useEffect, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { ArrowRight, ArrowDown, ArrowLeft, ArrowUpRight, CheckCircle2, XCircle, AlertTriangle, Download, Upload, Play, Save, Cpu, Sparkles } from "lucide-react";

const PHASES = [
  {
    id: 1,
    name: "Feature Sweep",
    summaryKey: "sweep" as const,
    description:
      "Trains a shallow Random Forest with Boruta-SHAP to eliminate non-predictive features from the candidate set. Outputs a locked feature list consumed by all downstream phases.",
  },
  {
    id: 2,
    name: "HPO Tuning",
    summaryKey: "hpo" as const,
    description:
      "Runs hyperparameter optimization with ASHA pruning on each surviving model. CPU models run in parallel via joblib; GPU and deep models run sequentially to avoid memory bottlenecks. Failed models are excluded from the committee.",
  },
  {
    id: 3,
    name: "Committee Build",
    summaryKey: "assembly" as const,
    description:
      "Assembles the committee by mapping each of 7 market regimes to a weighted ensemble of top-K models. Models are ranked by regime-specific Sharpe ratio. Weights are assigned via Sharpe-proportional or equal weighting.",
  },
  {
    id: 4,
    name: "Walk-Forward",
    summaryKey: "validation" as const,
    description:
      "Validates the assembled committee with a 36-month walk-forward backtest. Computes fold consistency CV, PBO (probability of backtest overfitting), trust score, and 3-seed robustness. A meta-learner is optionally trained to gate the committee.",
  },
  {
    id: 5,
    name: "Factory",
    summaryKey: "factory" as const,
    description:
      "Iterative committee optimization. The proposer suggests model swaps/adds/removes per regime. Each proposal is tested via proxy WFO. The loop stops on patience, budget, hard-gate, exhaustion, or divergence.",
  },
  {
    id: 6,
    name: "Snapshot",
    summaryKey: "snapshot" as const,
    description:
      "Trains the final committee on the full dataset and saves byte-for-byte reproducible model weights. Produces a deployment artifact ready for live trading.",
  },
] as const;

type PhaseKey = (typeof PHASES)[number]["summaryKey"];

interface PhaseConfig {
  proposer: string;
  llmBackend: string;
  hpoSampler: string;
  skipFeatureSweep: boolean;
  enablePhase3: boolean;
  enablePhase4: boolean;
  enablePhase5: boolean;
  enablePhase6: boolean;
  maxIterations: number;
  committeeTopK: number;
  selectedModelCount: number;
  lockedFeaturesCount?: number;
}

const PROPOSER_LABELS: Record<string, string> = {
  llm: "LLM",
  hybrid_llm_ucb1: "LLM+UCB1",
  ucb1: "UCB1",
  deterministic: "Greedy",
};

const SAMPLER_LABELS: Record<string, string> = {
  tpe: "TPE",
  random: "Random",
  grid: "Grid",
};

const BACKEND_LABELS: Record<string, string> = {
  deepseek: "DeepSeek",
  ollama: "Ollama",
  openai: "OpenAI",
  anthropic: "Claude",
};

function getSummary(key: PhaseKey, cfg: PhaseConfig): string {
  switch (key) {
    case "sweep":
      if (cfg.skipFeatureSweep) return "Skipped";
      return cfg.lockedFeaturesCount && cfg.lockedFeaturesCount > 0
        ? `${cfg.lockedFeaturesCount} features`
        : "Boruta-SHAP";
    case "hpo":
      return `Sampler: ${SAMPLER_LABELS[cfg.hpoSampler] || cfg.hpoSampler}`;
    case "assembly":
      return `Top-${cfg.committeeTopK} · ${cfg.selectedModelCount} models`;
    case "validation":
      return "3-seed robustness";
    case "factory": {
      const lbl = PROPOSER_LABELS[cfg.proposer] || cfg.proposer;
      const backend =
        (cfg.proposer === "llm" || cfg.proposer === "hybrid_llm_ucb1") && cfg.llmBackend
          ? BACKEND_LABELS[cfg.llmBackend] || cfg.llmBackend
          : null;
      return `Proposer: ${lbl}${backend ? ` (${backend})` : ""}`;
    }
    case "snapshot":
      return "Deploy artifact";
    default:
      return "";
  }
}

function isPhaseEnabled(id: number, cfg: PhaseConfig): boolean {
  switch (id) {
    case 1: return !cfg.skipFeatureSweep;
    case 2: return cfg.enablePhase3;
    case 3: return cfg.enablePhase4;
    case 4: return cfg.enablePhase5;
    case 5: return cfg.enablePhase6;
    case 6: return cfg.enablePhase6;
    default: return true;
  }
}

function getConfigLines(id: number, cfg: PhaseConfig): string[] {
  switch (id) {
    case 1:
      return cfg.skipFeatureSweep
        ? ["Feature sweep: Skipped"]
        : ["Boruta-SHAP Random Forest", "Estimators: 100 · Max Depth: 5"];
    case 2:
      return [
        `Sampler: ${SAMPLER_LABELS[cfg.hpoSampler] || cfg.hpoSampler}`,
        "ASHA pruning enabled · CPU parallel, GPU serial",
      ];
    case 3:
      return [
        `Top-${cfg.committeeTopK} models per regime`,
        `${cfg.selectedModelCount} models selected · 7 market regimes`,
      ];
    case 4:
      return ["Fold consistency CV · PBO", "3-seed robustness · Meta-learner gate"];
    case 5: {
      const lbl = PROPOSER_LABELS[cfg.proposer] || cfg.proposer;
      const backend =
        (cfg.proposer === "llm" || cfg.proposer === "hybrid_llm_ucb1") && cfg.llmBackend
          ? BACKEND_LABELS[cfg.llmBackend] || cfg.llmBackend
          : null;
      return [
        `Proposer: ${lbl}${backend ? ` (${backend})` : ""}`,
        `Iterations: ${cfg.maxIterations} · Patience: 5`,
      ];
    }
    case 6:
      return ["Byte-for-byte reproducible weights", "Ready for live trading deployment"];
    default:
      return [];
  }
}

/* ── Shared diagram primitives ── */

function PhaseArrow({ className }: { className?: string }) {
  return (
    <div className={cn("flex shrink-0 items-center justify-center", className)}>
      <ArrowRight size={20} strokeWidth={2} className="text-(--color-brand)/50" />
    </div>
  );
}

function PhaseArrowDown({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center justify-center", className)}>
      <ArrowDown size={20} strokeWidth={2} className="text-(--color-brand)/50" />
    </div>
  );
}

function DiagramBox({
  label,
  sub,
  color = "brand",
  icon,
  children,
  className,
}: {
  label: string;
  sub?: string;
  color?: "brand" | "success" | "warning" | "danger" | "classical" | "deep" | "ensemble";
  icon?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  const colorMap = {
    brand: "border-(--color-brand)/25 bg-(--color-brand-glow) text-(--color-stepper-active)",
    success: "border-(--color-accent-success)/25 bg-[rgba(8,153,129,0.06)] text-(--color-accent-success)",
    warning: "border-(--color-accent-warning)/25 bg-[rgba(245,158,11,0.05)] text-(--color-accent-warning)",
    danger: "border-(--color-accent-danger)/25 bg-[rgba(242,54,69,0.05)] text-(--color-accent-danger)",
    classical: "border-(--color-accent-classical)/25 bg-[rgba(34,211,238,0.05)] text-(--color-accent-classical)",
    deep: "border-(--color-accent-deep)/25 bg-[rgba(167,139,250,0.05)] text-(--color-accent-deep)",
    ensemble: "border-(--color-accent-ensemble)/25 bg-[rgba(236,72,153,0.05)] text-(--color-accent-ensemble)",
  };
  return (
    <div
      className={cn(
        "flex min-w-[110px] flex-col gap-1 rounded-lg border p-3 text-center shadow-sm",
        colorMap[color],
        className,
      )}
    >
      {icon && <div className="mb-1 flex justify-center">{icon}</div>}
      <span className="text-[12px] font-bold tracking-[0.04em]">{label}</span>
      {sub && <span className="text-[10px] font-medium opacity-70">{sub}</span>}
      {children && <div className="mt-1">{children}</div>}
    </div>
  );
}

function IoBadge({ type, label }: { type: "input" | "output"; label: string }) {
  if (type === "input") {
    return (
      <div className="inline-flex items-center gap-1.5 rounded-md border border-slate-600/20 bg-slate-800/40 px-2.5 py-1">
        <Download size={11} className="text-(--color-text-dim)/50" />
        <span className="text-[8px] font-bold tracking-[0.08em] text-(--color-text-dim)/50 uppercase">INPUT</span>
        <span className="text-[10px] text-(--color-text-secondary)">{label}</span>
      </div>
    );
  }
  return (
    <div className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-success)/20 bg-[rgba(8,153,129,0.06)] px-2.5 py-1">
      <Upload size={11} className="text-(--color-accent-success)/60" />
      <span className="text-[8px] font-bold tracking-[0.08em] text-(--color-accent-success)/60 uppercase">OUTPUT</span>
      <span className="text-[10px] text-(--color-text-secondary)">{label}</span>
    </div>
  );
}

function ConditionBadge({ label }: { label: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-warning)/20 bg-[rgba(245,158,11,0.05)] px-2.5 py-1">
      <AlertTriangle size={11} className="text-(--color-accent-warning)/70" />
      <span className="text-[8px] font-bold tracking-[0.08em] text-(--color-accent-warning)/70 uppercase">CONDITION</span>
      <span className="text-[10px] text-(--color-text-secondary)">{label}</span>
    </div>
  );
}

function StopBadge({ label }: { label: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-danger)/20 bg-[rgba(242,54,69,0.05)] px-2.5 py-1">
      <XCircle size={11} className="text-(--color-accent-danger)/60" />
      <span className="text-[8px] font-bold tracking-[0.08em] text-(--color-accent-danger)/60 uppercase">STOP</span>
      <span className="text-[10px] text-(--color-text-secondary)">{label}</span>
    </div>
  );
}

/* ── Sub-diagram renderers ── */

function SweepSubDiagram() {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <IoBadge type="input" label="OHLCV + 250+ TA indicators" />
        <IoBadge type="output" label="locked_features.json (~68 feature names) → Phases 2–6" />
      </div>

      {/* Desktop: horizontal */}
      <div className="hidden flex-wrap items-center justify-center gap-2 md:flex">
        <DiagramBox label="Raw Features" sub="250+ cols" color="classical" />
        <PhaseArrow />
        <DiagramBox label="Boruta-SHAP RF" sub="iterative shadow test" color="brand" icon={<Sparkles size={16} />} />
        <PhaseArrow />
        <DiagramBox label="Locked Features" sub="~68 survivors" color="success" icon={<CheckCircle2 size={16} />} />
      </div>

      {/* Mobile: stacked */}
      <div className="flex flex-col items-center gap-3 md:hidden">
        <div className="flex flex-wrap items-center justify-center gap-2">
          <DiagramBox label="Raw Features" sub="250+ cols" color="classical" />
          <PhaseArrow />
          <DiagramBox label="Boruta-SHAP RF" sub="iterative shadow test" color="brand" icon={<Sparkles size={16} />} />
        </div>
        <PhaseArrowDown />
        <DiagramBox label="Locked Features" sub="~68 survivors" color="success" icon={<CheckCircle2 size={16} />} />
      </div>

      <div className="flex items-center justify-center gap-2">
        <XCircle size={12} className="text-(--color-accent-danger)/50" />
        <span className="text-[10px] text-(--color-text-dim)">Eliminated features are discarded via the shadow-feature comparison loop</span>
      </div>
    </div>
  );
}

function HpoSubDiagram() {
  const cpuModels = ["RF", "XGB", "LGBM", "SVM", "CAT", "LOG"];
  const gpuModels = ["LSTM", "CNN", "Transformer", "GRU"];
  const ensModels = ["ENS", "ENS2"];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <IoBadge type="input" label="Feature set from Phase 1" />
        <IoBadge type="output" label="best_params.json (HPO params + validation scores) → Phase 3" />
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        {/* CPU pool */}
        <div className="min-w-[180px] rounded-lg border border-(--color-accent-classical)/20 bg-[rgba(34,211,238,0.03)] p-3">
          <div className="mb-2 flex items-center gap-2">
            <Cpu size={14} className="text-(--color-accent-classical)" />
            <span className="text-[11px] font-bold text-(--color-accent-classical)">CPU Parallel</span>
            <span className="ml-auto text-[9px] text-(--color-text-dim)">joblib · 4 workers</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {cpuModels.map((m) => (
              <span key={m} className="rounded border border-(--color-accent-classical)/15 bg-[rgba(34,211,238,0.08)] px-2 py-0.5 text-[10px] font-mono text-(--color-text-secondary)">
                {m}
              </span>
            ))}
          </div>
        </div>

        <PhaseArrow className="hidden md:flex" />
        <PhaseArrowDown className="flex md:hidden" />

        {/* GPU pool */}
        <div className="min-w-[180px] rounded-lg border border-(--color-accent-deep)/20 bg-[rgba(167,139,250,0.03)] p-3">
          <div className="mb-2 flex items-center gap-2">
            <Play size={14} className="text-(--color-accent-deep)" />
            <span className="text-[11px] font-bold text-(--color-accent-deep)">GPU Serial</span>
            <span className="ml-auto text-[9px] text-(--color-text-dim)">one at a time</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {gpuModels.map((m) => (
              <span key={m} className="rounded border border-(--color-accent-deep)/15 bg-[rgba(167,139,250,0.08)] px-2 py-0.5 text-[10px] font-mono text-(--color-text-secondary)">
                {m}
              </span>
            ))}
          </div>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {ensModels.map((m) => (
              <span key={m} className="rounded border border-(--color-accent-ensemble)/15 bg-[rgba(236,72,153,0.08)] px-2 py-0.5 text-[10px] font-mono text-(--color-accent-ensemble)">
                {m}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        <div className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-warning)/20 bg-[rgba(245,158,11,0.05)] px-2.5 py-1">
          <AlertTriangle size={11} className="text-(--color-accent-warning)/70" />
          <span className="text-[10px] text-(--color-text-secondary)">ASHA pruning terminates underperforming trials early</span>
        </div>
        <div className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-danger)/20 bg-[rgba(242,54,69,0.05)] px-2.5 py-1">
          <XCircle size={11} className="text-(--color-accent-danger)/60" />
          <span className="text-[10px] text-(--color-text-secondary)">Failed models excluded from committee</span>
        </div>
      </div>
    </div>
  );
}

function AssemblySubDiagram({ topK, modelCount }: { topK: number; modelCount: number }) {
  const regimes = [
    { name: "Trend Up", models: ["cnn", "xgb", "lstm"], w: [40, 35, 25] },
    { name: "Trend Down", models: ["xgb", "lstm", "cnn"], w: [45, 30, 25] },
    { name: "Sideways", models: ["svm", "log", "rf"], w: [50, 30, 20] },
    { name: "Volatile", models: ["lstm", "cnn", "xgb"], w: [40, 35, 25] },
    { name: "Quiet", models: ["rf", "svm", "log"], w: [45, 35, 20] },
    { name: "Reversal", models: ["log", "xgb", "svm"], w: [50, 30, 20] },
    { name: "Breakout", models: ["cnn", "lstm", "xgb"], w: [35, 35, 30] },
  ];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <IoBadge type="input" label="Best params + HPO scores from Phase 2" />
        <IoBadge type="output" label="regime_committee.json (7 regimes × top-K models + weights) → Phase 4" />
      </div>

      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-(--color-accent-classical)/15 bg-[rgba(34,211,238,0.03)] p-2.5">
        <span className="text-[10px] font-bold text-(--color-accent-classical)">Market Classification</span>
        <ArrowRight size={12} className="text-(--color-text-dim)/40" />
        <span className="text-[10px] text-(--color-text-secondary)">7 Market Regimes</span>
        <ArrowRight size={12} className="text-(--color-text-dim)/40" />
        <span className="text-[10px] text-(--color-text-secondary)">Rank by Sharpe</span>
        <ArrowRight size={12} className="text-(--color-text-dim)/40" />
        <span className="text-[10px] text-(--color-text-secondary)">Top-{topK} Selection</span>
        <ArrowRight size={12} className="text-(--color-text-dim)/40" />
        <span className="text-[10px] text-(--color-text-secondary)">Sharpe-Proportional Weights</span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-(--color-glass-border)">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-(--color-glass-border) bg-(--color-surface)">
              <th className="px-4 py-2.5 text-[10px] font-semibold uppercase tracking-[0.06em] text-(--color-text-dim)">Regime</th>
              <th className="px-4 py-2.5 text-[10px] font-semibold uppercase tracking-[0.06em] text-(--color-text-dim)">Models</th>
              <th className="px-4 py-2.5 text-[10px] font-semibold uppercase tracking-[0.06em] text-(--color-text-dim)">Weights</th>
            </tr>
          </thead>
          <tbody>
            {regimes.map((r) => (
              <tr key={r.name} className="border-b border-(--color-glass-border)/50 text-[12px]">
                <td className="px-4 py-2.5 font-mono text-(--color-text-secondary) whitespace-nowrap">{r.name}</td>
                <td className="px-4 py-2.5">
                  <span className="font-mono text-(--color-text-dim)">{r.models.slice(0, topK).join(" · ")}</span>
                </td>
                <td className="px-4 py-2.5">
                  <div className="flex items-center gap-2">
                    {r.w.slice(0, topK).map((w, i) => (
                      <div key={i} className="flex items-center gap-1.5">
                        <div className="h-2 w-10 overflow-hidden rounded-full" style={{ backgroundColor: "rgba(0,229,255,0.08)" }}>
                          <div className="h-full rounded-full" style={{ width: `${w}%`, backgroundColor: "var(--color-brand)", opacity: 0.6 }} />
                        </div>
                        <span className="font-mono text-[10px] text-(--color-text-dim)">{w}%</span>
                      </div>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="text-[10px] text-(--color-text-dim)">
        Using {modelCount} selected models across 7 regimes. Only the top-{topK} performers per regime are retained.
      </div>
    </div>
  );
}

function ValidateSubDiagram() {
  const steps = [
    { label: "WFO 36-mo", sub: "Sharpe · Equity" },
    { label: "Fold CV", sub: "Stability CV%" },
    { label: "PBO", sub: "P(overfit)" },
    { label: "Trust Score", sub: "Composite 0–1" },
  ];

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <IoBadge type="input" label="Committee from Phase 3" />
        <IoBadge type="output" label="validation_report.json (CV stability, PBO score, trust) → Phase 5" />
      </div>

      {/* Desktop */}
      <div className="hidden flex-wrap items-center justify-center gap-2 md:flex">
        {steps.map((s, i) => (
          <div key={s.label} className="flex items-center gap-2">
            <DiagramBox label={s.label} sub={s.sub} color="brand" />
            {i < steps.length - 1 && <PhaseArrow />}
          </div>
        ))}
        <PhaseArrow />
        <DiagramBox label="3-Seed" sub="branch ×3 merge" color="success" icon={<CheckCircle2 size={16} />} />
      </div>

      {/* Mobile */}
      <div className="flex flex-col gap-3 md:hidden">
        <div className="flex flex-wrap items-center justify-center gap-2">
          {steps.slice(0, 3).map((s, i) => (
            <div key={s.label} className="flex items-center gap-2">
              <DiagramBox label={s.label} sub={s.sub} color="brand" />
              {i < 2 && <PhaseArrow />}
            </div>
          ))}
        </div>
        <div className="flex items-center justify-center">
          <PhaseArrowDown />
        </div>
        <div className="flex flex-wrap items-center justify-center gap-2">
          <DiagramBox label="Trust Score" sub="Composite 0–1" color="brand" />
          <PhaseArrow />
          <DiagramBox label="3-Seed" sub="branch ×3 merge" color="success" icon={<CheckCircle2 size={16} />} />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        <ConditionBadge label="trust > threshold" />
        <span className="text-[10px] text-(--color-text-dim)">Only committees passing the trust gate proceed to Phase 5</span>
      </div>
    </div>
  );
}

function FactorySubDiagram({ proposer, backend, maxIterations }: { proposer: string; backend: string | null; maxIterations: number }) {
  const proposerLabel = PROPOSER_LABELS[proposer] || proposer;
  const backendLabel = backend ? `(${backend})` : "";

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <IoBadge type="input" label="Committee + validation from Phase 4" />
        <IoBadge type="output" label="optimized_committee.json (best config + ΔSharpe history) → Phase 6" />
      </div>

      <div className="flex items-center justify-center">
        <StopBadge label="patience · hard-gate · exhaustion · divergence" />
      </div>

      {/* Desktop */}
      <div className="hidden flex-wrap items-center justify-center gap-2 md:flex">
        <DiagramBox label="PROPOSE" sub={`${proposerLabel} ${backendLabel}`} color="brand" icon={<Sparkles size={16} />} />
        <PhaseArrow />
        <DiagramBox label="EXECUTE" sub="Proxy WFO" color="brand" />
        <PhaseArrow />
        <DiagramBox label="EVALUATE" sub="Δ Sharpe" color="brand" />
        <PhaseArrow />
        <DiagramBox label="DECIDE" sub="Accept / Reject" color="warning" />
      </div>

      {/* Mobile */}
      <div className="flex flex-col gap-3 md:hidden">
        <div className="flex flex-wrap items-center justify-center gap-2">
          <DiagramBox label="PROPOSE" sub={`${proposerLabel} ${backendLabel}`} color="brand" icon={<Sparkles size={16} />} />
          <PhaseArrow />
          <DiagramBox label="EXECUTE" sub="Proxy WFO" color="brand" />
        </div>
        <div className="flex items-center justify-center">
          <PhaseArrowDown />
        </div>
        <div className="flex flex-wrap items-center justify-center gap-2">
          <DiagramBox label="EVALUATE" sub="Δ Sharpe" color="brand" />
          <PhaseArrow />
          <DiagramBox label="DECIDE" sub="Accept / Reject" color="warning" />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-2">
        <div className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-success)/20 bg-[rgba(8,153,129,0.05)] px-2.5 py-1">
          <CheckCircle2 size={11} className="text-(--color-accent-success)/70" />
          <span className="text-[10px] text-(--color-text-secondary)">Accept → update committee</span>
        </div>
        <div className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-danger)/20 bg-[rgba(242,54,69,0.05)] px-2.5 py-1">
          <XCircle size={11} className="text-(--color-accent-danger)/60" />
          <span className="text-[10px] text-(--color-text-secondary)">Reject → patience++</span>
        </div>
        <span className="text-[10px] text-(--color-text-dim)">max {maxIterations} iterations</span>
      </div>
    </div>
  );
}

function SnapshotSubDiagram() {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <IoBadge type="input" label="Winning committee from Phase 5" />
        <IoBadge type="output" label="deploy_artifact.pkl (weights, features, regimes, scaler) → Live Trading" />
      </div>

      {/* Desktop */}
      <div className="hidden flex-wrap items-center justify-center gap-2 md:flex">
        <DiagramBox label="Best Iteration" sub="winning params" color="brand" />
        <PhaseArrow />
        <DiagramBox label="Train (full)" sub="entire dataset" color="brand" />
        <PhaseArrow />
        <DiagramBox label="Save Weights" sub="reproducible" color="brand" icon={<Save size={16} />} />
        <PhaseArrow />
        <DiagramBox label="Artifact" sub="Weights · Features · Regimes · Scaler" color="success" icon={<CheckCircle2 size={16} />} />
        <PhaseArrow />
        <DiagramBox label="Live Trading" sub="Engine ✓" color="success" icon={<ArrowUpRight size={16} />} />
      </div>

      {/* Mobile */}
      <div className="flex flex-col gap-3 md:hidden">
        <div className="flex flex-wrap items-center justify-center gap-2">
          <DiagramBox label="Best Iteration" sub="winning params" color="brand" />
          <PhaseArrow />
          <DiagramBox label="Train (full)" sub="entire dataset" color="brand" />
        </div>
        <div className="flex items-center justify-center">
          <PhaseArrowDown />
        </div>
        <div className="flex flex-wrap items-center justify-center gap-2">
          <DiagramBox label="Save Weights" sub="reproducible" color="brand" icon={<Save size={16} />} />
          <PhaseArrow />
          <DiagramBox label="Artifact" sub="Weights · Features · Regimes · Scaler" color="success" icon={<CheckCircle2 size={16} />} />
          <PhaseArrow />
          <DiagramBox label="Live Trading" sub="Engine ✓" color="success" icon={<ArrowUpRight size={16} />} />
        </div>
      </div>
    </div>
  );
}

/* ── Main pipeline cards ── */

interface PhaseCardProps {
  phase: typeof PHASES[number];
  enabled: boolean;
  isSelected: boolean;
  isHovered: boolean;
  summary: string;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  onClick: () => void;
}

function PhaseCard({ phase, enabled, isSelected, isHovered, summary, onMouseEnter, onMouseLeave, onClick }: PhaseCardProps) {
  const isHighlighted = isSelected || isHovered;
  return (
    <button
      type="button"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={cn(
        "relative flex w-full flex-col items-center gap-2 rounded-lg border p-3 text-center transition-all duration-200",
        isHighlighted
          ? "border-(--color-brand)/40 bg-[rgba(0,229,255,0.07)] shadow-[0_0_20px_rgba(0,229,255,0.08)]"
          : "border-(--color-glass-border) bg-(--color-surface)",
        !enabled && "opacity-45",
      )}
    >
      {/* Phase number circle */}
      <div
        className={cn(
          "flex h-10 w-10 items-center justify-center rounded-full border-2 text-[15px] font-bold transition-all",
          isHighlighted
            ? "border-(--color-stepper-active) bg-(--color-stepper-active)/15 text-(--color-stepper-active)"
            : enabled
              ? "border-(--color-stepper-completed) bg-(--color-stepper-completed)/10 text-(--color-stepper-completed)"
              : "border-(--color-stepper-upcoming) bg-transparent text-(--color-stepper-upcoming)",
        )}
      >
        {phase.id}
      </div>

      {/* Phase name */}
      <div
        className={cn(
          "text-[12px] font-semibold",
          enabled ? "text-(--color-text-primary)" : "text-(--color-text-dim)",
        )}
      >
        {phase.name}
      </div>

      {/* Summary line */}
      <div className="text-[10px] text-(--color-text-dim)">{summary}</div>

      {/* Disabled strikethrough */}
      {!enabled && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="h-px w-[70%] rotate-[-12deg] bg-(--color-text-dim)/20" />
        </div>
      )}
    </button>
  );
}

/* ── Main component ── */

interface Props {
  config: PhaseConfig;
  className?: string;
  selectedPhase?: number | null;
  onPhaseSelect?: (phaseId: number | null) => void;
}

export function PhaseArchitectureDiagram({ config, className, selectedPhase: externalPhase, onPhaseSelect }: Props) {
  const [hoveredPhase, setHoveredPhase] = useState<number | null>(null);
  const [internalPhase, setInternalPhase] = useState<number | null>(null);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => { if (hoverTimer.current) clearTimeout(hoverTimer.current); };
  }, []);

  const handleMouseEnter = useCallback((phaseId: number) => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = setTimeout(() => setHoveredPhase(phaseId), 80);
  }, []);

  const handleMouseLeave = useCallback(() => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = setTimeout(() => setHoveredPhase(null), 80);
  }, []);

  const selectedPhase = onPhaseSelect ? externalPhase ?? null : internalPhase;

  const handleNodeClick = useCallback(
    (phaseId: number) => {
      if (onPhaseSelect) {
        onPhaseSelect(externalPhase === phaseId ? null : phaseId);
      } else {
        setInternalPhase((prev) => (prev === phaseId ? null : phaseId));
      }
    },
    [onPhaseSelect, externalPhase],
  );

  const handleContainerClick = useCallback(() => {
    if (onPhaseSelect) {
      onPhaseSelect(null);
    } else {
      setInternalPhase(null);
    }
  }, [onPhaseSelect]);

  const activePhase = hoveredPhase ?? selectedPhase;
  const phase = activePhase ? PHASES.find((p) => p.id === activePhase) : null;
  const configLines = activePhase ? getConfigLines(activePhase, config) : [];

  const backendLabel =
    (config.proposer === "llm" || config.proposer === "hybrid_llm_ucb1") && config.llmBackend
      ? BACKEND_LABELS[config.llmBackend] || config.llmBackend
      : null;

  return (
    <div className={cn("flex flex-col gap-0", className)}>
      {/* ── Pipeline overview (responsive grid) ── */}
      <div
        className="rounded-lg border border-(--color-glass-border) bg-(--color-surface)/50 p-3"
        onClick={handleContainerClick}
      >
        {/* Desktop: horizontal 1×6 */}
        <div className="hidden items-center gap-2 lg:flex">
          {PHASES.map((p, i) => (
            <div key={p.id} className="flex flex-1 items-center gap-2">
              <PhaseCard
                phase={p}
                enabled={isPhaseEnabled(p.id, config)}
                isSelected={selectedPhase === p.id}
                isHovered={hoveredPhase === p.id}
                summary={getSummary(p.summaryKey, config)}
                onMouseEnter={() => handleMouseEnter(p.id)}
                onMouseLeave={handleMouseLeave}
                onClick={() => handleNodeClick(p.id)}
              />
              {i < PHASES.length - 1 && <PhaseArrow />}
            </div>
          ))}
        </div>

        {/* Tablet/Mobile: 2 rows × 3 columns with arrows */}
        <div className="flex flex-col gap-3 lg:hidden">
          {/* Row 1 */}
          <div className="flex items-center gap-1">
            {PHASES.slice(0, 3).map((p, i) => (
              <div key={p.id} className="flex flex-1 items-center gap-1">
                <PhaseCard
                  phase={p}
                  enabled={isPhaseEnabled(p.id, config)}
                  isSelected={selectedPhase === p.id}
                  isHovered={hoveredPhase === p.id}
                  summary={getSummary(p.summaryKey, config)}
                  onMouseEnter={() => handleMouseEnter(p.id)}
                  onMouseLeave={handleMouseLeave}
                  onClick={() => handleNodeClick(p.id)}
                />
                {i < 2 && <PhaseArrow />}
              </div>
            ))}
          </div>

          {/* Down arrow connector */}
          <div className="flex items-center justify-end pr-[16.666%]">
            <PhaseArrowDown />
          </div>

          {/* Row 2 */}
          <div className="flex items-center gap-1">
            {PHASES.slice(3, 6).map((p, i) => (
              <div key={p.id} className="flex flex-1 items-center gap-1">
                <PhaseCard
                  phase={p}
                  enabled={isPhaseEnabled(p.id, config)}
                  isSelected={selectedPhase === p.id}
                  isHovered={hoveredPhase === p.id}
                  summary={getSummary(p.summaryKey, config)}
                  onMouseEnter={() => handleMouseEnter(p.id)}
                  onMouseLeave={handleMouseLeave}
                  onClick={() => handleNodeClick(p.id)}
                />
                {i < 2 && <PhaseArrow />}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Detail Panel ── */}
      {selectedPhase !== null && phase && (
        <div
          className="mt-3 rounded-lg border px-5 py-4 transition-all duration-200"
          style={{
            borderColor: "rgba(0,229,255,0.2)",
            backgroundColor: "rgba(0,229,255,0.03)",
          }}
        >
          {/* Header */}
          <div className="mb-3 flex items-center gap-2">
            <span
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full font-mono text-[12px] font-bold"
              style={{ backgroundColor: "var(--color-brand-glow)", color: "var(--color-stepper-active)" }}
            >
              {phase.id}
            </span>
            <span
              className="text-[14px] font-semibold uppercase tracking-[0.06em]"
              style={{ color: "var(--color-stepper-active)" }}
            >
              {phase.name}
            </span>
          </div>

          {/* Description */}
          <p className="mb-4 max-w-[680px] text-[12px] leading-relaxed text-(--color-text-secondary)">
            {phase.description}
          </p>

          {/* Sub-diagram */}
          <div className="mb-4">
            {selectedPhase === 1 && <SweepSubDiagram />}
            {selectedPhase === 2 && <HpoSubDiagram />}
            {selectedPhase === 3 && (
              <AssemblySubDiagram topK={config.committeeTopK} modelCount={config.selectedModelCount} />
            )}
            {selectedPhase === 4 && <ValidateSubDiagram />}
            {selectedPhase === 5 && (
              <FactorySubDiagram proposer={config.proposer} backend={backendLabel} maxIterations={config.maxIterations} />
            )}
            {selectedPhase === 6 && <SnapshotSubDiagram />}
          </div>

          {/* Config lines */}
          {configLines.length > 0 && (
            <div className="flex flex-wrap gap-x-8 gap-y-2 border-t pt-3" style={{ borderColor: "var(--color-glass-border)" }}>
              {configLines.map((line, i) => (
                <span key={i} className="font-mono text-[11px] text-(--color-text-dim)">
                  {line}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
