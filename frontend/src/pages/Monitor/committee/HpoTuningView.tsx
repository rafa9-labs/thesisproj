import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import type { Phase2Cache } from "@/stores/useCommitteeMonitorStore";
import { Loader2, Check, X, Clock } from "lucide-react";

const CPU_MODELS = ["logistic", "svm", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"];
const GPU_MODELS = ["lstm", "cnn", "transformer", "gru", "gru_lstm"];
const ENSEMBLE_MODELS = ["ensemble_adaptive_regime", "ensemble_cnn_lstm_xgboost", "meta_ensemble", "stacking_ensemble"];

const MODEL_LABELS: Record<string, string> = {
  logistic: "Logistic",
  svm: "SVM",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  catboost: "CatBoost",
  lstm: "LSTM",
  cnn: "CNN",
  transformer: "Transformer",
  gru: "GRU",
  gru_lstm: "GRU-LSTM",
  ensemble_adaptive_regime: "Adaptive Regime",
  ensemble_cnn_lstm_xgboost: "CNN-LSTM-XGB",
  meta_ensemble: "Meta Ensemble",
  stacking_ensemble: "Stacking",
};

type HpoModelStatus = "success" | "timed_out" | "crashed" | "no_folds" | "skipped" | "running" | "pending";

function statusFromString(s: string): HpoModelStatus {
  switch (s.toLowerCase()) {
    case "success": return "success";
    case "timed_out": return "timed_out";
    case "crashed": return "crashed";
    case "no_folds": return "no_folds";
    case "skipped": return "skipped";
    default: return "pending";
  }
}

function statColor(s: HpoModelStatus): string {
  const map: Record<HpoModelStatus, string> = {
    success: "var(--color-accent-success)",
    timed_out: "var(--color-accent-warning)",
    crashed: "var(--color-accent-danger)",
    no_folds: "var(--color-accent-warning)",
    skipped: "var(--color-text-dim)",
    running: "var(--color-brand)",
    pending: "var(--color-text-dim)",
  };
  return map[s] || "var(--color-text-dim)";
}

function ModelCell({
  model,
  status,
  score,
}: {
  model: string;
  status: HpoModelStatus;
  score: number | null;
}) {
  const label = MODEL_LABELS[model] || model;

  const statusConfig: Record<
    HpoModelStatus,
    { bg: string; border: string; color: string; icon: React.ReactNode }
  > = {
    success: {
      bg: "rgba(16,185,129,0.06)",
      border: "rgba(16,185,129,0.2)",
      color: "var(--color-accent-success)",
      icon: <Check size={11} />,
    },
    timed_out: {
      bg: "rgba(245,158,11,0.06)",
      border: "rgba(245,158,11,0.2)",
      color: "var(--color-accent-warning)",
      icon: <Clock size={11} />,
    },
    crashed: {
      bg: "rgba(244,63,94,0.06)",
      border: "rgba(244,63,94,0.2)",
      color: "var(--color-accent-danger)",
      icon: <X size={11} />,
    },
    no_folds: {
      bg: "rgba(245,158,11,0.04)",
      border: "rgba(245,158,11,0.12)",
      color: "var(--color-accent-warning)",
      icon: <X size={11} />,
    },
    skipped: {
      bg: "transparent",
      border: "var(--color-text-dim)",
      color: "var(--color-text-dim)",
      icon: null,
    },
    running: {
      bg: "rgba(0,229,255,0.06)",
      border: "rgba(0,229,255,0.25)",
      color: "var(--color-brand)",
      icon: <Loader2 size={11} className="animate-spin" />,
    },
    pending: {
      bg: "transparent",
      border: "rgba(51,65,85,0.3)",
      color: "var(--color-text-dim)",
      icon: null,
    },
  };

  const cfg = statusConfig[status];

  return (
    <div
      className="flex items-center justify-between rounded-[2px] border px-2.5 py-2 transition-colors duration-200"
      style={{
        backgroundColor: cfg.bg,
        borderColor: cfg.border,
      }}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span style={{ color: cfg.color }}>{cfg.icon}</span>
        <span
          className="truncate font-mono text-[10px] font-semibold"
          style={{ color: cfg.color }}
        >
          {label}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {score != null && status === "success" && (
          <span className="font-mono text-[9px] text-(--color-accent-success)">
            {score >= 0 ? "+" : ""}{score.toFixed(3)}
          </span>
        )}
        {status !== "running" && status !== "pending" && (
          <span
            className="rounded-full px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-[0.04em]"
            style={{
              backgroundColor: cfg.border,
              color: cfg.color,
            }}
          >
            {status.replace(/_/g, " ")}
          </span>
        )}
      </div>
    </div>
  );
}

function LeaderboardBar({ model, score, maxScore, status }: { model: string; score: number; maxScore: number; status: HpoModelStatus }) {
  const pct = maxScore > 0 ? Math.max(3, (score / maxScore) * 100) : 0;
  const label = MODEL_LABELS[model] || model;
  return (
    <div className="flex items-center gap-2">
      <span className="w-[80px] shrink-0 truncate text-right font-mono text-[9px] text-(--color-text-secondary)">
        {label}
      </span>
      <div className="flex flex-1 items-center gap-1.5">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ backgroundColor: "rgba(0,229,255,0.06)" }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${pct}%`,
              backgroundColor: score >= 0.3 ? "var(--color-accent-success)" : "var(--color-brand)",
              opacity: 0.6 + (pct / 200),
            }}
          />
        </div>
        <span
          className="w-[34px] text-right font-mono text-[9px] font-semibold"
          style={{ color: statColor(status) }}
        >
          {score >= 0 ? "+" : ""}{score.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export function HpoTuningView() {
  const phaseCache = useCommitteeMonitorStore((s) => s.phaseCache);
  const phaseNumber = useCommitteeMonitorStore((s) => s.phaseNumber);
  const phaseProgress = useCommitteeMonitorStore((s) => s.phaseProgress);
  const survivingModels = useCommitteeMonitorStore((s) => s.survivingModels);
  const hpoScores = useCommitteeMonitorStore((s) => s.hpoScores);
  // Live HPO status from status endpoint (available during Phase 2 execution)
  const liveHpoStatus = useCommitteeMonitorStore((s) => s.liveHpoStatus);

  const cache = phaseCache[2] as Phase2Cache | null;
  // Prefer live status when available, fall back to results cache
  const hpoStatus = Object.keys(liveHpoStatus).length > 0 ? liveHpoStatus : (cache?.hpoStatus ?? {});

  const isRunning = phaseNumber === 2;

  const allModels = survivingModels.length > 0
    ? survivingModels
    : [...CPU_MODELS, ...GPU_MODELS, ...ENSEMBLE_MODELS];

  const modelStatuses: Record<string, HpoModelStatus> = {};
  for (const m of allModels) {
    if (hpoStatus[m]) {
      modelStatuses[m] = statusFromString(hpoStatus[m]);
    } else if (isRunning) {
      modelStatuses[m] = "pending";
    } else {
      modelStatuses[m] = "skipped";
    }
  }

  // Build leaderboard: only successful models with scores
  const leaderboard = allModels
    .filter((m) => modelStatuses[m] === "success" && hpoScores[m] != null)
    .map((m) => ({ model: m, score: hpoScores[m]! }))
    .sort((a, b) => b.score - a.score);

  const lblMaxScore = leaderboard.length > 0 ? Math.max(leaderboard[0].score, 0.5) : 1;

  const cpuModels = allModels.filter((m) => CPU_MODELS.includes(m));
  const gpuModels = allModels.filter((m) => GPU_MODELS.includes(m));
  const ensembleModels = allModels.filter((m) => ENSEMBLE_MODELS.includes(m));

  const gpuRunningIdx = gpuModels.findIndex(
    (m) => modelStatuses[m] === "running" || modelStatuses[m] === "pending",
  );

  return (
    <div className="flex flex-col gap-5 px-2 py-4 sm:px-4">
      {/* Progress header */}
      {isRunning && phaseProgress && (
        <div className="flex items-center gap-2">
          <Loader2 size={14} className="animate-spin text-(--color-brand)" />
          <span className="font-mono text-[10px] text-(--color-text-secondary)">
            HPO in progress: {phaseProgress} models complete
          </span>
        </div>
      )}

      {/* Ranked leaderboard bar chart */}
      {leaderboard.length > 1 && (
        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-text-dim)">
            HPO Leaderboard (Best Sharpe)
          </div>
          <div className="flex flex-col gap-1.5 rounded-[2px] border border-(--color-glass-border) bg-white/[0.02] p-3">
            {leaderboard.map(({ model, score }) => (
              <LeaderboardBar
                key={model}
                model={model}
                score={score}
                maxScore={lblMaxScore}
                status={modelStatuses[model]}
              />
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* CPU Models */}
        <div>
          <div className="mb-2 flex items-center gap-2">
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: "var(--color-accent-classical)" }}
            />
            <span className="text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-accent-classical)">
              CPU Models (Parallel)
            </span>
          </div>
          <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {cpuModels.map((m) => (
              <ModelCell
                key={m}
                model={m}
                status={modelStatuses[m] || "pending"}
                score={hpoScores[m] ?? null}
              />
            ))}
          </div>
        </div>

        {/* GPU + Ensemble */}
        <div className="flex flex-col gap-4">
          {gpuModels.length > 0 && (
            <div>
              <div className="mb-2 flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: "var(--color-accent-deep)" }}
                />
                <span className="text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-accent-deep)">
                  GPU Models (Serial)
                </span>
              </div>
              <div className="flex flex-col gap-1.5">
                {gpuModels.map((m, idx) => (
                  <ModelCell
                    key={m}
                    model={m}
                    status={
                      gpuRunningIdx >= 0 && idx > gpuRunningIdx
                        ? "skipped"
                        : modelStatuses[m] || "pending"
                    }
                    score={hpoScores[m] ?? null}
                  />
                ))}
              </div>
            </div>
          )}

          {ensembleModels.length > 0 && (
            <div>
              <div className="mb-2 flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: "var(--color-accent-ensemble)" }}
                />
                <span className="text-[10px] font-semibold uppercase tracking-[0.04em] text-(--color-accent-ensemble)">
                  Ensembles (Serial)
                </span>
              </div>
              <div className="flex flex-col gap-1.5">
                {ensembleModels.map((m) => (
                  <ModelCell
                    key={m}
                    model={m}
                    status={modelStatuses[m] || "pending"}
                    score={hpoScores[m] ?? null}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Summary footer */}
      {!isRunning && Object.keys(hpoStatus).length > 0 && (
        <div className="flex flex-wrap gap-3 text-[9px] font-mono text-(--color-text-dim)">
          <span>
            Success: {Object.values(hpoStatus).filter((s) => s === "success").length}
          </span>
          <span>
            Failed: {Object.values(hpoStatus).filter((s) => s !== "success" && s !== "skipped").length}
          </span>
          {leaderboard.length > 0 && (
            <span className="text-(--color-accent-success)">
              Best: {leaderboard[0].model.toUpperCase()} ({leaderboard[0].score >= 0 ? "+" : ""}{leaderboard[0].score.toFixed(2)})
            </span>
          )}
        </div>
      )}
    </div>
  );
}
