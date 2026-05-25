import { useCallback } from "react";
import { useLlmAnalysis } from "@/api/queries";
import type { LlmAnalysis } from "@/api/schemas";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { Bot, Zap, AlertTriangle, Lightbulb, ChevronRight } from "lucide-react";

interface Props {
  jobId: string | null;
  modelName: string | null;
}

export function LLMAdvisor({ jobId, modelName }: Props) {
  const { data, isLoading, refetch } = useLlmAnalysis(jobId, modelName);
  const applyQuickPreset = useBacktestStore((s) => s.applyQuickPreset);
  const setField = useBacktestStore((s) => s.setField);

  const handleApplyPreset = useCallback((presetKey: string) => {
    if (presetKey && jobId) {
      setField("parentJobId", jobId);
      applyQuickPreset(presetKey);
    }
  }, [applyQuickPreset, setField, jobId]);

  if (!jobId || !modelName) return null;

  const analysis: LlmAnalysis | null = data?.analysis ?? null;

  // Not yet analyzed
  if (!analysis && !isLoading) {
    return (
      <div
        className="rounded-xl border p-5"
        style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot size={14} style={{ color: "var(--color-accent)" }} />
            <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
              AI Analysis
            </h3>
          </div>
          <button
            onClick={() => refetch()}
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider transition-all"
            style={{
              backgroundColor: "var(--color-accent)",
              color: "var(--color-text-inverse)",
              cursor: "pointer",
            }}
          >
            <Zap size={11} />
            Analyze Results
          </button>
        </div>
      </div>
    );
  }

  // Loading
  if (isLoading) {
    return (
      <div
        className="rounded-xl border p-5"
        style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
      >
        <div className="flex items-center gap-2">
          <Bot size={14} style={{ color: "var(--color-accent)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            Analyzing results...
          </span>
        </div>
      </div>
    );
  }

  // Error or no analysis
  if (!analysis) return null;

  const presetColor = analysis.recommended_preset?.includes("prod") ? "var(--color-accent-danger)"
    : analysis.recommended_preset?.includes("deep") ? "var(--color-accent)"
    : "var(--color-brand)";

  return (
    <div
      className="rounded-xl border p-5"
      style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Bot size={14} style={{ color: "var(--color-accent)" }} />
          <h3 className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-secondary)" }}>
            AI Analysis
          </h3>
        </div>
        <button
          onClick={() => refetch()}
          className="text-xs"
          style={{ color: "var(--color-text-muted)", cursor: "pointer" }}
        >
          Re-analyze
        </button>
      </div>

      {analysis.error ? (
        <div className="flex items-start gap-2 p-3 rounded" style={{ backgroundColor: "rgba(239,68,68,0.05)" }}>
          <AlertTriangle size={12} style={{ color: "var(--color-accent-danger)", marginTop: 1 }} />
          <div className="flex flex-col gap-2">
            <span className="text-xs" style={{ color: "var(--color-accent-danger)" }}>
              {analysis.error}
            </span>
            {analysis.raw_text && (
              <span className="text-[10px] font-mono" style={{ color: "var(--color-text-muted)" }}>
                Raw response: {analysis.raw_text}
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {/* Insight */}
          {analysis.insight && (
            <div className="flex items-start gap-2">
              <Lightbulb size={12} style={{ color: "var(--color-accent-warning)", marginTop: 1, flexShrink: 0 }} />
              <span className="text-[11px] leading-relaxed" style={{ color: "var(--color-text-primary)" }}>
                {analysis.insight}
              </span>
            </div>
          )}

          {/* Recommendation with Apply button */}
          {analysis.recommended_preset && (
            <div
              className="flex items-center justify-between rounded-lg p-3"
              style={{ backgroundColor: "var(--color-elevated)" }}
            >
              <div className="flex flex-col gap-0.5">
                <span className="text-[10px] font-semibold uppercase tracking-[0.06em]" style={{ color: presetColor }}>
                  Next Study: {analysis.recommended_preset}
                </span>
                {analysis.reason && (
                  <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                    {analysis.reason}
                  </span>
                )}
                {analysis.predicted_improvement && (
                  <span className="text-[9px]" style={{ color: "var(--color-accent-success)", fontFamily: "var(--font-mono)" }}>
                    {analysis.predicted_improvement}
                  </span>
                )}
              </div>
              <button
                onClick={() => handleApplyPreset(analysis.recommended_preset!)}
                className="flex items-center gap-1 rounded-md px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider transition-all hover:brightness-110"
                style={{
                  backgroundColor: presetColor,
                  color: "var(--color-text-inverse)",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                Apply Study
                <ChevronRight size={10} />
              </button>
            </div>
          )}

          {/* Parameter changes */}
          {analysis.parameter_changes && analysis.parameter_changes.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-muted)" }}>
                Suggestions
              </span>
              {analysis.parameter_changes.map((s, i) => (
                <div key={i} className="flex items-start gap-1.5">
                  <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>·</span>
                  <span className="text-[10px]" style={{ color: "var(--color-text-secondary)" }}>{s}</span>
                </div>
              ))}
            </div>
          )}

          {/* Warning */}
          {analysis.warning && (
            <div className="flex items-start gap-1.5 p-2 rounded" style={{ backgroundColor: "rgba(245,158,11,0.05)" }}>
              <AlertTriangle size={10} style={{ color: "var(--color-accent-warning)", marginTop: 1, flexShrink: 0 }} />
              <span className="text-[10px]" style={{ color: "var(--color-accent-warning)" }}>
                {analysis.warning}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
