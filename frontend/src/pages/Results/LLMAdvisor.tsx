import { useCallback, useMemo } from "react";
import { useLlmAnalysis } from "@/api/queries";
import type { LlmAnalysis, AnalysisSection } from "@/api/schemas";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { Bot, Zap, AlertTriangle, Lightbulb, ChevronRight, Shield, TrendingDown, Layers } from "lucide-react";

interface Props {
  jobId: string | null;
  modelName: string | null;
}

const SEVERITY_BORDERS: Record<string, string> = {
  green: "border-l-4 border-l-[var(--color-accent-success)]",
  amber: "border-l-4 border-l-[var(--color-accent-warning)]",
  red: "border-l-4 border-l-[var(--color-accent-danger)]",
  info: "border-l-4 border-l-[var(--color-accent)]",
};

const SEVERITY_DOTS: Record<string, string> = {
  green: "var(--color-accent-success)",
  amber: "var(--color-accent-warning)",
  red: "var(--color-accent-danger)",
  info: "var(--color-accent)",
};

const SEVERITY_LABELS: Record<string, string> = {
  green: "PASS",
  amber: "WARN",
  red: "FAIL",
  info: "INFO",
};

function AnalysisCard({
  section,
  icon,
}: {
  section: AnalysisSection;
  icon: React.ReactNode;
}) {
  const borderClass = SEVERITY_BORDERS[section.severity] ?? SEVERITY_BORDERS.info;
  const dotColor = SEVERITY_DOTS[section.severity] ?? SEVERITY_DOTS.info;
  const label = SEVERITY_LABELS[section.severity] ?? "INFO";

  return (
    <div className={`rounded-sm border border-(--color-glass-border) bg-[rgba(15,18,26,0.65)] p-4 ${borderClass}`}>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-(--color-text-muted)">{icon}</span>
          <h4 className="text-[11px] font-semibold tracking-[0.06em] text-(--color-text-secondary) uppercase">
            {section.title}
          </h4>
        </div>
        <span
          className="rounded px-2 py-0.5 text-[9px] font-bold tracking-wider uppercase"
          style={{
            color: dotColor,
            backgroundColor: `${dotColor}15`,
            border: `1px solid ${dotColor}30`,
          }}
        >
          {label}
        </span>
      </div>
      <p className="mb-2 text-[11px] leading-relaxed text-(--color-text-dim)">
        {section.detail}
      </p>
      <div className="flex items-start gap-1.5 rounded bg-white/[0.02] px-2 py-1.5">
        <Lightbulb size={10} className="mt-[2px] shrink-0 text-(--color-accent-warning)" />
        <span className="text-[10px] text-(--color-text-muted)">{section.recommendation}</span>
      </div>
    </div>
  );
}

export function LLMAdvisor({ jobId, modelName }: Props) {
  const { data, isLoading, refetch } = useLlmAnalysis(jobId, modelName);
  const applyQuickPreset = useBacktestStore((s) => s.applyQuickPreset);
  const setField = useBacktestStore((s) => s.setField);

  const handleApplyPreset = useCallback(
    (presetKey: string) => {
      if (presetKey && jobId) {
        setField("parentJobId", jobId);
        applyQuickPreset(presetKey);
      }
    },
    [applyQuickPreset, setField, jobId],
  );

  if (!jobId || !modelName) return null;

  const analysis: LlmAnalysis | null = data?.analysis ?? null;

  const hasStructuredSections = useMemo(
    () =>
      !!(
        analysis?.dsr_analysis?.title ||
        analysis?.friction_analysis?.title ||
        analysis?.regime_analysis?.title
      ),
    [analysis],
  );

  if (!analysis && !isLoading) {
    return (
      <div className="rounded-sm border border-(--color-glass-border) bg-[rgba(15,18,26,0.5)] p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot size={14} className="text-(--color-accent)" />
            <h3 className="text-xs font-semibold tracking-[0.1em] text-(--color-text-secondary) uppercase">
              Statistical Diagnostics
            </h3>
            <span className="text-[10px] text-(--color-text-muted)">DSR &middot; Friction &middot; Regime</span>
          </div>
          <button
            onClick={() => refetch()}
            className="flex cursor-pointer items-center gap-1.5 rounded-md bg-(--color-accent) px-3 py-1.5 text-[10px] font-semibold tracking-wider text-(--color-text-inverse) uppercase transition-all"
          >
            <Zap size={11} />
            Analyze Results
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="rounded-sm border border-(--color-glass-border) bg-[rgba(15,18,26,0.5)] p-5">
        <div className="flex items-center gap-2">
          <Bot size={14} className="text-(--color-accent)" />
          <span className="text-xs text-(--color-text-muted)">Computing diagnostics...</span>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  const presetColor = analysis.recommended_preset?.includes("prod")
    ? "var(--color-accent-danger)"
    : analysis.recommended_preset?.includes("deep")
      ? "var(--color-accent)"
      : "var(--color-brand)";

  return (
    <div className="flex flex-col gap-4 rounded-sm border border-(--color-glass-border) bg-[rgba(15,18,26,0.4)] p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bot size={14} className="text-(--color-accent)" />
          <h3 className="text-xs font-semibold tracking-[0.1em] text-(--color-text-secondary) uppercase">
            Statistical Diagnostics
          </h3>
          <span className="text-[10px] text-(--color-text-muted)">DSR &middot; Friction &middot; Regime</span>
        </div>
        <button
          onClick={() => refetch()}
          className="cursor-pointer text-[10px] text-(--color-text-muted) hover:text-(--color-text-secondary)"
        >
          Re-analyze
        </button>
      </div>

      {analysis.error ? (
        <div className="flex items-start gap-2 rounded bg-red-500/[0.05] p-3">
          <AlertTriangle size={12} className="mt-[1px] text-(--color-accent-danger)" />
          <div className="flex flex-col gap-1">
            <span className="text-xs text-(--color-accent-danger)">{analysis.error}</span>
            {analysis.raw_text && (
              <span className="font-mono text-[10px] text-(--color-text-muted)">
                {analysis.raw_text}
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {/* Structured 3-card diagnostic grid */}
          {hasStructuredSections && (
            <div className="grid grid-cols-1 gap-3">
              {analysis.dsr_analysis?.title && (
                <AnalysisCard
                  section={analysis.dsr_analysis}
                  icon={<Shield size={12} />}
                />
              )}
              {analysis.friction_analysis?.title && (
                <AnalysisCard
                  section={analysis.friction_analysis}
                  icon={<TrendingDown size={12} />}
                />
              )}
              {analysis.regime_analysis?.title && (
                <AnalysisCard
                  section={analysis.regime_analysis}
                  icon={<Layers size={12} />}
                />
              )}
            </div>
          )}

          {/* LLM-enriched insight (optional) */}
          {analysis.insight && (
            <div className="flex items-start gap-2 rounded border border-(--color-glass-border) bg-white/[0.02] p-3">
              <Lightbulb size={12} className="mt-[1px] shrink-0 text-(--color-accent-warning)" />
              <span className="text-[11px] leading-relaxed text-(--color-text-primary)">
                {analysis.insight}
              </span>
            </div>
          )}

          {/* Recommended next study */}
          {analysis.recommended_preset && (
            <div className="flex items-center justify-between rounded-sm bg-(--color-elevated) p-3">
              <div className="flex flex-col gap-0.5">
                <span
                  className="text-[10px] font-semibold tracking-[0.06em] uppercase"
                  style={{ color: presetColor }}
                >
                  Next Study: {analysis.recommended_preset}
                </span>
                {analysis.reason && (
                  <span className="text-[10px] text-(--color-text-muted)">{analysis.reason}</span>
                )}
                {analysis.predicted_improvement && (
                  <span className="font-mono text-[9px] text-(--color-accent-success)">
                    {analysis.predicted_improvement}
                  </span>
                )}
              </div>
              <button
                onClick={() => handleApplyPreset(analysis.recommended_preset!)}
                className="flex cursor-pointer items-center gap-1 rounded-md px-3 py-1.5 text-[10px] font-semibold tracking-wider whitespace-nowrap text-(--color-text-inverse) uppercase transition-all hover:brightness-110"
                style={{ backgroundColor: presetColor }}
              >
                Apply Study
                <ChevronRight size={10} />
              </button>
            </div>
          )}

          {/* Parameter suggestions */}
          {analysis.parameter_changes && analysis.parameter_changes.length > 0 && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
                Suggestions
              </span>
              {analysis.parameter_changes.map((s, i) => (
                <div key={i} className="flex items-start gap-1.5">
                  <span className="text-[9px] text-(--color-text-muted)">&middot;</span>
                  <span className="text-[10px] text-(--color-text-secondary)">{s}</span>
                </div>
              ))}
            </div>
          )}

          {/* Warning */}
          {analysis.warning && (
            <div className="flex items-start gap-1.5 rounded bg-amber-500/[0.05] p-2">
              <AlertTriangle size={10} className="mt-[1px] shrink-0 text-(--color-accent-warning)" />
              <span className="text-[10px] text-(--color-accent-warning)">{analysis.warning}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
