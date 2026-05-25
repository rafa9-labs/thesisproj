import { TriangleAlert, Bookmark, ArrowRight } from "lucide-react";
import { useRuntimeEstimate } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

const TAB_SEQUENCE = [
  { key: "quickstart", label: "Quick Start" },
  { key: "asset", label: "Asset & Model" },
  { key: "study", label: "Study & HPO" },
  { key: "features", label: "Features" },
  { key: "hyperparams", label: "Hyperparameters" },
  { key: "execution", label: "Execution" },
  { key: "forwardtest", label: "Forward Test" },
];

interface Props {
  warnings: number;
  errors: number;
  canDeploy: boolean;
  isSubmitting: boolean;
  hasModels: boolean;
  hasPair: boolean;
  hasDates: boolean;
  onDeploy: () => void;
  onSavePreset?: () => void;
  activeTab?: string;
  onTabChange?: (tab: string) => void;
}

function RuntimeInline() {
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const testMonths = useBacktestStore((s) => s.testMonths);
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);

  const { data, isLoading } = useRuntimeEstimate(
    selectedModels as string[],
    testMonths as number,
    hpoIntensity,
  );

  if (selectedModels.length === 0 || isLoading || !data) return null;

  const { estimated_minutes_low, estimated_minutes_high } = data;

  const fmt = (mins: number) => {
    if (mins < 1) return `${Math.round(mins * 60)}s`;
    if (mins < 60) return `${Math.round(mins)} min`;
    const h = Math.floor(mins / 60);
    const m = Math.round(mins % 60);
    return m > 0 ? `~${h}h ${m}m` : `~${h}h`;
  };

  return (
    <div className="flex items-center gap-1.5">
      <span
        style={{
          fontSize: 9,
          fontWeight: 600,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--color-text-muted)",
          opacity: 0.7,
        }}
      >
        EST. RUNTIME
      </span>
      <span
        style={{
          fontSize: 12,
          fontFamily: "var(--font-mono)",
          color: "var(--color-text-primary)",
        }}
      >
        {fmt(estimated_minutes_low)} – {fmt(estimated_minutes_high)}
      </span>
    </div>
  );
}

export function ValidationBar({
  warnings,
  errors,
  canDeploy,
  isSubmitting,
  hasModels,
  hasPair,
  hasDates,
  onDeploy,
  onSavePreset,
  activeTab,
  onTabChange,
}: Props) {
  const currentIdx = TAB_SEQUENCE.findIndex((t) => t.key === activeTab);
  const nextTab = currentIdx >= 0 && currentIdx < TAB_SEQUENCE.length - 1
    ? TAB_SEQUENCE[currentIdx + 1]
    : null;
  const missingItems: string[] = [];
  if (!hasPair) missingItems.push("a currency pair");
  if (!hasModels) missingItems.push("at least one model");
  if (!hasDates) missingItems.push("date range");

  return (
    <div
      className="sticky bottom-0 z-20 flex items-center justify-between px-6 relative"
      style={{
        height: 56,
        borderTop: "1px solid #333",
        backgroundColor: "var(--color-app)",
      }}
    >
      {/* Left: status + runtime */}
      <div className="flex items-center gap-6">
        {/* Status text */}
        <div className="flex items-center gap-2">
          {missingItems.length > 0 ? (
            <span
              style={{
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--color-text-muted)",
              }}
            >
              Select {missingItems.join(" and ")} to start
            </span>
          ) : errors > 0 ? (
            <div className="flex items-center gap-1.5">
              <TriangleAlert size={12} style={{ color: "var(--color-accent-danger)" }} />
              <span
                style={{
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                  color: "var(--color-text-secondary)",
                }}
              >
                {errors} config error{errors > 1 ? "s" : ""}
              </span>
            </div>
          ) : warnings > 0 ? (
            <div className="flex items-center gap-1.5">
              <TriangleAlert size={12} style={{ color: "var(--color-accent-warning)" }} />
              <span
                style={{
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                  color: "var(--color-text-secondary)",
                }}
              >
                {warnings} warning{warnings > 1 ? "s" : ""}
              </span>
            </div>
          ) : (
            <span
              style={{
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--color-text-secondary)",
              }}
            >
              Ready to deploy
            </span>
          )}
        </div>

        {/* Divider + runtime */}
        <RuntimeInline />
      </div>

      {/* Centre: next step */}
      {nextTab && onTabChange && (
        <button
          type="button"
          onClick={() => onTabChange(nextTab.key)}
          className="absolute left-1/2 flex items-center gap-1.5 rounded border transition-colors"
          style={{
            transform: "translateX(-50%)",
            height: 28,
            padding: "0 12px",
            backgroundColor: "#1D4ED818",
            borderColor: "#3B82F655",
            color: "#60A5FA",
            fontSize: 11,
            fontFamily: "var(--font-mono)",
            cursor: "pointer",
            letterSpacing: "0.04em",
            whiteSpace: "nowrap",
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#1D4ED830"; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = "#1D4ED818"; }}
        >
          Next: {nextTab.label}
          <ArrowRight size={11} strokeWidth={1.5} />
        </button>
      )}

      {/* Right: save preset + deploy CTA */}
      <div className="flex items-center gap-3">
        {onSavePreset && canDeploy && (
          <button
            onClick={onSavePreset}
            className="flex items-center gap-1.5 rounded border px-3 text-[10px] font-semibold uppercase tracking-[0.06em] transition-colors duration-150 hover:brightness-110"
            style={{
              height: 34,
              borderColor: "var(--color-glass-border)",
              color: "var(--color-text-secondary)",
              background: "transparent",
            }}
          >
            <Bookmark size={11} />
            Save Preset
          </button>
        )}

        <button
          onClick={onDeploy}
          disabled={!canDeploy || isSubmitting}
          className="rounded text-[11px] font-bold uppercase tracking-[0.08em] transition-all duration-150 hover:brightness-110"
          style={{
            height: 34,
            paddingLeft: 24,
            paddingRight: 24,
            backgroundColor: "var(--color-brand)",
            color: "#0A0D12",
            cursor: canDeploy && !isSubmitting ? "pointer" : "not-allowed",
            opacity: canDeploy ? (isSubmitting ? 0.7 : 1) : 0.35,
            border: "none",
          }}
        >
          {isSubmitting ? "Submitting..." : "Deploy Backtest"}
        </button>
      </div>
    </div>
  );
}
