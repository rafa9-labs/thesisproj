import { TriangleAlert, Bookmark, Rocket, ArrowRight } from "lucide-react";
import { useRuntimeEstimate } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

interface Props {
  warnings: number;
  errors: number;
  canDeploy: boolean;
  isSubmitting: boolean;
  hasModels: boolean;
  hasPair: boolean;
  hasDates?: boolean;
  onDeploy: () => void;
  onSavePreset?: () => void;
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
}: Props) {
  const hasDatesValue = hasDates ?? true;
  const missingItems: string[] = [];
  if (!hasPair) missingItems.push("a currency pair");
  if (!hasModels) missingItems.push("at least one model");
  if (!hasDatesValue) missingItems.push("date range");

  const ready = missingItems.length === 0 && errors === 0;

  return (
    <div
      className="sticky bottom-0 z-20 flex items-center justify-between px-6"
      style={{
        height: 72,
        borderTop: "1px solid var(--color-glass-border)",
        backgroundColor: "var(--color-surface)",
      }}
    >
      {/* Left: deploy status with rocket */}
      <div className="flex items-center gap-4">
        <div
          className="flex items-center justify-center rounded-lg"
          style={{
            width: 40,
            height: 40,
            backgroundColor: ready ? "var(--color-brand-glow)" : "var(--color-glass)",
            border: `1px solid ${ready ? "var(--color-border-active)" : "var(--color-glass-border)"}`,
            color: ready ? "var(--color-brand)" : "var(--color-text-muted)",
          }}
        >
          <Rocket size={18} strokeWidth={1.75} />
        </div>

        <div className="flex flex-col gap-0.5">
          {missingItems.length > 0 ? (
            <>
              <span
                className="text-[13px] font-semibold"
                style={{ color: "var(--color-text-secondary)" }}
              >
                Setup incomplete
              </span>
              <span
                style={{
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                  color: "var(--color-text-muted)",
                }}
              >
                Select {missingItems.join(" and ")} to start
              </span>
            </>
          ) : errors > 0 ? (
            <>
              <div className="flex items-center gap-1.5">
                <TriangleAlert size={13} style={{ color: "var(--color-accent-danger)" }} />
                <span
                  className="text-[13px] font-semibold"
                  style={{ color: "var(--color-accent-danger)" }}
                >
                  {errors} config error{errors > 1 ? "s" : ""}
                </span>
              </div>
              <span
                style={{
                  fontSize: 11,
                  fontFamily: "var(--font-mono)",
                  color: "var(--color-text-muted)",
                }}
              >
                Resolve errors before deploying
              </span>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <span
                  className="text-[14px] font-bold"
                  style={{ color: "var(--color-text-primary)" }}
                >
                  Ready to Deploy
                </span>
                {warnings > 0 && (
                  <span className="flex items-center gap-1">
                    <TriangleAlert size={11} style={{ color: "var(--color-accent-warning)" }} />
                    <span
                      style={{
                        fontSize: 10,
                        fontFamily: "var(--font-mono)",
                        color: "var(--color-accent-warning)",
                      }}
                    >
                      {warnings} warning{warnings > 1 ? "s" : ""}
                    </span>
                  </span>
                )}
              </div>
              <RuntimeInline />
            </>
          )}
        </div>
      </div>

      {/* Right: save preset + deploy CTA */}
      <div className="flex items-center gap-3">
        {onSavePreset && canDeploy && (
          <button
            onClick={onSavePreset}
            className="flex items-center gap-1.5 rounded-md border px-4 text-[11px] font-semibold uppercase tracking-[0.06em] transition-colors duration-150"
            style={{
              height: 44,
              borderColor: "var(--color-glass-border)",
              color: "var(--color-text-secondary)",
              background: "transparent",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "var(--color-border-active)";
              e.currentTarget.style.color = "var(--color-text-primary)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--color-glass-border)";
              e.currentTarget.style.color = "var(--color-text-secondary)";
            }}
          >
            <Bookmark size={13} />
            Save Draft
          </button>
        )}

        <button
          onClick={onDeploy}
          disabled={!canDeploy || isSubmitting}
          className="flex items-center gap-2 rounded-md text-[12px] font-bold uppercase tracking-[0.08em] transition-all duration-150 hover:brightness-110"
          style={{
            height: 44,
            paddingLeft: 28,
            paddingRight: 28,
            backgroundColor: "var(--color-brand)",
            color: "var(--color-text-inverse)",
            cursor: canDeploy && !isSubmitting ? "pointer" : "not-allowed",
            opacity: canDeploy ? (isSubmitting ? 0.7 : 1) : 0.35,
            border: "none",
            boxShadow: canDeploy ? "0 0 20px rgba(0,229,255,0.25)" : "none",
          }}
        >
          {isSubmitting ? "Submitting..." : "Deploy Backtest"}
          {!isSubmitting && <ArrowRight size={15} strokeWidth={2.25} />}
        </button>
      </div>
    </div>
  );
}
