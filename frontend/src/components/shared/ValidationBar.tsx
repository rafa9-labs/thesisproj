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
      <span className="text-[9px] font-semibold tracking-[0.1em] text-(--color-text-muted) uppercase opacity-70">
        EST. RUNTIME
      </span>
      <span className="font-mono text-xs text-(--color-text-primary)">
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
    <div className="sticky bottom-0 z-20 flex h-[72px] items-center justify-between border-t border-(--color-glass-border) bg-(--color-surface) px-6">
      {/* Left: deploy status with rocket */}
      <div className="flex items-center gap-4">
        <div
          className="flex h-10 w-10 items-center justify-center rounded-lg"
          style={{
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
              <span className="text-[13px] font-semibold text-(--color-text-secondary)">
                Setup incomplete
              </span>
              <span className="font-mono text-[11px] text-(--color-text-muted)">
                Select {missingItems.join(" and ")} to start
              </span>
            </>
          ) : errors > 0 ? (
            <>
              <div className="flex items-center gap-1.5">
                <TriangleAlert size={13} className="text-(--color-accent-danger)" />
                <span className="text-[13px] font-semibold text-(--color-accent-danger)">
                  {errors} config error{errors > 1 ? "s" : ""}
                </span>
              </div>
              <span className="font-mono text-[11px] text-(--color-text-muted)">
                Resolve errors before deploying
              </span>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <span className="text-[14px] font-bold text-(--color-text-primary)">
                  Ready to Deploy
                </span>
                {warnings > 0 && (
                  <span className="flex items-center gap-1">
                    <TriangleAlert size={11} className="text-(--color-accent-warning)" />
                    <span className="font-mono text-[10px] text-(--color-accent-warning)">
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
            className="flex h-11 items-center gap-1.5 rounded-md border border-(--color-glass-border) bg-transparent px-4 text-[11px] font-semibold tracking-[0.06em] text-(--color-text-secondary) uppercase transition-colors duration-150 hover:border-(--color-border-active) hover:text-(--color-text-primary)"
          >
            <Bookmark size={13} />
            Save Draft
          </button>
        )}

        <button
          onClick={onDeploy}
          disabled={!canDeploy || isSubmitting}
          className="flex h-11 items-center gap-2 rounded-md border-0 bg-(--color-brand) pr-7 pl-7 text-[12px] font-bold tracking-[0.08em] text-(--color-text-inverse) uppercase transition-all duration-150 hover:brightness-110"
          style={{
            cursor: canDeploy && !isSubmitting ? "pointer" : "not-allowed",
            opacity: canDeploy ? (isSubmitting ? 0.7 : 1) : 0.35,
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
