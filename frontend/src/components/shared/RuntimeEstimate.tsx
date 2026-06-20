import { useRuntimeEstimate } from "@/api/queries";
import { useBacktestStore } from "@/stores/useBacktestStore";

export function RuntimeEstimate() {
  const selectedModels = useBacktestStore((s) => s.selectedModels);
  const testMonths = useBacktestStore((s) => s.testMonths);
  const hpoIntensity = useBacktestStore((s) => s.hpoIntensity);
  const nTrials = useBacktestStore((s) => s.nTrials);

  const { data, isLoading, isError } = useRuntimeEstimate(
    selectedModels as string[],
    testMonths as number,
    hpoIntensity,
    nTrials,
  );

  if (selectedModels.length === 0) return null;

  if (isLoading) {
    return <span className="text-xs text-(--color-text-muted)">Estimating runtime...</span>;
  }

  if (isError || !data) return null;

  const { estimated_minutes_low, estimated_minutes_high, total_trials } = data;

  let color = "var(--color-text-muted)";
  if (estimated_minutes_high <= 2) color = "var(--color-accent-success)";
  else if (estimated_minutes_high <= 10) color = "var(--color-accent-warning)";
  else if (estimated_minutes_high <= 60) color = "#f97316";
  else color = "var(--color-accent-danger)";

  const formatTime = (mins: number) => {
    if (mins < 1) return `${Math.round(mins * 60)}s`;
    if (mins < 60) return `${Math.round(mins)} min`;
    const h = Math.floor(mins / 60);
    const m = Math.round(mins % 60);
    return m > 0 ? `~${h}h ${m}m` : `~${h}h`;
  };

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs font-medium tracking-wide text-(--color-text-muted) uppercase">
        Est. Runtime:
      </span>
      <span className="text-sm font-bold" style={{ color }}>
        {formatTime(estimated_minutes_low)} – {formatTime(estimated_minutes_high)}
      </span>
      <span className="text-xs text-(--color-text-muted)">({total_trials} HPO trials)</span>
    </div>
  );
}
