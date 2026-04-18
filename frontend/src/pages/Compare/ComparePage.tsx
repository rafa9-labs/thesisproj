import { EmptyState } from "@/components/shared/EmptyState";
import { GitCompare } from "lucide-react";

export function ComparePage() {
  return (
    <div className="flex flex-col gap-6">
      <h2
        className="text-base font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        Model Comparison
      </h2>
      <EmptyState
        icon={<GitCompare size={48} />}
        title="No comparison data"
        description="Run multiple models in a single backtest to generate a leaderboard, equity overlay, and significance testing matrix."
      />
    </div>
  );
}
