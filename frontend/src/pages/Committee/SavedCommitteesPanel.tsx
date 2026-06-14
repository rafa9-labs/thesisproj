import {
  useSavedCommittees,
  useActivateSavedCommittee,
  useDeleteSavedCommittee,
} from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import apiClient from "@/api/client";
import { Trash2, Zap, Circle } from "lucide-react";

export function SavedCommitteesPanel() {
  const { data, isLoading } = useSavedCommittees();
  const activateMutation = useActivateSavedCommittee();
  const deleteMutation = useDeleteSavedCommittee();
  const store = useFullCycleStore();

  const handleDeploy = (committee: {
    id: string;
    full_cycle_job_id: string | null;
    pair: string;
    timeframe: string;
  }) => {
    apiClient
      .post("/trading/live/committee/start", {
        pair: committee.pair || store.deployedPair,
        timeframe: committee.timeframe || store.deployedTimeframe,
        initial_equity: 10000.0,
        confidence_threshold: 0.55,
        mode: store.executionMode,
        full_cycle_job_id: committee.full_cycle_job_id,
      })
      .then((r: { data: { session_id: string; pair: string; timeframe: string } }) => {
        store.setDeployedSession(r.data.session_id, r.data.pair, r.data.timeframe);
      })
      .catch(console.error);
  };

  if (isLoading) {
    return <div className="text-xs text-(--color-text-muted)">Loading saved committees...</div>;
  }

  const committees = data?.committees ?? [];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <span className="text-[13px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          Saved Committees{data?.total ? ` (${data.total})` : ""}
        </span>
      </div>

      {committees.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-4 rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-12">
          <Zap size={32} className="text-(--color-text-muted)" />
          <span className="text-xs text-(--color-text-muted)">
            No saved committees yet. Run a Full Cycle and click "Save Committee" on the results.
          </span>
        </div>
      )}

      <div className="flex flex-col gap-4">
        {committees.map((c) => {
          const regimeCount = c.config_json?.regimes
            ? Object.keys(c.config_json.regimes as Record<string, unknown>).length
            : 0;
          const modelSet = new Set<string>();
          if (c.config_json?.regimes) {
            Object.values(c.config_json.regimes as Record<string, { models?: string[] }>).forEach(
              (r) => {
                r.models?.forEach((m: string) => modelSet.add(m));
              },
            );
          }
          return (
            <div
              key={c.id}
              className="rounded-sm border p-4 transition-all"
              style={{
                borderColor: c.is_active ? "var(--color-brand)" : "var(--color-glass-border)",
                backgroundColor: c.is_active ? "rgba(0,229,255,0.02)" : "var(--color-glass)",
                boxShadow: c.is_active ? "0 0 8px rgba(0,229,255,0.06)" : "none",
              }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-[12px] font-semibold text-(--color-text-primary)">
                    {c.name}
                  </span>
                  {c.is_active && (
                    <span className="rounded bg-[rgba(0,229,255,0.12)] px-2 py-0.5 text-[9px] font-bold tracking-[0.06em] text-(--color-brand) uppercase">
                      ACTIVE
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => deleteMutation.mutate(c.id)}
                    className="cursor-pointer rounded border-none bg-transparent p-1 text-(--color-text-muted) hover:text-(--color-accent-danger)"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-4 gap-3 text-[10px]">
                <div>
                  <span className="text-(--color-text-muted)">Pair</span>
                  <div className="font-mono text-(--color-text-secondary)">{c.pair}</div>
                </div>
                <div>
                  <span className="text-(--color-text-muted)">TF</span>
                  <div className="font-mono text-(--color-text-secondary)">{c.timeframe}</div>
                </div>
                <div>
                  <span className="text-(--color-text-muted)">Regimes</span>
                  <div className="font-mono text-(--color-text-secondary)">{regimeCount}</div>
                </div>
                <div>
                  <span className="text-(--color-text-muted)">Models</span>
                  <div className="font-mono text-(--color-text-secondary)">{modelSet.size}</div>
                </div>
                {c.trust_score != null && (
                  <div>
                    <span className="text-(--color-text-muted)">Trust</span>
                    <div
                      className="font-mono"
                      style={{
                        color:
                          c.trust_score >= 0.8
                            ? "#089981"
                            : c.trust_score >= 0.6
                              ? "#F2B436"
                              : "#F23645",
                      }}
                    >
                      {(c.trust_score * 100).toFixed(0)}%
                    </div>
                  </div>
                )}
                {c.avg_sharpe != null && (
                  <div>
                    <span className="text-(--color-text-muted)">Sharpe</span>
                    <div
                      className="font-mono"
                      style={{
                        color: c.avg_sharpe >= 0 ? "#089981" : "#F23645",
                      }}
                    >
                      {c.avg_sharpe >= 0 ? "+" : ""}
                      {c.avg_sharpe.toFixed(2)}
                    </div>
                  </div>
                )}
                <div>
                  <span className="text-(--color-text-muted)">Created</span>
                  <div className="font-mono text-(--color-text-dim)">
                    {c.created_at?.slice(0, 10)}
                  </div>
                </div>
              </div>

              {modelSet.size > 0 && (
                <div className="mt-3 flex flex-wrap gap-1">
                  {[...modelSet].map((m) => (
                    <span
                      key={m}
                      className="rounded-[3px] bg-(--color-elevated) px-[6px] py-[1px] font-mono text-[9px] text-(--color-text-dim)"
                    >
                      {m}
                    </span>
                  ))}
                </div>
              )}

              {c.config_json?.regimes && (
                <div className="mt-3 flex flex-col gap-1">
                  {Object.entries(
                    c.config_json.regimes as Record<
                      string,
                      { models?: string[]; weights?: number[] }
                    >,
                  )
                    .slice(0, 3)
                    .map(([regime, assignment]) => (
                      <div key={regime} className="flex items-center gap-2 text-[9px]">
                        <span className="w-[100px] shrink-0 font-medium tracking-[0.04em] text-(--color-brand) uppercase">
                          {regime.replace(/_/g, " ")}
                        </span>
                        <span className="text-(--color-text-dim)">
                          {(assignment.models ?? [])
                            .map(
                              (m: string, i: number) =>
                                `${m} ${((assignment.weights?.[i] ?? 0) * 100).toFixed(0)}%`,
                            )
                            .join(" | ")}
                        </span>
                      </div>
                    ))}
                  {Object.keys(c.config_json.regimes as Record<string, unknown>).length > 3 && (
                    <span className="text-[9px] text-(--color-text-muted)">
                      +{Object.keys(c.config_json.regimes as Record<string, unknown>).length - 3}{" "}
                      more regimes
                    </span>
                  )}
                </div>
              )}

              <div className="mt-3 flex items-center gap-2">
                {!c.is_active && (
                  <button
                    onClick={() => activateMutation.mutate(c.id)}
                    disabled={activateMutation.isPending}
                    className="flex items-center gap-1 rounded bg-(--color-accent-success) px-3 py-1.5 text-[10px] font-semibold tracking-[0.06em] text-(--color-text-inverse) uppercase hover:brightness-110"
                  >
                    <Circle size={8} fill="currentColor" />
                    Activate
                  </button>
                )}
                <button
                  onClick={() => handleDeploy(c)}
                  className="rounded border border-(--color-glass-border) bg-(--color-elevated) px-3 py-1.5 text-[10px] font-medium tracking-[0.06em] text-(--color-text-secondary) uppercase hover:brightness-110"
                >
                  Deploy Live
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
