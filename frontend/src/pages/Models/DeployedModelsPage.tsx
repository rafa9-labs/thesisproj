import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { TagEditor } from "@/components/shared/TagEditor";
import { Box, Trash2, Star } from "lucide-react";

interface DeployedModel {
  id: string;
  model_type: string;
  best_sharpe: number | null;
  best_return: number | null;
  created_at: string;
  status: string;
  tags: string[];
  parent_job_id: string | null;
  missing_on_disk?: boolean;
}

const CATEGORY_COLORS: Record<string, string> = {
  logistic: "var(--color-accent)",
  svm: "var(--color-accent)",
  random_forest: "var(--color-accent-success)",
  decision_tree: "var(--color-accent-success)",
  xgboost: "var(--color-brand)",
  lightgbm: "var(--color-accent-success)",
  catboost: "var(--color-accent-success)",
  cnn: "var(--color-accent)",
  lstm: "var(--color-accent)",
  transformer: "var(--color-accent)",
  gru: "var(--color-accent)",
  gru_lstm: "var(--color-accent)",
  ensemble_adaptive_regime: "var(--color-accent-warning)",
  ensemble_cnn_lstm_xgboost: "var(--color-accent-warning)",
  meta_ensemble: "var(--color-accent-warning)",
  stacking_ensemble: "var(--color-accent-warning)",
  dqn: "var(--color-accent-danger)",
};

function useDeployedModels() {
  return useQuery({
    queryKey: ["deployed-models"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ models: DeployedModel[] }>("/models/deployed");
      return data.models;
    },
    refetchOnMount: true,
  });
}

function useActivateModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.post(`/models/deployed/${id}/activate`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["deployed-models"] }),
  });
}

function useDeactivateModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.post(`/models/deployed/${id}/deactivate`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["deployed-models"] }),
  });
}

function useDeleteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.delete(`/models/deployed/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["deployed-models"] }),
  });
}

function useUpdateTags() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, action, tag }: { id: string; action: string; tag: string }) =>
      apiClient.patch(`/models/deployed/${id}/tags`, { action, tag }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["deployed-models"] }),
  });
}

export function DeployedModelsPage() {
  const { data: models, isLoading } = useDeployedModels();
  const activateModel = useActivateModel();
  const deactivateModel = useDeactivateModel();
  const deleteModel = useDeleteModel();
  const updateTags = useUpdateTags();
  const [filter, setFilter] = useState<string>("all");

  const filtered = (models ?? []).filter((m) => {
    if (filter === "active") return m.status === "active";
    if (filter === "inactive") return m.status === "inactive";
    return true;
  });

  return (
    <div className="flex flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold uppercase tracking-[0.1em]" style={{ color: "var(--color-text-primary)" }}>
            Deployed Models
          </h2>
          <p className="text-[11px] mt-1" style={{ color: "var(--color-text-muted)" }}>
            Models you've saved from backtest results. Activate one per type for live prediction.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {(["all", "active", "inactive"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className="rounded-md px-3 py-1 text-[10px] font-medium uppercase tracking-[0.06em] transition-all"
              style={{
                backgroundColor: filter === f ? "var(--color-brand)" : "var(--color-glass-hover)",
                color: filter === f ? "var(--color-text-inverse)" : "var(--color-text-muted)",
                border: "1px solid",
                borderColor: filter === f ? "transparent" : "var(--color-glass-border)",
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {isLoading && (
        <div className="text-xs" style={{ color: "var(--color-text-muted)" }}>Loading models...</div>
      )}

      {!isLoading && filtered.length === 0 && (
        <div
          className="flex flex-col items-center justify-center gap-4 rounded-xl border p-12"
          style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
        >
          <Box size={40} style={{ color: "var(--color-text-muted)" }} />
          <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            No deployed models yet. Run a backtest, then click Save Model on the Results page.
          </span>
        </div>
      )}

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))" }}>
        {filtered.map((m) => (
          <div
            key={m.id}
            className="rounded-xl border p-4 transition-all"
            style={{
              borderColor: m.status === "active" ? "var(--color-brand)" : "var(--color-glass-border)",
              backgroundColor: m.status === "active" ? "rgba(0,229,255,0.03)" : "var(--color-glass)",
              boxShadow: m.status === "active" ? "0 0 12px rgba(0,229,255,0.08)" : "none",
            }}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div
                  className="rounded px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em]"
                  style={{
                    backgroundColor: `${CATEGORY_COLORS[m.model_type] ?? "var(--color-text-muted)"}18`,
                    color: CATEGORY_COLORS[m.model_type] ?? "var(--color-text-muted)",
                  }}
                >
                  {m.model_type}
                </div>
                {m.status === "active" && (
                  <div
                    className="rounded px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.06em]"
                    style={{ backgroundColor: "rgba(0,229,255,0.12)", color: "var(--color-brand)" }}
                  >
                    ACTIVE
                  </div>
                )}
                {m.missing_on_disk && (
                  <span className="text-[9px]" style={{ color: "var(--color-accent-danger)" }}>
                    missing
                  </span>
                )}
              </div>
            </div>

            <div className="flex items-center gap-3 text-[11px] mb-3" style={{ fontFamily: "var(--font-mono)" }}>
              <span style={{ color: "var(--color-text-secondary)" }}>
                SR: {m.best_sharpe != null ? (m.best_sharpe >= 0 ? "+" : "") + m.best_sharpe.toFixed(2) : "—"}
              </span>
              {m.best_return != null && (
                <span style={{ color: (m.best_return >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)") }}>
                  {(m.best_return >= 0 ? "+" : "")}{m.best_return.toFixed(1)}%
                </span>
              )}
            </div>

            <div className="mb-3">
              <TagEditor
                tags={m.tags}
                onAdd={(tag) => updateTags.mutate({ id: m.id, action: "add", tag })}
                onRemove={(tag) => updateTags.mutate({ id: m.id, action: "remove", tag })}
              />
            </div>

            <div className="text-[9px] mb-3" style={{ color: "var(--color-text-muted)" }}>
              {m.created_at?.slice(0, 10)}
            </div>

            <div className="flex items-center gap-2">
              {m.status !== "active" && (
                <button
                  onClick={() => activateModel.mutate(m.id)}
                  disabled={activateModel.isPending}
                  className="flex items-center gap-1 rounded-md px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.06em] transition-all hover:brightness-110"
                  style={{
                    backgroundColor: "var(--color-accent-success)",
                    color: "var(--color-text-inverse)",
                  }}
                >
                  <Star size={10} />
                  Activate
                </button>
              )}
              {m.status === "active" && (
                <button
                  onClick={() => deactivateModel.mutate(m.id)}
                  disabled={deactivateModel.isPending}
                  className="flex items-center gap-1 rounded-md px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.06em] transition-all hover:brightness-110"
                  style={{
                    backgroundColor: "var(--color-glass-hover)",
                    color: "var(--color-text-secondary)",
                    border: "1px solid var(--color-glass-border)",
                  }}
                >
                  <Star size={10} />
                  Deactivate
                </button>
              )}
              <button
                onClick={() => {
                  if (confirm(`Delete model ${m.model_type} (${m.id.slice(0, 8)}...)?`)) {
                    deleteModel.mutate(m.id);
                  }
                }}
                className="flex items-center gap-1 rounded-md px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.06em] transition-all hover:brightness-110"
                style={{
                  backgroundColor: "var(--color-glass-hover)",
                  color: "var(--color-accent-danger)",
                  border: "1px solid var(--color-glass-border)",
                }}
              >
                <Trash2 size={10} />
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
