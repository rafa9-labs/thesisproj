import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import apiClient from "@/api/client";
import { useBulkDeleteModels } from "@/api/queries";
import { TagEditor } from "@/components/shared/TagEditor";
import { Box, Trash2, Power, PowerOff, Play, Pencil, Star } from "lucide-react";
import { SavedCommitteesPanel } from "@/pages/Committee/SavedCommitteesPanel";

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
  win_rate: number | null;
  max_drawdown: number | null;
  total_trades: number | null;
  sortino: number | null;
  train_start: string | null;
  train_end: string | null;
  feature_count: number | null;
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

const ENSEMBLE_TYPES = new Set([
  "meta_ensemble",
  "stacking_ensemble",
  "ensemble_adaptive_regime",
  "ensemble_cnn_lstm_xgboost",
]);

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
  const bulkDelete = useBulkDeleteModels();
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [modelTypeTab, setModelTypeTab] = useState<string>("all");
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [bulkError, setBulkError] = useState<string | null>(null);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [customNames, setCustomNames] = useState<Record<string, string>>({});
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const navigate = useNavigate();

  const typeTabs = useMemo(() => {
    const unique = new Set<string>();
    for (const m of models ?? []) {
      unique.add(m.model_type);
    }
    const sorted = [...unique].sort();
    const hasEnsemble = sorted.some((t) => ENSEMBLE_TYPES.has(t));
    return { sorted, hasEnsemble };
  }, [models]);

  const filtered = useMemo(() => {
    let list = models ?? [];
    if (modelTypeTab === "favorites") {
      list = list.filter((m) => favorites.has(m.id));
    } else if (modelTypeTab === "committee") {
      list = list.filter((m) => ENSEMBLE_TYPES.has(m.model_type));
    } else if (modelTypeTab !== "all") {
      list = list.filter((m) => m.model_type === modelTypeTab);
    }
    if (statusFilter === "active") {
      list = list.filter((m) => m.status === "active");
    } else if (statusFilter === "inactive") {
      list = list.filter((m) => m.status === "inactive");
    }
    return list;
  }, [models, modelTypeTab, statusFilter, favorites]);

  function toggleCheck(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setBulkError(null);
  }

  function toggleFavorite(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function startRename(id: string, currentType: string, e: React.MouseEvent) {
    e.stopPropagation();
    setEditingNameId(id);
    setEditValue(customNames[id] || currentType.toUpperCase());
  }

  function commitRename() {
    if (editingNameId && editValue.trim()) {
      setCustomNames((prev) => ({ ...prev, [editingNameId]: editValue.trim() }));
    }
    setEditingNameId(null);
  }

  function getDisplayName(m: DeployedModel): string {
    return customNames[m.id] || m.model_type.toUpperCase();
  }

  function clearChecks() {
    setCheckedIds(new Set());
    setBulkError(null);
  }

  function handleBulkDelete() {
    const ids = [...checkedIds];
    if (ids.length === 0) return;
    if (!confirm(`Delete ${ids.length} model(s)?`)) return;
    bulkDelete.mutate(ids, {
      onSuccess: () => clearChecks(),
      onError: (e: unknown) => setBulkError((e as Error).message),
    });
  }

  function renderCard(m: DeployedModel) {
    const isChecked = checkedIds.has(m.id);
    const isFavorite = favorites.has(m.id);
    const isActive = m.status === "active";
    const isEditing = editingNameId === m.id;

    return (
      <div
        key={m.id}
        role="button"
        tabIndex={0}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("button, input, label")) return;
          navigate(`/models/${m.id}`);
        }}
        className={`relative cursor-pointer rounded-lg border p-5 transition-all hover:border-cyan-500/50 ${
          isActive
            ? "border-cyan-500 bg-cyan-950/20 shadow-[0_0_14px_rgba(6,182,212,0.08)]"
            : "border-slate-800 bg-slate-900/40"
        }`}
      >
        {/* ── Header: Title, Edit, ACTIVE badge ── */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex min-w-0 items-center gap-2">
            {isEditing ? (
              <input
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onBlur={commitRename}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitRename();
                  if (e.key === "Escape") setEditingNameId(null);
                }}
                onClick={(e) => e.stopPropagation()}
                className="w-[180px] rounded border border-cyan-500 bg-slate-800 px-2 py-0.5 text-lg font-bold tracking-tight text-slate-100 outline-none"
                autoFocus
              />
            ) : (
              <>
                <span className="truncate text-lg font-bold tracking-tight text-slate-100">
                  {getDisplayName(m)}
                </span>
                <button
                  onClick={(e) => startRename(m.id, m.model_type, e)}
                  className="shrink-0 cursor-pointer text-slate-500 transition-colors hover:text-cyan-400"
                  aria-label="Rename model"
                >
                  <Pencil size={13} />
                </button>
              </>
            )}
            {isActive && (
              <span className="shrink-0 rounded bg-cyan-500/15 px-2 py-0.5 text-[9px] font-bold tracking-[0.06em] text-cyan-400 uppercase">
                ACTIVE
              </span>
            )}
            {m.missing_on_disk && <span className="text-[9px] text-rose-400">missing</span>}
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <button
              onClick={(e) => toggleFavorite(m.id, e)}
              className="cursor-pointer transition-colors"
              aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
            >
              <Star
                size={15}
                className={
                  isFavorite ? "fill-cyan-400 text-cyan-400" : "text-slate-600 hover:text-amber-400"
                }
              />
            </button>
            <label
              className="flex shrink-0 cursor-pointer items-center"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={isChecked}
                onChange={() => {
                  const fakeEvent = { stopPropagation: () => {} } as React.MouseEvent;
                  toggleCheck(m.id, fakeEvent);
                }}
                className="h-5 w-5 cursor-pointer rounded border-slate-600 bg-slate-900/50 transition-colors checked:border-cyan-500 checked:bg-cyan-500 focus:ring-0 focus:ring-offset-0"
              />
            </label>
          </div>
        </div>

        {/* ── Stats Micro-Grid ── */}
        <div className="mt-4 mb-6 grid grid-cols-2 gap-x-4 gap-y-3">
          <div className="flex flex-col">
            <span className="text-[10px] tracking-wider text-slate-500 uppercase">Sharpe</span>
            <span
              className={`font-mono text-sm ${
                m.best_sharpe != null
                  ? m.best_sharpe >= 0
                    ? "text-emerald-400"
                    : "text-rose-400"
                  : "text-slate-600"
              }`}
            >
              {m.best_sharpe != null
                ? (m.best_sharpe >= 0 ? "+" : "") + m.best_sharpe.toFixed(2)
                : "\u2014"}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] tracking-wider text-slate-500 uppercase">Return</span>
            <span
              className={`font-mono text-sm ${
                m.best_return != null
                  ? m.best_return >= 0
                    ? "text-emerald-400"
                    : "text-rose-400"
                  : "text-slate-600"
              }`}
            >
              {m.best_return != null
                ? (m.best_return >= 0 ? "+" : "") + m.best_return.toFixed(1) + "%"
                : "\u2014"}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] tracking-wider text-slate-500 uppercase">Win Rate</span>
            <span className="font-mono text-sm text-slate-200">
              {m.win_rate != null ? (m.win_rate * 100).toFixed(0) + "%" : "\u2014"}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] tracking-wider text-slate-500 uppercase">Trades</span>
            <span className="font-mono text-sm text-slate-200">
              {m.total_trades != null ? m.total_trades : "\u2014"}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] tracking-wider text-slate-500 uppercase">Max DD</span>
            <span
              className={`font-mono text-sm ${
                m.max_drawdown != null
                  ? m.max_drawdown < -10
                    ? "text-rose-400"
                    : "text-slate-200"
                  : "text-slate-600"
              }`}
            >
              {m.max_drawdown != null ? m.max_drawdown.toFixed(1) + "%" : "\u2014"}
            </span>
          </div>

          <div className="flex flex-col">
            <span className="text-[10px] tracking-wider text-slate-500 uppercase">Sortino</span>
            <span
              className={`font-mono text-sm ${
                m.sortino != null
                  ? m.sortino >= 0
                    ? "text-emerald-400"
                    : "text-rose-400"
                  : "text-slate-600"
              }`}
            >
              {m.sortino != null ? (m.sortino >= 0 ? "+" : "") + m.sortino.toFixed(2) : "\u2014"}
            </span>
          </div>
        </div>

        {/* ── Training Period ── */}
        <div className="mb-2 font-mono text-[10px] text-slate-500">
          {m.train_start && m.train_end
            ? `${m.train_start.slice(0, 10)} \u2192 ${m.train_end.slice(0, 10)}`
            : "Period: N/A"}
          {m.feature_count != null && (
            <span className="ml-3 text-slate-600">Feat: {m.feature_count}</span>
          )}
        </div>

        {/* ── Tags ── */}
        <div className="mb-2">
          <TagEditor
            tags={m.tags}
            onAdd={(tag) => updateTags.mutate({ id: m.id, action: "add", tag })}
            onRemove={(tag) => updateTags.mutate({ id: m.id, action: "remove", tag })}
          />
        </div>

        {/* ── Created Date ── */}
        <div className="mb-4 text-[9px] text-slate-600">{m.created_at?.slice(0, 10)}</div>

        {/* ── Action Buttons ── */}
        <div className="grid grid-cols-3 gap-2">
          {m.status !== "active" && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                activateModel.mutate(m.id);
              }}
              disabled={activateModel.isPending}
              className="flex items-center justify-center gap-1 rounded-md bg-emerald-600 px-2 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-white uppercase transition-all hover:brightness-110"
            >
              <Power size={10} /> Activate
            </button>
          )}
          {m.status === "active" && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                deactivateModel.mutate(m.id);
              }}
              disabled={deactivateModel.isPending}
              className="flex items-center justify-center gap-1 rounded-md border border-slate-700 bg-slate-800 px-2 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-slate-300 uppercase transition-all hover:brightness-110"
            >
              <PowerOff size={10} /> Deactivate
            </button>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              apiClient
                .post("/trading/paper/start", {
                  model_id: m.id,
                  model_type: m.model_type,
                  pair: "EURUSD",
                  timeframe: "H1",
                  initial_equity: 10000,
                  position_sizing: "fixed",
                })
                .then((r: { data: { session_id: string } }) => {
                  window.location.href = `/trading?session=${r.data.session_id}`;
                })
                .catch(console.error);
            }}
            className="flex items-center justify-center gap-1 rounded-md border border-cyan-500 bg-cyan-500/8 px-2 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-cyan-400 uppercase transition-all hover:brightness-110"
          >
            <Play size={10} /> Deploy
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (confirm(`Delete model ${m.model_type} (${m.id.slice(0, 8)}...)?`)) {
                deleteModel.mutate(m.id);
              }
            }}
            className="flex items-center justify-center gap-1 rounded-md border border-slate-700 bg-slate-800 px-2 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-rose-400 uppercase transition-all hover:brightness-110"
          >
            <Trash2 size={10} /> Delete
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-[0.1em] text-slate-100 uppercase">
            Deployed Models
          </h2>
          <p className="mt-1 text-[11px] text-slate-400">
            Models saved from backtest results. Activate one globally as your trading model. Check
            models to bulk delete.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {checkedIds.size > 0 && (
            <>
              <span className="font-mono text-[10px] whitespace-nowrap text-slate-400">
                {checkedIds.size} selected
              </span>
              <button
                onClick={handleBulkDelete}
                disabled={bulkDelete.isPending}
                className="flex items-center gap-1 rounded border border-rose-500/25 bg-rose-500/8 px-2.5 py-1 text-[10px] font-semibold tracking-[0.06em] whitespace-nowrap text-rose-400 uppercase hover:brightness-110"
              >
                <Trash2 size={9} /> Delete
              </button>
              <button
                onClick={clearChecks}
                className="rounded border border-slate-700 px-2 py-1 text-[9px] text-slate-400 hover:text-slate-200"
              >
                Clear
              </button>
            </>
          )}
          {(["all", "active", "inactive"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setStatusFilter(f)}
              className="rounded-md px-3 py-1 text-[10px] font-medium tracking-[0.06em] uppercase transition-all"
              style={{
                backgroundColor: statusFilter === f ? "rgba(6,182,212,0.2)" : "rgba(30,41,59,0.6)",
                color: statusFilter === f ? "rgb(34,211,238)" : "rgb(148,163,184)",
                border: "1px solid",
                borderColor: statusFilter === f ? "rgba(6,182,212,0.3)" : "rgb(51,65,85)",
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Primary tabs: model types */}
      <div className="-mb-2 flex flex-wrap items-center gap-1 overflow-x-auto">
        <button
          onClick={() => {
            setModelTypeTab("all");
            clearChecks();
          }}
          className={`shrink-0 rounded-t px-3 py-1.5 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all ${
            modelTypeTab === "all"
              ? "border-b-2 border-cyan-400 bg-slate-800/50 text-cyan-400"
              : "border-b-2 border-transparent text-slate-500 hover:text-slate-300"
          }`}
        >
          All Models
        </button>
        <button
          onClick={() => {
            setModelTypeTab("favorites");
            clearChecks();
          }}
          className={`flex shrink-0 items-center gap-1 rounded-t px-3 py-1.5 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all ${
            modelTypeTab === "favorites"
              ? "border-b-2 border-amber-400 bg-slate-800/50 text-amber-400"
              : "border-b-2 border-transparent text-slate-500 hover:text-slate-300"
          }`}
        >
          <Star size={10} className={modelTypeTab === "favorites" ? "fill-amber-400" : ""} />
          Favorites
          {favorites.size > 0 && (
            <span className="ml-0.5 rounded bg-slate-700 px-1 py-0 text-[8px] text-slate-300">
              {favorites.size}
            </span>
          )}
        </button>
        {typeTabs.sorted.map((type) => (
          <button
            key={type}
            onClick={() => {
              setModelTypeTab(type);
              clearChecks();
            }}
            className={`shrink-0 rounded-t px-3 py-1.5 text-[10px] font-medium tracking-[0.04em] uppercase transition-all ${
              modelTypeTab === type
                ? "border-b-2 border-cyan-400 bg-slate-800/50 text-cyan-400"
                : "border-b-2 border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            {type}
          </button>
        ))}
        {typeTabs.hasEnsemble && (
          <button
            onClick={() => {
              setModelTypeTab("committee");
              clearChecks();
            }}
            className={`shrink-0 rounded-t px-3 py-1.5 text-[10px] font-medium tracking-[0.04em] uppercase transition-all ${
              modelTypeTab === "committee"
                ? "border-b-2 border-amber-400 bg-slate-800/50 text-amber-400"
                : "border-b-2 border-transparent text-slate-500 hover:text-slate-300"
            }`}
          >
            Committee
          </button>
        )}
        <button
          onClick={() => {
            setModelTypeTab("saved-committees");
            clearChecks();
          }}
          className={`shrink-0 rounded-t px-3 py-1.5 text-[10px] font-medium tracking-[0.04em] uppercase transition-all ${
            modelTypeTab === "saved-committees"
              ? "border-b-2 border-purple-400 bg-slate-800/50 text-purple-400"
              : "border-b-2 border-transparent text-slate-500 hover:text-slate-300"
          }`}
        >
          Saved Committees
        </button>
      </div>

      {modelTypeTab === "saved-committees" ? (
        <SavedCommitteesPanel />
      ) : (
        <>
          {!isLoading && filtered.length === 0 && (
            <div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-slate-800 bg-slate-900/40 p-12">
              <Box size={40} className="text-slate-600" />
              <span className="text-xs text-slate-400">
                {models?.length
                  ? "No models match the current filters."
                  : "No deployed models yet. Run a backtest, then click Save Model on the Results page."}
              </span>
            </div>
          )}

          {bulkError && (
            <div className="rounded border border-rose-500/20 bg-rose-500/6 px-3 py-2 font-mono text-[10px] text-rose-400">
              {bulkError}
            </div>
          )}

          {/* Card grid */}
          <div
            className="grid gap-4"
            style={{ gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))" }}
          >
            {filtered.map(renderCard)}
          </div>
        </>
      )}
    </div>
  );
}
