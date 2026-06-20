import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import apiClient from "@/api/client";
import {
  useBulkDeleteModels,
  useSavedCommittees,
  useDeleteSavedCommittee,
} from "@/api/queries";
import { Box, Trash2, Play, Pencil, Star, Search } from "lucide-react";

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
  calmar_ratio?: number | null;
  profit_factor?: number | null;
  cagr?: number | null;
  overfit_score?: number | null;
  risk_level?: string | null;
  train_start: string | null;
  train_end: string | null;
  feature_count: number | null;
  seed: number | null;
  calibrate_method: string | null;
  consensus_model_count?: number;
  is_consensus?: boolean;
  full_cycle_job_id?: string | null;
  pair?: string;
  timeframe?: string;
  avg_sharpe?: number | null;
  avg_return?: number | null;
  trust_score?: number | null;
  regime_count?: number;
}

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

function useDeleteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.delete(`/models/deployed/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["deployed-models"] }),
  });
}

export function DeployedModelsPage() {
  const { data: models, isLoading: isLoadingModels } = useDeployedModels();
  const { data: savedData, isLoading: isLoadingSaved } = useSavedCommittees();
  const deleteModel = useDeleteModel();
  const bulkDelete = useBulkDeleteModels();
  const deleteCommittee = useDeleteSavedCommittee();
  const [searchQuery, setSearchQuery] = useState("");
  const [viewMode, setViewMode] = useState<"all" | "favorites">("all");
  const [modelType, setModelType] = useState<"singular" | "committee" | "all">("singular");
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [bulkError, setBulkError] = useState<string | null>(null);
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [customNames, setCustomNames] = useState<Record<string, string>>({});
  const [editingNameId, setEditingNameId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const navigate = useNavigate();

  const modelsFromCommittees: DeployedModel[] = useMemo(() => {
    if (!savedData?.committees) return [];
    return savedData.committees
      .filter((c) => {
        const sharpe = c.avg_sharpe;
        return sharpe != null && isFinite(sharpe) && sharpe > -100;
      })
      .map((c) => {
        return {
          id: c.id,
          model_type: "consensus",
          best_sharpe: c.avg_sharpe,
          best_return: c.avg_return ?? null,
          created_at: c.created_at,
          status: c.is_active ? "active" : "inactive",
          tags: c.tags || [],
          parent_job_id: null,
          missing_on_disk: false,
          win_rate: c.win_rate ?? null,
          max_drawdown: c.max_drawdown ?? null,
          total_trades: c.total_trades ?? null,
          sortino: c.sortino ?? null,
          train_start: null,
          train_end: null,
          feature_count: null,
          seed: null,
          calibrate_method: null,
          consensus_model_count: c.consensus_model_count ?? 0,
          is_consensus: true,
          full_cycle_job_id: c.full_cycle_job_id,
          pair: c.pair,
          timeframe: c.timeframe,
          avg_sharpe: c.avg_sharpe,
          avg_return: c.avg_return ?? null,
          trust_score: c.trust_score ?? null,
          regime_count: c.regime_count ?? 0,
        } as DeployedModel;
      });
  }, [savedData]);

  const allDeployed: DeployedModel[] = useMemo(
    () => (Array.isArray(models) ? models : []),
    [models],
  );
  const modelsSafe: DeployedModel[] = useMemo(
    () => [...allDeployed, ...modelsFromCommittees],
    [allDeployed, modelsFromCommittees],
  );

  const filtered = useMemo(() => {
    let list = modelsSafe;
    if (modelType === "singular") {
      list = list.filter((m) => !m.is_consensus);
    } else if (modelType === "committee") {
      list = list.filter((m) => m.is_consensus);
    }
    if (viewMode === "favorites") {
      list = list.filter((m) => favorites.has(m.id));
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (m) =>
          (customNames[m.id] || m.model_type).toLowerCase().includes(q) ||
          m.tags.some((t) => t.toLowerCase().includes(q)) ||
          (m.pair && m.pair.toLowerCase().includes(q)),
      );
    }
    return list;
  }, [modelsSafe, modelType, viewMode, favorites, searchQuery, customNames]);

  function toggleCheck(id: string) {
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
    if (m.is_consensus) {
      return `Consensus +${m.consensus_model_count || 0}`;
    }
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

  const fmtPct = (v: number | null | undefined, decimals = 1) => {
    if (v == null) return "\u2014";
    return (v >= 0 ? "+" : "") + v.toFixed(decimals) + "%";
  };

  const fmtNum = (v: number | null | undefined, decimals = 2) => {
    if (v == null) return "\u2014";
    return (v >= 0 ? "+" : "") + v.toFixed(decimals);
  };

  function renderRow(m: DeployedModel) {
    const isChecked = checkedIds.has(m.id);
    const isFavorite = favorites.has(m.id);
    const isEditing = editingNameId === m.id;

    return (
      <div
        key={m.id}
        role={m.is_consensus ? undefined : "button"}
        tabIndex={m.is_consensus ? undefined : 0}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("button, input, label")) return;
          if (m.is_consensus) return;
          if (e.ctrlKey || e.metaKey) {
            toggleCheck(m.id);
            return;
          }
          navigate(`/models/${m.id}`);
        }}
        className={[
          "flex flex-row items-center justify-between p-4 rounded-lg transition-all border",
          m.is_consensus
            ? "border-(--color-glass-border) bg-(--color-glass) border-l-2 border-l-amber-500/40 cursor-default"
            : "border-(--color-glass-border) bg-(--color-glass) backdrop-blur-[12px] cursor-pointer hover:border-(--color-border-active)",
        ].join(" ")}
      >
        {/* ── LEFT: Identity ── */}
        <div className="flex items-center gap-3 min-w-0 w-[240px] shrink-0">
          {!m.is_consensus && (
            <label
              className="flex shrink-0 cursor-pointer items-center"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={isChecked}
                onChange={() => toggleCheck(m.id)}
                className="h-4 w-4 cursor-pointer rounded border-(--color-glass-border) bg-(--color-input-bg) transition-colors checked:border-cyan-500 checked:bg-cyan-500 focus:ring-0 focus:ring-offset-0"
              />
            </label>
          )}
          {!m.is_consensus && m.risk_level && (
            <span
              className="shrink-0 rounded-full"
              style={{
                width: 6,
                height: 6,
                backgroundColor:
                  m.risk_level === "low" ? "#34d399" :
                  m.risk_level === "medium" ? "#fbbf24" :
                  m.risk_level === "high" ? "#f87171" : "#475569",
              }}
              title={`Overfit risk: ${m.risk_level} (${m.overfit_score ?? "?"})`}
            />
          )}
          <button
            onClick={(e) => toggleFavorite(m.id, e)}
            className="shrink-0 cursor-pointer transition-colors"
            aria-label={isFavorite ? "Remove from favorites" : "Add to favorites"}
          >
            <Star
              size={14}
              className={
                isFavorite ? "fill-amber-400 text-amber-400" : "text-(--color-text-dim) hover:text-amber-400"
              }
            />
          </button>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
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
                  className="w-[140px] rounded border border-cyan-500 bg-(--color-glass) px-2 py-0.5 text-sm font-bold text-(--color-text-primary) outline-none"
                  autoFocus
                />
              ) : (
                <span className="truncate text-sm font-bold tracking-tight text-(--color-text-primary)">
                  {getDisplayName(m)}
                </span>
              )}
              {!m.is_consensus && (
                <button
                  onClick={(e) => startRename(m.id, m.model_type, e)}
                  className="shrink-0 cursor-pointer text-(--color-text-muted) transition-colors hover:text-cyan-400"
                  aria-label="Rename model"
                >
                  <Pencil size={11} />
                </button>
              )}
              {m.is_consensus && (
                <span className="shrink-0 rounded bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-bold text-amber-400 uppercase">
                  CONSENSUS
                </span>
              )}
            </div>
            <div className="mt-0.5 text-[10px] text-(--color-text-muted)">
              {m.is_consensus
                ? `${m.pair || "EURUSD"} / ${m.timeframe || "H1"}`
                : m.created_at?.slice(0, 10)}
            </div>
          </div>
        </div>

        {/* ── MIDDLE: Metrics ── */}
        <div className="flex flex-1 items-center gap-4 px-4 min-w-0 justify-center flex-wrap">
          {m.is_consensus ? (
            <>
              <Metric label="Sharpe" value={fmtNum(m.best_sharpe)} color={m.best_sharpe != null && m.best_sharpe >= 0 ? "text-emerald-400" : "text-rose-400"} />
              <Metric label="Regimes" value={String(m.regime_count ?? 0)} color="text-(--color-text-secondary)" />
              <Metric label="Models" value={String(m.consensus_model_count ?? 0)} color="text-(--color-text-secondary)" />
              <Metric label="Trust" value={m.trust_score != null ? m.trust_score.toFixed(2) : "\u2014"} color="text-(--color-text-secondary)" />
            </>
          ) : (
            <>
              <Metric
                label="Sharpe"
                value={m.best_sharpe != null ? fmtNum(m.best_sharpe) : "\u2014"}
                color={m.best_sharpe != null ? m.best_sharpe >= 0 ? "text-emerald-400" : "text-rose-400" : "text-(--color-text-dim)"}
              />
              <Metric
                label="Return"
                value={m.best_return != null ? fmtPct(m.best_return, 1) : "\u2014"}
                color={m.best_return != null ? m.best_return >= 0 ? "text-emerald-400" : "text-rose-400" : "text-(--color-text-dim)"}
              />
              <Metric
                label="Win Rate"
                value={m.win_rate != null ? (m.win_rate * 100).toFixed(0) + "%" : "\u2014"}
                color="text-(--color-text-secondary)"
              />
              <Metric
                label="Max DD"
                value={m.max_drawdown != null ? fmtPct(m.max_drawdown, 1) : "\u2014"}
                color={m.max_drawdown != null ? m.max_drawdown < -10 ? "text-rose-400" : "text-(--color-text-secondary)" : "text-(--color-text-dim)"}
              />
              <Metric label="Trades" value={m.total_trades != null ? String(m.total_trades) : "\u2014"} color="text-(--color-text-secondary)" />
              {m.risk_level ? (
                <Metric
                  label="Overfit"
                  value={m.risk_level.toUpperCase()}
                  color={
                    m.risk_level === "low" ? "text-emerald-400" :
                    m.risk_level === "medium" ? "text-amber-400" :
                    "text-rose-400"
                  }
                />
              ) : (
                <Metric label="Overfit" value={"\u2014"} color="text-(--color-text-dim)" />
              )}
              <div className="flex flex-col items-center gap-0.5">
                <span className="text-[9px] font-medium tracking-[0.04em] text-(--color-text-muted) uppercase">Info</span>
                <span className="font-mono text-[9px] text-(--color-text-muted) text-center leading-relaxed max-w-[160px] truncate">
                  {[
                    m.seed != null && `S:${m.seed}`,
                    m.calibrate_method && `Cal:${m.calibrate_method}`,
                    m.train_start && m.train_end && `${m.train_start.slice(0, 7)}→${m.train_end.slice(0, 7)}`,
                  ].filter(Boolean).join(" · ") || "\u2014"}
                </span>
              </div>
            </>
          )}
        </div>

        {/* ── RIGHT: Actions ── */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (m.is_consensus) {
                apiClient
                  .post("/trading/live/committee/start", {
                    pair: m.pair || "EURUSD",
                    timeframe: m.timeframe || "H1",
                    initial_equity: 10000,
                    confidence_threshold: 0.55,
                    mode: "paper",
                    full_cycle_job_id: m.full_cycle_job_id,
                  })
                  .then((r: { data: { session_id: string } }) => {
                    window.location.href = `/trading?session=${r.data.session_id}`;
                  })
                  .catch(console.error);
              } else {
                navigate(`/trading?modelId=${m.id}`);
              }
            }}
            className="flex items-center gap-1 rounded-md bg-cyan-600 px-3 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-white uppercase transition-all hover:brightness-110"
          >
            <Play size={10} /> {m.is_consensus ? "Deploy Committee" : "Deploy Model"}
          </button>
          {m.is_consensus ? (
            <button
              onClick={(e) => {
                e.stopPropagation();
                if (confirm(`Delete Consensus +${m.consensus_model_count} (${m.id.slice(0, 8)}...)?`)) {
                  deleteCommittee.mutate(m.id);
                }
              }}
              className="flex items-center gap-1 rounded-md border border-(--color-glass-border) bg-(--color-glass) px-3 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-rose-400 uppercase transition-all hover:brightness-110"
            >
              <Trash2 size={10} /> Delete
            </button>
          ) : (
            <>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  navigate(`/models/${m.id}`);
                }}
                className="flex items-center gap-1 rounded-md border border-(--color-glass-border) bg-(--color-glass) px-3 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-(--color-text-secondary) uppercase transition-all hover:brightness-110"
              >
                Details
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Delete model ${m.model_type} (${m.id.slice(0, 8)}...)?`)) {
                    deleteModel.mutate(m.id);
                  }
                }}
                className="flex items-center gap-1 rounded-md border border-(--color-glass-border) bg-(--color-glass) px-2 py-1.5 text-[10px] font-semibold tracking-[0.04em] text-rose-400 uppercase transition-all hover:brightness-110"
              >
                <Trash2 size={10} />
              </button>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 p-6">
      {/* ── Unified Control Bar ── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Search */}
          <div className="relative">
            <Search
              size={14}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-(--color-text-muted)"
            />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search models..."
              className="w-[220px] rounded-md border border-(--color-glass-border) bg-(--color-glass) py-1.5 pl-8 pr-3 text-[12px] text-(--color-text-secondary) outline-none transition placeholder:text-(--color-text-dim) focus:border-(--color-border-active)"
            />
          </div>

          {/* Bulk actions (visible when checked) */}
          {checkedIds.size > 0 && (
            <>
              <span className="font-mono text-[10px] whitespace-nowrap text-(--color-text-muted)">
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
                className="rounded border border-(--color-glass-border) px-2 py-1 text-[9px] text-(--color-text-muted) hover:text-(--color-text-secondary)"
              >
                Clear
              </button>
            </>
          )}
          {bulkError && (
            <span className="font-mono text-[10px] text-rose-400">{bulkError}</span>
          )}
        </div>

        {/* View mode pill toggle */}
        <div className="flex items-center gap-3">
          {/* Model type selector */}
          <div className="flex items-center rounded-lg bg-(--color-glass) p-0.5 border border-(--color-glass-border)">
            <button
              onClick={() => { setModelType("singular"); clearChecks(); }}
              className={`rounded-md px-4 py-1.5 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all ${
                modelType === "singular"
                  ? "bg-cyan-500/20 text-cyan-400"
                  : "text-(--color-text-muted) hover:text-(--color-text-secondary)"
              }`}
            >
              Singular
            </button>
            <button
              onClick={() => { setModelType("committee"); clearChecks(); }}
              className={`rounded-md px-4 py-1.5 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all ${
                modelType === "committee"
                  ? "bg-amber-500/20 text-amber-400"
                  : "text-(--color-text-muted) hover:text-(--color-text-secondary)"
              }`}
            >
              Committee
            </button>
            <button
              onClick={() => setModelType("all")}
              className={`rounded-md px-3 py-1.5 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all ${
                modelType === "all"
                  ? "bg-(--color-glass-hover) text-(--color-text-secondary)"
                  : "text-(--color-text-muted) hover:text-(--color-text-secondary)"
              }`}
            >
              All
            </button>
          </div>

          {/* Favorites pill */}
          <div className="flex items-center rounded-lg bg-(--color-glass) p-0.5 border border-(--color-glass-border)">
          <button
            onClick={() => setViewMode("all")}
            className={`rounded-md px-4 py-1.5 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all ${
              viewMode === "all"
                ? "bg-cyan-500/20 text-cyan-400"
                : "text-(--color-text-muted) hover:text-(--color-text-secondary)"
            }`}
          >
            All
          </button>
          <button
            onClick={() => {
              setViewMode("favorites");
              clearChecks();
            }}
            className={`flex items-center gap-1.5 rounded-md px-4 py-1.5 text-[10px] font-semibold tracking-[0.06em] uppercase transition-all ${
              viewMode === "favorites"
                ? "bg-amber-500/20 text-amber-400"
                : "text-(--color-text-muted) hover:text-(--color-text-secondary)"
            }`}
          >
            <Star size={10} className={viewMode === "favorites" ? "fill-amber-400" : ""} />
            Favorites
            {favorites.size > 0 && (
              <span className="rounded bg-(--color-glass) px-1 py-0 text-[8px] text-(--color-text-secondary)">
                {favorites.size}
              </span>
            )}
          </button>
        </div>
      </div>
    </div>

      {/* ── Empty State ── */}
      {!isLoadingModels && !isLoadingSaved && filtered.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-12">
          <Box size={40} className="text-(--color-text-dim)" />
          <span className="text-xs text-(--color-text-muted)">
            {modelsSafe.length > 0
              ? "No models match the current filters."
              : "No deployed models yet. Run a backtest, then click Save Model on the Results page."}
          </span>
        </div>
      )}

      {/* ── Model Rows ── */}
      <div className="flex flex-col gap-2">
        {filtered.map(renderRow)}
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className="text-[9px] font-medium tracking-[0.04em] text-(--color-text-muted) uppercase">
        {label}
      </span>
      <span className={`font-mono text-[12px] font-semibold ${color}`}>{value}</span>
    </div>
  );
}
