import { cn } from "@/lib/utils";
import { useState, useCallback } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Trash2,
  Download,
  Search,
  ChevronUp,
  ChevronDown,
  RefreshCw,
  Clock,
  MoreHorizontal,
  FileText,
} from "lucide-react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useResultsHistory, useBatchDeleteJobs, useRerunBacktest, useDeleteJob } from "@/api/queries";
import { formatRelativeTime, formatPercent } from "@/lib/formatters";
import { TabBar } from "@/components/shared/TabBar";
import { RunHistoryTable } from "@/pages/Committee/RunHistoryTable";
import type { BacktestSummaryItem } from "@/api/schemas";

// ── helpers ─────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null) return "—";
  return v.toFixed(digits);
}

function signColor(v: number | null | undefined): string {
  if (v == null) return "var(--color-text-muted)";
  return v >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)";
}

function StatusPill({ status }: { status: string }) {
  const map: Record<string, { bg: string; text: string; label: string }> = {
    completed: { bg: "rgba(8,153,129,0.15)", text: "var(--color-accent-success)", label: "done" },
    running: { bg: "rgba(0,229,255,0.10)", text: "var(--color-brand)", label: "running" },
    failed: { bg: "rgba(242,54,69,0.15)", text: "var(--color-accent-danger)", label: "failed" },
    pending: { bg: "rgba(245,158,11,0.12)", text: "var(--color-accent-warning)", label: "pending" },
    queued: { bg: "rgba(245,158,11,0.12)", text: "var(--color-accent-warning)", label: "queued" },
  };
  const s = map[status] ?? {
    bg: "rgba(255,255,255,0.05)",
    text: "var(--color-text-muted)",
    label: status,
  };
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[9px] font-bold tracking-wider uppercase"
      style={{ backgroundColor: s.bg, color: s.text }}
    >
      {s.label}
    </span>
  );
}

function ModelBadge({ model }: { model: string }) {
  const colorMap: Record<string, string> = {
    logistic: "var(--color-accent-classical)",
    xgboost: "var(--color-accent-classical)",
    svm: "var(--color-accent-classical)",
    random_forest: "var(--color-accent-classical)",
    decision_tree: "var(--color-accent-classical)",
    lightgbm: "var(--color-accent-classical)",
    catboost: "var(--color-accent-classical)",
    lstm: "var(--color-accent-deep)",
    cnn: "var(--color-accent-deep)",
    transformer: "var(--color-accent-deep)",
    gru: "var(--color-accent-deep)",
    gru_lstm: "var(--color-accent-deep)",
    dqn: "var(--color-accent-rl)",
    ensemble: "var(--color-accent-ensemble)",
  };
  const key = Object.keys(colorMap).find((k) => model.toLowerCase().includes(k));
  const color = key ? colorMap[key] : "var(--color-text-muted)";
  return (
    <span
      className="mr-0.5 inline-flex items-center rounded px-1.5 py-0.5 font-mono text-[9px] font-medium tracking-wider uppercase"
      style={{ backgroundColor: `${color}14`, border: `1px solid ${color}30` }}
    >
      {model}
    </span>
  );
}

// ── sort header ──────────────────────────────────────────────────────────────

function SortHeader({
  label,
  active,
  direction,
  onClick,
  align = "left",
}: {
  label: string;
  active: boolean;
  direction: "asc" | "desc";
  onClick: () => void;
  align?: "left" | "right";
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 text-[10px] font-medium tracking-[0.1em] uppercase transition-colors hover:text-[var(--color-text-primary)]",
        align === "right" && "ml-auto",
      )}
      style={{ color: active ? "var(--color-brand)" : "var(--color-text-muted)" }}
    >
      {align === "right" &&
        active &&
        (direction === "asc" ? <ChevronUp size={9} /> : <ChevronDown size={9} />)}
      {label}
      {align === "left" &&
        active &&
        (direction === "asc" ? <ChevronUp size={9} /> : <ChevronDown size={9} />)}
    </button>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export function ResultsHistoryPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const initTab = searchParams.get("tab") === "committee" ? "committee" : "backtest";
  const batchDelete = useBatchDeleteJobs();
  const rerun = useRerunBacktest();
  const deleteJob = useDeleteJob();

  const [historyTab, setHistoryTab] = useState(initTab);
  const [search, setSearch] = useState("");
  const [pairFilter, setPairFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("completed");
  const [sortBy, setSortBy] = useState("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [hideLegacy, setHideLegacy] = useState(true);

  const { data, isLoading } = useResultsHistory({
    pair: pairFilter,
    sort_by: sortBy,
    sort_order: sortOrder,
    limit: 100,
    status: statusFilter,
  });

  const results = data?.results ?? [];
  const filtered = (
    search
      ? results.filter(
          (r) =>
            r.job_id.toLowerCase().includes(search.toLowerCase()) ||
            r.pair?.toLowerCase().includes(search.toLowerCase()) ||
            r.models?.some((m) => m.toLowerCase().includes(search.toLowerCase())),
        )
      : results
  ).filter((r) => !(hideLegacy && r.legacy));

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    if (selected.size === filtered.length) setSelected(new Set());
    else setSelected(new Set(filtered.map((r) => r.job_id)));
  };

  const handleSort = (col: string) => {
    if (sortBy === col) setSortOrder((p) => (p === "asc" ? "desc" : "asc"));
    else {
      setSortBy(col);
      setSortOrder("desc");
    }
  };

  const exportCSV = useCallback(() => {
    const rows = filtered.filter((r) => selected.has(r.job_id));
    if (!rows.length) return;
    const csv = [
      "job_id,created_at,pair,timeframe,models,status,error,sharpe,return_pct,win_rate,max_dd_pct,trades",
      ...rows.map((r) =>
        [
          r.job_id,
          r.created_at,
          r.pair,
          r.timeframe,
          r.models.join(";"),
          r.status,
          (r.error ?? "").replace(/"/g, '""'),
          r.sharpe ?? "",
          r.total_return_pct ?? "",
          r.win_rate ?? "",
          r.max_drawdown_pct ?? "",
          r.total_trades ?? "",
        ].join(","),
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "backtest_results.csv";
    a.click();
    URL.revokeObjectURL(url);
  }, [filtered, selected]);

  const handleBatchDelete = useCallback(() => {
    const ids = [...selected];
    if (!ids.length) return;
    batchDelete.mutate(ids, {
      onSuccess: () => setSelected(new Set()),
    });
  }, [selected, batchDelete]);

  const handleRerun = useCallback(
    (row: BacktestSummaryItem) => {
      rerun.mutate(row.job_id);
    },
    [rerun],
  );

  const handleDeleteSingle = useCallback(
    (jobId: string) => {
      if (!confirm("Are you sure you want to delete this study?")) return;
      deleteJob.mutate(jobId);
    },
    [deleteJob],
  );

  const PAIRS = [...new Set(results.map((r) => r.pair).filter(Boolean))].sort();

  const completed = results.filter((r) => r.status === "completed");
  const avgSharpe = completed.length
    ? completed.reduce((a, b) => a + (b.sharpe ?? 0), 0) / completed.length
    : null;
  const bestSharpe = completed.length
    ? Math.max(...completed.map((r) => r.sharpe ?? -Infinity))
    : null;
  const pairsCount = [...new Set(results.map((r) => r.pair).filter(Boolean))].length;

  const HISTORY_TABS = [
    { key: "backtest", label: "Backtest Runs" },
    { key: "committee", label: "Committee Runs" },
  ];

  return (
    <div className="flex flex-col gap-3 py-6">
      {/* Tab bar for switching between Backtest and Committee history */}
      <TabBar tabs={HISTORY_TABS} activeTab={historyTab} onTabChange={(tab) => {
        setHistoryTab(tab);
        setSearchParams(tab === "committee" ? { tab: "committee" } : {}, { replace: true });
      }} />

      {historyTab === "committee" ? (
        <RunHistoryTable
          activeJobId={null}
          onSelect={(jobId: string) => {
            navigate(`/results/${jobId}?type=committee`);
          }}
        />
      ) : (
        <>
          {/* ── merged action bar: summary stats + filters + export ── */}
          <div className="flex flex-wrap items-center gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-surface) px-3 py-2">
        {!isLoading && results.length > 0 && (
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] tabular-nums text-(--color-text-muted)">
              <span className="text-(--color-text-primary)">{results.length}</span> runs
            </span>
            <span className="font-mono text-[10px] tabular-nums text-(--color-text-muted)">
              <span className="text-(--color-accent-success)">{completed.length}</span> done
            </span>
            <span className="font-mono text-[10px] tabular-nums text-(--color-text-muted)">
              <span className="text-(--color-text-primary)">{pairsCount}</span> pairs
            </span>
            <span className="font-mono text-[10px] tabular-nums" style={{ color: avgSharpe != null ? signColor(avgSharpe) : "var(--color-text-muted)" }}>
              Sharpe {avgSharpe != null ? avgSharpe.toFixed(2) : "—"}
            </span>
            <span className="font-mono text-[10px] tabular-nums" style={{ color: bestSharpe != null && isFinite(bestSharpe) ? signColor(bestSharpe) : "var(--color-text-muted)" }}>
              Best {bestSharpe != null && isFinite(bestSharpe) ? bestSharpe.toFixed(2) : "—"}
            </span>
          </div>
        )}
        {!isLoading && results.length > 0 && (
          <span className="h-4 w-px bg-(--color-glass-border)" />
        )}

        <div className="relative flex flex-1 items-center rounded-sm border border-(--color-glass-border) bg-(--color-input-bg) transition-colors focus-within:border-[var(--color-brand)]" style={{ maxWidth: 240 }}>
          <Search size={14} className="pointer-events-none absolute left-2.5 shrink-0 text-(--color-text-muted)" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search…"
            className="w-full bg-transparent py-1.5 pr-2.5 pl-8 font-mono text-[11px] text-(--color-text-primary) focus:outline-none"
          />
          {search && (
            <button onClick={() => setSearch("")} className="absolute right-1.5 rounded px-1 py-0.5 text-[10px] text-(--color-text-muted)" aria-label="Clear search">
              ✕
            </button>
          )}
        </div>

        <div className="relative rounded-sm border border-(--color-glass-border) bg-(--color-input-bg) transition-colors">
          <select
            value={pairFilter}
            onChange={(e) => setPairFilter(e.target.value)}
            className="cursor-pointer appearance-none bg-transparent py-1.5 pr-7 pl-2.5 font-mono text-[11px] text-(--color-text-primary) focus:outline-none"
          >
            <option value="">All Pairs</option>
            {PAIRS.map((p) => (<option key={p} value={p}>{p}</option>))}
          </select>
          <ChevronDown size={10} className="pointer-events-none absolute top-1/2 right-2 -translate-y-1/2 text-(--color-text-muted)" />
        </div>

        <div className="flex items-center gap-0.5 rounded-sm border border-(--color-glass-border) bg-(--color-input-bg) p-0.5">
          {(["completed", "failed"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className="rounded-sm px-2.5 py-1 text-[10px] font-medium capitalize transition-all"
              style={{
                backgroundColor: statusFilter === s ? "var(--color-brand-glow)" : "transparent",
                color: statusFilter === s ? "var(--color-brand)" : "var(--color-text-dim)",
              }}
            >
              {s}
            </button>
          ))}
        </div>

        {selected.size > 0 && (
          <div className="flex items-center gap-1.5">
            <button
              onClick={exportCSV}
              className="flex items-center gap-1.5 rounded-sm border border-(--color-glass-border) px-2.5 py-1 text-[10px] font-medium text-(--color-text-secondary) transition-colors hover:bg-[var(--color-glass-hover)]"
            >
              <Download size={12} />
              Export
            </button>
            <button
              onClick={handleBatchDelete}
              disabled={batchDelete.isPending}
              className="flex items-center gap-1.5 rounded-sm border border-[rgba(242,54,69,0.3)] px-2.5 py-1 text-[10px] font-medium text-[var(--color-accent-danger)] transition-colors hover:bg-[rgba(242,54,69,0.1)] disabled:opacity-50"
            >
              <Trash2 size={12} />
              Delete {selected.size}
            </button>
          </div>
        )}

        <button
          onClick={() => setHideLegacy((v) => !v)}
          className="rounded-sm border border-(--color-glass-border) px-2.5 py-1 text-[10px] font-medium transition-colors hover:bg-[var(--color-glass-hover)]"
          title="Legacy results were computed with inflated short-window Sharpe annualization"
          style={{ color: hideLegacy ? "var(--color-text-dim)" : "var(--color-accent-warning)" }}
        >
          {hideLegacy ? "Show legacy" : "Hide legacy"}
        </button>
      </div>

      {/* ── table ── */}
      <div className="overflow-hidden rounded-sm border border-(--color-glass-border)">
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          {/* thead */}
          <thead>
            <tr className="border-b border-(--color-glass-border) bg-(--color-surface)">
              <th className="w-9 px-4 py-3">
                <input
                  type="checkbox"
                  checked={selected.size === filtered.length && filtered.length > 0}
                  onChange={selectAll}
                  className="cursor-pointer accent-[var(--color-brand)]"
                />
              </th>
              <th className="px-4 py-3 text-left">
                <SortHeader
                  label="Date"
                  active={sortBy === "created_at"}
                  direction={sortOrder}
                  onClick={() => handleSort("created_at")}
                />
              </th>
              <th className="px-4 py-3 text-left">
                <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  Pair / TF
                </span>
              </th>
              <th className="hidden md:table-cell px-4 py-3 text-left">
                <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  Models
                </span>
              </th>
              <th className="hidden sm:table-cell px-4 py-3 text-left">
                <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  Status
                </span>
              </th>
              <th className="hidden md:table-cell px-4 py-3 text-right">
                <SortHeader
                  label="Sharpe"
                  active={sortBy === "sharpe"}
                  direction={sortOrder}
                  onClick={() => handleSort("sharpe")}
                  align="right"
                />
              </th>
              <th className="px-4 py-3 text-right">
                <SortHeader
                  label="Return"
                  active={sortBy === "total_return_pct"}
                  direction={sortOrder}
                  onClick={() => handleSort("total_return_pct")}
                  align="right"
                />
              </th>
              <th className="hidden lg:table-cell px-4 py-3 text-right">
                <SortHeader
                  label="Win %"
                  active={sortBy === "win_rate"}
                  direction={sortOrder}
                  onClick={() => handleSort("win_rate")}
                  align="right"
                />
              </th>
              <th className="hidden lg:table-cell px-4 py-3 text-right">
                <SortHeader
                  label="Max DD"
                  active={sortBy === "max_drawdown_pct"}
                  direction={sortOrder}
                  onClick={() => handleSort("max_drawdown_pct")}
                  align="right"
                />
              </th>
              <th className="hidden lg:table-cell px-4 py-3 text-right">
                <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  Trades
                </span>
              </th>
              <th className="px-3 py-3 text-right">
                <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  &nbsp;
                </span>
              </th>
            </tr>
          </thead>

          {/* tbody */}
          <tbody>
            {isLoading ? (
              /* skeleton rows */
              Array.from({ length: 4 }).map((_, i) => (
                <tr key={i} className="border-b border-(--color-glass-border)">
                  {Array.from({ length: 11 }).map((_, j) => (
                    <td key={j} className="px-4 py-3">
                      <div
                        className="h-3 animate-pulse rounded bg-(--color-elevated)"
                        style={{ width: j === 0 ? 16 : j === 3 ? 120 : 60 }}
                      />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={11} className="px-4 py-16 text-center text-(--color-text-muted)">
                  <div className="flex flex-col items-center gap-2">
                    <Clock size={32} className="text-(--color-glass-border)" />
                    <span className="text-xs">
                      No backtests found. Run one from the Backtest Setup tab.
                    </span>
                  </div>
                </td>
              </tr>
            ) : (
              filtered.map((row) => {
                const isSel = selected.has(row.job_id);
                return (
                  <tr
                    key={`${row.job_id}-${row.models.join("-")}`}
                    onClick={() => navigate(`/results/${row.job_id}`)}
                    className={cn(
                      "group cursor-pointer border-b border-(--color-glass-border) transition-colors duration-100 hover:bg-(--color-glass-hover)",
                      isSel && "bg-[rgba(0,229,255,0.04)]",
                    )}
                  >
                    {/* checkbox */}
                    <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={isSel}
                        onChange={() => toggleSelect(row.job_id)}
                        className="cursor-pointer accent-[var(--color-brand)]"
                      />
                    </td>

                    {/* date */}
                    <td className="px-4 py-3 font-mono whitespace-nowrap text-(--color-text-muted)">
                      <div className="flex items-center gap-1.5">
                        <Clock size={11} className="text-(--color-glass-border)" />
                        {formatRelativeTime(row.created_at)}
                      </div>
                    </td>

                    {/* pair / timeframe */}
                    <td className="px-4 py-3">
                      <div className="flex flex-col gap-0.5">
                        <span className="font-mono text-[12px] font-semibold text-(--color-text-primary)">
                          {row.pair}
                          {row.legacy && (
                            <span
                              className="ml-1.5 rounded-sm border border-[rgba(245,158,11,0.4)] px-1 py-0.5 align-middle font-mono text-[8px] tracking-wider text-(--color-accent-warning) uppercase"
                              title="Computed with inflated short-window Sharpe annualization (legacy metric format)"
                            >
                              Legacy
                            </span>
                          )}
                        </span>
                        {row.timeframe && (
                          <span className="font-mono text-[9px] text-(--color-text-muted)">
                            {row.timeframe}
                          </span>
                        )}
                      </div>
                    </td>

                    {/* models */}
                    <td className="hidden md:table-cell max-w-[220px] px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {row.models.map((m) => (
                          <ModelBadge key={m} model={m} />
                        ))}
                      </div>
                    </td>

                    {/* status */}
                    <td
                      className="hidden sm:table-cell px-4 py-3"
                      title={row.status === "failed" ? (row.error ?? "Job failed") : undefined}
                    >
                      <StatusPill status={row.status} />
                    </td>

                    {/* sharpe */}
                    <td
                      className="hidden md:table-cell px-4 py-3 text-right font-mono tabular-nums"
                      style={{ color: signColor(row.sharpe) }}
                    >
                      {fmt(row.sharpe)}
                    </td>

                    {/* return */}
                    <td
                      className="px-4 py-3 text-right font-mono tabular-nums"
                      style={{ color: signColor(row.total_return_pct) }}
                    >
                      {row.total_return_pct != null ? formatPercent(row.total_return_pct) : "—"}
                    </td>

                    {/* win rate */}
                    <td className="hidden lg:table-cell px-4 py-3 text-right font-mono text-(--color-text-primary) tabular-nums">
                      {row.win_rate != null ? formatPercent(row.win_rate) : "—"}
                    </td>

                    {/* max dd */}
                    <td className="hidden lg:table-cell px-4 py-3 text-right font-mono text-(--color-accent-danger) tabular-nums">
                      {row.max_drawdown_pct != null ? formatPercent(row.max_drawdown_pct) : "—"}
                    </td>

                    {/* trades */}
                    <td className="hidden lg:table-cell px-4 py-3 text-right font-mono text-(--color-text-secondary) tabular-nums">
                      {row.total_trades ?? "—"}
                    </td>

                    {/* actions — kebab menu */}
                    <td className="px-3 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                      <DropdownMenu.Root>
                        <DropdownMenu.Trigger asChild>
                          <button
                            aria-label="Row actions"
                            className="rounded-md p-1.5 text-[var(--color-text-muted)] opacity-0 transition-all hover:bg-slate-700/50 hover:text-[var(--color-text-primary)] group-hover:opacity-100"
                          >
                            <MoreHorizontal size={14} />
                          </button>
                        </DropdownMenu.Trigger>
                        <DropdownMenu.Portal>
                          <DropdownMenu.Content
                            sideOffset={4}
                            align="end"
                            className="z-50 min-w-[160px] rounded-md border border-slate-700 bg-slate-850 p-1 shadow-xl"
                          >
                            <DropdownMenu.Item
                              onClick={() => handleRerun(row)}
                              className="flex cursor-pointer items-center gap-2 rounded-sm px-2.5 py-1.5 text-[11px] text-slate-200 outline-none transition-colors hover:bg-slate-700/50 hover:text-white"
                            >
                              <RefreshCw size={12} />
                              Re-run Backtest
                            </DropdownMenu.Item>
                            <DropdownMenu.Item
                              onClick={() => navigate(`/results/${row.job_id}?tab=logs`)}
                              className="flex cursor-pointer items-center gap-2 rounded-sm px-2.5 py-1.5 text-[11px] text-slate-200 outline-none transition-colors hover:bg-slate-700/50 hover:text-white"
                            >
                              <FileText size={12} />
                              View Detailed Logs
                            </DropdownMenu.Item>
                            <DropdownMenu.Separator className="my-1 h-px bg-slate-700" />
                            <DropdownMenu.Item
                              onClick={() => handleDeleteSingle(row.job_id)}
                              className="flex cursor-pointer items-center gap-2 rounded-sm px-2.5 py-1.5 text-[11px] text-rose-400 outline-none transition-colors hover:bg-rose-500/10 hover:text-rose-300"
                            >
                              <Trash2 size={12} />
                              Delete Study
                            </DropdownMenu.Item>
                          </DropdownMenu.Content>
                        </DropdownMenu.Portal>
                      </DropdownMenu.Root>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>

        {/* table footer */}
        {!isLoading && filtered.length > 0 && (
          <div className="flex items-center justify-between border-t border-(--color-glass-border) bg-(--color-surface) px-4 py-2.5 text-[10px] text-(--color-text-muted)">
            <span>
              {filtered.length} row{filtered.length !== 1 ? "s" : ""}
              {search && ` matching "${search}"`}
            </span>
            <span className="font-mono">
              {selected.size > 0 ? `${selected.size} selected — Click row to view` : "Click row to view"}
            </span>
          </div>
        )}
      </div>
        </>
      )}
    </div>
  );
}
