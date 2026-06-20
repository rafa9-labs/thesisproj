import { useState, useMemo, useCallback } from "react";
import {
  Trash2,
  Search,
  Play,
  Clock,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import { useFullCycleHistory, useBatchDeleteCommittees } from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import apiClient from "@/api/client";
import type { FullCycleHistoryEntry } from "@/api/schemas";

interface Props {
  activeJobId: string | null;
  onSelect: (jobId: string) => void;
}

function cn(...classes: (string | boolean | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}

function SortChevron({
  active,
  direction,
}: {
  active: boolean;
  direction: "asc" | "desc";
}) {
  if (!active) return null;
  return direction === "asc" ? (
    <ChevronUp size={9} className="ml-0.5" />
  ) : (
    <ChevronDown size={9} className="ml-0.5" />
  );
}

export function RunHistoryTable({ activeJobId, onSelect }: Props) {
  const { data: history } = useFullCycleHistory();
  const store = useFullCycleStore();
  const batchDelete = useBatchDeleteCommittees();

  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("completed");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<"started_at" | "avg_sharpe" | "trust_score">("started_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const entries = history?.entries ?? [];

  const handleSort = (col: "started_at" | "avg_sharpe" | "trust_score") => {
    if (sortBy === col) setSortOrder((p) => (p === "asc" ? "desc" : "asc"));
    else {
      setSortBy(col);
      setSortOrder("desc");
    }
  };

  const filtered = useMemo(() => {
    let list = search
      ? entries.filter(
          (e) =>
            e.job_id.toLowerCase().includes(search.toLowerCase()) ||
            e.survivors?.some((s) => s.toLowerCase().includes(search.toLowerCase())),
        )
      : entries;

    list = list.filter((e) => {
      if (statusFilter === "completed") return e.status === "completed";
      if (statusFilter === "failed")
        return ["failed", "validation_failed", "cancelled", "orphaned"].includes(e.status);
      return true;
    });

    list.sort((a, b) => {
      let av: number, bv: number;
      if (sortBy === "started_at") {
        av = new Date(a.started_at ?? 0).getTime();
        bv = new Date(b.started_at ?? 0).getTime();
      } else if (sortBy === "avg_sharpe") {
        av = a.avg_sharpe ?? 0;
        bv = b.avg_sharpe ?? 0;
      } else {
        av = a.trust_score ?? 0;
        bv = b.trust_score ?? 0;
      }
      return sortOrder === "asc" ? av - bv : bv - av;
    });

    return list;
  }, [entries, search, statusFilter, sortBy, sortOrder]);

  const terminalPhases = new Set([
    "completed",
    "failed",
    "validation_failed",
    "cancelled",
    "orphaned",
  ]);

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
    else setSelected(new Set(filtered.map((e) => e.job_id)));
  };

  const handleBatchDelete = useCallback(() => {
    const ids = [...selected];
    if (!ids.length) return;
    batchDelete.mutate(ids, {
      onSuccess: () => setSelected(new Set()),
    });
  }, [selected, batchDelete]);

  const handleDeploy = (entry: FullCycleHistoryEntry) => {
    apiClient
      .post("/trading/live/committee/start", {
        pair: store.deployedPair,
        timeframe: store.deployedTimeframe,
        initial_equity: 10000.0,
        confidence_threshold: 0.55,
        mode: store.executionMode,
        full_cycle_job_id: entry.job_id,
      })
      .then((r: { data: { session_id: string; pair: string; timeframe: string } }) => {
        store.setDeployedSession(r.data.session_id, r.data.pair, r.data.timeframe);
        store.setDeployedJobId(entry.job_id);
      })
      .catch(console.error);
  };

  const completedCount = entries.filter((e) => e.status === "completed").length;
  const failedCount = entries.filter((e) =>
    ["failed", "validation_failed", "cancelled", "orphaned"].includes(e.status),
  ).length;

  const TH =
    "px-4 py-3 text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase";
  const TD = "px-4 py-3 font-mono text-[11px]";

  return (
    <div className="flex flex-col gap-3">
      {/* ── action bar ── */}
      <div className="flex flex-wrap items-center gap-3 rounded-sm border border-(--color-glass-border) bg-(--color-surface) px-3 py-2">
        {entries.length > 0 && (
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] tabular-nums text-(--color-text-muted)">
              <span className="text-(--color-text-primary)">{entries.length}</span> runs
            </span>
            <span className="font-mono text-[10px] tabular-nums text-(--color-text-muted)">
              <span className="text-(--color-accent-success)">{completedCount}</span> done
            </span>
            <span className="font-mono text-[10px] tabular-nums text-(--color-text-muted)">
              <span className="text-(--color-accent-danger)">{failedCount}</span> failed
            </span>
          </div>
        )}
        {entries.length > 0 && <span className="h-4 w-px bg-(--color-glass-border)" />}

        <div
          className="relative flex flex-1 items-center rounded-sm border border-(--color-glass-border) bg-(--color-input-bg) transition-colors focus-within:border-[var(--color-brand)]"
          style={{ maxWidth: 240 }}
        >
          <Search
            size={14}
            className="pointer-events-none absolute left-2.5 shrink-0 text-(--color-text-muted)"
          />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search…"
            className="w-full bg-transparent py-1.5 pr-2.5 pl-8 font-mono text-[11px] text-(--color-text-primary) focus:outline-none"
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-1.5 rounded px-1 py-0.5 text-[10px] text-(--color-text-muted)"
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        <div className="flex items-center gap-0.5 rounded-sm border border-(--color-glass-border) bg-(--color-input-bg) p-0.5">
          {(["completed", "failed"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className="rounded-sm px-2.5 py-1 text-[10px] font-medium capitalize transition-all"
              style={{
                backgroundColor:
                  statusFilter === s ? "var(--color-brand-glow)" : "transparent",
                color:
                  statusFilter === s ? "var(--color-brand)" : "var(--color-text-dim)",
              }}
            >
              {s}
            </button>
          ))}
        </div>

        {selected.size > 0 && (
          <button
            onClick={handleBatchDelete}
            disabled={batchDelete.isPending}
            className="flex items-center gap-1.5 rounded-sm border border-[rgba(242,54,69,0.3)] px-2.5 py-1 text-[10px] font-medium text-[var(--color-accent-danger)] transition-colors hover:bg-[rgba(242,54,69,0.1)] disabled:opacity-50"
          >
            <Trash2 size={12} />
            Delete {selected.size}
          </button>
        )}
      </div>

      {/* ── table ── */}
      <div className="overflow-hidden rounded-sm border border-(--color-glass-border)">
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr className="border-b border-(--color-glass-border) bg-(--color-surface)">
              {/* checkbox — always */}
              <th className="w-9 px-4 py-3">
                <input
                  type="checkbox"
                  checked={selected.size === filtered.length && filtered.length > 0}
                  onChange={selectAll}
                  className="cursor-pointer accent-[var(--color-brand)]"
                />
              </th>
              {/* date — always, sortable */}
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort("started_at")}
                  className={cn(
                    "flex items-center gap-1 text-[10px] font-medium tracking-[0.1em] uppercase transition-colors hover:text-[var(--color-text-primary)]",
                  )}
                  style={{
                    color: sortBy === "started_at" ? "var(--color-brand)" : "var(--color-text-muted)",
                  }}
                >
                  Date
                  <SortChevron active={sortBy === "started_at"} direction={sortOrder} />
                </button>
              </th>
              {/* status — always */}
              <th className={cn(TH, "hidden sm:table-cell")}>Status</th>
              {/* time — lg */}
              <th className={cn(TH, "hidden lg:table-cell")}>Time</th>
              {/* locked — lg */}
              <th className={cn(TH, "hidden lg:table-cell")}>Locked</th>
              {/* survivors — md */}
              <th className={cn(TH, "hidden md:table-cell")}>Survivors</th>
              {/* sharpe — always */}
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort("avg_sharpe")}
                  className={cn(
                    "flex items-center gap-1 ml-auto text-[10px] font-medium tracking-[0.1em] uppercase transition-colors hover:text-[var(--color-text-primary)]",
                  )}
                  style={{
                    color: sortBy === "avg_sharpe" ? "var(--color-brand)" : "var(--color-text-muted)",
                  }}
                >
                  Sharpe
                  <SortChevron active={sortBy === "avg_sharpe"} direction={sortOrder} />
                </button>
              </th>
              {/* trust — lg */}
              <th className={cn(TH, "hidden lg:table-cell text-right")}>
                <button
                  onClick={() => handleSort("trust_score")}
                  className={cn(
                    "flex items-center gap-1 ml-auto text-[10px] font-medium tracking-[0.1em] uppercase transition-colors hover:text-[var(--color-text-primary)]",
                  )}
                  style={{
                    color: sortBy === "trust_score" ? "var(--color-brand)" : "var(--color-text-muted)",
                  }}
                >
                  Trust
                  <SortChevron active={sortBy === "trust_score"} direction={sortOrder} />
                </button>
              </th>
              {/* actions — hidden on mobile, deploy only on lg */}
              <th className="px-4 py-3 text-right">
                <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
                  &nbsp;
                </span>
              </th>
            </tr>
          </thead>

          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-4 py-16 text-center text-(--color-text-muted)">
                  <div className="flex flex-col items-center gap-2">
                    <Clock size={32} className="text-(--color-glass-border)" />
                    <span className="text-xs">
                      No committee runs found. Run a Full Cycle from the Committee tab.
                    </span>
                  </div>
                </td>
              </tr>
            ) : (
              filtered.map((entry) => {
                const isActive = entry.job_id === activeJobId;
                const isTerminal = terminalPhases.has(entry.status);
                const isInProgress = !isTerminal && entry.status !== "unknown";
                const isSel = selected.has(entry.job_id);

                const started = entry.started_at
                  ? new Date(entry.started_at).toLocaleString(undefined, {
                      month: "2-digit",
                      day: "2-digit",
                      hour: "2-digit",
                      minute: "2-digit",
                    })
                  : "--";

                const timeStr =
                  entry.total_time_s > 0
                    ? entry.total_time_s >= 3600
                      ? `${Math.floor(entry.total_time_s / 3600)}h ${Math.floor((entry.total_time_s % 3600) / 60)}m`
                      : `${Math.floor(entry.total_time_s / 60)}m ${Math.floor(entry.total_time_s % 60)}s`
                    : "--";

                const sharpeStr =
                  entry.avg_sharpe !== 0 ? entry.avg_sharpe.toFixed(2) : "--";
                const sharpeColor =
                  entry.avg_sharpe > 0
                    ? "var(--color-accent-success)"
                    : entry.avg_sharpe < 0
                      ? "var(--color-accent-danger)"
                      : "var(--color-text-dim)";

                const trustStr =
                  entry.trust_score != null && entry.trust_score > 0
                    ? entry.trust_score.toFixed(2)
                    : "--";
                const trustColor =
                  (entry.trust_score ?? 0) > 0.4
                    ? "var(--color-accent-success)"
                    : "var(--color-text-dim)";

                const statusColor = isInProgress
                  ? "var(--color-brand)"
                  : entry.status === "completed"
                    ? "var(--color-accent-success)"
                    : entry.status === "validation_failed"
                      ? "var(--color-accent-warning)"
                      : entry.status === "cancelled"
                        ? "var(--color-accent-warning)"
                        : entry.status === "orphaned"
                          ? "var(--color-accent-warning)"
                          : "var(--color-accent-danger)";

                const statusLabel = isInProgress
                  ? "Running"
                  : entry.status === "cancelled"
                    ? "Cancelled"
                    : entry.status === "validation_failed"
                      ? "Validation Failed"
                      : entry.status === "failed"
                        ? "Failed"
                        : entry.status === "completed"
                          ? "Completed"
                          : entry.status === "orphaned"
                            ? "Interrupted"
                            : entry.status;

                const survivorsStr =
                  entry.survivors_count > 0
                    ? entry.survivors.slice(0, 3).join(", ") +
                      (entry.survivors_count > 3 ? ` +${entry.survivors_count - 3}` : "")
                    : "--";

                return (
                  <tr
                    key={entry.job_id}
                    onClick={() => onSelect(entry.job_id)}
                    className={cn(
                      "group cursor-pointer border-b border-(--color-glass-border) transition-colors duration-100 hover:bg-(--color-glass-hover)",
                      isSel && "bg-[rgba(0,229,255,0.04)]",
                    )}
                    style={{
                      background: isActive
                        ? "rgba(0,229,255,0.04)"
                        : isSel
                          ? "rgba(0,229,255,0.04)"
                          : undefined,
                    }}
                  >
                    {/* checkbox — always */}
                    <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={isSel}
                        onChange={() => toggleSelect(entry.job_id)}
                        className="cursor-pointer accent-[var(--color-brand)]"
                      />
                    </td>

                    {/* date — always */}
                    <td className="px-4 py-3 font-mono whitespace-nowrap text-(--color-text-muted)">
                      <div className="flex items-center gap-1.5">
                        <Clock size={11} className="text-(--color-glass-border)" />
                        {started}
                      </div>
                    </td>

                    {/* status — sm */}
                    <td className={cn(TD, "hidden sm:table-cell")}>
                      <span className="inline-flex items-center gap-[6px]">
                        <span
                          className="inline-block h-[6px] w-[6px] rounded-full"
                          style={{
                            background: statusColor,
                            animation: isInProgress ? "pulse 1.5s infinite" : "none",
                          }}
                        />
                        <span style={{ color: statusColor }}>{statusLabel}</span>
                      </span>
                    </td>

                    {/* time — lg */}
                    <td className={cn(TD, "hidden lg:table-cell text-(--color-text-secondary)")}>
                      {timeStr}
                    </td>

                    {/* locked — lg */}
                    <td className={cn(TD, "hidden lg:table-cell text-(--color-text-secondary)")}>
                      {entry.locked_features_count > 0 ? entry.locked_features_count : "--"}
                    </td>

                    {/* survivors — md */}
                    <td
                      className={cn(
                        TD,
                        "hidden md:table-cell max-w-[180px] truncate text-(--color-text-secondary)",
                      )}
                    >
                      {survivorsStr}
                    </td>

                    {/* sharpe — always */}
                    <td
                      className="px-4 py-3 text-right font-mono text-[11px] tabular-nums"
                      style={{ color: sharpeColor }}
                    >
                      {sharpeStr}
                    </td>

                    {/* trust — lg */}
                    <td
                      className={cn(
                        "px-4 py-3 text-right font-mono text-[11px] tabular-nums hidden lg:table-cell",
                      )}
                      style={{ color: trustColor }}
                    >
                      {trustStr}
                    </td>

                    {/* actions — deploy only for high-trust completed */}
                    <td className="px-4 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                      {entry.status === "completed" && (entry.trust_score ?? 0) > 0.4 && (
                        <button
                          onClick={() => handleDeploy(entry)}
                          title="Deploy"
                          aria-label="Deploy committee"
                          className="rounded-md p-1.5 text-[var(--color-accent-success)] opacity-0 transition-all hover:bg-[rgba(8,153,129,0.12)] group-hover:opacity-100"
                        >
                          <Play size={14} />
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>

        {filtered.length > 0 && (
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
    </div>
  );
}
