import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  Eye,
  Trash2,
  Download,
  Search,
  ChevronUp,
  ChevronDown,
  RefreshCw,
  Clock,
  BarChart2,
} from "lucide-react";
import { useResultsHistory, useDeleteJob } from "@/api/queries";
import { formatRelativeTime, formatPercent } from "@/lib/formatters";
import { useBacktestStore } from "@/stores/useBacktestStore";
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
    running:   { bg: "rgba(0,229,255,0.10)", text: "var(--color-brand)",          label: "running" },
    failed:    { bg: "rgba(242,54,69,0.15)", text: "var(--color-accent-danger)",  label: "failed" },
    pending:   { bg: "rgba(245,158,11,0.12)", text: "var(--color-accent-warning)", label: "pending" },
  };
  const s = map[status] ?? { bg: "rgba(255,255,255,0.05)", text: "var(--color-text-muted)", label: status };
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider"
      style={{ backgroundColor: s.bg, color: s.text, fontFamily: "var(--font-mono)" }}
    >
      {s.label}
    </span>
  );
}

function ModelBadge({ model }: { model: string }) {
  const colorMap: Record<string, string> = {
    logistic:      "var(--color-accent-classical)",
    xgboost:       "var(--color-accent-classical)",
    svm:           "var(--color-accent-classical)",
    random_forest: "var(--color-accent-classical)",
    decision_tree: "var(--color-accent-classical)",
    lightgbm:      "var(--color-accent-classical)",
    catboost:      "var(--color-accent-classical)",
    lstm:          "var(--color-accent-deep)",
    cnn:           "var(--color-accent-deep)",
    transformer:   "var(--color-accent-deep)",
    gru:           "var(--color-accent-deep)",
    gru_lstm:      "var(--color-accent-deep)",
    dqn:           "var(--color-accent-rl)",
    ensemble:      "var(--color-accent-ensemble)",
  };
  const key = Object.keys(colorMap).find((k) => model.toLowerCase().includes(k));
  const color = key ? colorMap[key] : "var(--color-text-muted)";
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wider mr-0.5"
      style={{ backgroundColor: `${color}14`, color, border: `1px solid ${color}30`, fontFamily: "var(--font-mono)" }}
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
      className={`flex items-center gap-1 text-[10px] font-medium uppercase tracking-[0.1em] transition-colors hover:text-[var(--color-text-primary)] ${align === "right" ? "ml-auto" : ""}`}
      style={{ color: active ? "var(--color-brand)" : "var(--color-text-muted)" }}
    >
      {align === "right" && active && (direction === "asc" ? <ChevronUp size={9} /> : <ChevronDown size={9} />)}
      {label}
      {align === "left" && active && (direction === "asc" ? <ChevronUp size={9} /> : <ChevronDown size={9} />)}
    </button>
  );
}

// ── summary strip ─────────────────────────────────────────────────────────────

function SummaryStrip({ results }: { results: BacktestSummaryItem[] }) {
  const completed = results.filter((r) => r.status === "completed");
  const avgSharpe = completed.length
    ? completed.reduce((a, b) => a + (b.sharpe ?? 0), 0) / completed.length
    : null;
  const bestSharpe = completed.length
    ? Math.max(...completed.map((r) => r.sharpe ?? -Infinity))
    : null;
  const pairs = [...new Set(results.map((r) => r.pair).filter(Boolean))].length;

  const stats = [
    { label: "Total Runs",   value: results.length.toString(),           mono: false },
    { label: "Completed",    value: completed.length.toString(),         mono: false },
    { label: "Pairs Tested", value: pairs.toString(),                    mono: false },
    { label: "Avg Sharpe",   value: avgSharpe != null ? avgSharpe.toFixed(2) : "—", mono: true, color: signColor(avgSharpe) },
    { label: "Best Sharpe",  value: bestSharpe != null && isFinite(bestSharpe) ? bestSharpe.toFixed(2) : "—", mono: true, color: signColor(bestSharpe) },
  ];

  return (
    <div className="grid grid-cols-5 gap-3">
      {stats.map((s) => (
        <div
          key={s.label}
          className="rounded-sm border px-4 py-3 flex flex-col gap-1"
          style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-elevated)" }}
        >
          <span className="text-[9px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>
            {s.label}
          </span>
          <span
            className="text-base font-semibold"
            style={{
              color: s.color ?? "var(--color-text-primary)",
              fontFamily: s.mono ? "var(--font-mono)" : undefined,
            }}
          >
            {s.value}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export function ResultsHistoryPage() {
  const navigate = useNavigate();
  const deleteJob = useDeleteJob();
  const setField = useBacktestStore((s) => s.setField);

  const [search, setSearch] = useState("");
  const [pairFilter, setPairFilter] = useState("");
  const [sortBy, setSortBy] = useState("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const { data, isLoading } = useResultsHistory({
    pair: pairFilter,
    sort_by: sortBy,
    sort_order: sortOrder,
    limit: 100,
  });

  const results = data?.results ?? [];
  const filtered = search
    ? results.filter(
        (r) =>
          r.job_id.toLowerCase().includes(search.toLowerCase()) ||
          r.pair?.toLowerCase().includes(search.toLowerCase()) ||
          r.models?.some((m) => m.toLowerCase().includes(search.toLowerCase()))
      )
    : results;

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
    else { setSortBy(col); setSortOrder("desc"); }
  };

  const exportCSV = useCallback(() => {
    const rows = filtered.filter((r) => selected.has(r.job_id));
    if (!rows.length) return;
    const csv = [
      "job_id,created_at,pair,timeframe,models,status,sharpe,return_pct,win_rate,max_dd_pct,trades",
      ...rows.map((r) =>
        [r.job_id, r.created_at, r.pair, r.timeframe, r.models.join(";"), r.status, r.sharpe ?? "", r.total_return_pct ?? "", r.win_rate ?? "", r.max_drawdown_pct ?? "", r.total_trades ?? ""].join(",")
      ),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "backtest_results.csv"; a.click();
    URL.revokeObjectURL(url);
  }, [filtered, selected]);

  const handleRerun = useCallback(
    async (row: BacktestSummaryItem) => {
      setField("parentJobId", row.job_id);
      setField("pair", row.pair || "EURUSD");
      if (row.models?.length) setField("selectedModels", row.models as string[]);
      navigate("/backtest");
    },
    [setField, navigate]
  );

  const PAIRS = [...new Set(results.map((r) => r.pair).filter(Boolean))].sort();

  return (
    <div className="flex flex-col gap-5 p-6">

      {/* ── page header ── */}
      <div className="flex items-end justify-between">
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <BarChart2 size={16} style={{ color: "var(--color-brand)" }} />
            <h2
              className="text-sm font-semibold uppercase tracking-[0.08em]"
              style={{ color: "var(--color-text-primary)" }}
            >
              Results History
            </h2>
          </div>
          <p className="text-[11px] pl-6" style={{ color: "var(--color-text-muted)" }}>
            {isLoading
              ? "Loading..."
              : `${data?.total ?? filtered.length} completed backtest${(data?.total ?? filtered.length) !== 1 ? "s" : ""}`}
          </p>
        </div>

        {/* export button — only shown when rows selected */}
        {selected.size > 0 && (
          <button
            onClick={exportCSV}
            className="flex items-center gap-2 rounded-sm border px-3 py-2 text-[11px] font-medium transition-colors hover:bg-[var(--color-glass-hover)]"
            style={{ borderColor: "var(--color-glass-border)", color: "var(--color-text-secondary)" }}
          >
            <Download size={14} />
            Export {selected.size} row{selected.size !== 1 ? "s" : ""}
          </button>
        )}
      </div>

      {/* ── summary strip ── */}
      {!isLoading && results.length > 0 && <SummaryStrip results={results} />}

      {/* ── toolbar: search + pair filter ── */}
      <div className="flex items-center gap-3">
        {/* search */}
        <div
          className="relative flex items-center flex-1 max-w-sm rounded-sm border transition-colors focus-within:border-[var(--color-brand)]"
          style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-surface)" }}
        >
          <Search
            size={16}
            className="absolute left-3 pointer-events-none shrink-0"
            style={{ color: "var(--color-text-muted)" }}
          />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search pair, model, or job ID…"
            className="w-full bg-transparent py-2.5 pl-10 pr-3 text-xs focus:outline-none"
            style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-2.5 text-[10px] px-1.5 py-0.5 rounded"
              style={{ color: "var(--color-text-muted)" }}
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        {/* pair selector */}
        <div
          className="relative rounded-sm border transition-colors"
          style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-surface)" }}
        >
          <select
            value={pairFilter}
            onChange={(e) => setPairFilter(e.target.value)}
            className="appearance-none bg-transparent pl-3 pr-8 py-2.5 text-xs focus:outline-none cursor-pointer"
            style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}
          >
            <option value="">All Pairs</option>
            {PAIRS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <ChevronDown
            size={12}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none"
            style={{ color: "var(--color-text-muted)" }}
          />
        </div>
      </div>

      {/* ── table ── */}
      <div
        className="overflow-hidden rounded-sm border"
        style={{ borderColor: "var(--color-glass-border)" }}
      >
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          {/* thead */}
          <thead>
            <tr style={{ backgroundColor: "var(--color-surface)", borderBottom: "1px solid var(--color-glass-border)" }}>
              <th className="px-4 py-3 w-9">
                <input
                  type="checkbox"
                  checked={selected.size === filtered.length && filtered.length > 0}
                  onChange={selectAll}
                  className="cursor-pointer accent-[var(--color-brand)]"
                />
              </th>
              <th className="px-4 py-3 text-left">
                <SortHeader label="Date" active={sortBy === "created_at"} direction={sortOrder} onClick={() => handleSort("created_at")} />
              </th>
              <th className="px-4 py-3 text-left">
                <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Pair / TF</span>
              </th>
              <th className="px-4 py-3 text-left">
                <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Models</span>
              </th>
              <th className="px-4 py-3 text-left">
                <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Status</span>
              </th>
              <th className="px-4 py-3 text-right">
                <SortHeader label="Sharpe" active={sortBy === "sharpe"} direction={sortOrder} onClick={() => handleSort("sharpe")} align="right" />
              </th>
              <th className="px-4 py-3 text-right">
                <SortHeader label="Return" active={sortBy === "total_return_pct"} direction={sortOrder} onClick={() => handleSort("total_return_pct")} align="right" />
              </th>
              <th className="px-4 py-3 text-right">
                <SortHeader label="Win %" active={sortBy === "win_rate"} direction={sortOrder} onClick={() => handleSort("win_rate")} align="right" />
              </th>
              <th className="px-4 py-3 text-right">
                <SortHeader label="Max DD" active={sortBy === "max_drawdown_pct"} direction={sortOrder} onClick={() => handleSort("max_drawdown_pct")} align="right" />
              </th>
              <th className="px-4 py-3 text-right">
                <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Trades</span>
              </th>
              <th className="px-4 py-3 text-right">
                <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Actions</span>
              </th>
            </tr>
          </thead>

          {/* tbody */}
          <tbody>
            {isLoading ? (
              /* skeleton rows */
              Array.from({ length: 4 }).map((_, i) => (
                <tr key={i} style={{ borderBottom: "1px solid var(--color-glass-border)" }}>
                  {Array.from({ length: 11 }).map((_, j) => (
                    <td key={j} className="px-4 py-3">
                      <div
                        className="h-3 rounded animate-pulse"
                        style={{ backgroundColor: "var(--color-elevated)", width: j === 0 ? 16 : j === 3 ? 120 : 60 }}
                      />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td
                  colSpan={11}
                  className="px-4 py-16 text-center"
                  style={{ color: "var(--color-text-muted)" }}
                >
                  <div className="flex flex-col items-center gap-2">
                    <Clock size={32} style={{ color: "var(--color-glass-border)" }} />
                    <span className="text-xs">No backtests found. Run one from the Backtest Setup tab.</span>
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
                    className="group cursor-pointer transition-colors duration-100"
                    style={{
                      borderBottom: "1px solid var(--color-glass-border)",
                      backgroundColor: isSel ? "rgba(0,229,255,0.04)" : undefined,
                    }}
                    onMouseEnter={(e) => { (e.currentTarget as HTMLTableRowElement).style.backgroundColor = "var(--color-glass-hover)"; }}
                    onMouseLeave={(e) => { (e.currentTarget as HTMLTableRowElement).style.backgroundColor = isSel ? "rgba(0,229,255,0.04)" : ""; }}
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
                    <td className="px-4 py-3 whitespace-nowrap" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                      <div className="flex items-center gap-1.5">
                        <Clock size={11} style={{ color: "var(--color-glass-border)" }} />
                        {formatRelativeTime(row.created_at)}
                      </div>
                    </td>

                    {/* pair / timeframe */}
                    <td className="px-4 py-3">
                      <div className="flex flex-col gap-0.5">
                        <span className="font-semibold text-[12px]" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                          {row.pair}
                        </span>
                        {row.timeframe && (
                          <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                            {row.timeframe}
                          </span>
                        )}
                      </div>
                    </td>

                    {/* models */}
                    <td className="px-4 py-3 max-w-[220px]">
                      <div className="flex flex-wrap gap-1">
                        {row.models.map((m) => (
                          <ModelBadge key={m} model={m} />
                        ))}
                      </div>
                    </td>

                    {/* status */}
                    <td className="px-4 py-3">
                      <StatusPill status={row.status} />
                    </td>

                    {/* sharpe */}
                    <td className="px-4 py-3 text-right tabular-nums" style={{ color: signColor(row.sharpe), fontFamily: "var(--font-mono)" }}>
                      {fmt(row.sharpe)}
                    </td>

                    {/* return */}
                    <td className="px-4 py-3 text-right tabular-nums" style={{ color: signColor(row.total_return_pct), fontFamily: "var(--font-mono)" }}>
                      {row.total_return_pct != null ? formatPercent(row.total_return_pct) : "—"}
                    </td>

                    {/* win rate */}
                    <td className="px-4 py-3 text-right tabular-nums" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                      {row.win_rate != null ? formatPercent(row.win_rate) : "—"}
                    </td>

                    {/* max dd */}
                    <td className="px-4 py-3 text-right tabular-nums" style={{ color: "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>
                      {row.max_drawdown_pct != null ? formatPercent(row.max_drawdown_pct) : "—"}
                    </td>

                    {/* trades */}
                    <td className="px-4 py-3 text-right tabular-nums" style={{ color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>
                      {row.total_trades ?? "—"}
                    </td>

                    {/* actions */}
                    <td className="px-4 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => handleRerun(row)}
                          title="Re-run"
                          aria-label="Re-run backtest"
                          className="rounded-md p-1.5 transition-colors hover:bg-[var(--color-primary-glow)]"
                          style={{ color: "var(--color-brand)" }}
                        >
                          <RefreshCw size={14} />
                        </button>
                        <button
                          onClick={() => navigate(`/results/${row.job_id}`)}
                          title="View"
                          aria-label="View results"
                          className="rounded-md p-1.5 transition-colors hover:bg-[var(--color-glass-hover)]"
                          style={{ color: "var(--color-text-muted)" }}
                        >
                          <Eye size={14} />
                        </button>
                        <button
                          onClick={() => deleteJob.mutate(row.job_id)}
                          title="Delete"
                          aria-label="Delete backtest"
                          className="rounded-md p-1.5 transition-colors hover:bg-[rgba(242,54,69,0.12)]"
                          style={{ color: "var(--color-text-muted)" }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>

        {/* table footer */}
        {!isLoading && filtered.length > 0 && (
          <div
            className="flex items-center justify-between px-4 py-2.5 border-t text-[10px]"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-surface)", color: "var(--color-text-muted)" }}
          >
            <span>{filtered.length} row{filtered.length !== 1 ? "s" : ""}{search && ` matching "${search}"`}</span>
            <span style={{ fontFamily: "var(--font-mono)" }}>{selected.size > 0 ? `${selected.size} selected` : "Click row to view"}</span>
          </div>
        )}
      </div>
    </div>
  );
}
