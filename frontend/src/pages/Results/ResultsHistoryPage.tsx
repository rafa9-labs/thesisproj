import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, Trash2, Download, Search, ChevronUp, ChevronDown, Filter, RefreshCw } from "lucide-react";
import { useResultsHistory, useDeleteJob, useHeatmap } from "@/api/queries";
import { formatRelativeTime, formatPercent } from "@/lib/formatters";
import { PerformanceHeatmapSection } from "@/pages/Dashboard/PerformanceHeatmapSection";
import { useBacktestStore } from "@/stores/useBacktestStore";
import type { BacktestSummaryItem } from "@/api/schemas";

const SORT_COLS = [
  { key: "created_at", label: "Date" },
  { key: "sharpe", label: "Sharpe" },
  { key: "total_return_pct", label: "Return" },
  { key: "win_rate", label: "Win Rate" },
  { key: "max_drawdown_pct", label: "Max DD" },
];

const STATUS_STYLES: Record<string, { dot: string; bg: string; text: string }> = {
  completed: { dot: "var(--color-accent-success)", bg: "rgba(34,197,94,0.10)", text: "var(--color-accent-success)" },
  failed: { dot: "var(--color-accent-danger)", bg: "rgba(239,68,68,0.10)", text: "var(--color-accent-danger)" },
  running: { dot: "var(--color-brand)", bg: "var(--color-brand-glow)", text: "var(--color-brand)" },
  pending: { dot: "var(--color-accent-warning)", bg: "rgba(245,158,11,0.10)", text: "var(--color-accent-warning)" },
};

const CARD_STYLE: React.CSSProperties = {
  backgroundColor: "var(--color-glass)",
  border: "1px solid var(--color-glass-border)",
  borderRadius: 10,
  backdropFilter: "blur(12px)",
};

const TH_STYLE: React.CSSProperties = {
  backgroundColor: "var(--color-surface)",
  color: "var(--color-text-muted)",
  borderBottom: "1px solid var(--color-glass-border)",
};

function SortHeader({ label, active, direction, onClick }: { label: string; active: boolean; direction: "asc" | "desc"; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-[0.1em] cursor-pointer hover:text-[var(--color-text-primary)] transition-colors"
    >
      {label}
      {active && (direction === "asc" ? <ChevronUp size={10} /> : <ChevronDown size={10} />)}
    </button>
  );
}

export function ResultsHistoryPage() {
  const navigate = useNavigate();
  const deleteJob = useDeleteJob();
  const setField = useBacktestStore((s) => s.setField);
  const applyPreset = useBacktestStore((s) => s.applyPreset);
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

  const { data: heatmapData, isLoading: heatmapLoading } = useHeatmap();

  const results = data?.results ?? [];
  const filtered = search
    ? results.filter((r) => r.job_id.includes(search) || r.pair.toLowerCase().includes(search.toLowerCase()))
    : results;

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    if (selected.size === filtered.length) setSelected(new Set());
    else setSelected(new Set(filtered.map((r) => r.job_id)));
  };

  const handleSort = (col: string) => {
    if (sortBy === col) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(col);
      setSortOrder("desc");
    }
  };

  const exportSelectedCSV = useCallback(() => {
    const rows = filtered.filter((r) => selected.has(r.job_id));
    if (!rows.length) return;
    const csv = [
      "job_id,created_at,pair,models,sharpe,total_return_pct,win_rate,max_drawdown_pct,total_trades",
      ...rows.map((r) => [r.job_id, r.created_at, r.pair, r.models.join(";"), r.sharpe ?? "", r.total_return_pct ?? "", r.win_rate ?? "", r.max_drawdown_pct ?? "", r.total_trades ?? ""].join(",")),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "backtest_results.csv"; a.click();
    URL.revokeObjectURL(url);
  }, [filtered, selected]);

  const exportSelectedJSON = useCallback(() => {
    const rows = filtered.filter((r) => selected.has(r.job_id));
    if (!rows.length) return;
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "backtest_results.json"; a.click();
    URL.revokeObjectURL(url);
  }, [filtered, selected]);

  const handleRerun = useCallback(async (row: BacktestSummaryItem) => {
    setField("parentJobId", row.job_id);
    setField("pair", row.pair || "EURUSD");
    if (row.models?.length) setField("selectedModels", row.models as string[]);
    navigate("/backtest");
  }, [setField, navigate]);

  const PAIRS = [...new Set(results.map((r) => r.pair).filter(Boolean))].sort();

  return (
    <div className="flex flex-col gap-4 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>Results History</h2>
          <p className="text-xs mt-1" style={{ color: "var(--color-text-muted)" }}>
            {isLoading ? "Loading..." : `${data?.total ?? filtered.length} completed backtests`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {selected.size > 0 && (
            <>
              <button onClick={exportSelectedCSV} className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-medium transition hover:bg-[var(--color-glass-hover)]" style={{ borderColor: "var(--color-glass-border)", color: "var(--color-text-secondary)" }}>
                <Download size={12} /> CSV ({selected.size})
              </button>
              <button onClick={exportSelectedJSON} className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-[11px] font-medium transition hover:bg-[var(--color-glass-hover)]" style={{ borderColor: "var(--color-glass-border)", color: "var(--color-text-secondary)" }}>
                <Download size={12} /> JSON ({selected.size})
              </button>
            </>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: "var(--color-text-muted)" }} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by job ID or pair..."
            className="w-full rounded-md border py-1.5 pl-8 pr-3 text-xs transition focus:outline-none"
            style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)" }}
          />
        </div>
        <select
          value={pairFilter}
          onChange={(e) => setPairFilter(e.target.value)}
          className="rounded-md border px-3 py-1.5 text-xs transition focus:outline-none"
          style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)", color: "var(--color-text-primary)" }}
        >
          <option value="">All Pairs</option>
          {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </div>

      <div className="overflow-hidden rounded-lg border" style={{ borderColor: "var(--color-glass-border)" }}>
        <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={TH_STYLE}>
              <th className="px-3 py-2.5 w-8">
                <input type="checkbox" checked={selected.size === filtered.length && filtered.length > 0} onChange={selectAll} className="cursor-pointer" />
              </th>
              <th className="px-3 py-2.5 text-left">
                <SortHeader label="Date" active={sortBy === "created_at"} direction={sortOrder} onClick={() => handleSort("created_at")} />
              </th>
              <th className="px-3 py-2.5 text-left">
                <button className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)", display: "flex", alignItems: "center", gap: 4 }}>
                  <Filter size={10} /> Pair
                </button>
              </th>
              <th className="px-3 py-2.5 text-left font-medium uppercase tracking-[0.1em] text-[10px]">Models</th>
              <th className="px-3 py-2.5 text-right">
                <SortHeader label="Sharpe" active={sortBy === "sharpe"} direction={sortOrder} onClick={() => handleSort("sharpe")} />
              </th>
              <th className="px-3 py-2.5 text-right">
                <SortHeader label="Return" active={sortBy === "total_return_pct"} direction={sortOrder} onClick={() => handleSort("total_return_pct")} />
              </th>
              <th className="px-3 py-2.5 text-right">
                <SortHeader label="Win Rate" active={sortBy === "win_rate"} direction={sortOrder} onClick={() => handleSort("win_rate")} />
              </th>
              <th className="px-3 py-2.5 text-right">
                <SortHeader label="Max DD" active={sortBy === "max_drawdown_pct"} direction={sortOrder} onClick={() => handleSort("max_drawdown_pct")} />
              </th>
              <th className="px-3 py-2.5 text-right font-medium uppercase tracking-[0.1em] text-[10px]">Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr><td colSpan={10} className="px-3 py-8 text-center" style={{ color: "var(--color-text-muted)" }}>Loading results...</td></tr>
            ) : filtered.length === 0 ? (
              <tr><td colSpan={10} className="px-3 py-8 text-center" style={{ color: "var(--color-text-muted)" }}>No results found. Run a backtest to see results here.</td></tr>
            ) : (
              filtered.map((row) => {
                const status = STATUS_STYLES[row.status] || STATUS_STYLES.completed;
                const isSel = selected.has(row.job_id);
                return (
                  <tr
                    key={`${row.job_id}-${row.models.join("-")}`}
                    className="group transition-colors hover:bg-[var(--color-glass-hover)]"
                    style={{ borderBottom: "1px solid var(--color-glass-border)", cursor: "pointer" }}
                    onClick={() => navigate(`/results/${row.job_id}`)}
                  >
                    <td className="px-3 py-2.5" onClick={(e) => e.stopPropagation()}>
                      <input type="checkbox" checked={isSel} onChange={() => toggleSelect(row.job_id)} className="cursor-pointer" />
                    </td>
                    <td className="px-3 py-2.5" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>{formatRelativeTime(row.created_at)}</td>
                    <td className="px-3 py-2.5" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>{row.pair}</td>
                    <td className="px-3 py-2.5 max-w-[200px] truncate" style={{ color: "var(--color-text-secondary)" }}>{row.models.join(", ")}</td>
                    <td className="px-3 py-2.5 text-right" style={{ color: row.sharpe && row.sharpe > 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>
                      {row.sharpe != null ? row.sharpe.toFixed(2) : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right" style={{ color: row.total_return_pct && row.total_return_pct > 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>
                      {row.total_return_pct != null ? formatPercent(row.total_return_pct) : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                      {row.win_rate != null ? formatPercent(row.win_rate) : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right" style={{ color: "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}>
                      {row.max_drawdown_pct != null ? formatPercent(row.max_drawdown_pct) : "—"}
                    </td>
                    <td className="px-3 py-2.5 text-right" onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-1">
                        <button onClick={() => handleRerun(row)} className="rounded p-1.5 transition-colors hover:bg-[var(--color-primary-glow)]" style={{ color: "var(--color-accent)", cursor: "pointer" }} title="Re-run"><RefreshCw size={14} /></button>
                        <button onClick={() => navigate(`/results/${row.job_id}`)} className="rounded p-1.5 transition-colors hover:bg-[var(--color-primary-glow)]" style={{ color: "var(--color-text-muted)", cursor: "pointer" }} title="View"><Eye size={14} /></button>
                        <button onClick={() => deleteJob.mutate(row.job_id)} className="rounded p-1.5 transition-colors hover:bg-[var(--color-accent-danger)]" style={{ color: "var(--color-text-muted)", cursor: "pointer" }} title="Delete"><Trash2 size={14} /></button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      <PerformanceHeatmapSection data={heatmapData} isLoading={heatmapLoading} />
    </div>
  );
}