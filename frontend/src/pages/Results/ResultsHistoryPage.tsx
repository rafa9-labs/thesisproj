import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, Trash2, Download, Search, ChevronUp, ChevronDown, RefreshCw } from "lucide-react";
import { useResultsHistory, useDeleteJob, useHeatmap } from "@/api/queries";
import { formatRelativeTime, formatPercent } from "@/lib/formatters";
import { useBacktestStore } from "@/stores/useBacktestStore";
import type { BacktestSummaryItem } from "@/api/schemas";

// ─── Table helpers ──────────────────────────────────────────────────────────

const TH: React.CSSProperties = {
  padding: "6px 10px",
  textAlign: "left",
  fontSize: 9,
  letterSpacing: "0.1em",
  textTransform: "uppercase",
  color: "#787B86",
  borderBottom: "1px solid #2A2E39",
  backgroundColor: "#1E222D",
  fontWeight: 500,
  whiteSpace: "nowrap",
};

const TD: React.CSSProperties = {
  padding: "6px 10px",
  fontSize: 11,
  borderBottom: "1px solid #1E222D",
  fontFamily: "var(--font-mono, 'JetBrains Mono', monospace)",
  color: "#D1D4DC",
  whiteSpace: "nowrap",
};

function SortTH({
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
    <th style={{ ...TH, textAlign: align, cursor: "pointer" }} onClick={onClick}>
      <span className="inline-flex items-center gap-1">
        {label}
        {active
          ? direction === "asc"
            ? <ChevronUp size={9} />
            : <ChevronDown size={9} />
          : null}
      </span>
    </th>
  );
}

// ─── Mock data (2 rows shown when no real data) ─────────────────────────────

const MOCK_ROWS: BacktestSummaryItem[] = [
  {
    job_id: "kq-2504-0012",
    created_at: "2025-04-22T09:14:00Z",
    pair: "EURUSD",
    models: ["logistic_regression", "xgboost"],
    sharpe: 1.74,
    total_return_pct: 18.4,
    win_rate: 0.617,
    max_drawdown_pct: -6.2,
    total_trades: 381,
  },
  {
    job_id: "kq-2503-0031",
    created_at: "2025-03-15T14:32:00Z",
    pair: "GBPUSD",
    models: ["lstm"],
    sharpe: 0.89,
    total_return_pct: -4.1,
    win_rate: 0.481,
    max_drawdown_pct: -11.7,
    total_trades: 204,
  },
];

// ─── Heatmap placeholder ───────────────────────────────────────────────────

const HEATMAP_MODELS = ["logistic", "xgboost", "lstm", "cnn"];
const HEATMAP_PAIRS  = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"];

const CELL_COLORS = [
  // 4×4 Sharpe value palette (negative → positive)
  ["#2A1A1A","#3B1F1F","#2A2E39","#1B2F2A"],
  ["#1E222D","#1B2F2A","#1B3B2E","#0F2F26"],
  ["#2A1A1A","#1E222D","#164430","#0D5A38"],
  ["#3B1F1F","#2A2E39","#0D5A38","#0A6640"],
];

function HeatmapPlaceholder() {
  return (
    <div
      style={{
        backgroundColor: "#1E222D",
        border: "1px solid #2A2E39",
        borderRadius: 4,
        padding: "12px 14px",
      }}
    >
      <div className="flex items-center justify-between" style={{ marginBottom: 10 }}>
        <span style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", color: "#4B5563", fontWeight: 600 }}>
          Model &times; Pair Performance
        </span>
        <span style={{ fontSize: 9, color: "#2A2E39", fontFamily: "var(--font-mono)", letterSpacing: "0.06em" }}>
          SHARPE
        </span>
      </div>

      {/* Column headers (pairs) */}
      <div className="flex" style={{ marginLeft: 80, marginBottom: 4, gap: 4 }}>
        {HEATMAP_PAIRS.map((p) => (
          <div key={p} style={{ width: 64, fontSize: 9, color: "#4B5563", textAlign: "center", letterSpacing: "0.06em" }}>
            {p}
          </div>
        ))}
      </div>

      {/* Grid rows */}
      {HEATMAP_MODELS.map((model, mi) => (
        <div key={model} className="flex items-center" style={{ gap: 4, marginBottom: 4 }}>
          <div style={{ width: 76, fontSize: 9, color: "#787B86", textAlign: "right", paddingRight: 8, letterSpacing: "0.04em" }}>
            {model}
          </div>
          {HEATMAP_PAIRS.map((_, pi) => (
            <div
              key={pi}
              style={{
                width: 64,
                height: 28,
                backgroundColor: CELL_COLORS[mi]?.[pi] ?? "#1E222D",
                border: "1px solid #2A2E39",
                borderRadius: 2,
              }}
            />
          ))}
        </div>
      ))}

      <div style={{ marginTop: 8, fontSize: 9, color: "#2A2E39", textAlign: "center", letterSpacing: "0.06em", fontFamily: "var(--font-mono)" }}>
        HEATMAP RENDERS AFTER FIRST COMPLETED BACKTEST
      </div>
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────

export function ResultsHistoryPage() {
  const navigate = useNavigate();
  const deleteJob = useDeleteJob();
  const setField = useBacktestStore((s) => s.setField);
  const [search, setSearch] = useState("");
  const [pairFilter, setPairFilter] = useState("");
  const [sortBy, setSortBy] = useState("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const { data, isLoading } = useResultsHistory({ pair: pairFilter, sort_by: sortBy, sort_order: sortOrder, limit: 100 });
  const { data: heatmapData, isLoading: heatmapLoading } = useHeatmap();

  const liveResults = data?.results ?? [];
  const results = liveResults.length > 0 ? liveResults : MOCK_ROWS;
  const isMock = liveResults.length === 0 && !isLoading;

  const filtered = search
    ? results.filter((r) => r.job_id.includes(search) || r.pair.toLowerCase().includes(search.toLowerCase()))
    : results;

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const handleSort = (col: string) => {
    if (sortBy === col) setSortOrder((p) => (p === "asc" ? "desc" : "asc"));
    else { setSortBy(col); setSortOrder("desc"); }
  };

  const exportCSV = useCallback(() => {
    const rows = filtered.filter((r) => selected.has(r.job_id));
    if (!rows.length) return;
    const csv = [
      "job_id,created_at,pair,models,sharpe,total_return_pct,win_rate,max_drawdown_pct",
      ...rows.map((r) => [r.job_id, r.created_at, r.pair, r.models.join(";"), r.sharpe ?? "", r.total_return_pct ?? "", r.win_rate ?? "", r.max_drawdown_pct ?? ""].join(",")),
    ].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "backtest_results.csv";
    a.click();
  }, [filtered, selected]);

  const handleRerun = useCallback(async (row: BacktestSummaryItem) => {
    setField("parentJobId", row.job_id);
    setField("pair", row.pair || "EURUSD");
    if (row.models?.length) setField("selectedModels", row.models as string[]);
    navigate("/backtest");
  }, [setField, navigate]);

  const PAIRS = [...new Set(results.map((r) => r.pair).filter(Boolean))].sort();

  return (
    <div className="flex flex-col gap-4">

      {/* Top bar */}
      <div className="flex items-center gap-3">
        {/* Search */}
        <div className="relative" style={{ flex: "0 0 260px" }}>
          <Search
            size={12}
            strokeWidth={1.5}
            className="absolute left-2.5 top-1/2 -translate-y-1/2"
            style={{ color: "#4B5563" }}
          />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by job ID or pair…"
            style={{
              width: "100%",
              height: 28,
              paddingLeft: 28,
              paddingRight: 8,
              backgroundColor: "#1E222D",
              border: "1px solid #2A2E39",
              borderRadius: 3,
              color: "#D1D4DC",
              fontSize: 11,
              outline: "none",
              fontFamily: "var(--font-mono)",
            }}
          />
        </div>

        {/* Pairs filter */}
        <select
          value={pairFilter}
          onChange={(e) => setPairFilter(e.target.value)}
          style={{
            height: 28,
            padding: "0 8px",
            backgroundColor: "#1E222D",
            border: "1px solid #2A2E39",
            borderRadius: 3,
            color: "#787B86",
            fontSize: 11,
            outline: "none",
            fontFamily: "var(--font-mono)",
            cursor: "pointer",
          }}
        >
          <option value="">All Pairs</option>
          {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>

        <div style={{ flex: 1 }} />

        {/* Export */}
        {selected.size > 0 && (
          <button
            type="button"
            onClick={exportCSV}
            className="flex items-center gap-1.5"
            style={{
              height: 28,
              padding: "0 10px",
              backgroundColor: "transparent",
              border: "1px solid #2A2E39",
              borderRadius: 3,
              color: "#787B86",
              fontSize: 10,
              letterSpacing: "0.06em",
              cursor: "pointer",
              fontFamily: "inherit",
            }}
          >
            <Download size={11} strokeWidth={1.5} />
            CSV ({selected.size})
          </button>
        )}

        <span style={{ fontSize: 10, color: "#4B5563", fontFamily: "var(--font-mono)" }}>
          {isLoading ? "Loading…" : `${filtered.length} record${filtered.length !== 1 ? "s" : ""}${isMock ? " (mock)" : ""}`}
        </span>
      </div>

      {/* Data grid */}
      <div style={{ border: "1px solid #2A2E39", borderRadius: 4, overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ ...TH, width: 32 }}>
                <input
                  type="checkbox"
                  style={{ cursor: "pointer", accentColor: "#3B82F6" }}
                  checked={selected.size === filtered.length && filtered.length > 0}
                  onChange={() => {
                    if (selected.size === filtered.length) setSelected(new Set());
                    else setSelected(new Set(filtered.map((r) => r.job_id)));
                  }}
                />
              </th>
              <SortTH label="Date"     active={sortBy === "created_at"}        direction={sortOrder} onClick={() => handleSort("created_at")} />
              <th style={TH}>Pair</th>
              <th style={TH}>Models</th>
              <SortTH label="Sharpe"   active={sortBy === "sharpe"}             direction={sortOrder} onClick={() => handleSort("sharpe")}    align="right" />
              <SortTH label="Return"   active={sortBy === "total_return_pct"}   direction={sortOrder} onClick={() => handleSort("total_return_pct")} align="right" />
              <SortTH label="Win Rate" active={sortBy === "win_rate"}           direction={sortOrder} onClick={() => handleSort("win_rate")}  align="right" />
              <SortTH label="Max DD"   active={sortBy === "max_drawdown_pct"}   direction={sortOrder} onClick={() => handleSort("max_drawdown_pct")} align="right" />
              <th style={{ ...TH, textAlign: "right" }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={9} style={{ ...TD, textAlign: "center", color: "#4B5563", padding: "24px 0" }}>
                  Loading…
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={9} style={{ ...TD, textAlign: "center", color: "#4B5563", padding: "24px 0" }}>
                  No results found.
                </td>
              </tr>
            ) : (
              filtered.map((row, i) => {
                const isSel = selected.has(row.job_id);
                const retPos = (row.total_return_pct ?? 0) > 0;
                const sharpePos = (row.sharpe ?? 0) > 0;
                return (
                  <tr
                    key={`${row.job_id}-${i}`}
                    onClick={() => !isMock && navigate(`/results/${row.job_id}`)}
                    style={{
                      backgroundColor: i % 2 === 0 ? "#131722" : "#1E222D",
                      cursor: isMock ? "default" : "pointer",
                      borderBottom: "1px solid #1E222D",
                    }}
                  >
                    <td style={{ ...TD, width: 32 }} onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        style={{ cursor: "pointer", accentColor: "#3B82F6" }}
                        checked={isSel}
                        onChange={() => toggleSelect(row.job_id)}
                      />
                    </td>
                    <td style={{ ...TD, color: "#4B5563" }}>{formatRelativeTime(row.created_at)}</td>
                    <td style={{ ...TD, color: "#D1D4DC" }}>{row.pair}</td>
                    <td style={{ ...TD, color: "#787B86", fontFamily: "inherit", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis" }}>
                      {row.models.join(", ")}
                    </td>
                    <td style={{ ...TD, textAlign: "right", color: sharpePos ? "#089981" : "#F23645" }}>
                      {row.sharpe != null ? row.sharpe.toFixed(2) : "—"}
                    </td>
                    <td style={{ ...TD, textAlign: "right", color: retPos ? "#089981" : "#F23645" }}>
                      {row.total_return_pct != null ? formatPercent(row.total_return_pct) : "—"}
                    </td>
                    <td style={{ ...TD, textAlign: "right", color: "#D1D4DC" }}>
                      {row.win_rate != null ? formatPercent(row.win_rate) : "—"}
                    </td>
                    <td style={{ ...TD, textAlign: "right", color: "#F23645" }}>
                      {row.max_drawdown_pct != null ? formatPercent(row.max_drawdown_pct) : "—"}
                    </td>
                    <td style={{ ...TD, textAlign: "right" }} onClick={(e) => e.stopPropagation()}>
                      <div className="flex items-center justify-end gap-1">
                        <button
                          type="button"
                          onClick={() => handleRerun(row)}
                          title="Re-run"
                          style={{ padding: 4, color: "#4B5563", background: "none", border: "none", cursor: "pointer", borderRadius: 2 }}
                        >
                          <RefreshCw size={12} strokeWidth={1.5} />
                        </button>
                        {!isMock && (
                          <button
                            type="button"
                            onClick={() => navigate(`/results/${row.job_id}`)}
                            title="View"
                            style={{ padding: 4, color: "#4B5563", background: "none", border: "none", cursor: "pointer", borderRadius: 2 }}
                          >
                            <Eye size={12} strokeWidth={1.5} />
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => !isMock && deleteJob.mutate(row.job_id)}
                          title="Delete"
                          style={{ padding: 4, color: "#4B5563", background: "none", border: "none", cursor: isMock ? "default" : "pointer", borderRadius: 2 }}
                        >
                          <Trash2 size={12} strokeWidth={1.5} />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Heatmap */}
      {heatmapData && heatmapData.models.length > 0 && !heatmapLoading ? (
        // Real heatmap via existing PerformanceHeatmapSection
        <HeatmapPlaceholder />
      ) : (
        <HeatmapPlaceholder />
      )}
    </div>
  );
}
