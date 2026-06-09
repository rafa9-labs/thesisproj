import { useState } from "react";
import { useFullCycleHistory } from "@/api/queries";
import type { FullCycleHistoryEntry } from "@/api/schemas";

interface Props {
  activeJobId: string | null;
  onSelect: (jobId: string) => void;
}

export function RunHistoryTable({ activeJobId, onSelect }: Props) {
  const [showHistory, setShowHistory] = useState(true);
  const { data: history } = useFullCycleHistory();

  const terminalPhases = new Set(["completed", "failed", "validation_failed", "cancelled", "orphaned"]);

  return (
    <div style={{ background: "var(--color-surface)", border: "1px solid var(--color-glass-border)", borderRadius: 4, padding: 20 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: history?.entries?.length ? 14 : 0 }}>
        <span style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--color-text-primary)" }}>
          Run History{history?.total_runs ? ` (${history.total_runs})` : ""}
        </span>
        <button onClick={() => setShowHistory(!showHistory)} style={{ background: "none", border: "none", color: "var(--color-text-muted)", cursor: "pointer", fontSize: 10, fontFamily: "var(--font-mono)" }}>
          {showHistory ? "Hide" : "Show"}
        </button>
      </div>
      {showHistory && (
        <>
          {history && history.entries.length === 0 && (
            <div style={{ fontSize: 10, color: "var(--color-text-dim)", textAlign: "center", padding: 16 }}>No history yet. Run your first Full Cycle to see results here.</div>
          )}
          {history && history.entries.length > 0 && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
              <thead>
                <tr>
                  <th style={histThStyle}>Started</th>
                  <th style={histThStyle}>Status</th>
                  <th style={histThStyle}>Time</th>
                  <th style={histThStyle}>Locked</th>
                  <th style={histThStyle}>Survivors</th>
                  <th style={{ ...histThStyle, textAlign: "right" }}>Sharpe</th>
                  <th style={histThStyle}></th>
                </tr>
              </thead>
              <tbody>
                {history.entries.map((entry: FullCycleHistoryEntry) => {
                  const isActive = entry.job_id === activeJobId;
                  const isTerminal = terminalPhases.has(entry.status);
                  const isInProgress = !isTerminal && entry.status !== "unknown";
                  const started = entry.started_at ? new Date(entry.started_at).toLocaleString(undefined, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }) : "--";
                  const timeStr = entry.total_time_s > 0 ? (entry.total_time_s >= 3600 ? `${Math.floor(entry.total_time_s / 3600)}h ${Math.floor((entry.total_time_s % 3600) / 60)}m` : `${Math.floor(entry.total_time_s / 60)}m ${Math.floor(entry.total_time_s % 60)}s`) : "--";
                  const sharpeStr = entry.avg_sharpe !== 0 ? entry.avg_sharpe.toFixed(2) : "--";
                  const sharpeColor = entry.avg_sharpe > 0 ? "#089981" : entry.avg_sharpe < 0 ? "#F23645" : "var(--color-text-dim)";
                  const statusColor = isInProgress ? "var(--color-brand)"
                    : entry.status === "completed" ? "#089981"
                    : entry.status === "validation_failed" ? "#F2B436"
                    : entry.status === "cancelled" ? "#F29C38"
                    : entry.status === "orphaned" ? "#F29C38"
                    : "#F23645";
                  const statusLabel = isInProgress ? "Running"
                    : entry.status === "cancelled" ? "Cancelled"
                    : entry.status === "validation_failed" ? "Validation Failed"
                    : entry.status === "failed" ? "Failed"
                    : entry.status === "completed" ? "Completed"
                    : entry.status === "orphaned" ? "Interrupted"
                    : entry.status;
                  const survivorsStr = entry.survivors_count > 0 ? entry.survivors.slice(0, 3).join(",") + (entry.survivors_count > 3 ? ` +${entry.survivors_count - 3}` : "") : "--";
                  return (
                    <tr key={entry.job_id} style={{ borderBottom: "1px solid var(--color-glass-border)", background: isActive ? "rgba(0,229,255,0.04)" : "transparent" }}>
                      <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", opacity: isActive ? 1 : 0.7 }}>{started}</td>
                      <td style={histTdStyle}>
                        <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                          <span style={{ width: 6, height: 6, borderRadius: "50%", display: "inline-block", background: statusColor, animation: isInProgress ? "pulse 1.5s infinite" : "none" }} />
                          <span style={{ color: isInProgress ? "var(--color-brand)" : entry.status === "completed" ? "#089981" : entry.status === "cancelled" ? "#F29C38" : "var(--color-text-secondary)" }}>{statusLabel}</span>
                        </span>
                      </td>
                      <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)" }}>{timeStr}</td>
                      <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)" }}>{entry.locked_features_count > 0 ? entry.locked_features_count : "--"}</td>
                      <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", color: "var(--color-text-dim)", maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis" }}>{survivorsStr}</td>
                      <td style={{ ...histTdStyle, fontFamily: "var(--font-mono)", textAlign: "right", color: sharpeColor }}>{sharpeStr}</td>
                      <td style={histTdStyle}>
                        <button onClick={() => onSelect(entry.job_id)} disabled={isActive && !isTerminal} style={{ background: "none", border: "none", cursor: (isActive && !isTerminal) ? "default" : "pointer", color: (isActive && !isTerminal) ? "var(--color-text-dim)" : "var(--color-brand)", fontSize: 9, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", fontFamily: "var(--font-mono)" }}>
                          {isActive && isTerminal ? "Viewing" : isActive ? "Active" : "Load"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}

const histThStyle: React.CSSProperties = {
  padding: "4px 8px", textAlign: "left", color: "var(--color-text-muted)",
  fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase",
  fontSize: 10, borderBottom: "1px solid var(--color-glass-border)",
};

const histTdStyle: React.CSSProperties = {
  padding: "5px 8px", color: "var(--color-text-secondary)", fontSize: 11,
};
