import { useState } from "react";
import { useFullCycleHistory } from "@/api/queries";
import { useFullCycleStore } from "@/stores/useFullCycleStore";
import apiClient from "@/api/client";
import type { FullCycleHistoryEntry } from "@/api/schemas";

interface Props {
  activeJobId: string | null;
  onSelect: (jobId: string) => void;
}

export function RunHistoryTable({ activeJobId, onSelect }: Props) {
  const [showHistory, setShowHistory] = useState(true);
  const { data: history } = useFullCycleHistory();
  const store = useFullCycleStore();

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

  const terminalPhases = new Set([
    "completed",
    "failed",
    "validation_failed",
    "cancelled",
    "orphaned",
  ]);

  return (
    <div className="rounded-[4px] border border-(--color-glass-border) bg-(--color-surface) p-[20px]">
      <div
        className="flex items-center justify-between"
        style={{ marginBottom: history?.entries?.length ? 14 : 0 }}
      >
        <span className="text-[11px] font-semibold tracking-[0.08em] text-(--color-text-primary) uppercase">
          Run History{history?.total_runs ? ` (${history.total_runs})` : ""}
        </span>
        <button
          onClick={() => setShowHistory(!showHistory)}
          className="cursor-pointer border-none bg-transparent font-mono text-[10px] text-(--color-text-muted)"
        >
          {showHistory ? "Hide" : "Show"}
        </button>
      </div>
      {showHistory && (
        <>
          {history && history.entries.length === 0 && (
            <div className="p-[16px] text-center text-[10px] text-(--color-text-dim)">
              No history yet. Run your first Full Cycle to see results here.
            </div>
          )}
          {history && history.entries.length > 0 && (
            <table className="w-full border-collapse text-[10px]">
              <thead>
                <tr>
                  <th className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES}`}>
                    Started
                  </th>
                  <th className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES}`}>
                    Status
                  </th>
                  <th className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES}`}>
                    Time
                  </th>
                  <th className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES}`}>
                    Locked
                  </th>
                  <th className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES}`}>
                    Survivors
                  </th>
                  <th
                    className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES} text-right`}
                  >
                    Sharpe
                  </th>
                  <th className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES}`}></th>
                  <th
                    className={`border-b border-(--color-glass-border) ${HIST_TH_CLASSES} text-right`}
                  ></th>
                </tr>
              </thead>
              <tbody>
                {history.entries.map((entry: FullCycleHistoryEntry) => {
                  const isActive = entry.job_id === activeJobId;
                  const isTerminal = terminalPhases.has(entry.status);
                  const isInProgress = !isTerminal && entry.status !== "unknown";
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
                  const sharpeStr = entry.avg_sharpe !== 0 ? entry.avg_sharpe.toFixed(2) : "--";
                  const sharpeColor =
                    entry.avg_sharpe > 0
                      ? "#089981"
                      : entry.avg_sharpe < 0
                        ? "#F23645"
                        : "var(--color-text-dim)";
                  const statusColor = isInProgress
                    ? "var(--color-brand)"
                    : entry.status === "completed"
                      ? "#089981"
                      : entry.status === "validation_failed"
                        ? "#F2B436"
                        : entry.status === "cancelled"
                          ? "#F29C38"
                          : entry.status === "orphaned"
                            ? "#F29C38"
                            : "#F23645";
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
                      ? entry.survivors.slice(0, 3).join(",") +
                        (entry.survivors_count > 3 ? ` +${entry.survivors_count - 3}` : "")
                      : "--";
                  return (
                    <tr
                      key={entry.job_id}
                      className="border-b border-(--color-glass-border)"
                      style={{
                        background: isActive ? "rgba(0,229,255,0.04)" : "transparent",
                      }}
                    >
                      <td className="font-mono" style={{ opacity: isActive ? 1 : 0.7 }}>
                        {started}
                      </td>
                      <td className={HIST_TD_CLASSES}>
                        <span className="inline-flex items-center gap-[6px]">
                          <span
                            className="inline-block h-[6px] w-[6px] rounded-full"
                            style={{
                              background: statusColor,
                              animation: isInProgress ? "pulse 1.5s infinite" : "none",
                            }}
                          />
                          <span
                            style={{
                              color: isInProgress
                                ? "var(--color-brand)"
                                : entry.status === "completed"
                                  ? "#089981"
                                  : entry.status === "cancelled"
                                    ? "#F29C38"
                                    : "var(--color-text-secondary)",
                            }}
                          >
                            {statusLabel}
                          </span>
                        </span>
                      </td>
                      <td className="font-mono text-(--color-text-dim)">{timeStr}</td>
                      <td className="font-mono text-(--color-text-dim)">
                        {entry.locked_features_count > 0 ? entry.locked_features_count : "--"}
                      </td>
                      <td className="max-w-[140px] overflow-hidden font-mono text-ellipsis text-(--color-text-dim)">
                        {survivorsStr}
                      </td>
                      <td className="text-right font-mono" style={{ color: sharpeColor }}>
                        {sharpeStr}
                      </td>
                      <td className={HIST_TD_CLASSES}>
                        <button
                          onClick={() => onSelect(entry.job_id)}
                          disabled={isActive && !isTerminal}
                          className="border-none bg-transparent font-mono text-[9px] font-semibold tracking-[0.06em] uppercase"
                          style={{
                            cursor: isActive && !isTerminal ? "default" : "pointer",
                            color:
                              isActive && !isTerminal
                                ? "var(--color-text-dim)"
                                : "var(--color-brand)",
                          }}
                        >
                          {isActive && isTerminal ? "Viewing" : isActive ? "Active" : "Load"}
                        </button>
                      </td>
                      <td className="text-right">
                        {entry.status === "completed" && (entry.trust_score ?? 0) > 0.4 && (
                          <button
                            onClick={() => handleDeploy(entry)}
                            className="rounded border border-[rgba(8,153,129,0.25)] bg-[rgba(8,153,129,0.08)] px-2 py-1 font-mono text-[9px] font-semibold tracking-[0.06em] text-[#089981] uppercase hover:brightness-110"
                          >
                            Deploy
                          </button>
                        )}
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

const HIST_TH_CLASSES =
  "px-2 py-1 text-left font-medium tracking-[0.06em] uppercase text-[10px] text-(--color-text-muted)";
const HIST_TD_CLASSES = "px-2 py-1.5 text-[11px] text-(--color-text-secondary)";
