import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { X } from "lucide-react";
import { useJobStore } from "@/stores/useJobStore";
import { useCommitteeMonitorStore } from "@/stores/useCommitteeMonitorStore";
import { useFullCycleHistory } from "@/api/queries";
import { wsManager } from "@/api/websocket";

const MAX_VISIBLE = 5;

const PHASE_MAP: Record<string, { badge: string; label: string; color: string }> = {
  feature_sweep: { badge: "P1", label: "Sweep", color: "#089981" },
  phase1_hpo: { badge: "P2", label: "HPO", color: "#2962FF" },
  phase2_assembly: { badge: "P3", label: "Build", color: "#E5A014" },
  phase3_validation: { badge: "P4", label: "Valid", color: "#A78BFA" },
  phase4_factory: { badge: "P5", label: "Factory", color: "#F23645" },
  completed: { badge: "OK", label: "Done", color: "#089981" },
  failed: { badge: "ERR", label: "Fail", color: "#F23645" },
  validation_failed: { badge: "ERR", label: "Fail", color: "#F23645" },
};

function phaseInfo(status: string) {
  return PHASE_MAP[status] ?? { badge: "?", label: status.slice(0, 6), color: "var(--color-text-dim)" };
}

function loadDismissed(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem("dismissedCommitteeJobs") || "[]"));
  } catch {
    return new Set();
  }
}

function saveDismissed(ids: Set<string>) {
  localStorage.setItem("dismissedCommitteeJobs", JSON.stringify([...ids]));
}

type BarItem =
  | { kind: "backtest"; jobId: string; label: string; progress: string; wsDot: string }
  | { kind: "committee_running"; jobId: string; label: string; progress: string; badge: string; color: string }
  | { kind: "committee_done"; jobId: string; label: string; progress: string; badge: string; color: string };

export function RunningJobsBar() {
  const navigate = useNavigate();
  const activeJobs = useJobStore((s) => s.activeJobs);
  const selectJob = useJobStore((s) => s.selectJob);
  const setActiveTab = useJobStore((s) => s.setActiveTab);
  const cmSelectJob = useCommitteeMonitorStore((s) => s.selectJob);
  const { data: fcHistory } = useFullCycleHistory();
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(loadDismissed);

  const dismiss = useCallback(
    (jobId: string) => {
      const next = new Set(dismissedIds);
      next.add(jobId);
      setDismissedIds(next);
      saveDismissed(next);
    },
    [dismissedIds],
  );

  // ── Backtest items ──
  const rawJobs = activeJobs instanceof Map ? activeJobs : new Map();
  const backtestRunning = [...rawJobs.values()]
    .filter((j) => j.status === "pending" || j.status === "running")
    .map<BarItem>((j) => ({
      kind: "backtest",
      jobId: j.jobId,
      label:
        (Array.isArray(j.models) ? j.models : []).slice(0, 2).join("+") +
        (Array.isArray(j.models) && j.models.length > 2 ? "..." : ""),
      progress: `${Math.round(j.progress)}%`,
      wsDot: wsManager.isConnected(j.jobId) ? "var(--color-brand)" : "var(--color-text-dim)",
    }));

  // ── Committee items ──
  const allEntries = (fcHistory?.entries ?? [])
    .filter((e) => e.status !== "orphaned" && e.status !== "cancelled" && e.status !== "unknown" && !dismissedIds.has(e.job_id))
    .sort((a, b) => b.started_at.localeCompare(a.started_at));

  const committeeItems: BarItem[] = [];
  for (const e of allEntries) {
    const isRunning =
      e.status !== "completed" && e.status !== "failed" && e.status !== "validation_failed";
    const info = phaseInfo(e.status);
    committeeItems.push({
      kind: isRunning ? "committee_running" : "committee_done",
      jobId: e.job_id,
      label: e.job_id.slice(-8),
      progress: info.label,
      badge: info.badge,
      color: info.color,
    });
  }

  // ── Build priority-ordered list and cap at MAX_VISIBLE ──
  const committeeRunning = committeeItems.filter((i) => i.kind === "committee_running");
  const committeeDone = committeeItems.filter((i) => i.kind === "committee_done");

  const allItems: BarItem[] = [
    ...backtestRunning,
    ...committeeRunning,
    ...committeeDone,
  ];

  const visible = allItems.slice(0, MAX_VISIBLE);
  const overflow = Math.max(0, allItems.length - MAX_VISIBLE);
  const totalRunning = backtestRunning.length + committeeRunning.length;

  if (visible.length === 0) {
    return (
      <div className="flex h-8 shrink-0 items-center border-b border-(--color-glass-border) bg-(--color-app)/60 px-6">
        <span className="text-[10px] text-(--color-text-dim)">No jobs running</span>
      </div>
    );
  }

  const handleBacktest = (jobId: string) => {
    selectJob(jobId);
    setActiveTab(jobId, "hpo-and-results");
    navigate("/monitor");
  };

  const handleCommittee = (jobId: string) => {
    cmSelectJob(jobId);
    navigate(`/committee?jobId=${jobId}`);
  };

  return (
    <div className="flex h-8 shrink-0 items-center gap-2 overflow-x-hidden border-b border-(--color-glass-border) bg-(--color-app)/60 px-6">
      <span className="shrink-0 text-[9px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
        {totalRunning > 0 ? `Running (${totalRunning})` : "Recent"}
      </span>

      {visible.map((item) => {
        if (item.kind === "backtest") {
          return (
            <JobPill
              key={`bt-${item.jobId}`}
              onClick={() => handleBacktest(item.jobId)}
              dotColor={item.wsDot}
              badge="SINGLE"
              badgeColor="var(--color-text-dim)"
              badgeBg="var(--color-border-subtle)"
              label={item.label}
              progress={item.progress}
            />
          );
        }
        return (
          <JobPill
            key={`cm-${item.jobId}`}
            onClick={() => handleCommittee(item.jobId)}
            dotColor={item.color}
            badge={item.badge}
            badgeColor={item.color}
            badgeBg={`${item.color}18`}
            label={item.label}
            progress={item.progress}
            onDismiss={() => dismiss(item.jobId)}
          />
        );
      })}

      {overflow > 0 && (
        <span className="shrink-0 text-[9px] font-mono text-(--color-text-dim)">
          +{overflow} more
        </span>
      )}
    </div>
  );
}

function JobPill({
  onClick,
  dotColor,
  badge,
  badgeColor,
  badgeBg,
  label,
  progress,
  onDismiss,
}: {
  onClick: () => void;
  dotColor: string;
  badge: string;
  badgeColor: string;
  badgeBg: string;
  label: string;
  progress: string;
  onDismiss?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group flex shrink-0 items-center gap-1.5 rounded px-2 py-1 text-[10px] transition"
      style={{
        border: "1px solid var(--color-glass-border)",
        backgroundColor: "var(--color-glass)",
        color: "var(--color-text-primary)",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor = "var(--color-brand)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.borderColor = "var(--color-glass-border)";
      }}
    >
      <span
        className="h-1.5 w-1.5 shrink-0 rounded-full"
        style={{ backgroundColor: dotColor }}
      />
      <span
        className="rounded-full px-1.5 py-px text-[7px] font-semibold uppercase tracking-[0.04em]"
        style={{ backgroundColor: badgeBg, color: badgeColor }}
      >
        {badge}
      </span>
      <span className="max-w-[100px] truncate font-mono text-[10px]">{label}</span>
      <span className="min-w-[22px] text-right font-mono text-[10px] tabular-nums text-(--color-text-dim)">
        {progress}
      </span>
      {onDismiss && (
        <span
          className="ml-0.5 hidden shrink-0 rounded-full p-[2px] opacity-0 transition-opacity group-hover:inline-flex group-hover:opacity-100"
          style={{ color: "var(--color-text-dim)" }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.color = "var(--color-accent-danger)";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = "var(--color-text-dim)";
          }}
          onClick={(e) => {
            e.stopPropagation();
            onDismiss();
          }}
          title="Dismiss"
        >
          <X size={10} strokeWidth={2.5} />
        </span>
      )}
    </button>
  );
}
