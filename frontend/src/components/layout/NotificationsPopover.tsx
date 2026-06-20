import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, CheckCircle2, XCircle, ExternalLink } from "lucide-react";
import { useJobStore } from "@/stores/useJobStore";
import { useSettingsStore } from "@/stores/useSettingsStore";

function relativeTime(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function NotificationsPopover({ onClose }: { onClose: () => void }) {
  const navigate = useNavigate();
  const completedJobs = useJobStore((s) => s.completedJobs);
  const settings = useSettingsStore();

  return (
    <div
      className="absolute right-0 top-full z-50 mt-2 w-80 rounded-sm border border-(--color-glass-border) bg-(--color-surface) shadow-[0_12px_32px_rgba(0,0,0,0.4)]"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between border-b border-(--color-glass-border) px-4 py-3">
        <div className="flex items-center gap-2">
          <Bell size={14} className="text-(--color-text-muted)" />
          <span className="text-[11px] font-semibold tracking-[0.06em] text-(--color-text-primary) uppercase">
            Notifications
          </span>
        </div>
        <button
          onClick={onClose}
          className="cursor-pointer text-[16px] text-(--color-text-muted) leading-none"
          style={{ border: "none", background: "none" }}
        >
          ×
        </button>
      </div>

      <div className="max-h-[320px] overflow-y-auto">
        {completedJobs.length === 0 ? (
          <div className="flex flex-col items-center gap-1 py-10 text-(--color-text-muted)">
            <Bell size={20} strokeWidth={1} />
            <span className="text-[11px]">No recent notifications</span>
            <span className="text-[9px] text-(--color-text-dim)">
              Job completions will appear here
            </span>
          </div>
        ) : (
          completedJobs
            .slice()
            .reverse()
            .map((job, i) => (
              <button
                key={job.jobId || i}
                onClick={() => {
                  onClose();
                  const type = job.jobId.includes("full-cycle") ? "committee" : "";
                  navigate(`/results/${job.jobId}${type ? `?type=${type}` : ""}`);
                }}
                className="flex w-full cursor-pointer items-start gap-3 border-b border-(--color-glass-border) px-4 py-3 text-left transition-colors hover:bg-(--color-elevated)"
                style={{ border: "none", background: "none" }}
              >
                {job.status === "completed" ? (
                  <CheckCircle2 size={14} className="mt-0.5 shrink-0 text-(--color-accent-success)" />
                ) : (
                  <XCircle size={14} className="mt-0.5 shrink-0 text-(--color-accent-danger)" />
                )}
                <div className="flex flex-1 flex-col gap-0.5">
                  <span className="text-[11px] font-medium text-(--color-text-primary)">
                    {job.models.length > 2
                      ? `${job.models.length} models`
                      : job.models.join(", ")}
                    {job.pair ? ` on ${job.pair}` : ""}
                  </span>
                  <span className="text-[10px] text-(--color-text-muted)">
                    {job.status === "completed" ? "Completed" : "Failed"} ·{" "}
                    {job.completedAt ? relativeTime(new Date(job.completedAt)) : "unknown"}
                  </span>
                  <span className="font-mono text-[9px] text-(--color-text-dim)">
                    {job.jobId.slice(0, 12)}...
                  </span>
                </div>
                <ExternalLink size={11} className="mt-0.5 shrink-0 text-(--color-text-dim)" />
              </button>
            ))
        )}
      </div>

      <div className="flex items-center justify-between border-t border-(--color-glass-border) px-4 py-2.5">
        <div className="flex flex-col">
          <span className="text-[10px] font-medium text-(--color-text-secondary)">
            Notification sounds
          </span>
          <span className="text-[9px] text-(--color-text-dim)">
            Play a chime when jobs finish
          </span>
        </div>
        <button
          onClick={() => {
            settings.setField("notificationSound", !settings.notificationSound);
            if (!settings.notificationSound && "Notification" in window && Notification.permission === "default") {
              Notification.requestPermission();
            }
          }}
          className="relative h-5 w-9 cursor-pointer rounded-full transition-colors"
          style={{
            border: "none",
            backgroundColor: settings.notificationSound
              ? "var(--color-brand)"
              : "var(--color-glass-border)",
          }}
          aria-label="Toggle notification sounds"
        >
          <div
            className="absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all"
            style={{
              left: settings.notificationSound ? "calc(100% - 18px)" : "2px",
            }}
          />
        </button>
      </div>
    </div>
  );
}
