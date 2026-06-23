import { Outlet } from "react-router-dom";
import { useState, useEffect, useMemo, useRef } from "react";

import { TerminalPanel } from "./TerminalPanel";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { RunningJobsBar } from "./RunningJobsBar";
import { useHealth, useActiveBacktests, _progressCursors } from "@/api/queries";
import { apiClient } from "@/api/client";
import { wsManager } from "@/api/websocket";
import { useBacktestWebSocket } from "@/hooks/useBacktestWebSocket";
import { useJobNotifications } from "@/hooks/useJobNotifications";
import { useJobStore } from "@/stores/useJobStore";
import { UpdateNotification } from "../UpdateNotification/UpdateNotification";
import { DataSourceModal } from "../onboarding/DataSourceModal";
import { useAppStore } from "@/stores/useAppStore";
import type { WsEvent } from "@/api/schemas";

const DS_CHOSEN_KEY = "fx-datasource-chosen";

function hasChosenDataSource(): boolean {
  try {
    return localStorage.getItem(DS_CHOSEN_KEY) === "true";
  } catch {
    return false;
  }
}

function markDataSourceChosen(): void {
  try {
    localStorage.setItem(DS_CHOSEN_KEY, "true");
  } catch {
    /* ignore */
  }
}

export function AppShell() {
  const [wsConnected, setWsConnected] = useState(false);
  const [showDataSource, setShowDataSource] = useState(!hasChosenDataSource());
  const { data: health } = useHealth();
  const { data: activeData } = useActiveBacktests();
  const { setDemoMode } = useAppStore();
  const activeJobs = useJobStore((s) => s.activeJobs);
  const ensureJob = useJobStore((s) => s.ensureJob);

  // Job completion notifications (sound + desktop)
  useJobNotifications();

  useEffect(() => {
    const interval = setInterval(() => {
      setWsConnected(wsManager.connected);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const jobs = activeData?.jobs ?? [];
    for (const job of jobs) {
      const jobsMap = activeJobs instanceof Map ? activeJobs : new Map();
      const local = jobsMap.get(job.job_id);
      if (!local || local.status === "stale") {
        ensureJob(job.job_id, job.pair ?? "", job.models ?? []);
      }
    }
  }, [activeData?.jobs, activeJobs, ensureJob]);

  const runningIds = useMemo(() => {
    const ids: string[] = [];
    if (activeJobs instanceof Map) {
      activeJobs.forEach((job) => {
        if (job.status === "pending" || job.status === "running") {
          ids.push(job.jobId);
        }
      });
    }
    return ids;
  }, [activeJobs]);

  useBacktestWebSocket(runningIds);

  // Continuous REST fallback poll — keeps event cursor advancing even when
  // MonitorPage is not mounted, so returning to Monitor shows current data.
  // On first poll for a job, skip replaying old events by advancing cursor
  // to the current total. This prevents rehydrated jobs from appearing to
  // restart by replaying historical events.
  const handleWsEventRef = useRef(useJobStore.getState().handleWsEvent);
  useEffect(() => {
    handleWsEventRef.current = useJobStore.getState().handleWsEvent;
  });
  const syncedCursorRef = useRef(new Set<string>());
  useEffect(() => {
    if (runningIds.length === 0) return;
    const interval = setInterval(() => {
      for (const id of runningIds) {
        const synced = syncedCursorRef.current.has(id);
        const cursor = _progressCursors.get(id) ?? 0;
        if (!synced && cursor === 0) {
          // First fetch: get total count, advance cursor past all old events
          apiClient
            .get<{ events: WsEvent[]; total: number }>(`/backtest/${id}/events?after=0`)
            .then(({ data }) => {
              _progressCursors.set(id, data.total);
              syncedCursorRef.current.add(id);
            })
            .catch(() => {});
        } else {
          apiClient
            .get<{ events: WsEvent[]; total: number }>(`/backtest/${id}/events?after=${cursor}`)
            .then(({ data }) => {
              if (data.events && data.events.length > 0) {
                for (const evt of data.events) {
                  handleWsEventRef.current(evt);
                }
                _progressCursors.set(id, data.total);
              }
            })
            .catch(() => {});
        }
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [runningIds]);

  // ── Dead-study redirect removed — MonitorPage handles empty state with routing UI ──

  const handleStart = (mode: string) => {
    markDataSourceChosen();
    if (mode === "demo") {
      setDemoMode(true);
    }
    setShowDataSource(false);
  };

  return (
    <div className="flex h-full w-full bg-(--color-app)">
      <DataSourceModal
        isOpen={showDataSource}
        onBack={() => setShowDataSource(false)}
        onStart={handleStart}
      />

      <Sidebar />

      <div className="flex flex-1 flex-col overflow-hidden" style={{ minWidth: 0 }}>
        <TopBar />

        <RunningJobsBar />

        <div className="flex-1 animate-fade-in overflow-y-auto bg-(--color-app) px-6 py-4">
          <Outlet />
        </div>

        <TerminalPanel apiOk={health?.status === "ok"} wsConnected={wsConnected} />
        <UpdateNotification />
      </div>
    </div>
  );
}
