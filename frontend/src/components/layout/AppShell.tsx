import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { useState, useEffect, useMemo } from "react";

import { TerminalPanel } from "./TerminalPanel";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { RunningJobsBar } from "./RunningJobsBar";
import { useHealth, useActiveBacktests, useFullCycleHistory } from "@/api/queries";
import { wsManager } from "@/api/websocket";
import { useBacktestWebSocket } from "@/hooks/useBacktestWebSocket";
import { useJobNotifications } from "@/hooks/useJobNotifications";
import { useJobStore } from "@/stores/useJobStore";
import { UpdateNotification } from "../UpdateNotification/UpdateNotification";
import { DataSourceModal } from "../onboarding/DataSourceModal";
import { useAppStore } from "@/stores/useAppStore";

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
  const { data: fcHistory } = useFullCycleHistory();
  const { setDemoMode } = useAppStore();
  const activeJobs = useJobStore((s) => s.activeJobs);
  const ensureJob = useJobStore((s) => s.ensureJob);
  const navigate = useNavigate();
  const location = useLocation();

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
      if (!(activeJobs instanceof Map) || !activeJobs.has(job.job_id)) {
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

  // Redirect to Dashboard when no visible jobs and user is viewing a dead study
  useEffect(() => {
    if (!location.pathname.startsWith("/monitor")) return;
    const rawJobs = activeJobs instanceof Map ? activeJobs : new Map();
    const hasBacktestJobs = [...rawJobs.values()].some(
      (j) => j.status === "pending" || j.status === "running",
    );
    const hasCommitteeJobs = (fcHistory?.entries ?? []).some(
      (e) => e.status !== "orphaned" && e.status !== "cancelled",
    );
    if (!hasBacktestJobs && !hasCommitteeJobs) {
      navigate("/", { replace: true });
    }
  }, [activeJobs, fcHistory, location.pathname, navigate]);

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
