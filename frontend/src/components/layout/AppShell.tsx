import { Outlet } from "react-router-dom";
import { useState, useEffect } from "react";

import { TerminalPanel } from "./TerminalPanel";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { useHealth } from "@/api/queries";
import { wsManager } from "@/api/websocket";
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
  const { setDemoMode } = useAppStore();

  useEffect(() => {
    const interval = setInterval(() => {
      setWsConnected(wsManager.connected);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

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

        <div className="flex-1 animate-fade-in overflow-y-auto bg-(--color-app) px-6 py-4">
          <Outlet />
        </div>

        <TerminalPanel apiOk={health?.status === "ok"} wsConnected={wsConnected} />
        <UpdateNotification />
      </div>
    </div>
  );
}
