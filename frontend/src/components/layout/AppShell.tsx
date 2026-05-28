import { Outlet } from "react-router-dom";
import { useState, useEffect } from "react";

import { TerminalPanel } from "./TerminalPanel";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { useHealth } from "@/api/queries";
import { wsManager } from "@/api/websocket";
import { UpdateNotification } from "../UpdateNotification/UpdateNotification";
import { DataSourceModal } from "../onboarding/DataSourceModal";

export function AppShell() {
  const [wsConnected, setWsConnected] = useState(false);
  const [showDataSource, setShowDataSource] = useState(true);
  const { data: health } = useHealth();

  useEffect(() => {
    const interval = setInterval(() => {
      setWsConnected(wsManager.connected);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-full w-full" style={{ backgroundColor: "var(--color-app)" }}>
      <DataSourceModal
        isOpen={showDataSource}
        onBack={() => setShowDataSource(false)}
        onStart={(_mode, _value) => setShowDataSource(false)}
        onSkip={() => setShowDataSource(false)}
      />

      <Sidebar />

      <div className="flex flex-1 flex-col overflow-hidden" style={{ minWidth: 0 }}>
        <TopBar />

        <div
          className="flex-1 overflow-y-auto px-6 py-4 animate-fade-in"
          style={{ backgroundColor: "var(--color-app)" }}
        >
          <Outlet />
        </div>

        <TerminalPanel
          apiOk={health?.status === "ok"}
          wsConnected={wsConnected}
        />
        <UpdateNotification />
      </div>
    </div>
  );
}
