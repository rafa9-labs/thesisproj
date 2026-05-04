import { Outlet } from "react-router-dom";
import { useState, useEffect } from "react";
import { layout } from "@/lib/tokens";
import { TerminalPanel } from "./TerminalPanel";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { useHealth } from "@/api/queries";
import { wsManager } from "@/api/websocket";
import { UpdateNotification } from "../UpdateNotification/UpdateNotification";

export function AppShell() {
  const [collapsed, setCollapsed] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const { data: health } = useHealth();

  useEffect(() => {
    const interval = setInterval(() => {
      setWsConnected(wsManager.connected);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex h-full w-full" style={{ backgroundColor: "var(--color-app)" }}>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />

      <div className="flex flex-1 flex-col overflow-hidden" style={{ minWidth: 0 }}>
        <TopBar sidebarCollapsed={collapsed} onToggleSidebar={() => setCollapsed(!collapsed)} />

        <div
          className="flex-1 overflow-y-auto px-6 py-6 animate-fade-in"
          style={{ backgroundColor: "var(--color-app)" }}
        >
          <Outlet />
        </div>

        <TerminalPanel />
        <UpdateNotification />

        <footer
          className="flex items-center justify-between px-5"
          style={{
            height: layout.statusBarHeight,
            borderTop: "1px solid rgba(255,255,255,0.04)",
            backgroundColor: "var(--color-surface)",
            color: "var(--color-text-muted)",
            fontSize: "10px",
            fontWeight: 300,
            fontFamily: "var(--font-mono)",
            flexShrink: 0,
          }}
        >
          <div className="flex items-center gap-3"
          >
            <StatusDot
              color={health?.status === "ok" ? "var(--color-brand)" : "var(--color-accent-danger)"}
              label="API"
              pulse={health?.status === "ok"}
            />
            <StatusDot
              color={wsConnected ? "var(--color-brand)" : "var(--color-text-muted)"}
              label="WS"
              pulse={wsConnected}
            />
          </div>
          <span style={{ letterSpacing: "0.04em" }}>v1.0.0 — KodaQuant</span>
        </footer>
      </div>
    </div>
  );
}

function StatusDot({
  color,
  label,
  pulse,
}: {
  color: string;
  label: string;
  pulse?: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="relative">
        <div className="h-[5px] w-[5px] rounded-full" style={{ backgroundColor: color }} />
        {pulse && (
          <div
            className="absolute inset-0 animate-ping-brand rounded-full"
            style={{ backgroundColor: color, opacity: 0.4 }}
          />
        )}
      </div>
      <span className="text-[10px]" style={{ color: "var(--color-text-muted)", fontWeight: 300 }}>
        {label}
      </span>
    </div>
  );
}
