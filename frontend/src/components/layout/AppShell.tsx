import { Outlet } from "react-router-dom";
import { useState, useEffect } from "react";
import { layout } from "@/lib/tokens";
import { TerminalPanel } from "./TerminalPanel";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";
import { useHealth } from "@/api/queries";
import { wsManager } from "@/api/websocket";

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

      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar sidebarCollapsed={collapsed} onToggleSidebar={() => setCollapsed(!collapsed)} />

        <div
          className="flex-1 overflow-y-auto px-6 py-5 animate-fade-in"
          style={{ backgroundColor: "var(--color-app)" }}
        >
          <Outlet />
        </div>

        <TerminalPanel />

        <footer
          className="flex items-center justify-between border-t px-4"
          style={{
            height: layout.statusBarHeight,
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-surface)",
            color: "var(--color-text-muted)",
            fontSize: "11px",
            fontFamily: "var(--font-mono)",
            flexShrink: 0,
          }}
        >
          <div className="flex items-center gap-3"
          >
            <StatusDot
              color={health?.status === "ok" ? "var(--color-accent-success)" : "var(--color-accent-danger)"}
              label="API"
              pulse={health?.status === "ok"}
            />
            <StatusDot
              color={wsConnected ? "var(--color-accent-success)" : "var(--color-text-muted)"}
              label="WS"
              pulse={wsConnected}
            />
          </div>
          <span>v1.0.0 — KodaQuant</span>
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
        <div className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
        {pulse && (
          <div
            className="absolute inset-0 animate-ping-brand rounded-full"
            style={{ backgroundColor: color, opacity: 0.4 }}
          />
        )}
      </div>
      <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
        {label}
      </span>
    </div>
  );
}
