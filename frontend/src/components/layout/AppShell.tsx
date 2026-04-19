import { Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  BarChart3,
  GitCompare,
  Newspaper,
  Settings,
} from "lucide-react";
import { useState } from "react";
import { layout } from "@/lib/tokens";
import { TerminalPanel } from "./TerminalPanel";
import { useHealth } from "@/api/queries";
import { wsManager } from "@/api/websocket";

const navItems = [
  { icon: LayoutDashboard, label: "DASHBOARD", path: "/" },
  { icon: FlaskConical, label: "BACKTEST", path: "/backtest" },
  { icon: BarChart3, label: "RESULTS", path: "/results" },
  { icon: GitCompare, label: "COMPARE", path: "/compare" },
  { icon: Newspaper, label: "NEWS", path: "/news" },
  { icon: Settings, label: "SETTINGS", path: "/settings" },
];

export function AppShell() {
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const { data: health } = useHealth();
  const sidebarWidth = collapsed ? layout.sidebarCollapsed : layout.sidebarExpanded;

  useState(() => {
    const interval = setInterval(() => {
      setWsConnected(wsManager.connected);
    }, 3000);
    return () => clearInterval(interval);
  });

  return (
    <div className="flex h-full w-full flex-col" style={{ backgroundColor: "var(--color-app)" }}>
      <div className="flex flex-1 overflow-hidden">
        <aside
          className="flex flex-col border-r transition-all duration-200"
          style={{
            width: sidebarWidth,
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-surface)",
          }}
          onMouseEnter={() => setCollapsed(false)}
          onMouseLeave={() => setCollapsed(true)}
        >
          {navItems.map((item) => {
            const isActive =
              item.path === "/"
                ? location.pathname === "/"
                : location.pathname.startsWith(item.path);
            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className="flex items-center gap-3 px-4 py-3 text-left transition-colors duration-150 hover:bg-[var(--color-elevated)]"
                style={{
                  borderLeft: isActive ? "3px solid var(--color-accent)" : "3px solid transparent",
                  color: isActive ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                  backgroundColor: isActive ? "var(--color-elevated)" : "transparent",
                }}
                title={item.label}
              >
                <item.icon size={20} style={{ minWidth: 20 }} />
                {!collapsed && (
                  <span
                    className="whitespace-nowrap text-[11px] font-semibold uppercase tracking-[0.1em]"
                    style={{ fontFamily: "var(--font-sans)" }}
                  >
                    {item.label}
                  </span>
                )}
              </button>
            );
          })}
        </aside>

        <main className="flex flex-1 flex-col overflow-hidden">
          <header
            className="flex items-center justify-between border-b px-4"
            style={{
              height: layout.headerHeight,
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-surface)",
            }}
          >
            <div className="flex items-center gap-3">
              <h1 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
                FX ML Backtester
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <StatusDot color={health?.status === "ok" ? "var(--color-accent-success)" : "var(--color-accent-danger)}"} label="Backend" pulse={health?.status === "ok"} />
              <StatusDot color={wsConnected ? "var(--color-accent-success)" : "var(--color-text-muted)"} label="WS" />
            </div>
          </header>

          <div className="flex-1 overflow-y-auto p-6" style={{ backgroundColor: "var(--color-app)" }}>
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
            }}
          >
            <span>Idle</span>
            <span>v1.0.0</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

function StatusDot({ color, label, pulse }: { color: string; label: string; pulse?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="relative">
        <div
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: color }}
        />
        {pulse && (
          <div
            className="absolute inset-0 animate-ping rounded-full"
            style={{ backgroundColor: color, opacity: 0.3 }}
          />
        )}
      </div>
      <span className="text-[11px]" style={{ color: "var(--color-text-muted)" }}>
        {label}
      </span>
    </div>
  );
}
