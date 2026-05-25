import { useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  Eye,
  BarChart3,
  Newspaper,
  Settings,
  PanelLeftOpen,
  PanelLeftClose,
  Zap,
  Box,
} from "lucide-react";
import { KodaLogo } from "@/components/shared/KodaLogo";
import { layout } from "@/lib/tokens";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

interface NavItem {
  icon: React.ElementType;
  label: string;
  path: string;
}

interface NavGroup {
  label: string;
  items: NavItem[];
}

const navGroups: NavGroup[] = [
  {
    label: "Analyze",
    items: [
      { icon: LayoutDashboard, label: "Dashboard", path: "/" },
      { icon: FlaskConical, label: "Backtest Setup", path: "/backtest" },
      { icon: Eye, label: "Monitor", path: "/monitor" },
      { icon: BarChart3, label: "Results", path: "/results" },
      { icon: Box, label: "Models", path: "/models" },
    ],
  },
  {
    label: "Intelligence",
    items: [
      { icon: Zap, label: "Live", path: "/live-trading" },
      { icon: Newspaper, label: "News", path: "/news" },
    ],
  },
  {
    label: "System",
    items: [
      { icon: Settings, label: "Settings", path: "/settings" },
    ],
  },
];

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const width = collapsed ? layout.sidebarCollapsed : layout.sidebarExpanded;

  return (
    <aside
      className="flex flex-col border-r transition-all duration-200"
      style={{
        width,
        borderColor: "var(--color-border-subtle)",
        backgroundColor: "var(--color-surface)",
        flexShrink: 0,
        overflowX: "hidden",
        position: "relative",
        zIndex: 10,
      }}
    >
      {/* Logo area */}
      <div
        className="flex items-center border-b"
        style={{
          height: layout.headerHeight,
          borderColor: "var(--color-border-subtle)",
          padding: collapsed ? "0 20px" : "0 16px",
          justifyContent: collapsed ? "center" : "flex-start",
        }}
      >
        <KodaLogo size="sm" collapsed={collapsed} />
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto py-3">
        {navGroups.map((group) => (
          <div key={group.label} className="mb-1">
            {!collapsed && (
              <div
                className="px-4 py-2 text-[10px] font-medium uppercase tracking-[0.14em]"
                style={{
                  color: "var(--color-text-muted)",
                  fontFamily: "var(--font-sans)",
                }}
              >
                {group.label}
              </div>
            )}
            {group.items.map((item) => {
              const isActive =
                item.path === "/"
                  ? location.pathname === "/"
                  : location.pathname === item.path || location.pathname.startsWith(item.path + "/");
              return (
                <button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  className="flex items-center text-left transition-all duration-200"
                  style={{
                    width: "100%",
                    height: 38,
                    gap: collapsed ? 0 : 12,
                    paddingLeft: collapsed ? 0 : 16,
                    paddingRight: collapsed ? 0 : 12,
                    justifyContent: collapsed ? "center" : "flex-start",
                    borderLeft: isActive
                      ? "2px solid var(--color-brand)"
                      : "2px solid transparent",
                    backgroundColor: isActive
                      ? "var(--color-brand-glow)"
                      : "transparent",
                    color: isActive
                      ? "var(--color-text-primary)"
                      : "var(--color-text-secondary)",
                  }}
                  title={item.label}
                >
                  <item.icon
                    size={18}
                    strokeWidth={isActive ? 2 : 1.5}
                    style={{
                      minWidth: 18,
                      color: isActive
                        ? "var(--color-brand)"
                        : "var(--color-text-muted)",
                    }}
                  />
                  {!collapsed && (
                    <span
                      className="whitespace-nowrap text-[11px] font-medium uppercase tracking-[0.1em]"
                      style={{ fontFamily: "var(--font-sans)" }}
                    >
                      {item.label}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </div>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="flex items-center justify-center border-t transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
        style={{
          height: 36,
          borderColor: "var(--color-border-subtle)",
          color: "var(--color-text-muted)",
        }}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {collapsed ? (
          <PanelLeftOpen size={14} strokeWidth={1.5} style={{ color: "var(--color-text-muted)" }} />
        ) : (
          <PanelLeftClose size={14} strokeWidth={1.5} style={{ color: "var(--color-text-muted)" }} />
        )}
      </button>
    </aside>
  );
}
