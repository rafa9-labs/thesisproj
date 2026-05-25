import { useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  Eye,
  BarChart3,
  Newspaper,
  Settings,
  Zap,
  Box,
} from "lucide-react";
import { KodaLogo } from "@/components/shared/KodaLogo";
import { layout } from "@/lib/tokens";

interface NavItem {
  icon: React.ElementType;
  label: string;
  path: string;
}

const navItems: NavItem[] = [
  { icon: LayoutDashboard, label: "Dashboard", path: "/" },
  { icon: FlaskConical, label: "Backtest Setup", path: "/backtest" },
  { icon: Eye, label: "Monitor", path: "/monitor" },
  { icon: BarChart3, label: "Results", path: "/results" },
  { icon: Box, label: "Models", path: "/models" },
  { icon: Zap, label: "Live", path: "/live-trading" },
  { icon: Newspaper, label: "News", path: "/news" },
  { icon: Settings, label: "Settings", path: "/settings" },
];

export function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <aside
      className="flex flex-col border-r"
      style={{
        width: layout.sidebarCollapsed,
        borderColor: "var(--color-border-subtle)",
        backgroundColor: "var(--color-surface)",
        flexShrink: 0,
        overflowX: "hidden",
        position: "relative",
        zIndex: 10,
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center justify-center border-b"
        style={{
          height: layout.headerHeight,
          borderColor: "var(--color-border-subtle)",
        }}
      >
        <KodaLogo size="sm" collapsed />
      </div>

      {/* Navigation — icons only, native tooltip via title */}
      <div className="flex flex-1 flex-col items-center gap-0.5 py-3">
        {navItems.map((item) => {
          const isActive =
            item.path === "/"
              ? location.pathname === "/"
              : location.pathname === item.path ||
                location.pathname.startsWith(item.path + "/");
          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              title={item.label}
              className="flex items-center justify-center transition-colors duration-150"
              style={{
                width: 40,
                height: 38,
                borderRadius: 6,
                borderLeft: isActive
                  ? "2px solid var(--color-brand)"
                  : "2px solid transparent",
                backgroundColor: isActive
                  ? "var(--color-brand-glow)"
                  : "transparent",
                color: isActive
                  ? "var(--color-brand)"
                  : "var(--color-text-muted)",
              }}
            >
              <item.icon
                size={18}
                strokeWidth={isActive ? 2 : 1.5}
              />
            </button>
          );
        })}
      </div>
    </aside>
  );
}
