import { useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  Eye,
  BarChart3,
  Layers,
  Newspaper,
  Settings,
  Zap,
  Box,
} from "lucide-react";
import { layout } from "@/lib/tokens";

interface NavItem {
  icon: React.ElementType;
  label: string;
  path: string;
}

const primaryNav: NavItem[] = [
  { icon: LayoutDashboard, label: "Dashboard", path: "/" },
  { icon: FlaskConical, label: "Backtest", path: "/backtest" },
  { icon: Eye, label: "Monitor", path: "/monitor" },
  { icon: BarChart3, label: "Results", path: "/results" },
  { icon: Layers, label: "Committee", path: "/committee" },
  { icon: Box, label: "Models", path: "/models" },
  { icon: Zap, label: "Trading", path: "/trading" },
  { icon: Newspaper, label: "News", path: "/news" },
];

const settingsItem: NavItem = { icon: Settings, label: "Settings", path: "/settings" };

function NavRow({
  item,
  isActive,
  onClick,
}: {
  item: NavItem;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group flex items-center gap-3 rounded-md px-3 transition-colors duration-150"
      style={{
        height: 42,
        width: "100%",
        borderLeft: isActive
          ? "2px solid var(--color-brand)"
          : "2px solid transparent",
        backgroundColor: isActive ? "var(--color-brand-glow)" : "transparent",
        color: isActive ? "var(--color-brand)" : "var(--color-text-secondary)",
      }}
      onMouseEnter={(e) => {
        if (!isActive) e.currentTarget.style.backgroundColor = "var(--color-glass-hover)";
      }}
      onMouseLeave={(e) => {
        if (!isActive) e.currentTarget.style.backgroundColor = "transparent";
      }}
    >
      <item.icon size={18} strokeWidth={isActive ? 2 : 1.75} style={{ flexShrink: 0 }} />
      <span
        className="text-[13px]"
        style={{ fontWeight: isActive ? 600 : 500, letterSpacing: "0.01em" }}
      >
        {item.label}
      </span>
    </button>
  );
}

export function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();

  const isItemActive = (path: string) =>
    path === "/"
      ? location.pathname === "/"
      : location.pathname === path || location.pathname.startsWith(path + "/");

  return (
    <aside
      className="flex flex-col border-r"
      style={{
        width: layout.sidebarExpanded,
        borderColor: "var(--color-border-subtle)",
        backgroundColor: "var(--color-surface)",
        flexShrink: 0,
        overflowX: "hidden",
        position: "relative",
        zIndex: 10,
      }}
    >
      {/* Brand wordmark */}
      <div
        className="flex flex-col justify-center border-b px-5"
        style={{
          height: layout.headerHeight + 32,
          borderColor: "var(--color-border-subtle)",
        }}
      >
        <span
          className="font-bold leading-none"
          style={{
            fontFamily: "var(--font-sans)",
            fontSize: 22,
            letterSpacing: "0.02em",
            color: "var(--color-brand)",
          }}
        >
          KODAQUANT
        </span>
        <span
          className="mt-1.5 text-[10px] font-medium uppercase tracking-[0.18em]"
          style={{ color: "var(--color-text-muted)" }}
        >
          Institutional Terminal
        </span>
      </div>

      {/* Primary navigation */}
      <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
        {primaryNav.map((item) => (
          <NavRow
            key={item.path}
            item={item}
            isActive={isItemActive(item.path)}
            onClick={() => navigate(item.path)}
          />
        ))}
      </nav>

      {/* Settings pinned to bottom */}
      <div
        className="border-t px-3 py-3"
        style={{ borderColor: "var(--color-border-subtle)" }}
      >
        <NavRow
          item={settingsItem}
          isActive={isItemActive(settingsItem.path)}
          onClick={() => navigate(settingsItem.path)}
        />
      </div>
    </aside>
  );
}
