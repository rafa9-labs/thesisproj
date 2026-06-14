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
import { cn } from "@/lib/utils";

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
      className={cn(
        "group flex h-[42px] w-full items-center gap-3 rounded-md px-3 transition-colors duration-150",
        isActive ? "bg-(--color-brand-glow)" : "bg-transparent hover:bg-(--color-glass-hover)",
      )}
      style={{
        borderLeft: isActive ? "2px solid var(--color-brand)" : "2px solid transparent",
        color: isActive ? "var(--color-brand)" : "var(--color-text-secondary)",
      }}
    >
      <item.icon size={18} strokeWidth={isActive ? 2 : 1.75} className="shrink-0" />
      <span className="text-[13px] tracking-[0.01em]" style={{ fontWeight: isActive ? 600 : 500 }}>
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
      className="relative z-10 flex shrink-0 flex-col overflow-x-hidden border-r border-(--color-border-subtle) bg-(--color-surface)"
      style={{
        width: layout.sidebarExpanded,
      }}
    >
      {/* Brand wordmark */}
      <div
        className="flex flex-col justify-center border-b border-(--color-border-subtle) px-5"
        style={{ height: layout.headerHeight + 32 }}
      >
        <span className="font-sans text-[22px] leading-none font-bold tracking-[0.02em] text-(--color-brand)">
          KODAQUANT
        </span>
        <span className="mt-1.5 text-[10px] font-medium tracking-[0.18em] text-(--color-text-muted) uppercase">
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
      <div className="border-t border-(--color-border-subtle) px-3 py-3">
        <NavRow
          item={settingsItem}
          isActive={isItemActive(settingsItem.path)}
          onClick={() => navigate(settingsItem.path)}
        />
      </div>
    </aside>
  );
}
