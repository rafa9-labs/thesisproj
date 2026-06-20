import { useLocation, useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import {
  LayoutDashboard,
  FlaskConical,
  BarChart3,
  Layers,
  Newspaper,
  Settings,
  Zap,
  Box,
  Activity,
} from "lucide-react";
import apiClient from "@/api/client";
import { layout } from "@/lib/tokens";
import { cn } from "@/lib/utils";

interface NavItem {
  icon: React.ElementType;
  label: string;
  path: string;
  badge?: string;
}

const primaryNav: NavItem[] = [
  { icon: LayoutDashboard, label: "Dashboard", path: "/" },
  { icon: Newspaper, label: "News", path: "/news" },
  { icon: FlaskConical, label: "Backtest", path: "/backtest" },
  { icon: Layers, label: "Committee", path: "/committee", badge: "BETA" },
  { icon: Activity, label: "Monitor", path: "/monitor" },
  { icon: BarChart3, label: "Results", path: "/results" },
  { icon: Box, label: "Models", path: "/models" },
  { icon: Zap, label: "Trading", path: "/trading" },
];

const settingsItem: NavItem = { icon: Settings, label: "Settings", path: "/settings" };

function NavRow({
  item,
  isActive,
  onClick,
  onMouseEnter,
}: {
  item: NavItem;
  isActive: boolean;
  onClick: () => void;
  onMouseEnter?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      className={cn(
        "group flex h-[42px] w-full items-center gap-3 rounded-md px-3 transition-colors duration-150",
        isActive ? "bg-[rgba(0,229,255,0.10)]" : "bg-transparent hover:bg-(--color-glass-hover)",
      )}
      style={{
        borderLeft: isActive ? "3px solid var(--color-brand)" : "3px solid transparent",
        color: isActive ? "var(--color-brand)" : "var(--color-text-secondary)",
        boxShadow: isActive ? "inset 0 0 24px rgba(0,229,255,0.04)" : undefined,
      }}
    >
      <item.icon size={18} strokeWidth={isActive ? 2 : 1.75} className="shrink-0" />
      <span className="text-[13px] tracking-[0.01em]" style={{ fontWeight: isActive ? 600 : 500 }}>
        {item.label}
      </span>
      {item.badge && (
        <span className="ml-auto rounded-full border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-amber-500 uppercase">
          {item.badge}
        </span>
      )}
    </button>
  );
}

const PREFETCH_MAP: Record<string, { key: unknown[]; fn: () => Promise<unknown>; staleTime: number }> = {
  "/results": {
    key: ["results-history", { pair: "", sort_by: "created_at", sort_order: "desc", limit: 100, status: "completed" }],
    fn: async () => {
      const { data } = await apiClient.get("/backtest/results/summary", {
        params: { pair: "", sort_by: "created_at", sort_order: "desc", limit: 100, status: "completed" },
      });
      return data;
    },
    staleTime: 5 * 60_000,
  },
  "/models": {
    key: ["deployed-models"],
    fn: async () => {
      const { data } = await apiClient.get<{ models: unknown[] }>("/models/deployed");
      return data.models;
    },
    staleTime: 5 * 60_000,
  },
  "/news": {
    key: ["news-status"],
    fn: async () => {
      const { data } = await apiClient.get("/news/status");
      return data;
    },
    staleTime: 5 * 60_000,
  },
};

export function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const isItemActive = (path: string) =>
    path === "/"
      ? location.pathname === "/"
      : location.pathname === path || location.pathname.startsWith(path + "/");

  const buildPrefetch = (path: string) => () => {
    const wasPrefetched = sessionStorage.getItem(`nav-prefetch-${path}`);
    if (wasPrefetched) return;
    sessionStorage.setItem(`nav-prefetch-${path}`, "1");

    const entry = PREFETCH_MAP[path];
    if (!entry) return;

    queryClient.prefetchQuery({
      queryKey: entry.key,
      queryFn: entry.fn,
      staleTime: entry.staleTime,
    });
  };

  const forcePrefetch = (path: string) => {
    const entry = PREFETCH_MAP[path];
    if (!entry) return;

    queryClient.prefetchQuery({
      queryKey: entry.key,
      queryFn: entry.fn,
      staleTime: entry.staleTime,
    });
  };

  const handleClick = (path: string) => {
    forcePrefetch(path);
    navigate(path);
  };

  return (
    <aside
      className="relative z-10 flex shrink-0 flex-col overflow-x-hidden border-r border-(--color-border-subtle) bg-(--color-surface)"
      style={{ width: layout.sidebarExpanded }}
    >
      <div
        className="flex flex-col items-center justify-center border-b border-(--color-border-subtle) px-5"
        style={{ height: layout.headerHeight + 32 }}
      >
        <div className="flex items-center leading-none">
          <span className="font-sans text-[22px] font-bold tracking-[0.02em] text-(--color-brand)">K</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="mx-[1px]">
            <polygon points="12,2 22,12 12,22 2,12" fill="var(--color-brand)" />
          </svg>
          <span className="font-sans text-[22px] font-bold tracking-[0.02em] text-(--color-brand)">DAQUANT</span>
        </div>
        <span className="mt-1.5 text-[10px] font-medium tracking-[0.18em] text-(--color-text-muted) uppercase">
          Institutional Terminal
        </span>
      </div>

      <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
        {primaryNav.map((item) => (
          <NavRow
            key={item.path}
            item={item}
            isActive={isItemActive(item.path)}
            onClick={() => handleClick(item.path)}
            onMouseEnter={buildPrefetch(item.path)}
          />
        ))}
      </nav>

      <div className="border-t border-(--color-border-subtle) px-3 py-3">
        <NavRow
          item={settingsItem}
          isActive={isItemActive(settingsItem.path)}
          onClick={() => handleClick(settingsItem.path)}
        />
      </div>
    </aside>
  );
}
