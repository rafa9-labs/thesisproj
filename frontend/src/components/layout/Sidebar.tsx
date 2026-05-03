import { useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  BarChart3,
  GitCompare,
  Newspaper,
  Settings,
  Diamond,
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
      { icon: FlaskConical, label: "Backtest", path: "/backtest" },
      { icon: BarChart3, label: "Results", path: "/results" },
      { icon: GitCompare, label: "Compare", path: "/compare" },
    ],
  },
  {
    label: "Intelligence",
    items: [
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
        borderColor: "var(--color-border)",
        backgroundColor: "var(--color-surface)",
        flexShrink: 0,
      }}
    >
      {/* Logo area */}
      <div
        className="flex items-center border-b"
        style={{
          height: layout.headerHeight,
          borderColor: "var(--color-border)",
          padding: collapsed ? "0 20px" : "0 16px",
          justifyContent: collapsed ? "center" : "flex-start",
        }}
      >
        <KodaLogo size="sm" collapsed={collapsed} />
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto py-2">
        {navGroups.map((group) => (
          <div key={group.label} className="mb-2">
            {!collapsed && (
              <div
                className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-[0.12em]"
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
                  : location.pathname.startsWith(item.path);
              return (
                <button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  className="flex items-center text-left transition-colors duration-150"
                  style={{
                    width: "100%",
                    height: 40,
                    gap: collapsed ? 0 : 12,
                    paddingLeft: collapsed ? 0 : 16,
                    paddingRight: collapsed ? 0 : 12,
                    justifyContent: collapsed ? "center" : "flex-start",
                    borderLeft: isActive
                      ? "3px solid var(--color-brand)"
                      : "3px solid transparent",
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
                    style={{
                      minWidth: 18,
                      color: isActive
                        ? "var(--color-brand)"
                        : "var(--color-text-secondary)",
                    }}
                  />
                  {!collapsed && (
                    <span
                      className="whitespace-nowrap text-[11px] font-semibold uppercase tracking-[0.08em]"
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
        className="flex items-center justify-center border-t transition-colors duration-150 hover:bg-[var(--color-elevated)]"
        style={{
          height: 36,
          borderColor: "var(--color-border)",
          color: "var(--color-text-muted)",
        }}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <Diamond
          size={14}
          style={{
            color: "var(--color-text-muted)",
            transform: collapsed ? "rotate(0deg)" : "rotate(45deg)",
            transition: "transform 200ms ease",
          }}
        />
      </button>
    </aside>
  );
}
