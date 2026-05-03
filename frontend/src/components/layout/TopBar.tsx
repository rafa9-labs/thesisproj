import { Search, Bell, User, Menu, PanelLeftClose } from "lucide-react";
import { KodaLogo } from "@/components/shared/KodaLogo";
import { Breadcrumb } from "./Breadcrumb";
import { layout } from "@/lib/tokens";

interface TopBarProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}

export function TopBar({ sidebarCollapsed, onToggleSidebar }: TopBarProps) {
  const unreadCount = 0;
  return (
    <header
      className="flex flex-col border-b"
      style={{
        borderColor: "var(--color-border)",
        backgroundColor: "var(--color-surface)",
        flexShrink: 0,
      }}
    >
      {/* Main top bar */}
      <div
        className="flex items-center justify-between px-4"
        style={{ height: layout.headerHeight }}
      >
        <div className="flex items-center gap-3">
          {/* Sidebar toggle */}
          <button
            onClick={onToggleSidebar}
            className="flex items-center justify-center rounded-md p-1.5 transition-colors hover:bg-[var(--color-elevated)]"
            style={{ color: "var(--color-text-secondary)" }}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <Menu size={16} /> : <PanelLeftClose size={16} />}
          </button>

          {/* Only show logo text in top bar when sidebar is collapsed */}
          {sidebarCollapsed && (
            <KodaLogo size="sm" />
          )}
          <Breadcrumb />
        </div>

        <div className="flex items-center gap-4">
          {/* Command palette placeholder */}
          <button
            className="hidden items-center gap-2 rounded-md border px-3 py-1.5 transition-colors hover:border-[var(--color-border-active)] md:flex"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-elevated)",
              color: "var(--color-text-muted)",
              fontSize: 12,
            }}
            onClick={() => {
              // TODO: wire up global command palette
            }}
          >
            <Search size={13} />
            <span>Search…</span>
            <span
              className="rounded px-1 text-[10px]"
              style={{
                backgroundColor: "var(--color-app)",
                border: "1px solid var(--color-border)",
                fontFamily: "var(--font-mono)",
              }}
            >
              Ctrl+K
            </span>
          </button>

          {/* Notifications */}
          <button
            className="relative flex items-center justify-center rounded-md p-1.5 transition-colors hover:bg-[var(--color-elevated)]"
            style={{ color: "var(--color-text-secondary)" }}
            title="Notifications"
          >
            <Bell size={16} />
            {unreadCount > 0 && (
              <span
                className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: "var(--color-accent-danger)" }}
              />
            )}
          </button>

          {/* User avatar */}
          <button
            className="flex items-center justify-center rounded-full border p-1 transition-colors hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "var(--color-border)",
              backgroundColor: "var(--color-elevated)",
              color: "var(--color-text-secondary)",
            }}
            title="User menu"
          >
            <User size={14} />
          </button>
        </div>
      </div>
    </header>
  );
}
