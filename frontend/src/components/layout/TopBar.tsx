import { Search, Bell, User, Menu, PanelLeftClose, Info } from "lucide-react";
import { useState } from "react";
import { KodaLogo } from "@/components/shared/KodaLogo";
import { Breadcrumb } from "./Breadcrumb";
import { AboutDialog } from "@/components/shared/AboutDialog";
import { layout } from "@/lib/tokens";

interface TopBarProps {
  sidebarCollapsed: boolean;
  onToggleSidebar: () => void;
}

export function TopBar({ sidebarCollapsed, onToggleSidebar }: TopBarProps) {
  const [aboutOpen, setAboutOpen] = useState(false);
  const unreadCount = 0;
  return (
    <header
      className="flex flex-col"
      style={{
        borderBottom: "1px solid rgba(255,255,255,0.04)",
        backgroundColor: "var(--color-surface)",
        flexShrink: 0,
        position: "relative",
        zIndex: 20,
      }}
    >
      {/* Main top bar */}
      <div
        className="flex items-center justify-between px-5"
        style={{ height: layout.headerHeight }}
      >
        <div className="flex items-center gap-3">
          {/* Sidebar toggle */}
          <button
            onClick={onToggleSidebar}
            className="flex items-center justify-center rounded-md p-1.5 transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
            style={{ color: "var(--color-text-muted)" }}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <Menu size={16} strokeWidth={1.5} /> : <PanelLeftClose size={16} strokeWidth={1.5} />}
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
            className="hidden items-center gap-2 rounded-md border px-3 py-1.5 transition-all duration-200 hover:border-[var(--color-border-active)] md:flex"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "var(--color-glass)",
              color: "var(--color-text-muted)",
              fontSize: 12,
              fontWeight: 400,
            }}
            onClick={() => {
              // TODO: wire up global command palette
            }}
          >
            <Search size={13} strokeWidth={1.5} />
            <span>Search…</span>
            <span
              className="rounded px-1 text-[10px]"
              style={{
                backgroundColor: "rgba(0,229,255,0.08)",
                border: "1px solid rgba(0,229,255,0.15)",
                color: "var(--color-brand)",
                fontFamily: "var(--font-mono)",
              }}
            >
              Ctrl+K
            </span>
          </button>

          {/* Notifications */}
          <button
            className="relative flex items-center justify-center rounded-md p-1.5 transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
            style={{ color: "var(--color-text-muted)" }}
            title="Notifications"
          >
            <Bell size={16} strokeWidth={1.5} />
            {unreadCount > 0 && (
              <span
                className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: "var(--color-brand)", boxShadow: "0 0 6px rgba(0,229,255,0.5)" }}
              />
            )}
          </button>

          {/* About */}
          <button
            className="flex items-center justify-center rounded-md p-1.5 transition-colors duration-200 hover:bg-[var(--color-glass-hover)]"
            style={{ color: "var(--color-text-muted)" }}
            title="About KodaQuant"
            onClick={() => setAboutOpen(true)}
          >
            <Info size={16} strokeWidth={1.5} />
          </button>

          {/* User avatar */}
          <button
            className="flex items-center justify-center rounded-full border p-1 transition-all duration-200 hover:border-[var(--color-border-active)]"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "var(--color-glass)",
              color: "var(--color-text-muted)",
            }}
            title="User menu"
          >
            <User size={14} strokeWidth={1.5} />
          </button>
        </div>
      </div>

      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </header>
  );
}
