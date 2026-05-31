import { Search, Bell, User, Info } from "lucide-react";
import { useState } from "react";
import { useLocation } from "react-router-dom";
import { AboutDialog } from "@/components/shared/AboutDialog";
import { layout } from "@/lib/tokens";

const routeTitles: Record<string, string> = {
  "": "Dashboard",
  "backtest": "Backtest Setup",
  "monitor": "Monitor",
  "results": "Results",
  "models": "Models",
  "trading": "Trading",
  "news": "News",
  "settings": "Settings",
};

function usePageTitle(): string {
  const location = useLocation();
  const segment = location.pathname.split("/").filter(Boolean)[0] ?? "";
  return routeTitles[segment] ?? segment;
}

export function TopBar() {
  const [aboutOpen, setAboutOpen] = useState(false);
  const title = usePageTitle();
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
      <div
        className="flex items-center justify-between px-6"
        style={{ height: layout.headerHeight }}
      >
        {/* Left: page title */}
        <span
          className="text-[13px] font-semibold uppercase tracking-[0.1em]"
          style={{
            fontFamily: "var(--font-sans)",
            color: "var(--color-text-primary)",
          }}
        >
          {title}
        </span>

        {/* Right: search + actions */}
        <div className="flex items-center gap-3">
          {/* Search bar — prominent, wider */}
          <button
            className="flex items-center gap-2.5 rounded-md border px-4 py-1.5 transition-all duration-200 hover:border-[var(--color-border-active)]"
            style={{
              width: 260,
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
            <Search size={13} strokeWidth={1.5} style={{ flexShrink: 0 }} />
            <span className="flex-1 text-left">Search…</span>
            <span
              className="rounded px-1.5 text-[10px]"
              style={{
                backgroundColor: "rgba(0,229,255,0.08)",
                border: "1px solid rgba(0,229,255,0.15)",
                color: "var(--color-brand)",
                fontFamily: "var(--font-mono)",
                flexShrink: 0,
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
                style={{
                  backgroundColor: "var(--color-brand)",
                  boxShadow: "0 0 6px rgba(0,229,255,0.5)",
                }}
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
