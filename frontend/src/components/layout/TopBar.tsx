import { Search, Bell, User, HelpCircle } from "lucide-react";
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

function IconButton({
  label,
  onClick,
  children,
  hasBadge,
}: {
  label: string;
  onClick?: () => void;
  children: React.ReactNode;
  hasBadge?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      aria-label={label}
      className="relative flex items-center justify-center rounded-full border transition-all duration-200"
      style={{
        width: 34,
        height: 34,
        borderColor: "var(--color-glass-border)",
        backgroundColor: "transparent",
        color: "var(--color-text-secondary)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "var(--color-border-active)";
        e.currentTarget.style.color = "var(--color-text-primary)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "var(--color-glass-border)";
        e.currentTarget.style.color = "var(--color-text-secondary)";
      }}
    >
      {children}
      {hasBadge && (
        <span
          className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full"
          style={{
            backgroundColor: "var(--color-brand)",
            boxShadow: "0 0 6px rgba(0,229,255,0.5)",
          }}
        />
      )}
    </button>
  );
}

export function TopBar() {
  const [aboutOpen, setAboutOpen] = useState(false);
  const title = usePageTitle();
  const unreadCount = 0;

  return (
    <header
      className="flex flex-col"
      style={{
        borderBottom: "1px solid var(--color-border-subtle)",
        backgroundColor: "var(--color-surface)",
        flexShrink: 0,
        position: "relative",
        zIndex: 20,
      }}
    >
      <div
        className="flex items-center justify-between px-6"
        style={{ height: layout.headerHeight + 32 }}
      >
        {/* Left: brand / page title */}
        <span
          className="text-[18px] font-semibold tracking-tight"
          style={{
            fontFamily: "var(--font-sans)",
            color: "var(--color-text-primary)",
          }}
        >
          {title}
        </span>

        {/* Right: action cluster */}
        <div className="flex items-center gap-2.5">
          <IconButton label="Search (Ctrl+K)">
            <Search size={16} strokeWidth={1.75} />
          </IconButton>
          <IconButton label="Notifications" hasBadge={unreadCount > 0}>
            <Bell size={16} strokeWidth={1.75} />
          </IconButton>
          <IconButton label="About KodaQuant" onClick={() => setAboutOpen(true)}>
            <HelpCircle size={16} strokeWidth={1.75} />
          </IconButton>
          <IconButton label="User menu">
            <User size={16} strokeWidth={1.75} />
          </IconButton>
        </div>
      </div>

      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </header>
  );
}
