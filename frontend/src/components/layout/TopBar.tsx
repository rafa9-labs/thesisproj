import { Search, Bell, HelpCircle, BookOpen } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { AboutDialog } from "@/components/shared/AboutDialog";
import { CommandPalette } from "./CommandPalette";
import { NotificationsPopover } from "./NotificationsPopover";
import { GuidePopover } from "./GuidePopover";
import { useAppStore } from "@/stores/useAppStore";
import { useJobStore } from "@/stores/useJobStore";

export function TopBar() {
  const [aboutOpen, setAboutOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const demoMode = useAppStore((s) => s.demoMode);
  const unreadCount = useJobStore((s) => s.unreadCompletedCount);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        searchRef.current?.focus();
        setPaletteOpen(true);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  // Close popovers on outside click
  useEffect(() => {
    if (!notifOpen && !guideOpen) return;
    const onClick = () => {
      setNotifOpen(false);
      setGuideOpen(false);
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, [notifOpen, guideOpen]);

  return (
    <>
      <header className="sticky top-0 z-50 flex h-16 shrink-0 items-center border-b border-(--color-glass-border) bg-(--color-app)/80 px-6 backdrop-blur-md">
        {/* Left: Search bar */}
        <div className="relative w-48 sm:w-56 lg:w-96">
          <Search
            size={14}
            className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-(--color-text-muted)"
          />
          <input
            ref={searchRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              if (!paletteOpen) setPaletteOpen(true);
            }}
            onFocus={() => setPaletteOpen(true)}
            onBlur={() => setTimeout(() => setPaletteOpen(false), 200)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                searchRef.current?.blur();
                setPaletteOpen(false);
              }
            }}
            placeholder="Search assets, models, or settings..."
            aria-label="Global search"
            className="w-full rounded-md border border-(--color-glass-border) bg-(--color-input-bg) py-1.5 pr-16 pl-9 text-xs text-(--color-text-primary) transition outline-none placeholder:text-(--color-text-dim) focus:border-(--color-border-active)"
          />
          <kbd className="pointer-events-none absolute top-1/2 right-2 -translate-y-1/2 rounded border border-(--color-glass-border) px-1.5 py-0.5 text-[9px] font-medium text-(--color-text-muted)">
            Ctrl+K
          </kbd>
          <CommandPalette
            open={paletteOpen}
            onClose={() => {
              setPaletteOpen(false);
              setQuery("");
            }}
            query={query}
          />
        </div>

        <div className="flex-1" />

        {demoMode && (
          <span className="rounded-full border border-[rgba(0,229,255,0.3)] bg-[rgba(0,229,255,0.06)] px-3 py-1 text-[10px] font-semibold tracking-[0.05em] text-(--color-brand) uppercase">
            Local Data Mode
          </span>
        )}

        {/* Right: Actions */}
        <div className="flex items-center gap-2 sm:gap-4">
          {/* Guide */}
          <div className="relative">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setGuideOpen((v) => !v);
                setNotifOpen(false);
              }}
              aria-label="Guide"
              className="rounded-full border border-(--color-glass-border) p-1.5 text-(--color-text-secondary) transition hover:border-(--color-border-active) hover:text-(--color-text-primary)"
            >
              <BookOpen size={16} strokeWidth={1.75} />
            </button>
            {guideOpen && <GuidePopover onClose={() => setGuideOpen(false)} />}
          </div>

          {/* Notifications */}
          <div className="relative">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setNotifOpen((v) => !v);
                setGuideOpen(false);
                if (unreadCount > 0) useJobStore.getState().clearUnreadCount();
              }}
              aria-label="Notifications"
              className="relative rounded-full border border-(--color-glass-border) p-1.5 text-(--color-text-secondary) transition hover:border-(--color-border-active) hover:text-(--color-text-primary)"
            >
              <Bell size={16} strokeWidth={1.75} />
              {unreadCount > 0 && (
                <span
                  className="absolute -top-0.5 -right-0.5 flex h-3.5 min-w-[14px] items-center justify-center rounded-full bg-(--color-brand) px-[3px] text-[8px] font-bold text-(--color-text-inverse)"
                  style={{ boxShadow: "0 0 8px rgba(0,229,255,0.6)" }}
                >
                  {unreadCount > 9 ? "9+" : unreadCount}
                </span>
              )}
            </button>
            {notifOpen && <NotificationsPopover onClose={() => setNotifOpen(false)} />}
          </div>

          {/* Help / About */}
          <button
            onClick={() => setAboutOpen(true)}
            aria-label="Help"
            className="rounded-full border border-(--color-glass-border) p-1.5 text-(--color-text-secondary) transition hover:border-(--color-border-active) hover:text-(--color-text-primary)"
          >
            <HelpCircle size={16} strokeWidth={1.75} />
          </button>
        </div>
      </header>

      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </>
  );
}
