import { Search, Bell, HelpCircle, BookOpen, ChevronDown } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { AboutDialog } from "@/components/shared/AboutDialog";
import { CommandPalette } from "./CommandPalette";

export function TopBar() {
  const [aboutOpen, setAboutOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const unreadCount = 3;

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

        {/* Right: Actions */}
        <div className="flex items-center gap-2 sm:gap-4">
          <button
            onClick={() => setAboutOpen(true)}
            aria-label="Library"
            className="rounded-full border border-(--color-glass-border) p-1.5 text-(--color-text-secondary) transition hover:border-(--color-border-active) hover:text-(--color-text-primary)"
          >
            <BookOpen size={16} strokeWidth={1.75} />
          </button>

          <button
            aria-label="Notifications"
            className="relative rounded-full border border-(--color-glass-border) p-1.5 text-(--color-text-secondary) transition hover:border-(--color-border-active) hover:text-(--color-text-primary)"
          >
            <Bell size={16} strokeWidth={1.75} />
            {unreadCount > 0 && (
              <span
                className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 animate-pulse rounded-full bg-(--color-brand)"
                style={{ boxShadow: "0 0 8px rgba(0,229,255,0.6)" }}
              />
            )}
          </button>

          <button
            onClick={() => setAboutOpen(true)}
            aria-label="Help"
            className="rounded-full border border-(--color-glass-border) p-1.5 text-(--color-text-secondary) transition hover:border-(--color-border-active) hover:text-(--color-text-primary)"
          >
            <HelpCircle size={16} strokeWidth={1.75} />
          </button>

          <button className="hidden items-center gap-1.5 rounded-full border border-(--color-glass-border) py-1 pr-2 pl-1 transition hover:border-(--color-border-active) sm:flex">
            <span
              className="flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold text-(--color-text-inverse)"
              style={{
                background: "linear-gradient(135deg, var(--color-brand), #a78bfa)",
              }}
            >
              KQ
            </span>
            <ChevronDown size={12} className="text-(--color-text-muted)" />
          </button>
        </div>
      </header>

      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </>
  );
}
