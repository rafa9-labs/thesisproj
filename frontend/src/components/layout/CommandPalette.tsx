import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";

interface SearchItem {
  label: string;
  path: string;
  keywords: string[];
}

const SEARCH_INDEX: SearchItem[] = [
  { label: "Dashboard", path: "/", keywords: ["home", "overview", "metrics", "kpi"] },
  {
    label: "Backtest Setup",
    path: "/backtest",
    keywords: ["run", "study", "hpo", "tuning", "train"],
  },
  { label: "Monitor", path: "/monitor", keywords: ["live", "running", "jobs", "watch"] },
  { label: "Results", path: "/results", keywords: ["history", "sharpe", "leaderboard", "compare"] },
  { label: "Committee", path: "/committee", keywords: ["ensemble", "voting", "members"] },
  { label: "Models", path: "/models", keywords: ["deploy", "activate", "registry", "training"] },
  {
    label: "Trading",
    path: "/trading",
    keywords: ["live", "paper", "order", "equity", "position", "execute"],
  },
  {
    label: "News",
    path: "/news",
    keywords: ["sentiment", "articles", "macro", "calendar", "feed"],
  },
  {
    label: "Settings",
    path: "/settings",
    keywords: ["config", "api", "oanda", "preferences", "account"],
  },
  {
    label: "Risk Configuration",
    path: "/trading",
    keywords: ["risk", "drawdown", "kill switch", "dd"],
  },
  {
    label: "Model Registry",
    path: "/models",
    keywords: ["deploy", "activate", "training", "manage"],
  },
];

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  query: string;
}

export function CommandPalette({ open, onClose, query }: CommandPaletteProps) {
  const navigate = useNavigate();

  const results = useMemo(() => {
    if (!query.trim()) return SEARCH_INDEX;
    const q = query.toLowerCase();
    return SEARCH_INDEX.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        item.keywords.some((k) => k.toLowerCase().includes(q)),
    );
  }, [query]);

  if (!open) return null;

  const handleSelect = (item: SearchItem) => {
    navigate(item.path);
    onClose();
  };

  return (
    <div
      className="absolute top-full left-0 z-50 mt-1 w-72 overflow-hidden rounded-lg border border-(--color-glass-border) bg-(--color-elevated) shadow-2xl lg:w-96"
      style={{ maxHeight: 320, boxShadow: "0 16px 48px rgba(0,0,0,0.4)" }}
    >
      {results.length === 0 ? (
        <div className="flex flex-col items-center gap-2 px-4 py-8">
          <Search size={18} className="text-(--color-text-dim) opacity-40" />
          <span className="text-[11px] text-(--color-text-muted)">
            No results for &ldquo;{query}&rdquo;
          </span>
        </div>
      ) : (
        <div className="overflow-y-auto" style={{ maxHeight: 320 }}>
          {results.map((item) => (
            <button
              key={item.label}
              onClick={() => handleSelect(item)}
              className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-xs transition-colors hover:bg-(--color-glass-hover)"
            >
              <span className="min-w-0 flex-1 truncate text-(--color-text-primary)">
                {item.label}
              </span>
              <span className="shrink-0 font-mono text-[10px] text-(--color-text-dim)">
                {item.path === "/" ? "/" : item.path}
              </span>
            </button>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 border-t border-(--color-glass-border) px-4 py-2">
        <kbd className="rounded border border-(--color-glass-border) px-1.5 py-0.5 text-[9px] text-(--color-text-dim)">
          Esc
        </kbd>
        <span className="text-[9px] text-(--color-text-dim)">to close</span>
        <kbd className="rounded border border-(--color-glass-border) px-1.5 py-0.5 text-[9px] text-(--color-text-dim)">
          Enter
        </kbd>
        <span className="text-[9px] text-(--color-text-dim)">to navigate</span>
      </div>
    </div>
  );
}
