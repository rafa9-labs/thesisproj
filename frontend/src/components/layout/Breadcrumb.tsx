import { useLocation } from "react-router-dom";
import { ChevronRight } from "lucide-react";

const routeLabels: Record<string, string> = {
  "": "Dashboard",
  backtest: "Backtest",
  results: "Results",
  models: "Models",
  settings: "Settings",
};

export function Breadcrumb() {
  const location = useLocation();

  const segments = location.pathname.split("/").filter(Boolean);
  if (segments.length === 0) {
    segments.push("");
  }

  return (
    <nav className="flex items-center gap-1.5" aria-label="Breadcrumb">
      {segments.map((seg, idx) => {
        const isLast = idx === segments.length - 1;
        const label = routeLabels[seg] ?? seg;
        return (
          <div key={idx} className="flex items-center gap-1.5">
            {idx > 0 && <ChevronRight size={12} className="text-(--color-text-muted)" />}
            <span
              className="font-mono text-[11px] font-medium"
              style={{
                color: isLast ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              {label}
            </span>
          </div>
        );
      })}
    </nav>
  );
}
