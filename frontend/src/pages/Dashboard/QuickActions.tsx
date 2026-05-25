import { useNavigate } from "react-router-dom";
import { Play, RotateCcw } from "lucide-react";

export function QuickActions() {
  const navigate = useNavigate();

  return (
    <div className="flex items-center gap-3">
      <h3
        className="text-[11px] font-medium uppercase tracking-[0.12em] mr-2"
        style={{ color: "var(--color-text-muted)" }}
      >
        Quick Actions
      </h3>
      <button
        onClick={() => navigate("/backtest")}
        className="flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-medium transition-all duration-200 hover:scale-[1.02]"
        style={{
          backgroundColor: "var(--color-brand)",
          color: "#fff",
        }}
      >
        <Play size={13} />
        New Backtest
      </button>
      <button
        onClick={() => {
          navigate("/backtest");
        }}
        className="flex items-center gap-1.5 rounded-lg border px-3.5 py-1.5 text-xs font-medium transition-all duration-200 hover:bg-[var(--color-glass-hover)]"
        style={{
          borderColor: "var(--color-glass-border)",
          color: "var(--color-text-secondary)",
        }}
      >
        <RotateCcw size={13} />
        Re-run Last
      </button>
    </div>
  );
}