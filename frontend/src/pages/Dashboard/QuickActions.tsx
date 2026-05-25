import { useNavigate } from "react-router-dom";
import { Plus, RotateCcw } from "lucide-react";

export function QuickActions() {
  const navigate = useNavigate();

  return (
    <div className="flex items-center gap-1.5">
      {/* Primary: New Backtest */}
      <button
        onClick={() => navigate("/backtest")}
        className="flex items-center gap-1.5 rounded border px-3 text-[11px] font-semibold uppercase tracking-[0.06em] transition-colors duration-150 hover:brightness-110"
        style={{
          height: 28,
          backgroundColor: "var(--color-brand)",
          borderColor: "var(--color-brand)",
          color: "#0A0D12",
        }}
      >
        <Plus size={12} strokeWidth={2.5} />
        New Backtest
      </button>

      {/* Icon-only: Re-run Last */}
      <button
        onClick={() => navigate("/backtest")}
        className="flex items-center justify-center rounded border transition-colors duration-150 hover:bg-[var(--color-glass-hover)]"
        style={{
          height: 28,
          width: 28,
          borderColor: "var(--color-glass-border)",
          color: "var(--color-text-muted)",
        }}
        title="Re-run last backtest"
      >
        <RotateCcw size={13} strokeWidth={1.5} />
      </button>
    </div>
  );
}
