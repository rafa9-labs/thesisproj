import { BookOpen, BarChart3, Settings, TrendingUp, Layers, Rocket } from "lucide-react";

interface Props {
  onClose: () => void;
}

const STEPS = [
  {
    icon: Settings,
    title: "1. Configure",
    desc: "Select your currency pair, timeframe, and model types. Adjust hyperparameters or use presets.",
  },
  {
    icon: BarChart3,
    title: "2. Run Backtest",
    desc: "Launch the backtest. Monitor real-time progress via WebSocket or the Running Jobs bar.",
  },
  {
    icon: TrendingUp,
    title: "3. View Results",
    desc: "Explore equity curves, walk-forward metrics, overfitting diagnostics, and trade logs.",
  },
  {
    icon: Layers,
    title: "4. Compare Models",
    desc: "Use the leaderboard to rank models by Sharpe, Sortino, or other metrics. Run significance tests.",
  },
  {
    icon: Rocket,
    title: "5. Deploy via Committee",
    desc: "Combine top models into a regime-routing committee. Validate, optimize, and deploy to live trading.",
  },
];

export function GuidePopover({ onClose }: Props) {
  return (
    <div
      className="absolute right-0 top-full z-50 mt-2 w-80 rounded-sm border border-(--color-glass-border) bg-(--color-surface) shadow-[0_12px_32px_rgba(0,0,0,0.4)]"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between border-b border-(--color-glass-border) px-4 py-3">
        <div className="flex items-center gap-2">
          <BookOpen size={14} className="text-(--color-text-muted)" />
          <span className="text-[11px] font-semibold tracking-[0.06em] text-(--color-text-primary) uppercase">
            Quick Guide
          </span>
        </div>
        <button
          onClick={onClose}
          className="cursor-pointer text-[16px] text-(--color-text-muted) leading-none"
          style={{ border: "none", background: "none" }}
        >
          ×
        </button>
      </div>

      <div className="flex flex-col gap-0 px-4 py-3">
        {STEPS.map((step) => (
          <div
            key={step.title}
            className="flex items-start gap-3 border-b border-(--color-glass-border) py-2.5 last:border-b-0"
          >
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-(--color-elevated)">
              <step.icon size={13} className="text-(--color-brand)" />
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="text-[11px] font-semibold text-(--color-text-primary)">
                {step.title}
              </span>
              <span className="text-[10px] leading-relaxed text-(--color-text-muted)">
                {step.desc}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div className="border-t border-(--color-glass-border) px-4 py-2.5 text-center">
        <span className="text-[9px] text-(--color-text-dim)">
          More at{" "}
          <a
            href="https://github.com/rafa9-labs/thesisproj"
            target="_blank"
            rel="noopener noreferrer"
            className="text-(--color-brand) underline"
          >
            github.com/rafa9-labs/thesisproj
          </a>
        </span>
      </div>
    </div>
  );
}
