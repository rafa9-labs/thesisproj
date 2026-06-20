import { TooltipIcon } from "@/components/shared/TooltipLabel";

interface ParamSliderProps {
  label: string;
  paramKey?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  tooltip?: string;
  onChange: (value: number) => void;
}

const INPUT_STYLE: React.CSSProperties = {
  backgroundColor: "var(--color-glass)",
  borderColor: "var(--color-glass-border)",
  color: "var(--color-brand)",
};

export function ParamSlider({
  label,
  value,
  min,
  max,
  step = 1,
  tooltip,
  onChange,
}: ParamSliderProps) {
  const filledPct = max > min ? ((value - min) / (max - min)) * 100 : 0;

  const handleStep = (delta: number) => {
    const raw = value + delta;
    const clamped = Math.round(raw / step) * step;
    onChange(Math.max(min, Math.min(max, clamped)));
  };

  return (
    <div className="flex flex-col py-4">
      <span className="flex items-center gap-1.5 text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
        {label}
        {tooltip && <TooltipIcon text={tooltip} />}
      </span>

      <div className="relative mt-2">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full rounded border px-3 py-2 pr-8 text-left font-mono text-sm font-medium backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
          style={INPUT_STYLE}
        />
        <div className="absolute right-0 top-0 flex h-full w-6 flex-col rounded-r overflow-hidden">
          <button
            type="button"
            tabIndex={-1}
            onClick={() => handleStep(step)}
            className="flex h-1/2 w-full items-center justify-center text-(--color-text-muted) transition-colors hover:text-(--color-brand) hover:bg-[rgba(0,229,255,0.08)]"
          >
            <svg width="8" height="5" viewBox="0 0 8 5"><path d="M4 0L8 5H0z" fill="currentColor" /></svg>
          </button>
          <button
            type="button"
            tabIndex={-1}
            onClick={() => handleStep(-step)}
            className="flex h-1/2 w-full items-center justify-center text-(--color-text-muted) transition-colors hover:text-(--color-brand) hover:bg-[rgba(0,229,255,0.08)]"
          >
            <svg width="8" height="5" viewBox="0 0 8 5"><path d="M0 0L4 5L8 0z" fill="currentColor" /></svg>
          </button>
        </div>
      </div>

      <div className="relative mt-3 flex h-5 items-center">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="slider-thumb h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-800 outline-none"
          style={{
            background: `linear-gradient(to right, var(--color-brand) 0%, var(--color-brand) ${filledPct}%, #1e293b ${filledPct}%, #1e293b 100%)`,
          }}
        />
      </div>
    </div>
  );
}
