import { useState } from "react";
import { TooltipIcon } from "@/components/shared/TooltipLabel";

interface Tier1SliderProps {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hpoMin?: number;
  hpoMax?: number;
  onChange: (value: number) => void;
  onHpoToggle: (enabled: boolean, minVal: number, maxVal: number) => void;
}

export function Tier1Slider({
  label, description,
  value, min, max, step,
  hpoMin, hpoMax, onChange, onHpoToggle,
}: Tier1SliderProps) {
  const hasHpoRange = hpoMin !== undefined && hpoMax !== undefined;
  const [hpoEnabled, setHpoEnabled] = useState(hasHpoRange);

  const handleToggle = () => {
    const next = !hpoEnabled;
    setHpoEnabled(next);
    if (next) {
      onHpoToggle(true, min, max);
    } else {
      onHpoToggle(false, value, value);
    }
  };

  const filledPct = max > min ? ((value - min) / (max - min)) * 100 : 0;

  const handleStep = (delta: number) => {
    const raw = value + delta;
    const clamped = Math.round(raw / step) * step;
    onChange(Math.max(min, Math.min(max, clamped)));
  };

  return (
    <div className="flex flex-col gap-2 py-2">
      <div className="flex justify-between items-center">
        <span className="flex items-center gap-1.5 text-[11px] font-semibold tracking-[0.1em] text-(--color-text-secondary) uppercase">
          <span className="truncate">{label}</span>
          <TooltipIcon text={description} />
        </span>
        <button
          type="button"
          onClick={handleToggle}
          className="shrink-0 rounded px-2 py-0.5 text-[9px] font-bold tracking-[0.08em] uppercase transition-all duration-200"
          style={{
            backgroundColor: hpoEnabled ? "rgba(0,229,255,0.12)" : "transparent",
            color: hpoEnabled ? "var(--color-brand)" : "var(--color-text-muted)",
            border: `1px solid ${hpoEnabled ? "var(--color-brand)" : "var(--color-glass-border)"}`,
            boxShadow: hpoEnabled ? "0 0 8px rgba(0,229,255,0.15)" : "none",
          }}
        >
          HPO {hpoEnabled ? "ON" : "OFF"}
        </button>
      </div>

      {hpoEnabled ? (
        <div className="flex items-center gap-4">
          <div className="flex-1 min-w-0">
            <label className="mb-1 block text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted)/60 uppercase">
              Min
            </label>
            <input
              type="number"
              value={hpoMin ?? min}
              min={min}
              max={max}
              step={step}
              onChange={(e) => onHpoToggle(true, Number(e.target.value), hpoMax ?? max)}
              className="w-full rounded-md border border-(--color-glass-border) bg-(--color-input-bg) px-2 py-1.5 text-left font-mono text-xs font-medium text-(--color-brand) transition-all duration-200 focus:outline-none focus:border-(--color-brand)"
            />
          </div>
          <span className="text-xs text-(--color-text-muted)">—</span>
          <div className="flex-1 min-w-0">
            <label className="mb-1 block text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted)/60 uppercase">
              Max
            </label>
            <input
              type="number"
              value={hpoMax ?? max}
              min={min}
              max={max}
              step={step}
              onChange={(e) => onHpoToggle(true, hpoMin ?? min, Number(e.target.value))}
              className="w-full rounded-md border border-(--color-glass-border) bg-(--color-input-bg) px-2 py-1.5 text-left font-mono text-xs font-medium text-(--color-brand) transition-all duration-200 focus:outline-none focus:border-(--color-brand)"
            />
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-4">
          <div className="flex flex-1 flex-col gap-1">
            <div className="relative flex h-5 items-center">
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
          <div className="relative w-24">
            <input
              type="number"
              value={value}
              min={min}
              max={max}
              step={step}
              onChange={(e) => onChange(Number(e.target.value))}
              className="w-full rounded-md border border-(--color-glass-border) bg-(--color-input-bg) px-3 py-2 pr-8 text-left font-mono text-xs font-medium text-(--color-brand) transition-all duration-200 focus:outline-none focus:border-(--color-brand)"
            />
            <div className="absolute right-0 top-0 flex h-full w-6 flex-col rounded-r-md overflow-hidden">
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
        </div>
      )}
    </div>
  );
}
