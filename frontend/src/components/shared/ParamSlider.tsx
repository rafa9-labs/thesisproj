interface ParamSliderProps {
  label: string;
  paramKey?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  description?: string;
  onChange: (value: number) => void;
}

export function ParamSlider({
  label,
  value,
  min,
  max,
  step = 1,
  description,
  onChange,
}: ParamSliderProps) {
  return (
    <div className="flex flex-col gap-2 py-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
          {label}
        </span>
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-20 rounded border border-(--color-glass-border) bg-(--color-glass) px-2 py-1 text-right font-mono text-xs font-medium text-(--color-brand) backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
        />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 w-full cursor-pointer appearance-none rounded-full"
        style={{
          accentColor: "var(--color-brand)",
          background: `linear-gradient(to right, var(--color-brand) 0%, var(--color-brand) ${((value - min) / (max - min)) * 100}%, var(--color-glass-border) ${((value - min) / (max - min)) * 100}%, var(--color-glass-border) 100%)`,
        }}
      />
      {description && (
        <p className="text-[11px] leading-relaxed font-light text-(--color-text-muted)">
          {description}
        </p>
      )}
    </div>
  );
}
