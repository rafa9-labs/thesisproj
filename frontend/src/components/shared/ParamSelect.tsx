interface SelectOption {
  value: string;
  label: string;
}

interface ParamSelectProps {
  label: string;
  paramKey?: string;
  value: string;
  options: SelectOption[];
  description?: string;
  onChange: (value: string) => void;
}

export function ParamSelect({ label, value, options, description, onChange }: ParamSelectProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-(--color-glass-border) bg-(--color-glass) px-2.5 py-2 font-sans text-sm font-normal text-(--color-text-primary) backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {description && (
        <p className="text-[11px] leading-relaxed font-light text-(--color-text-muted)">
          {description}
        </p>
      )}
    </div>
  );
}
