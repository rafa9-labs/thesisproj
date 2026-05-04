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

export function ParamSelect({
  label,
  value,
  options,
  description,
  onChange,
}: ParamSelectProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <span
        className="text-[11px] font-medium uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-muted)" }}
      >
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border px-2.5 py-2 text-sm transition-all duration-200 focus:outline-none"
        style={{
          fontFamily: "var(--font-sans)",
          fontWeight: 400,
          backgroundColor: "var(--color-glass)",
          borderColor: "var(--color-glass-border)",
          color: "var(--color-text-primary)",
          backdropFilter: "blur(8px)",
        }}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {description && (
        <p className="text-[11px] font-light leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          {description}
        </p>
      )}
    </div>
  );
}
