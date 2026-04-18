interface SelectOption {
  value: string;
  label: string;
}

interface ParamSelectProps {
  label: string;
  paramKey: string;
  value: string;
  options: SelectOption[];
  description?: string;
  onChange: (value: string) => void;
}

export function ParamSelect({
  label,
  paramKey,
  value,
  options,
  description,
  onChange,
}: ParamSelectProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span className="text-sm" style={{ color: "var(--color-text-primary)" }}>
          {label}
        </span>
        <span
          className="text-xs"
          style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}
        >
          {paramKey}
        </span>
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border px-2 py-1.5 text-sm"
        style={{
          fontFamily: "var(--font-sans)",
          backgroundColor: "var(--color-surface)",
          borderColor: "var(--color-border)",
          color: "var(--color-text-primary)",
        }}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {description && (
        <p className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
          {description}
        </p>
      )}
    </div>
  );
}
