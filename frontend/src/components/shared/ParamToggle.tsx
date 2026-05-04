interface ParamToggleProps {
  label: string;
  paramKey?: string;
  checked: boolean;
  description?: string;
  onChange: (checked: boolean) => void;
}

export function ParamToggle({
  label,
  checked,
  description,
  onChange,
}: ParamToggleProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span
          className="text-[11px] font-medium uppercase tracking-[0.1em]"
          style={{ color: "var(--color-text-muted)" }}
        >
          {label}
        </span>
        <button
          role="switch"
          aria-checked={checked}
          onClick={() => onChange(!checked)}
          className="relative h-5 w-9 rounded-full transition-all duration-200"
          style={{
            backgroundColor: checked
              ? "var(--color-brand)"
              : "var(--color-glass-border)",
            boxShadow: checked ? "0 0 8px rgba(0,229,255,0.25)" : "none",
          }}
        >
          <span
            className="absolute top-0.5 h-4 w-4 rounded-full transition-transform duration-200"
            style={{
              left: checked ? "18px" : "2px",
              backgroundColor: checked
                ? "var(--color-text-inverse)"
                : "var(--color-text-muted)",
            }}
          />
        </button>
      </div>
      {description && (
        <p className="text-[11px] font-light leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          {description}
        </p>
      )}
    </div>
  );
}
