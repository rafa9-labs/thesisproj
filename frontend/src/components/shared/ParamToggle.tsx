interface ParamToggleProps {
  label: string;
  paramKey?: string;
  checked: boolean;
  description?: string;
  onChange: (checked: boolean) => void;
}

export function ParamToggle({ label, checked, description, onChange }: ParamToggleProps) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex w-full items-start justify-between gap-3 rounded-lg border px-4 py-3.5 text-left transition-all duration-200"
      style={{
        borderColor: checked ? "var(--color-border-active)" : "var(--color-glass-border)",
        backgroundColor: checked ? "rgba(0,229,255,0.04)" : "var(--color-input-bg)",
      }}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="text-[13px] font-medium text-(--color-text-primary)">{label}</span>
        {description && (
          <p className="text-[11px] leading-relaxed font-light text-(--color-text-muted)">
            {description}
          </p>
        )}
      </div>

      <span
        role="switch"
        aria-checked={checked}
        aria-label={label}
        className="relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition-all duration-200"
        style={{
          backgroundColor: checked ? "var(--color-brand)" : "var(--color-glass-border)",
          boxShadow: checked ? "0 0 10px rgba(0,229,255,0.35)" : "none",
        }}
      >
        <span
          className="absolute top-0.5 h-4 w-4 rounded-full transition-all duration-200"
          style={{
            left: checked ? "18px" : "2px",
            backgroundColor: checked ? "#0B1220" : "var(--color-text-muted)",
          }}
        />
      </span>
    </button>
  );
}
