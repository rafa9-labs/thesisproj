interface ParamToggleProps {
  label: string;
  paramKey: string;
  checked: boolean;
  description?: string;
  onChange: (checked: boolean) => void;
}

export function ParamToggle({
  label,
  paramKey,
  checked,
  description,
  onChange,
}: ParamToggleProps) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
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
        <button
          role="switch"
          aria-checked={checked}
          onClick={() => onChange(!checked)}
          className="relative h-5 w-9 rounded-full transition-colors duration-200"
          style={{
            backgroundColor: checked
              ? "var(--color-accent)"
              : "var(--color-border)",
          }}
        >
          <span
            className="absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform duration-200"
            style={{ left: checked ? "18px" : "2px" }}
          />
        </button>
      </div>
      {description && (
        <p className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
          {description}
        </p>
      )}
    </div>
  );
}
