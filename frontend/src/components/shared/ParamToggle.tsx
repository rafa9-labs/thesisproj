import { TooltipIcon } from "@/components/shared/TooltipLabel";

interface ParamToggleProps {
  label: string;
  paramKey?: string;
  checked: boolean;
  tooltip?: string;
  compact?: boolean;
  onChange: (checked: boolean) => void;
}

const TOGGLE_UNCHECKED_BG = "var(--color-glass)";
const TOGGLE_BORDER = "var(--color-glass-border)";

export function ParamToggle({ label, checked, tooltip, compact, onChange }: ParamToggleProps) {
  if (compact) {
    return (
      <button
        type="button"
        onClick={() => onChange(!checked)}
        className="flex w-full items-center justify-between gap-2 rounded border px-3 py-2 text-left backdrop-blur-[8px] transition-all duration-200"
        style={{
          backgroundColor: checked ? "rgba(0,229,255,0.06)" : TOGGLE_UNCHECKED_BG,
          borderColor: checked ? "rgba(0,229,255,0.3)" : TOGGLE_BORDER,
        }}
      >
        <span className="flex min-w-0 items-center gap-1.5 text-[12px] font-medium tracking-[0.04em] text-(--color-text-primary)">
          <span className="truncate">{label}</span>
          {tooltip && <TooltipIcon text={tooltip} />}
        </span>
        <span
          role="switch"
          aria-checked={checked}
          aria-label={label}
          className="relative h-5 w-9 shrink-0 rounded-full transition-all duration-200"
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

  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex w-full items-start justify-between gap-3 rounded border px-4 py-3.5 text-left backdrop-blur-[8px] transition-all duration-200"
      style={{
        borderColor: checked ? "var(--color-border-active)" : TOGGLE_BORDER,
        backgroundColor: checked ? "rgba(0,229,255,0.04)" : TOGGLE_UNCHECKED_BG,
      }}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="flex items-center gap-1.5 text-[13px] font-medium tracking-[0.04em] text-(--color-text-primary) uppercase">
          {label}
          {tooltip && <TooltipIcon text={tooltip} />}
        </span>
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
