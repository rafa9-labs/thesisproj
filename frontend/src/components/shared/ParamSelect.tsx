import { TooltipIcon } from "@/components/shared/TooltipLabel";

interface SelectOption {
  value: string;
  label: string;
}

interface ParamSelectProps {
  label: string;
  paramKey?: string;
  value: string;
  options: SelectOption[];
  tooltip?: string;
  onChange: (value: string) => void;
}

const SELECT_STYLE: React.CSSProperties = {
  backgroundColor: "var(--color-glass)",
  borderColor: "var(--color-glass-border)",
  color: "var(--color-text-primary)",
};

export function ParamSelect({ label, value, options, tooltip, onChange }: ParamSelectProps) {
  return (
    <div className="flex flex-col py-4">
      <span className="flex items-center gap-1.5 text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
        {label}
        {tooltip && <TooltipIcon text={tooltip} />}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-2 w-full rounded border px-3 py-2 font-mono text-sm backdrop-blur-[8px] transition-all duration-200 focus:outline-none"
        style={SELECT_STYLE}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      <div className="mt-3 h-5" aria-hidden="true" />
    </div>
  );
}
