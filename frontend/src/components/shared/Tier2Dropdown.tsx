import { TooltipIcon } from "@/components/shared/TooltipLabel";

interface Tier2DropdownProps {
  label: string;
  description: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}

export function Tier2Dropdown({ label, description, value, options, onChange }: Tier2DropdownProps) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="flex items-center gap-1.5">
        <span className="min-w-0 truncate text-[11px] font-semibold tracking-[0.1em] text-(--color-text-secondary) uppercase">
          {label}
        </span>
        <span className="rounded bg-(--color-brand)/10 px-1.5 py-px text-[9px] font-medium text-(--color-brand) uppercase tracking-wider">
          Fixed
        </span>
        <TooltipIcon text={description} />
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-(--color-glass-border) bg-(--color-glass) px-3 py-2 font-mono text-sm font-medium text-(--color-text-primary) backdrop-blur-[8px] transition-all duration-200 focus:outline-none focus:border-(--color-border-active)"
        style={{ borderColor: "var(--color-border-active, var(--color-glass-border))" }}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {String(opt) === "None" ? "None (unlimited)" : String(opt)}
          </option>
        ))}
      </select>
    </div>
  );
}
