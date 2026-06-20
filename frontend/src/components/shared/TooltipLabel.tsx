import { cn } from "@/lib/utils";

interface TooltipLabelProps {
  label?: string;
  tooltip?: string;
  className?: string;
}

export function TooltipLabel({ label, tooltip, className }: TooltipLabelProps) {
  if (!tooltip && !label) return null;

  return (
    <span className={cn("inline-flex items-center gap-1.5 text-[11px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase", className)}>
      {label && <span>{label}</span>}
      {tooltip && <TooltipIcon text={tooltip} />}
    </span>
  );
}

export function TooltipIcon({ text }: { text: string }) {
  if (!text) return null;
  return (
    <div className="group/tip relative inline-flex items-center">
      <span className="cursor-help text-[11px] text-(--color-text-muted)/50 hover:text-(--color-text-muted)">
        (?)
      </span>
      <div className="pointer-events-none absolute bottom-full left-0 z-[100] mb-2 opacity-0 transition-opacity group-hover/tip:opacity-100">
        <div
          className="rounded-md border border-slate-700 bg-slate-800 p-3 text-[11px] leading-relaxed text-(--color-text-secondary) shadow-xl"
          style={{ width: "max-content", maxWidth: "min(16rem, calc(100vw - 2rem))" }}
        >
          {text}
        </div>
      </div>
    </div>
  );
}
