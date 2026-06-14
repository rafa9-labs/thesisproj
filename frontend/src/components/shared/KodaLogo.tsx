import { cn } from "@/lib/utils";

interface KodaLogoProps {
  size?: "sm" | "md" | "lg";
  collapsed?: boolean;
  className?: string;
}

export function KodaLogo({ size = "md", collapsed = false, className }: KodaLogoProps) {
  const height = size === "sm" ? 18 : size === "md" ? 22 : 28;
  const fontSize = size === "sm" ? 14 : size === "md" ? 17 : 22;
  const diamondSize = size === "sm" ? 12 : size === "md" ? 14 : 18;

  if (collapsed) {
    return (
      <div className={cn("flex items-center justify-center", className)} style={{ height }}>
        <svg
          width={diamondSize}
          height={diamondSize}
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <polygon points="12,2 22,12 12,22 2,12" fill="var(--color-brand)" />
        </svg>
      </div>
    );
  }

  return (
    <div
      className={cn("inline-flex items-center gap-[1px] select-none", className)}
      style={{ height }}
    >
      <span className="font-sans font-bold tracking-[0.04em] tracking-tight text-(--color-text-primary)">
        K
      </span>
      <svg
        width={diamondSize}
        height={diamondSize}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="mx-[1px]"
      >
        <polygon points="12,2 22,12 12,22 2,12" fill="var(--color-brand)" />
      </svg>
      <span className="font-sans font-bold tracking-[0.04em] tracking-tight text-(--color-text-primary)">
        DA
      </span>
      <span
        className="ml-[3px] font-sans font-semibold tracking-[0.12em] tracking-wide text-(--color-text-secondary)"
        style={{ fontSize: Math.round(fontSize * 0.82) }}
      >
        QUANT
      </span>
    </div>
  );
}
