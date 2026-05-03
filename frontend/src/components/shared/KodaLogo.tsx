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
          <polygon
            points="12,2 22,12 12,22 2,12"
            fill="var(--color-brand)"
          />
        </svg>
      </div>
    );
  }

  return (
    <div className={cn("inline-flex items-center select-none", className)} style={{ height, gap: 1 }}>
      <span
        className="font-bold tracking-tight"
        style={{
          fontFamily: "var(--font-sans)",
          fontSize,
          color: "var(--color-text-primary)",
          letterSpacing: "0.04em",
        }}
      >
        K
      </span>
      <svg
        width={diamondSize}
        height={diamondSize}
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ marginInline: 1 }}
      >
        <polygon
          points="12,2 22,12 12,22 2,12"
          fill="var(--color-brand)"
        />
      </svg>
      <span
        className="font-bold tracking-tight"
        style={{
          fontFamily: "var(--font-sans)",
          fontSize,
          color: "var(--color-text-primary)",
          letterSpacing: "0.04em",
        }}
      >
        DA
      </span>
      <span
        className="font-semibold tracking-wide"
        style={{
          fontFamily: "var(--font-sans)",
          fontSize: Math.round(fontSize * 0.82),
          color: "var(--color-text-secondary)",
          letterSpacing: "0.12em",
          marginLeft: 3,
        }}
      >
        QUANT
      </span>
    </div>
  );
}
