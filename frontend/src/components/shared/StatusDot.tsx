interface StatusDotProps {
  color: string;
  label?: string;
  size?: number;
  pulse?: boolean;
}

export function StatusDot({ color, label, size = 8, pulse = false }: StatusDotProps) {
  return (
    <div className="flex items-center gap-1.5">
      <div className="relative">
        <div
          className="rounded-full"
          style={{ width: size, height: size, backgroundColor: color }}
        />
        {pulse && (
          <div
            className="absolute inset-0 animate-ping rounded-full"
            style={{ backgroundColor: color, opacity: 0.4 }}
          />
        )}
      </div>
      {label && <span className="text-[11px] text-(--color-text-muted)">{label}</span>}
    </div>
  );
}
