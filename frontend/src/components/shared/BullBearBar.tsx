interface BullBearBarProps {
  position: number;
  articleCount?: number;
  confidence?: number | null;
  compact?: boolean;
  vaderContribution?: number | null;
  llmContribution?: number | null;
}

export function BullBearBar({
  position,
  articleCount = 0,
  confidence = null,
  compact = false,
  vaderContribution,
  llmContribution,
}: BullBearBarProps) {
  const clamped = Math.max(-1, Math.min(1, position));
  const pct = ((clamped + 1) / 2) * 100;
  const hasData = articleCount > 0;
  const isLowData = hasData && articleCount < 3;
  const color =
    clamped > 0
      ? "var(--color-accent-success)"
      : clamped < 0
        ? "var(--color-accent-danger)"
        : "var(--color-text-muted)";
  const barOpacity = hasData
    ? isLowData
      ? 0.4
      : confidence != null && confidence < 0.3
        ? 0.6
        : 1
    : 0.15;

  const bar = (
    <div
      className="relative w-full overflow-hidden rounded-full bg-(--color-glass-hover)"
      style={{ height: compact ? 4 : 8 }}
    >
      {hasData && (
        <div
          className="absolute top-0 h-full rounded-full transition-all duration-500"
          style={{
            left: clamped >= 0 ? "50%" : `${pct}%`,
            width: `${Math.abs(clamped) * 50}%`,
            backgroundColor: color,
            opacity: barOpacity,
          }}
        />
      )}
      <div className="absolute top-0 h-full w-px bg-(--color-text-muted)" style={{ left: "50%" }} />
    </div>
  );

  if (compact) return bar;

  const showVaderLlm = !compact && vaderContribution != null && llmContribution != null;

  return (
    <div className="flex flex-col gap-1">
      {bar}
      <div className="flex items-center justify-between text-[9px]">
        <span className="text-(--color-accent-danger)">SHORT</span>
        {isLowData && (
          <span className="font-mono text-(--color-accent-warning) tabular-nums">low data</span>
        )}
        <span
          className="font-mono tabular-nums"
          style={{
            color:
              clamped > 0.3
                ? "var(--color-accent-success)"
                : clamped < -0.3
                  ? "var(--color-accent-danger)"
                  : "var(--color-text-muted)",
            opacity: isLowData ? 0.55 : 1,
          }}
        >
          {clamped > 0 ? "+" : ""}
          {clamped.toFixed(2)}
        </span>
        <span className="text-(--color-accent-success)">LONG</span>
      </div>
      {showVaderLlm && (
        <div className="flex items-center justify-center gap-2 text-[8px] text-(--color-text-dim)">
          <span>
            VADER: {vaderContribution! > 0 ? "+" : ""}
            {vaderContribution!.toFixed(2)}
          </span>
          <span>·</span>
          <span>
            LLM: {llmContribution! > 0 ? "+" : ""}
            {llmContribution!.toFixed(2)}
          </span>
        </div>
      )}
    </div>
  );
}
