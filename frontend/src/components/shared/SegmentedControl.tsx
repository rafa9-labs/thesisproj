interface Segment {
  key: string;
  label: string;
  badge?: number;
}

interface Props {
  segments: Segment[];
  active: string;
  onChange: (key: string) => void;
}

/**
 * Pill-style segmented control used for sub-navigation inside a step.
 * Matches the institutional terminal aesthetic (cyan active state).
 */
export function SegmentedControl({ segments, active, onChange }: Props) {
  return (
    <div
      className="inline-flex items-center gap-1 rounded-lg border p-1"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "rgba(255,255,255,0.02)",
      }}
      role="tablist"
    >
      {segments.map((seg) => {
        const isActive = seg.key === active;
        return (
          <button
            key={seg.key}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(seg.key)}
            className="flex items-center gap-1.5 rounded-md px-3.5 py-1.5 text-[12px] font-semibold tracking-[0.01em] transition-all duration-150"
            style={{
              backgroundColor: isActive ? "var(--color-brand)" : "transparent",
              color: isActive ? "var(--color-text-inverse)" : "var(--color-text-muted)",
              boxShadow: isActive ? "0 0 12px rgba(0,229,255,0.25)" : "none",
              cursor: "pointer",
              border: "none",
              outline: "none",
              whiteSpace: "nowrap",
            }}
          >
            {seg.label}
            {seg.badge != null && seg.badge > 0 && (
              <span
                className="flex items-center justify-center rounded-full text-[10px] font-bold"
                style={{
                  minWidth: 16,
                  height: 16,
                  padding: "0 4px",
                  fontFamily: "var(--font-mono)",
                  backgroundColor: isActive ? "rgba(0,0,0,0.25)" : "var(--color-brand-glow)",
                  color: isActive ? "var(--color-text-inverse)" : "var(--color-brand)",
                }}
              >
                {seg.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
