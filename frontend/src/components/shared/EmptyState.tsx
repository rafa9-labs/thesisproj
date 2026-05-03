import type { ReactNode } from "react";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
}

export function EmptyState({
  icon,
  title,
  description,
  actionLabel,
  onAction,
}: EmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 py-20">
      {icon && (
        <div style={{ color: "var(--color-text-muted)" }}>{icon}</div>
      )}
      <h2
        className="text-lg font-semibold"
        style={{ color: "var(--color-text-secondary)" }}
      >
        {title}
      </h2>
      {description && (
        <p className="max-w-md text-center text-sm" style={{ color: "var(--color-text-muted)" }}>
          {description}
        </p>
      )}
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="mt-2 rounded-md px-6 py-2 text-xs font-bold uppercase transition-colors duration-150"
          style={{
            backgroundColor: "var(--color-brand)",
            color: "var(--color-text-inverse)",
            letterSpacing: "0.05em",
          }}
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
