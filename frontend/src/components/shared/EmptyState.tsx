import type { ReactNode } from "react";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
}

export function EmptyState({ icon, title, description, actionLabel, onAction }: EmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 py-20">
      {icon && <div className="text-(--color-text-muted)">{icon}</div>}
      <h2 className="text-lg font-semibold text-(--color-text-secondary)">{title}</h2>
      {description && (
        <p className="max-w-md text-center text-sm text-(--color-text-muted)">{description}</p>
      )}
      {actionLabel && onAction && (
        <button
          onClick={onAction}
          className="mt-2 rounded-md bg-(--color-brand) px-6 py-2 text-xs font-bold text-(--color-text-inverse) uppercase transition-colors duration-150"
          style={{ letterSpacing: "0.05em" }}
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
