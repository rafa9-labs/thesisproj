import { useEffect, useRef } from "react";
import { AlertTriangle, X } from "lucide-react";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "warning";
  isLoading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "danger",
  isLoading = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) {
      const handler = (e: KeyboardEvent) => {
        if (e.key === "Escape") onCancel();
      };
      document.addEventListener("keydown", handler);
      setTimeout(() => confirmRef.current?.focus(), 50);
      return () => document.removeEventListener("keydown", handler);
    }
  }, [open, onCancel]);

  if (!open) return null;

  const iconColor = variant === "danger" ? "text-rose-400" : "text-amber-400";
  const buttonBg = variant === "danger"
    ? "bg-rose-500/15 border-rose-500/25 hover:bg-rose-500/25 text-rose-400"
    : "bg-amber-500/15 border-amber-500/25 hover:bg-amber-500/25 text-amber-400";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.6] backdrop-blur-[4px]"
      onClick={onCancel}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="w-[420px] animate-fade-in rounded-sm border border-(--color-glass-border) bg-(--color-elevated) p-6 shadow-2xl"
        style={{ boxShadow: "0 0 40px rgba(0,229,255,0.06)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <AlertTriangle size={16} className={iconColor} />
            <span className="text-[11px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
              {title}
            </span>
          </div>
          <button
            onClick={onCancel}
            aria-label="Close"
            className="cursor-pointer rounded p-1 text-(--color-text-muted) transition-colors hover:text-(--color-text-secondary)"
          >
            <X size={14} />
          </button>
        </div>

        <p className="mb-6 text-[12px] leading-relaxed text-(--color-text-secondary)">
          {message}
        </p>

        <div className="flex items-center justify-end gap-3">
          <button
            onClick={onCancel}
            disabled={isLoading}
            className="cursor-pointer rounded-md border border-(--color-glass-border) bg-(--color-glass) px-4 py-2 text-[10px] font-semibold tracking-[0.08em] text-(--color-text-muted) uppercase transition-colors hover:border-(--color-text-dim) hover:text-(--color-text-secondary)"
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            onClick={onConfirm}
            disabled={isLoading}
            className={`cursor-pointer rounded-md border px-4 py-2 text-[10px] font-semibold tracking-[0.08em] uppercase transition-all ${buttonBg}`}
            style={{ opacity: isLoading ? 0.6 : 1, cursor: isLoading ? "not-allowed" : "pointer" }}
          >
            {isLoading ? "Deleting..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
