import { TriangleAlert, CheckCircle } from "lucide-react";

interface Props {
  warnings: number;
  errors: number;
  canDeploy: boolean;
  isSubmitting: boolean;
  onDeploy: () => void;
}

export function ValidationBar({ warnings, errors, canDeploy, isSubmitting, onDeploy }: Props) {
  return (
    <div
      className="flex items-center justify-between rounded-lg border px-5 py-3"
      style={{
        backgroundColor: "var(--color-surface)",
        borderColor: "var(--color-border)",
      }}
    >
      <div className="flex items-center gap-4">
        {errors > 0 && (
          <div className="flex items-center gap-1.5">
            <TriangleAlert size={13} style={{ color: "var(--color-accent-danger)" }} />
            <span
              className="text-[11px] font-semibold"
              style={{ color: "var(--color-accent-danger)", fontFamily: "var(--font-mono)" }}
            >
              {errors} error{errors > 1 ? "s" : ""}
            </span>
          </div>
        )}
        {warnings > 0 && (
          <div className="flex items-center gap-1.5">
            <TriangleAlert size={13} style={{ color: "var(--color-accent-warning)" }} />
            <span
              className="text-[11px] font-semibold"
              style={{ color: "var(--color-accent-warning)", fontFamily: "var(--font-mono)" }}
            >
              {warnings} warning{warnings > 1 ? "s" : ""}
            </span>
          </div>
        )}
        {errors === 0 && warnings === 0 && (
          <div className="flex items-center gap-1.5">
            <CheckCircle size={13} style={{ color: "var(--color-accent-success)" }} />
            <span
              className="text-[11px]"
              style={{ color: "var(--color-accent-success)", fontFamily: "var(--font-mono)" }}
            >
              Ready
            </span>
          </div>
        )}
      </div>

      <button
        onClick={onDeploy}
        disabled={!canDeploy || isSubmitting}
        className="rounded-md px-6 py-2 text-xs font-bold uppercase transition-all duration-300 hover:brightness-110"
        style={{
          background: canDeploy
            ? "linear-gradient(135deg, #00E5FF 0%, #22D3EE 100%)"
            : "var(--color-glass-border)",
          color: canDeploy ? "var(--color-text-inverse)" : "var(--color-text-muted)",
          letterSpacing: "0.08em",
          cursor: canDeploy ? "pointer" : "not-allowed",
          opacity: isSubmitting ? 0.7 : 1,
        }}
      >
        {isSubmitting ? "Submitting..." : "Deploy Backtest"}
      </button>
    </div>
  );
}
