interface ValidationAlertProps {
  warnings: string[];
  errors: string[];
}

export function ValidationAlert({ warnings, errors }: ValidationAlertProps) {
  if (errors.length === 0 && warnings.length === 0) return null;

  return (
    <div className="flex flex-col gap-2">
      {errors.length > 0 && (
        <div
          className="rounded-md border p-3"
          style={{
            borderColor: "var(--color-accent-danger)",
            backgroundColor: "rgba(239, 68, 68, 0.08)",
          }}
        >
          <div className="mb-1 text-xs font-semibold" style={{ color: "var(--color-accent-danger)" }}>
            {errors.length} Error{errors.length > 1 ? "s" : ""}
          </div>
          <ul className="flex flex-col gap-1">
            {errors.map((err, i) => (
              <li key={i} className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
                {err}
              </li>
            ))}
          </ul>
        </div>
      )}
      {warnings.length > 0 && (
        <div
          className="rounded-md border p-3"
          style={{
            borderColor: "var(--color-accent-warning)",
            backgroundColor: "rgba(245, 158, 11, 0.08)",
          }}
        >
          <div className="mb-1 text-xs font-semibold" style={{ color: "var(--color-accent-warning)" }}>
            {warnings.length} Warning{warnings.length > 1 ? "s" : ""}
          </div>
          <ul className="flex flex-col gap-1">
            {warnings.map((w, i) => (
              <li key={i} className="text-xs" style={{ color: "var(--color-text-secondary)" }}>
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
