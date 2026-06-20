function isVoid(v: unknown): v is (null | undefined) {
  return v == null;
}

function isBadNumber(v: unknown): boolean {
  return typeof v !== "number" || !Number.isFinite(v);
}

export function formatMetric(value: number | null | undefined, decimals = 2): string {
  if (isVoid(value) || isBadNumber(value)) return "—";
  return value.toFixed(decimals);
}

export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (isVoid(value) || isBadNumber(value)) return "—";
  const pct = Math.abs(value) <= 1 ? value * 100 : value;
  if (isBadNumber(pct)) return "—";
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(decimals)}%`;
}

export function formatPrice(value: number | null | undefined, decimals = 5): string {
  if (isVoid(value) || isBadNumber(value)) return "—";
  return value.toFixed(decimals);
}

export function formatPips(value: number | null | undefined): string {
  if (isVoid(value) || isBadNumber(value)) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(1)} pips`;
}

export function formatInt(value: number | null | undefined): string {
  if (isVoid(value) || isBadNumber(value)) return "—";
  return value.toLocaleString();
}

export function formatRelativeTime(isoDate: string | null | undefined): string {
  if (!isoDate) return "—";
  const date = new Date(isoDate);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

export function colorForReturn(value: number | null | undefined): string {
  if (isVoid(value) || isBadNumber(value)) return "var(--color-text-muted)";
  return value >= 0 ? "var(--color-accent-success)" : "var(--color-accent-danger)";
}
