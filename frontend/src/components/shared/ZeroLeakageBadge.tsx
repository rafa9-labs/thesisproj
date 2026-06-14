import { ShieldCheck } from "lucide-react";

export function ZeroLeakageBadge() {
  return (
    <div
      className="inline-flex items-center gap-1.5 rounded-md border border-(--color-accent-success) px-3 py-1 text-(--color-accent-success)"
      style={{ backgroundColor: "rgba(8, 153, 129, 0.1)" }}
    >
      <ShieldCheck size={14} />
      <span className="text-xs font-medium">Zero Lookahead Bias Confirmed</span>
    </div>
  );
}
