import { ShieldCheck } from "lucide-react";

export function ZeroLeakageBadge() {
  return (
    <div
      className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1"
      style={{
        borderColor: "var(--color-accent-success)",
        backgroundColor: "rgba(8, 153, 129, 0.1)",
        color: "var(--color-accent-success)",
      }}
    >
      <ShieldCheck size={14} />
      <span className="text-xs font-medium">Zero Lookahead Bias Confirmed</span>
    </div>
  );
}
