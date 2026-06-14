import { useState, useEffect } from "react";
import { Clock, AlertTriangle } from "lucide-react";

interface TrialCountdownProps {
  daysLeft: number;
  onExpired?: () => void;
}

export function TrialCountdown({ daysLeft, onExpired }: TrialCountdownProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (daysLeft <= 0 && onExpired) {
      onExpired();
    }
  }, [daysLeft, onExpired]);

  if (!visible || daysLeft <= 0) return null;

  const isUrgent = daysLeft <= 3;
  const bgColor = isUrgent ? "rgba(239,68,68,0.1)" : "rgba(41,98,255,0.08)";
  const borderColor = isUrgent ? "rgba(239,68,68,0.3)" : "rgba(41,98,255,0.2)";
  const textColor = isUrgent ? "#ef4444" : "var(--color-accent)";
  const Icon = isUrgent ? AlertTriangle : Clock;

  return (
    <div
      className="flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs"
      style={{ backgroundColor: bgColor, borderColor, color: textColor }}
    >
      <Icon size={14} />
      <span>
        {isUrgent
          ? `Trial expires in ${daysLeft} day${daysLeft === 1 ? "" : "s"}! Activate now.`
          : `Trial: ${daysLeft} day${daysLeft === 1 ? "" : "s"} remaining`}
      </span>
      <button
        onClick={() => setVisible(false)}
        style={{ color: textColor, cursor: "pointer", marginLeft: 4 }}
        className="text-xs opacity-60 hover:opacity-100"
      >
        &times;
      </button>
    </div>
  );
}
