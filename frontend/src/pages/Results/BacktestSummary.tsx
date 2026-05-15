import { useState } from "react";
import { Clipboard, Check, BookOpen } from "lucide-react";

interface Props {
  text: string | null;
}

export function BacktestSummary({ text }: Props) {
  const [copied, setCopied] = useState(false);

  if (!text) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className="rounded-xl border p-5"
      style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <BookOpen size={14} style={{ color: "var(--color-brand)" }} />
          <h3
            className="text-xs font-semibold uppercase tracking-[0.1em]"
            style={{ color: "var(--color-text-secondary)" }}
          >
            Backtest Summary
          </h3>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium uppercase tracking-wider transition-all"
          style={{
            border: "1px solid var(--color-border)",
            color: copied ? "var(--color-accent-success)" : "var(--color-text-muted)",
          }}
        >
          {copied ? <Check size={10} /> : <Clipboard size={10} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <p
        className="text-xs leading-relaxed"
        style={{ color: "var(--color-text-primary)", maxWidth: "72ch" }}
      >
        {text}
      </p>
    </div>
  );
}
