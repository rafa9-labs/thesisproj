import { useState, useCallback } from "react";
import { Plus, X } from "lucide-react";

const SUGGESTIONS = ["good", "overfit", "failed", "high_sharpe", "verified", "production"];

interface Props {
  tags: string[];
  onAdd: (tag: string) => void;
  onRemove: (tag: string) => void;
}

export function TagEditor({ tags, onAdd, onRemove }: Props) {
  const [input, setInput] = useState("");
  const [showInput, setShowInput] = useState(false);

  const handleAdd = useCallback(() => {
    const t = input.trim().toLowerCase().replace(/\s+/g, "_");
    if (t && !tags.includes(t)) {
      onAdd(t);
    }
    setInput("");
    setShowInput(false);
  }, [input, tags, onAdd]);

  return (
    <div className="flex flex-wrap items-center gap-1">
      {tags.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] font-medium cursor-pointer hover:brightness-110"
          style={{
            backgroundColor: "var(--color-glass-hover)",
            color: "var(--color-text-secondary)",
            border: "1px solid var(--color-glass-border)",
          }}
          onClick={() => onRemove(tag)}
          title="Click to remove"
        >
          {tag}
          <X size={8} />
        </span>
      ))}
      {showInput ? (
        <input
          autoFocus
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleAdd();
            if (e.key === "Escape") { setShowInput(false); setInput(""); }
          }}
          onBlur={() => { if (input.trim()) handleAdd(); else setShowInput(false); }}
          className="rounded px-1.5 py-0.5 text-[9px] outline-none"
          style={{
            width: 70,
            backgroundColor: "var(--color-elevated)",
            color: "var(--color-text-primary)",
            border: "1px solid var(--color-brand)",
          }}
          placeholder="tag..."
        />
      ) : (
        <button
          onClick={() => setShowInput(true)}
          className="inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] transition hover:brightness-110"
          style={{
            backgroundColor: "transparent",
            color: "var(--color-text-muted)",
            border: "1px dashed var(--color-glass-border)",
          }}
        >
          <Plus size={8} />
          tag
        </button>
      )}
      {showInput && (
        <div className="flex flex-wrap gap-1 mt-1 w-full">
          {SUGGESTIONS.filter((s) => !tags.includes(s) && s.startsWith(input))
            .slice(0, 4)
            .map((s) => (
              <button
                key={s}
                onClick={() => { onAdd(s); setInput(""); setShowInput(false); }}
                className="rounded px-1.5 py-0.5 text-[8px] font-medium transition hover:brightness-110"
                style={{
                  backgroundColor: "var(--color-glass-hover)",
                  color: "var(--color-text-muted)",
                  border: "1px solid var(--color-glass-border)",
                }}
              >
                {s}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
