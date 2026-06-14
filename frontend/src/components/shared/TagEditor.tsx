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
          className="inline-flex cursor-pointer items-center gap-0.5 rounded bg-(--color-glass-hover) px-1.5 py-0.5 text-[9px] font-medium text-(--color-text-secondary) hover:brightness-110"
          style={{ border: "1px solid var(--color-glass-border)" }}
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
            if (e.key === "Escape") {
              setShowInput(false);
              setInput("");
            }
          }}
          onBlur={() => {
            if (input.trim()) handleAdd();
            else setShowInput(false);
          }}
          className="rounded bg-(--color-elevated) px-1.5 py-0.5 text-[9px] text-(--color-text-primary) outline-none"
          style={{ width: 70, border: "1px solid var(--color-brand)" }}
          placeholder="tag..."
        />
      ) : (
        <button
          onClick={() => setShowInput(true)}
          className="inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 text-[9px] text-(--color-text-muted) transition hover:brightness-110"
          style={{ backgroundColor: "transparent", border: "1px dashed var(--color-glass-border)" }}
        >
          <Plus size={8} />
          tag
        </button>
      )}
      {showInput && (
        <div className="mt-1 flex w-full flex-wrap gap-1">
          {SUGGESTIONS.filter((s) => !tags.includes(s) && s.startsWith(input))
            .slice(0, 4)
            .map((s) => (
              <button
                key={s}
                onMouseDown={(e) => {
                  e.preventDefault();
                  onAdd(s);
                  setInput("");
                  setShowInput(false);
                }}
                className="rounded bg-(--color-glass-hover) px-1.5 py-0.5 text-[8px] font-medium text-(--color-text-muted) transition hover:brightness-110"
                style={{ border: "1px solid var(--color-glass-border)" }}
              >
                {s}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
