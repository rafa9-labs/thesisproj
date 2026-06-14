import { useState } from "react";
import { X, Bookmark, Star } from "lucide-react";
import type { StudyMeta, StudyMetaRequest } from "@/api/schemas";

interface Props {
  open: boolean;
  onClose: () => void;
  onSave: (meta: StudyMetaRequest) => Promise<void>;
  defaultName: string;
  existingMeta?: StudyMeta | null;
  isSaving: boolean;
}

export function StudySaveModal({
  open,
  onClose,
  onSave,
  defaultName,
  existingMeta,
  isSaving,
}: Props) {
  const [displayName, setDisplayName] = useState(existingMeta?.display_name ?? defaultName);
  const [tagInput, setTagInput] = useState((existingMeta?.tags ?? []).join(", "));
  const [notes, setNotes] = useState(existingMeta?.notes ?? "");
  const [isFavorite, setIsFavorite] = useState(existingMeta?.is_favorite ?? false);

  if (!open) return null;

  const tags = tagInput
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);

  const handleSave = async () => {
    await onSave({
      display_name: displayName.trim() || undefined,
      tags,
      notes: notes.trim() || undefined,
      is_favorite: isFavorite,
    });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.6] backdrop-blur-[4px]"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Save Study"
    >
      <div
        className="w-[420px] animate-fade-in rounded-sm border border-(--color-glass-border) bg-(--color-elevated) p-6 shadow-2xl"
        style={{ boxShadow: "0 0 40px rgba(0,229,255,0.08)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bookmark size={14} className="text-(--color-brand)" />
            <span className="text-[11px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
              Save Study
            </span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-(--color-text-muted) hover:brightness-110"
            style={{ background: "none", border: "none" }}
            className="cursor-pointer"
          >
            <X size={14} />
          </button>
        </div>

        <div className="flex flex-col gap-4">
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
              Display Name
            </span>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder={defaultName}
              className="rounded border border-(--color-glass-border) bg-(--color-input-bg) px-3 py-2 font-mono text-xs text-(--color-text-primary)"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
              Tags{" "}
              <span className="font-normal tracking-normal text-(--color-text-dim) normal-case">
                (comma-separated)
              </span>
            </span>
            <input
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              placeholder="e.g. high-vol, US session"
              className="rounded border border-(--color-glass-border) bg-(--color-input-bg) px-3 py-2 font-mono text-xs text-(--color-text-primary)"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
              Notes
            </span>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="What was special about this run?"
              rows={3}
              className="resize-none rounded border border-(--color-glass-border) bg-(--color-input-bg) px-3 py-2 font-mono text-xs text-(--color-text-primary)"
            />
          </label>

          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              checked={isFavorite}
              onChange={(e) => setIsFavorite(e.target.checked)}
              className="accent-[var(--color-accent-warning)]"
            />
            <Star
              size={12}
              style={{
                color: isFavorite ? "var(--color-accent-warning)" : "var(--color-text-dim)",
              }}
            />
            <span className="text-[10px] text-(--color-text-secondary)">Mark as favorite</span>
          </label>
        </div>

        <div className="mt-6 flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            className="rounded border border-(--color-glass-border) px-4 py-2 text-[10px] font-semibold tracking-[0.08em] text-(--color-text-muted) uppercase"
            style={{ backgroundColor: "transparent" }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="rounded bg-(--color-accent-success) px-4 py-2 text-[10px] font-semibold tracking-[0.08em] text-(--color-text-inverse) uppercase"
            style={{
              border: "none",
              cursor: isSaving ? "not-allowed" : "pointer",
              opacity: isSaving ? 0.6 : 1,
            }}
          >
            {isSaving ? "Saving..." : "Save Study"}
          </button>
        </div>
      </div>
    </div>
  );
}
