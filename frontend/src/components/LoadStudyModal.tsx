import { useState, useMemo } from "react";
import { X, Search, Star, BarChart2 } from "lucide-react";
import { useBacktestStudies, useFullCycleHistory } from "@/api/queries";
import type { BacktestSummaryItem, FullCycleHistoryEntry } from "@/api/schemas";

interface Props {
  open: boolean;
  onClose: () => void;
  onSelect: (jobId: string) => void;
  studyType: "backtest" | "committee";
}

export function LoadStudyModal({ open, onClose, onSelect, studyType }: Props) {
  const [search, setSearch] = useState("");
  const [favoriteOnly, setFavoriteOnly] = useState(false);

  const { data: backtestData, isLoading: btLoading } = useBacktestStudies(
    studyType === "backtest"
      ? { limit: 200, search: search || undefined, favorite_only: favoriteOnly || undefined }
      : {},
  );
  const { data: committeeData, isLoading: fcLoading } =
    studyType === "committee" ? useFullCycleHistory() : { data: null, isLoading: false };

  const isLoading = studyType === "backtest" ? btLoading : fcLoading;

  const items = useMemo(() => {
    if (studyType === "backtest" && backtestData?.results) {
      return backtestData.results.filter((r) => {
        if (favoriteOnly && !r.study_meta?.is_favorite) return false;
        if (search) {
          const dn = (r.study_meta?.display_name ?? "").toLowerCase();
          const jid = r.job_id.toLowerCase();
          const q = search.toLowerCase();
          if (!dn.includes(q) && !jid.includes(q)) return false;
        }
        return r.study_meta != null;
      });
    }
    if (studyType === "committee" && committeeData?.entries) {
      return committeeData.entries.filter((e) => {
        if (favoriteOnly && !e.study_meta?.is_favorite) return false;
        if (search) {
          const dn = (e.study_meta?.display_name ?? "").toLowerCase();
          const jid = e.job_id.toLowerCase();
          const q = search.toLowerCase();
          if (!dn.includes(q) && !jid.includes(q)) return false;
        }
        return e.study_meta != null;
      });
    }
    return [];
  }, [studyType, backtestData, committeeData, search, favoriteOnly]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/[0.6] backdrop-blur-[4px]"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Load Previous Study"
    >
      <div
        className="flex max-h-[70vh] w-[640px] animate-fade-in flex-col rounded-sm border border-(--color-glass-border) bg-(--color-elevated) p-6 shadow-2xl shadow-[0_0_40px_rgba(0,229,255,0.08)]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart2 size={14} className="text-(--color-brand)" />
            <span className="text-[11px] font-semibold tracking-[0.1em] text-(--color-text-primary) uppercase">
              Load {studyType === "backtest" ? "Backtest" : "Consensus"} Study
            </span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="cursor-pointer rounded border-none bg-none p-1 text-(--color-text-muted) hover:brightness-110"
          >
            <X size={14} />
          </button>
        </div>

        <div className="mb-4 flex items-center gap-3">
          <div className="flex flex-1 items-center gap-2 rounded border border-(--color-glass-border) bg-(--color-input-bg) px-3 py-2">
            <Search size={12} className="text-(--color-text-muted)" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by name or job ID..."
              className="flex-1 border-none bg-transparent font-mono text-xs text-(--color-text-primary) outline-none"
            />
          </div>
          <label className="flex cursor-pointer items-center gap-1.5">
            <input
              type="checkbox"
              checked={favoriteOnly}
              onChange={(e) => setFavoriteOnly(e.target.checked)}
              className="accent-[var(--color-accent-warning)]"
            />
            <Star
              size={12}
              style={{
                color: favoriteOnly ? "var(--color-accent-warning)" : "var(--color-text-dim)",
              }}
            />
            <span className="text-[10px] text-(--color-text-secondary)">Favorites</span>
          </label>
        </div>

        <div className="max-h-[50vh] flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <span className="text-[10px] text-(--color-text-muted)">Loading studies...</span>
            </div>
          ) : items.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-12">
              <span className="text-[10px] text-(--color-text-muted)">
                {search
                  ? "No saved studies match your search."
                  : "No saved studies yet. Save a study from the results page first."}
              </span>
            </div>
          ) : (
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-(--color-glass-border)">
                  <th className="px-4 py-2 text-left text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
                    Name
                  </th>
                  <th className="px-4 py-2 text-left text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
                    Tags
                  </th>
                  <th className="px-4 py-2 text-left text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
                    Date
                  </th>
                  <th className="px-4 py-2 text-right text-[10px] font-medium tracking-[0.08em] text-(--color-text-muted) uppercase">
                    Sharpe
                  </th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => {
                  const isBacktest = studyType === "backtest";
                  const displayName =
                    item.study_meta?.display_name ??
                    (isBacktest
                      ? (item as BacktestSummaryItem).job_id
                      : (item as FullCycleHistoryEntry).job_id);
                  const tags = item.study_meta?.tags ?? [];
                  const isFav = item.study_meta?.is_favorite ?? false;
                  const dateStr = isBacktest
                    ? new Date((item as BacktestSummaryItem).created_at).toLocaleDateString()
                    : new Date((item as FullCycleHistoryEntry).started_at).toLocaleDateString();
                  const sharpe = isBacktest
                    ? (item as BacktestSummaryItem).sharpe
                    : (item as FullCycleHistoryEntry).avg_sharpe;
                  const sharpeColor =
                    sharpe != null && sharpe > 0
                      ? "#089981"
                      : sharpe != null && sharpe < 0
                        ? "#F23645"
                        : "var(--color-text-dim)";

                  return (
                    <tr
                      key={item.job_id}
                      onClick={() => onSelect(item.job_id)}
                      className="cursor-pointer border-b border-(--color-glass-border) transition-colors duration-100 hover:bg-(--color-glass-hover)"
                    >
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-1.5">
                          {isFav && (
                            <Star
                              size={10}
                              className="text-(--color-accent-warning)"
                              style={{ fill: "var(--color-accent-warning)" }}
                            />
                          )}
                          <span className="font-mono font-medium text-(--color-text-primary)">
                            {displayName}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5">
                        <div className="flex flex-wrap gap-1">
                          {tags.slice(0, 3).map((t) => (
                            <span
                              key={t}
                              className="rounded bg-(--color-glass) px-1.5 py-0.5 font-mono text-[9px] text-(--color-text-muted)"
                            >
                              {t}
                            </span>
                          ))}
                          {tags.length > 3 && (
                            <span className="text-[9px] text-(--color-text-dim)">
                              +{tags.length - 3}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-2.5 font-mono text-(--color-text-muted)">{dateStr}</td>
                      <td
                        className="px-4 py-2.5 text-right font-mono font-medium"
                        style={{ color: sharpeColor }}
                      >
                        {sharpe != null ? sharpe.toFixed(2) : "--"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
