import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { useFullCycleResults } from "@/api/queries";
import { FullCycleResults } from "../Committee/FullCycleResults";

export function CommitteeResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const { data: results, isLoading, isError } = useFullCycleResults(jobId ?? null);

  if (isLoading) {
    return (
      <div className="flex h-full animate-fade-in flex-col gap-4">
        <div className="flex min-h-[36px] flex-wrap items-center gap-2">
          <button
            onClick={() => navigate("/results?tab=committee")}
            className="flex cursor-pointer items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1.5 text-[11px] leading-none text-(--color-text-muted)"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Committee
          </button>
        </div>
        <div className="flex animate-pulse flex-col gap-3">
          <div className="h-5 w-48 rounded bg-(--color-elevated)" />
          <div className="h-[320px] rounded bg-(--color-surface)" />
          <div className="grid grid-cols-3 gap-3">
            {Array.from({ length: 3 }, (_, i) => (
              <div key={i} className="h-[150px] rounded bg-(--color-surface)" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (isError || !results) {
    return (
      <div className="flex h-full animate-fade-in flex-col gap-4">
        <div className="flex min-h-[36px] flex-wrap items-center gap-2">
          <button
            onClick={() => navigate("/results?tab=committee")}
            className="flex cursor-pointer items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1.5 text-[11px] leading-none text-(--color-text-muted)"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Committee
          </button>
        </div>
        <div className="flex flex-col items-center justify-center gap-2 rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-8 py-12 text-(--color-text-muted)">
          <span className="font-mono text-sm">Failed to load committee results</span>
          <span className="text-[11px] text-(--color-text-dim)">
            The job may have been deleted or the results are unavailable.
          </span>
          <button
            onClick={() => navigate("/results?tab=committee")}
            className="mt-2 cursor-pointer rounded-sm border border-(--color-glass-border) bg-(--color-elevated) px-4 py-1.5 text-[11px] text-(--color-text-secondary)"
          >
            Back to Committee
          </button>
        </div>
      </div>
    );
  }

  if (results.status === "cancelled") {
    return (
      <div className="flex h-full animate-fade-in flex-col gap-4">
        <div className="flex min-h-[36px] flex-wrap items-center gap-2">
          <button
            onClick={() => navigate("/results?tab=committee")}
            className="flex cursor-pointer items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1.5 text-[11px] leading-none text-(--color-text-muted)"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Committee
          </button>
        </div>
        <div className="flex flex-col items-center justify-center gap-2 rounded-sm border border-(--color-glass-border) bg-(--color-surface) p-8 py-12 text-(--color-text-muted)">
          <span className="font-mono text-sm">Committee run was cancelled</span>
          <span className="text-[11px] text-(--color-text-dim)">
            This run was cancelled before completion. No results are available.
          </span>
          <button
            onClick={() => navigate("/committee")}
            className="mt-2 cursor-pointer rounded-sm border border-(--color-brand) bg-[rgba(0,229,255,0.08)] px-4 py-1.5 text-[11px] text-(--color-brand)"
          >
            Start New Committee
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full animate-fade-in flex-col gap-4">
      {/* Top ribbon */}
      <div className="flex min-h-[36px] flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-3 leading-none">
          <button
            onClick={() => navigate("/results?tab=committee")}
            className="flex cursor-pointer items-center gap-1 rounded-md border border-(--color-glass-border) bg-transparent px-2 py-1.5 text-[11px] leading-none text-(--color-text-muted)"
          >
            <ArrowLeft size={12} strokeWidth={1.5} /> Committee
          </button>
          <span className="font-mono text-[11px] leading-none text-(--color-text-secondary)">
            Committee
          </span>
          <span className="text-[10px] text-(--color-text-muted)">
            {results.status === "validation_failed" ? "VALIDATION FAILED" : results.status.toUpperCase()}
          </span>
        </div>
      </div>

      <FullCycleResults jobId={jobId ?? ""} onRunAgain={() => navigate("/committee")} />
    </div>
  );
}
