import { EmptyState } from "@/components/shared/EmptyState";
import { Newspaper, TrendingUp, Calendar, Brain } from "lucide-react";

export function NewsPage() {
  return (
    <div className="flex flex-col gap-6">
      <h2
        className="text-base font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        News &amp; Sentiment
      </h2>

      <div className="grid grid-cols-3 gap-4">
        <div
          className="rounded-lg border p-4"
          style={{
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-surface)",
          }}
        >
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={16} style={{ color: "var(--color-accent-success)" }} />
            <span className="text-xs font-semibold uppercase" style={{ color: "var(--color-text-secondary)" }}>
              Sentiment Engine
            </span>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Backend</span>
              <span className="text-xs" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                VADER
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>finBERT</span>
              <span className="text-xs" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
                Not installed
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Window</span>
              <span className="text-xs" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                6 bars
              </span>
            </div>
          </div>
        </div>

        <div
          className="rounded-lg border p-4"
          style={{
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-surface)",
          }}
        >
          <div className="flex items-center gap-2 mb-3">
            <Calendar size={16} style={{ color: "var(--color-accent-warning)" }} />
            <span className="text-xs font-semibold uppercase" style={{ color: "var(--color-text-secondary)" }}>
              Upcoming Events
            </span>
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              No upcoming events in the next 7 days.
            </span>
            <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              Events: NFP, FOMC, CPI, PMI
            </span>
          </div>
        </div>

        <div
          className="rounded-lg border p-4"
          style={{
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-surface)",
          }}
        >
          <div className="flex items-center gap-2 mb-3">
            <Brain size={16} style={{ color: "var(--color-accent-deep)" }} />
            <span className="text-xs font-semibold uppercase" style={{ color: "var(--color-text-secondary)" }}>
              Sentiment Features
            </span>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>VADER Compound</span>
              <span className="text-xs" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                Disabled
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>Event Flags</span>
              <span className="text-xs" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                Disabled
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--color-text-muted)" }}>News Window</span>
              <span className="text-xs" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                24 bars
              </span>
            </div>
          </div>
        </div>
      </div>

      <EmptyState
        icon={<Newspaper size={48} />}
        title="Sentiment data requires backend connection"
        description="Start the FastAPI backend and ensure news data has been scraped. Sentiment scores, event markers, and news-driven features will appear here."
      />
    </div>
  );
}
