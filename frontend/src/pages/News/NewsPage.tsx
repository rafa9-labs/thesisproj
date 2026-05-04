import { EmptyState } from "@/components/shared/EmptyState";
import { Newspaper, TrendingUp, Calendar, Brain, Activity } from "lucide-react";
import { useNewsStatus } from "@/api/queries";

export function NewsPage() {
  const { data: status, isLoading, isError } = useNewsStatus();

  const backend = status?.sentiment_backend ?? "vader";
  const finbertOk = status?.finbert_available ?? false;
  const articles = status?.cached_articles ?? 0;
  const eventTypes = status?.event_types ?? [];
  const features = status?.features;

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-4 gap-4">
        <Card icon={<TrendingUp size={15} strokeWidth={1.5} style={{ color: "var(--color-accent-success)" }} />} title="Sentiment Engine">
          <Row label="Backend" value={backend.toUpperCase()} />
          <Row label="finBERT" value={finbertOk ? "Available" : "Not installed"} muted={!finbertOk} />
          <Row label="Cached Articles" value={isLoading ? "…" : String(articles)} />
        </Card>

        <Card icon={<Calendar size={15} strokeWidth={1.5} style={{ color: "var(--color-accent-warning)" }} />} title="Event Calendar">
          {eventTypes.length > 0 ? (
            <div className="flex flex-col gap-1.5">
              <span className="text-[11px] font-light" style={{ color: "var(--color-text-muted)" }}>Tracked event types:</span>
              <div className="flex flex-wrap gap-1.5">
                {eventTypes.map((e) => (
                  <span
                    key={e}
                    className="rounded px-1.5 py-0.5 text-[10px] font-medium tracking-[0.04em]"
                    style={{
                      backgroundColor: "rgba(245,158,11,0.08)",
                      color: "var(--color-accent-warning)",
                      fontFamily: "var(--font-mono)",
                      border: "1px solid rgba(245,158,11,0.12)",
                    }}
                  >
                    {e}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <span className="text-[11px] font-light" style={{ color: "var(--color-text-muted)" }}>No event types configured.</span>
          )}
        </Card>

        <Card icon={<Brain size={15} strokeWidth={1.5} style={{ color: "var(--color-accent-deep)" }} />} title="Sentiment Features">
          {features ? (
            <>
              <Row label="VADER Compound" value={features.vader_compound ? "Active" : "Disabled"} muted={!features.vader_compound} />
              <Row label="Event Flags" value={features.event_flags ? "Active" : "Disabled"} muted={!features.event_flags} />
              <Row label="Windows" value={features.news_volume_windows.join(", ") + " bars"} />
            </>
          ) : (
            <span className="text-[11px] font-light" style={{ color: "var(--color-text-muted)" }}>Feature info requires backend.</span>
          )}
        </Card>

        <Card icon={<Activity size={15} strokeWidth={1.5} style={{ color: "var(--color-accent-info)" }} />} title="Data Status">
          <Row label="Articles" value={String(articles)} />
          <Row label="Sentiment" value={backend} />
          <Row label="finBERT" value={finbertOk ? "Ready" : "N/A"} muted={!finbertOk} />
        </Card>
      </div>

      {isError && (
        <EmptyState
          icon={<Newspaper size={48} strokeWidth={1} />}
          title="Backend not reachable"
          description="Start the FastAPI backend to see live sentiment data. The news pipeline scrapes RSS feeds, computes VADER/finBERT scores, and generates event proximity markers."
        />
      )}
    </div>
  );
}

function Card({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <div
      className="flex flex-col gap-3 rounded-lg border p-5 transition-all duration-300 hover:border-[var(--color-border-active)]"
      style={{
        borderColor: "var(--color-glass-border)",
        backgroundColor: "var(--color-glass)",
        backdropFilter: "blur(12px)",
      }}
    >
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-[11px] font-medium uppercase tracking-[0.12em]" style={{ color: "var(--color-text-muted)" }}>
          {title}
        </span>
      </div>
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

function Row({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px] font-light" style={{ color: "var(--color-text-muted)" }}>{label}</span>
      <span
        className="text-[11px] font-medium"
        style={{
          color: muted ? "var(--color-text-muted)" : "var(--color-text-primary)",
          fontFamily: "var(--font-mono)",
        }}
      >
        {value}
      </span>
    </div>
  );
}
