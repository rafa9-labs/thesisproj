import { EmptyState } from "@/components/shared/EmptyState";
import { Newspaper } from "lucide-react";

export function NewsPage() {
  return (
    <div className="flex flex-col gap-6">
      <h2
        className="text-base font-semibold uppercase tracking-[0.1em]"
        style={{ color: "var(--color-text-secondary)" }}
      >
        News &amp; Sentiment
      </h2>
      <EmptyState
        icon={<Newspaper size={48} />}
        title="Sentiment Dashboard"
        description="VADER and finBERT sentiment analysis, economic event calendar, and news-driven feature configuration. Configure sentiment sources in Settings."
      />
    </div>
  );
}
