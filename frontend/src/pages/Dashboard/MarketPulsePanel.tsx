import { useLiveSentiment, useNewsStatus } from "@/api/queries";
import { formatPercent } from "@/lib/formatters";

interface ArticleItem {
  title: string;
  source: string;
  sentiment_score: number;
  timestamp: string;
}

function SentimentGauge({ value, label, pair }: { value: number | null; label: string; pair: string }) {
  const clamped = Math.max(-1, Math.min(1, value ?? 0));
  const pct = ((clamped + 1) / 2) * 100;
  const isBullish = clamped > 0.3;
  const isBearish = clamped < -0.3;
  const color = isBullish ? "var(--color-accent-success)" : isBearish ? "var(--color-accent-danger)" : "var(--color-accent-warning)";
  const bg = isBullish ? "rgba(34,197,94,0.12)" : isBearish ? "rgba(239,68,68,0.12)" : "rgba(245,158,11,0.12)";

  return (
    <div className="flex items-center gap-4">
      <div className="relative w-20 h-20 flex-shrink-0">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r="52" fill="none" stroke="var(--color-glass-hover)" strokeWidth="8" />
          <circle
            cx="60" cy="60" r="52" fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${(pct / 100) * 327} 327`}
            style={{ transition: "stroke-dasharray 0.6s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-sm font-semibold" style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
            {clamped > 0 ? "+" : ""}{clamped.toFixed(2)}
          </span>
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium" style={{ color: "var(--color-text-primary)" }}>{pair}</span>
        <span
          className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[10px] font-medium uppercase"
          style={{ backgroundColor: bg, color, width: "fit-content" }}
        >
          {isBullish ? "Bullish" : isBearish ? "Bearish" : "Neutral"}
        </span>
        <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>{label}</span>
      </div>
    </div>
  );
}

function ArticleFeedItem({ article }: { article: ArticleItem }) {
  const isPositive = article.sentiment_score > 0;
  const color = isPositive ? "var(--color-accent-success)" : article.sentiment_score < 0 ? "var(--color-accent-danger)" : "var(--color-text-muted)";
  return (
    <div className="flex items-center gap-2 py-1.5" style={{ borderBottom: "1px solid var(--color-glass-border)" }}>
      <span className="font-mono text-[10px] font-medium w-10 text-right flex-shrink-0" style={{ color }}>
        {article.sentiment_score > 0 ? "+" : ""}{article.sentiment_score.toFixed(2)}
      </span>
      <div className="flex-1 min-w-0">
        <div className="text-[11px] truncate leading-tight" style={{ color: "var(--color-text-secondary)" }}>{article.title}</div>
        <div className="text-[9px] mt-0.5" style={{ color: "var(--color-text-muted)" }}>{article.source}</div>
      </div>
    </div>
  );
}

export function MarketPulsePanel({ pair = "EURUSD" }: { pair?: string }) {
  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(pair);
  const { data: newsStatus } = useNewsStatus();

  const pairData = sentiment?.pairs?.[pair];
  const blended = pairData?.blended_sentiment ?? pairData?.vader_sentiment ?? 0;
  const topArticles: ArticleItem[] = sentiment?.top_articles ?? [];

  return (
    <div className="flex flex-col gap-4">
      <div
        className="rounded-lg border p-4"
        style={{
          borderColor: "var(--color-glass-border)",
          backgroundColor: "var(--color-glass)",
          backdropFilter: "blur(12px)",
        }}
      >
        <div className="flex flex-col gap-4">
          {sentLoading ? (
            <div className="flex items-center justify-center py-4" style={{ color: "var(--color-text-muted)" }}>
              <span className="text-xs">Loading sentiment data...</span>
            </div>
          ) : (
            <>
              <SentimentGauge value={blended} label={sentiment?.backend === "vader" ? "VADER" : `LLM (${sentiment?.model ?? "?"})`} pair={pair} />

              <div className="flex flex-col gap-1">
                <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Sentiment Details</span>
                <div className="flex justify-between text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                  <span>LLM</span>
                  <span>{pairData?.llm_sentiment != null ? (pairData.llm_sentiment > 0 ? "+" : "") + pairData.llm_sentiment.toFixed(2) : "—"}</span>
                </div>
                <div className="flex justify-between text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                  <span>VADER</span>
                  <span>{pairData?.vader_sentiment != null ? (pairData.vader_sentiment > 0 ? "+" : "") + pairData.vader_sentiment.toFixed(2) : "—"}</span>
                </div>
                <div className="flex justify-between text-[10px]" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>
                  <span>Confidence</span>
                  <span>{pairData?.llm_confidence != null ? formatPercent(pairData.llm_confidence) : "—"}</span>
                </div>
              </div>

              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>Top Articles</span>
                  <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>
                    {newsStatus?.cached_articles ?? 0} cached
                  </span>
                </div>
                {topArticles.length > 0 ? (
                  topArticles.map((a, i) => <ArticleFeedItem key={`${a.title}-${i}`} article={a} />)
                ) : (
                  <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>No articles available. Run a backtest with news features enabled.</span>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}