import { useLiveSentiment, useNewsStatus } from "@/api/queries";
import { formatPercent } from "@/lib/formatters";

interface ArticleItem {
  title: string;
  source: string;
  sentiment_score: number;
  timestamp: string;
}

/** Horizontal Bull/Bear progress bar replacing circular gauge */
function BullBearBar({ position }: { position: number }) {
  const clamped = Math.max(-1, Math.min(1, position));
  const pct = ((clamped + 1) / 2) * 100;
  const isLong = clamped > 0;
  const color = isLong ? "var(--color-accent-success)" : clamped < 0 ? "var(--color-accent-danger)" : "var(--color-text-muted)";

  return (
    <div className="flex flex-col gap-1">
      <div className="relative h-1.5 w-full rounded-full overflow-hidden" style={{ backgroundColor: "var(--color-glass-hover)" }}>
        <div
          className="absolute top-0 h-full rounded-full transition-all duration-500"
          style={{
            left: clamped >= 0 ? "50%" : `${pct}%`,
            width: `${Math.abs(clamped) * 50}%`,
            backgroundColor: color,
          }}
        />
        <div className="absolute top-0 h-full w-px" style={{ left: "50%", backgroundColor: "var(--color-text-muted)" }} />
      </div>
      <div className="flex items-center justify-between text-[9px]">
        <span style={{ color: "var(--color-accent-danger)" }}>SHORT</span>
        <span style={{ color: clamped > 0.3 ? "var(--color-accent-success)" : clamped < -0.3 ? "var(--color-accent-danger)" : "var(--color-text-muted)" }}>
          {clamped > 0 ? "+" : ""}{clamped.toFixed(2)}
        </span>
        <span style={{ color: "var(--color-accent-success)" }}>LONG</span>
      </div>
    </div>
  );
}

/** Compact sentiment scores in grid format */
function SentimentScores({ llm, vader, confidence }: { llm: number | null; vader: number | null; confidence: number | null }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="grid grid-cols-3 gap-2 text-[10px]">
        <div>
          <div style={{ color: "var(--color-text-muted)" }}>LLM</div>
          <div className="font-mono" style={{ color: "var(--color-text-secondary)" }}>
            {llm != null ? (llm > 0 ? "+" : "") + llm.toFixed(2) : "—"}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--color-text-muted)" }}>VADER</div>
          <div className="font-mono" style={{ color: "var(--color-text-secondary)" }}>
            {vader != null ? (vader > 0 ? "+" : "") + vader.toFixed(2) : "—"}
          </div>
        </div>
        <div>
          <div style={{ color: "var(--color-text-muted)" }}>Conf</div>
          <div className="font-mono" style={{ color: "var(--color-text-secondary)" }}>
            {confidence != null ? formatPercent(confidence) : "—"}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Article row with colored left border and relative timestamp */
function ArticleRow({ article }: { article: ArticleItem }) {
  const score = article.sentiment_score;
  const isBullish = score > 0.2;
  const isBearish = score < -0.2;
  const borderColor = isBullish ? "var(--color-accent-success)" : isBearish ? "var(--color-accent-danger)" : "var(--color-text-muted)";

  const now = new Date().getTime();
  const ts = new Date(article.timestamp).getTime();
  const msDiff = now - ts;
  const minsDiff = Math.floor(msDiff / 60000);
  const hoursDiff = Math.floor(minsDiff / 60);
  const daysDiff = Math.floor(hoursDiff / 24);
  const timeStr =
    daysDiff > 0 ? `${daysDiff}d ago` : hoursDiff > 0 ? `${hoursDiff}h ago` : minsDiff > 0 ? `${minsDiff}m ago` : "now";

  return (
    <div
      className="flex gap-1.5 py-1.5 text-[9px] border-l-2 pl-1.5"
      style={{ borderLeftColor: borderColor, color: "var(--color-text-secondary)" }}
    >
      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <div className="truncate" style={{ color: "var(--color-text-primary)" }}>
          {article.title}
        </div>
        <div style={{ color: "var(--color-text-muted)" }} className="flex items-center justify-between">
          <span>{article.source}</span>
          <span>{timeStr}</span>
        </div>
      </div>
    </div>
  );
}

/** Sentiment + bias only — used in Col 2 */
export function MarketPulsePanel({ pair = "EURUSD" }: { pair?: string }) {
  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(pair);

  const pairData = sentiment?.pairs?.[pair];
  const recommendedPosition = pairData?.recommended_position ?? 0;

  if (sentLoading) {
    return (
      <div className="flex flex-col gap-3 animate-pulse">
        <div className="h-8 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
        <div className="h-16 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <BullBearBar position={recommendedPosition} />
      <SentimentScores
        llm={pairData?.llm_sentiment}
        vader={pairData?.vader_sentiment}
        confidence={pairData?.llm_confidence}
      />
      {sentiment?.backend === "vader" && (
        <span className="text-[9px]" style={{ color: "var(--color-accent-warning)" }}>
          LLM unavailable — using VADER fallback.
        </span>
      )}
    </div>
  );
}

/** News articles only — used in Col 3 */
export function NewsArticlesPanel({ pair = "EURUSD" }: { pair?: string }) {
  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(pair);
  const { data: newsStatus } = useNewsStatus();

  const topArticles: ArticleItem[] = sentiment?.top_articles ?? [];

  if (sentLoading) {
    return (
      <div className="flex flex-col gap-2 animate-pulse">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i} className="h-8 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-0 h-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] uppercase tracking-[0.1em] font-medium" style={{ color: "var(--color-text-muted)" }}>
          Top Articles
        </span>
        <span className="text-[9px]" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          {newsStatus?.cached_articles ?? 0} cached
        </span>
      </div>
      {topArticles.length > 0 ? (
        topArticles.slice(0, 8).map((a, i) => (
          <ArticleRow key={`${a.title}-${i}`} article={a} />
        ))
      ) : (
        <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
          No articles available. Run a backtest with news features enabled.
        </span>
      )}
    </div>
  );
}
