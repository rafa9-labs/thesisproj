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

/** Compact sentiment scores — strict 3-column grid, perfectly aligned */
function SentimentScores({ llm, vader, confidence }: { llm: number | null; vader: number | null; confidence: number | null }) {
  const items = [
    { label: "LLM", value: llm != null ? (llm > 0 ? "+" : "") + llm.toFixed(2) : "—" },
    { label: "VADER", value: vader != null ? (vader > 0 ? "+" : "") + vader.toFixed(2) : "—" },
    { label: "CONF", value: confidence != null ? formatPercent(confidence) : "—" },
  ];
  return (
    <div
      className="grid grid-cols-3 mt-3 pt-3"
      style={{ borderTop: "1px solid var(--color-glass-border)" }}
    >
      {items.map(({ label, value }) => (
        <div key={label} className="flex flex-col items-center gap-0.5">
          <span
            className="text-[9px] font-medium uppercase tracking-[0.1em]"
            style={{ color: "var(--color-text-muted)" }}
          >
            {label}
          </span>
          <span
            className="text-[12px] tabular-nums"
            style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}
          >
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Article row — fixed height, strict 1-line title truncation, uniform sizing */
function ArticleRow({ article }: { article: ArticleItem }) {
  const score = article.sentiment_score;
  const isBullish = score > 0.2;
  const isBearish = score < -0.2;
  const borderColor = isBullish
    ? "var(--color-accent-success)"
    : isBearish
    ? "var(--color-accent-danger)"
    : "var(--color-glass-border)";

  const now = new Date().getTime();
  const ts = new Date(article.timestamp).getTime();
  const msDiff = now - ts;
  const minsDiff = Math.floor(msDiff / 60000);
  const hoursDiff = Math.floor(minsDiff / 60);
  const daysDiff = Math.floor(hoursDiff / 24);
  const timeStr =
    daysDiff > 0
      ? `${daysDiff}d`
      : hoursDiff > 0
      ? `${hoursDiff}h`
      : minsDiff > 0
      ? `${minsDiff}m`
      : "now";

  return (
    <div
      className="flex items-center gap-2 border-l-2 pl-2"
      style={{
        borderLeftColor: borderColor,
        height: 32,
        flexShrink: 0,
      }}
    >
      {/* title — strict 1 line, no wrapping */}
      <span
        className="flex-1 min-w-0 text-[10px] truncate"
        style={{ color: "var(--color-text-primary)", lineHeight: "1.3" }}
      >
        {article.title}
      </span>
      {/* meta: source + time, never wraps */}
      <span
        className="shrink-0 text-[9px] tabular-nums"
        style={{
          color: "var(--color-text-muted)",
          fontFamily: "var(--font-mono)",
          whiteSpace: "nowrap",
        }}
      >
        {article.source} · {timeStr}
      </span>
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
