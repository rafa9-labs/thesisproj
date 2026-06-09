import { useState } from "react";
import { useLiveSentiment, useNewsStatus } from "@/api/queries";
import { formatPercent } from "@/lib/formatters";
import type { LiveSentimentPairData, LiveSentimentResponse } from "@/api/schemas";
import { ChevronDown, ChevronRight } from "lucide-react";

interface ArticleItem {
  title: string;
  source: string;
  sentiment_score: number;
  timestamp: string;
  relevance_tier?: number;
}

function formatStaleness(ageHours: number | undefined | null): string {
  if (ageHours == null) return "";
  if (ageHours < 0.02) return "just now";
  if (ageHours < 1) return `${Math.round(ageHours * 60)}m ago`;
  if (ageHours < 24) return `${Math.round(ageHours)}h ago`;
  return `${Math.round(ageHours / 24)}d ago`;
}

/** Confidence-adjusted horizontal Bull/Bear bar — dims when data is thin */
function BullBearBar({
  position,
  articleCount = 0,
  confidence = null,
  compact = false,
  vaderContribution,
  llmContribution,
}: {
  position: number;
  articleCount?: number;
  confidence?: number | null;
  compact?: boolean;
  vaderContribution?: number | null;
  llmContribution?: number | null;
}) {
  const clamped = Math.max(-1, Math.min(1, position));
  const pct = ((clamped + 1) / 2) * 100;
  const isLong = clamped > 0;
  const hasData = articleCount > 0;
  const isLowData = hasData && articleCount < 3;
  const color = isLong
    ? "var(--color-accent-success)"
    : clamped < 0
      ? "var(--color-accent-danger)"
      : "var(--color-text-muted)";
  const barOpacity = hasData
    ? isLowData
      ? 0.4
      : confidence != null && confidence < 0.3
        ? 0.6
        : 1
    : 0.15;

  const bar = (
    <div
      className="relative w-full rounded-full overflow-hidden"
      style={{
        height: compact ? 3 : 6,
        backgroundColor: "var(--color-glass-hover)",
      }}
    >
      {hasData && (
        <div
          className="absolute top-0 h-full rounded-full transition-all duration-500"
          style={{
            left: clamped >= 0 ? "50%" : `${pct}%`,
            width: `${Math.abs(clamped) * 50}%`,
            backgroundColor: color,
            opacity: barOpacity,
          }}
        />
      )}
      <div className="absolute top-0 h-full w-px" style={{ left: "50%", backgroundColor: "var(--color-text-muted)" }} />
    </div>
  );

  if (compact) return bar;

  return (
    <div className="flex flex-col gap-1">
      {bar}
      <div className="flex items-center justify-between text-[9px]">
        <span style={{ color: "var(--color-accent-danger)" }}>SHORT</span>
        {isLowData && (
          <span className="tabular-nums" style={{ color: "var(--color-accent-warning)", fontFamily: "var(--font-mono)" }}>
            low data
          </span>
        )}
        <span
          className="tabular-nums"
          style={{
            fontFamily: "var(--font-mono)",
            color:
              clamped > 0.3
                ? "var(--color-accent-success)"
                : clamped < -0.3
                  ? "var(--color-accent-danger)"
                  : "var(--color-text-muted)",
            opacity: isLowData ? 0.55 : 1,
          }}
        >
          {clamped > 0 ? "+" : ""}
          {clamped.toFixed(2)}
        </span>
        <span style={{ color: "var(--color-accent-success)" }}>LONG</span>
      </div>
      {!compact && vaderContribution != null && llmContribution != null && (
        <div className="flex items-center justify-center gap-2 text-[8px]" style={{ color: "var(--color-text-dim)" }}>
          <span>VADER: {vaderContribution > 0 ? "+" : ""}{vaderContribution.toFixed(2)}</span>
          <span>·</span>
          <span>LLM: {llmContribution > 0 ? "+" : ""}{llmContribution.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

/** Compact row showing one pair in the overview grid */
function MiniPairRow({ pair, data, isOther = false }: { pair: string; data: LiveSentimentPairData; isOther?: boolean }) {
  const pos = data.recommended_position ?? 0;
  const count = data.article_count ?? 0;
  const scoreText = pos > 0 ? `+${pos.toFixed(2)}` : pos.toFixed(2);
  const scoreColor =
    pos > 0.05 ? "var(--color-accent-success)" : pos < -0.05 ? "var(--color-accent-danger)" : "var(--color-text-muted)";

  return (
    <div
      className="flex items-center gap-2 py-1.5"
      style={{ borderBottom: isOther ? "none" : "0.5px dashed var(--color-glass-border)" }}
    >
      <span
        className="text-[10px] font-mono w-14 shrink-0 tabular-nums"
        style={{ color: isOther ? "var(--color-text-dim)" : "var(--color-text-secondary)" }}
      >
        {pair}
      </span>
      <div className="flex-1">
        <BullBearBar position={pos} articleCount={count} compact />
      </div>
      <span className="text-[9px] font-mono w-10 text-right tabular-nums" style={{ color: scoreColor }}>
        {scoreText}
      </span>
      <span
        className="text-[8px] font-mono w-5 text-right tabular-nums"
        style={{
          color: count === 0 ? "var(--color-accent-warning)" : count < 3 ? "var(--color-text-muted)" : "var(--color-text-dim)",
        }}
      >
        {count}
      </span>
    </div>
  );
}

/** Overview grid of all pairs' sentiment — compact, fits above the detail view */
function SentimentOverview({ sentiment }: { sentiment: LiveSentimentResponse | undefined }) {
  if (!sentiment?.pairs) return null;

  const entries = Object.entries(sentiment.pairs);
  if (entries.length === 0) return null;

  const majorRows = entries.filter(([pair]) => pair !== "OTHER");
  const otherEntry = entries.find(([pair]) => pair === "OTHER");

  return (
    <div className="flex flex-col -mx-1">
      {majorRows.map(([pair, data]) => (
        <MiniPairRow key={pair} pair={pair} data={data} />
      ))}
      {otherEntry && (
        <>
          <div className="my-1 mx-1" style={{ borderTop: "1px dashed var(--color-glass-border)" }} />
          <MiniPairRow pair={otherEntry[0]} data={otherEntry[1]} isOther />
        </>
      )}
    </div>
  );
}

/** Compact sentiment scores — strict 3-column grid, perfectly aligned */
function SentimentScores({
  llm,
  vader,
  confidence,
}: {
  llm: number | null;
  vader: number | null;
  confidence: number | null;
}) {
  const items = [
    { label: "LLM", value: llm != null ? (llm > 0 ? "+" : "") + llm.toFixed(2) : "—" },
    { label: "VADER", value: vader != null ? (vader > 0 ? "+" : "") + vader.toFixed(2) : "—" },
    { label: "CONF", value: confidence != null ? formatPercent(confidence) : "—" },
  ];
  return (
    <div className="grid grid-cols-3 mt-3 pt-3" style={{ borderTop: "1px solid var(--color-glass-border)" }}>
      {items.map(({ label, value }) => (
        <div key={label} className="flex flex-col items-center gap-0.5">
          <span className="text-[9px] font-medium uppercase tracking-[0.1em]" style={{ color: "var(--color-text-muted)" }}>
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

/** Article row — fixed height, strict 1-line title truncation */
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
    daysDiff > 0 ? `${daysDiff}d` : hoursDiff > 0 ? `${hoursDiff}h` : minsDiff > 0 ? `${minsDiff}m` : "now";

  return (
    <div
      className="flex items-center gap-2 border-l-2 pl-2"
      style={{
        borderLeftColor: borderColor,
        height: 32,
        flexShrink: 0,
      }}
    >
      <span className="flex-1 min-w-0 text-[10px] truncate" style={{ color: "var(--color-text-primary)", lineHeight: "1.3" }}>
        {article.title}
      </span>
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

  const pairData = sentiment?.pairs?.[pair] ?? sentiment?.pairs?.[pair.toUpperCase()];
  const recommendedPosition = pairData?.recommended_position ?? 0;
  const articleCount = pairData?.article_count ?? 0;
  const cacheAge = pairData?.cache_age_hours;
  const positionConfidence = pairData?.position_confidence ?? null;
  const isError = sentiment?.status === "error";
  const isNoData = sentiment?.status === "no_articles" || (articleCount === 0 && !sentLoading);

  if (sentLoading) {
    return (
      <div className="flex flex-col gap-3 animate-pulse">
        <div className="h-8 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
        <div className="h-16 rounded" style={{ backgroundColor: "var(--color-glass-hover)" }} />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center gap-2 py-4">
        <span className="text-[10px]" style={{ color: "var(--color-accent-danger)" }}>
          Sentiment unavailable
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <SentimentOverview sentiment={sentiment} />

      <div className="pt-2" style={{ borderTop: "1px solid var(--color-glass-border)" }}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-mono font-semibold" style={{ color: "var(--color-text-secondary)" }}>
            {pair}
          </span>
          {pairData?.currencies_affected && pairData.currencies_affected.length > 0 && (
            <span className="text-[8px]" style={{ color: "var(--color-text-dim)" }}>
              affects: {pairData.currencies_affected.join(", ")}
            </span>
          )}
        </div>

        {isNoData ? (
          <div className="flex flex-col items-center gap-1 py-3">
            <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
              No news data available
            </span>
            <span className="text-[9px]" style={{ color: "var(--color-text-muted)", opacity: 0.6 }}>
              RSS feeds will populate over time
            </span>
          </div>
        ) : (
          <>
            <BullBearBar position={recommendedPosition} articleCount={articleCount} confidence={positionConfidence}
              vaderContribution={pairData?.vader_contribution ?? null}
              llmContribution={pairData?.llm_contribution ?? null}
            />
            <SentimentScores
              llm={pairData?.llm_sentiment ?? null}
              vader={pairData?.vader_sentiment ?? null}
              confidence={pairData?.llm_confidence ?? null}
            />
            <div className="flex items-center justify-between mt-2">
              <span
                className="text-[9px] tabular-nums"
                style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}
              >
                {articleCount} articles
                {articleCount > 0 && articleCount < 3 && (
                  <span className="ml-1" style={{ color: "var(--color-accent-warning)" }}>
                    — limited
                  </span>
                )}
              </span>
              {cacheAge != null && (
                <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>
                  {formatStaleness(cacheAge)}
                </span>
              )}
            </div>
          </>
        )}
        {sentiment?.backend === "vader" && !isNoData && (
          <span className="text-[9px] block mt-2" style={{ color: "var(--color-accent-warning)" }}>
            LLM unavailable — using VADER fallback.
          </span>
        )}
      </div>
    </div>
  );
}

/** News articles only — used in Col 3 */
export function NewsArticlesPanel({ pair = "EURUSD" }: { pair?: string }) {
  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(pair);
  const { data: newsStatus } = useNewsStatus();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ exact: true });

  const articles = (sentiment?.top_articles ?? []) as (ArticleItem & { relevance_tier?: number })[];
  const tierCounts = sentiment?.article_count_by_tier;

  const tiers = [
    { key: "exact" as const, label: `${pair} articles`, filter: (a: typeof articles[number]) => a.relevance_tier === 1 },
    { key: "partial" as const, label: "Related currency news", filter: (a: typeof articles[number]) => a.relevance_tier === 2 },
    { key: "other" as const, label: "Other / untagged", filter: (a: typeof articles[number]) => a.relevance_tier === 0 },
  ];

  const toggle = (key: string) => setExpanded((p) => ({ ...p, [key]: !p[key] }));

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
      {articles.length > 0 ? (
        <div className="flex flex-col gap-1 overflow-y-auto max-h-[480px]">
          {tiers.map(({ key, label, filter }) => {
            const tierArticles = articles.filter(filter);
            if (tierArticles.length === 0) return null;
            const count = tierCounts?.[key] ?? tierArticles.length;
            const isExpanded = expanded[key];
            return (
              <div key={key}>
                <button
                  onClick={() => toggle(key)}
                  className="flex items-center gap-1 w-full text-left py-1"
                  style={{ color: "var(--color-text-muted)" }}
                >
                  {isExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                  <span className="text-[9px] uppercase tracking-[0.06em] font-medium">{label}</span>
                  <span className="text-[8px] tabular-nums" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-dim)" }}>
                    ({count})
                  </span>
                </button>
                {isExpanded && tierArticles.map((a, i) => (
                  <ArticleRow key={`${a.title}-${i}`} article={a} />
                ))}
              </div>
            );
          })}
        </div>
      ) : (
        <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
          No articles available. RSS feeds will populate over time.
        </span>
      )}
    </div>
  );
}
