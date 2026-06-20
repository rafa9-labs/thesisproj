import { useState } from "react";
import { useLiveSentiment, useNewsStatus } from "@/api/queries";
import { formatPercent } from "@/lib/formatters";
import { BullBearBar } from "@/components/shared/BullBearBar";
import { useDashboardStore } from "@/stores/useDashboardStore";
import { useAppStore } from "@/stores/useAppStore";
import type { LiveSentimentPairData, LiveSentimentResponse } from "@/api/schemas";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { SENTIMENT_THRESHOLDS, BIAS_THRESHOLD } from "@/lib/sentiment-thresholds";

interface ArticleItem {
  title: string;
  source: string;
  sentiment_score: number;
  timestamp: string;
  relevance_tier?: number;
  url?: string;
  body?: string;
  summary?: string;
  llm_sentiment?: number | null;
  llm_confidence?: number | null;
}

function formatStaleness(ageHours: number | undefined | null): string {
  if (ageHours == null) return "";
  if (ageHours < 0.02) return "just now";
  if (ageHours < 1) return `${Math.round(ageHours * 60)}m ago`;
  if (ageHours < 24) return `${Math.round(ageHours)}h ago`;
  return `${Math.round(ageHours / 24)}d ago`;
}

function getSourceDomain(source: string): string {
  const map: Record<string, string> = {
    reuters: "reuters.com",
    bloomberg: "bloomberg.com",
    forexlive: "forexlive.com",
    fxstreet: "fxstreet.com",
    dailyfx: "dailyfx.com",
    investing: "investing.com",
    marketwatch: "marketwatch.com",
    cnbc: "cnbc.com",
    financialtimes: "ft.com",
  };
  const key = source.toLowerCase().replace(/\s+/g, "");
  return map[key] || `${key}.com`;
}

function getImpactLabel(score: number, magnitude?: number): { label: string; className: string } {
  const impact = Math.abs(score) * (magnitude ?? 1);
  if (impact >= SENTIMENT_THRESHOLDS.IMPACT_HIGH) return { label: "High", className: "bg-(--color-accent-danger) text-white" };
  if (impact >= SENTIMENT_THRESHOLDS.IMPACT_MED) return { label: "Med", className: "bg-(--color-accent-warning) text-black" };
  return { label: "Low", className: "bg-(--color-text-dim) text-(--color-text-primary)" };
}

function FaviconImg({ source }: { source: string }) {
  const domain = getSourceDomain(source);
  return (
    <img
      src={`https://www.google.com/s2/favicons?domain=${domain}&sz=16`}
      alt=""
      className="h-3.5 w-3.5 shrink-0"
      onError={(e) => {
        (e.target as HTMLImageElement).style.display = "none";
      }}
    />
  );
}

/** Compact row showing one pair in the overview grid */
function MiniPairRow({
  pair,
  data,
  isActive = false,
  isOther = false,
}: {
  pair: string;
  data: LiveSentimentPairData;
  isActive?: boolean;
  isOther?: boolean;
}) {
  const pos = data.recommended_position ?? 0;
  const count = data.article_count ?? 0;
  const scoreText = pos > 0 ? `+${pos.toFixed(2)}` : pos.toFixed(2);
  const scoreColor =
    pos > BIAS_THRESHOLD
      ? "var(--color-accent-success)"
      : pos < -BIAS_THRESHOLD
        ? "var(--color-accent-danger)"
        : "var(--color-text-muted)";

  return (
    <div
      className={cn(
        "flex items-center gap-2 py-1.5 transition-colors duration-150",
        isActive
          ? "bg-(--color-glass-hover) border-l-[3px] border-l-(--color-brand) pl-1.5"
          : "border-l-[3px] border-l-transparent pl-1.5",
      )}
      style={{ borderBottom: isOther ? "none" : "1px solid var(--color-glass-border)" }}
    >
      <span
        className={cn(
          "w-14 shrink-0 font-mono text-[10px] tabular-nums",
          isOther ? "text-(--color-text-dim)" : "text-(--color-text-secondary)",
        )}
      >
        {pair}
      </span>
      <div className="flex-1">
        <BullBearBar position={pos} articleCount={count} compact />
      </div>
      <span
        className="w-10 text-right font-mono text-[10px] font-semibold tabular-nums"
        style={{ color: scoreColor }}
      >
        {scoreText}
      </span>
      <span
        className="w-5 text-right font-mono text-[9px] font-medium tabular-nums"
        style={{
          color:
            count === 0
              ? "var(--color-accent-warning)"
              : count < 3
                ? "var(--color-text-muted)"
                : "var(--color-text-secondary)",
        }}
      >
        {count}
      </span>
    </div>
  );
}

/** Overview grid of all pairs' sentiment — compact, fits above the detail view */
function SentimentOverview({
  sentiment,
  activePair,
}: {
  sentiment: LiveSentimentResponse | undefined;
  activePair: string;
}) {
  if (!sentiment?.pairs) return null;

  const entries = Object.entries(sentiment.pairs);
  if (entries.length === 0) return null;

  const majorRows = entries.filter(([pair]) => pair !== "OTHER");
  const otherEntry = entries.find(([pair]) => pair === "OTHER");

  return (
    <div className="-mx-1 flex flex-col">
      {majorRows.map(([pair, data]) => (
        <MiniPairRow
          key={pair}
          pair={pair}
          data={data}
          isActive={pair === activePair}
        />
      ))}
      {otherEntry && (
        <>
          <div className="mx-1 my-1 border-t border-(--color-glass-border)" />
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
    <div className="mt-3 grid grid-cols-3 gap-2 border-t border-(--color-glass-border) pt-3">
      {items.map(({ label, value }) => (
        <div key={label} className="flex flex-col items-center gap-0.5">
          <span className="text-[9px] font-medium tracking-[0.1em] text-(--color-text-dim) uppercase">
            {label}
          </span>
          <span className="font-mono text-sm font-semibold text-(--color-text-primary) tabular-nums">
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Article row — clickable title, expandable body, impact tag, favicon */
function ArticleRow({ article }: { article: ArticleItem }) {
  const [expanded, setExpanded] = useState(false);
  const score = article.sentiment_score;
  const isBullish = score > SENTIMENT_THRESHOLDS.BULLISH;
  const isBearish = score < SENTIMENT_THRESHOLDS.BEARISH;
  const borderColor = isBullish
    ? "var(--color-accent-success)"
    : isBearish
      ? "var(--color-accent-danger)"
      : "var(--color-glass-border)";
  const hasBody = !!(article.body || article.summary);
  const bodyText = article.body || article.summary || "";
  const impact = getImpactLabel(score);

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

  const handleClick = () => {
    if (hasBody) setExpanded(!expanded);
  };

  return (
    <div
      className="flex shrink-0 cursor-pointer flex-col border-l-2 transition-colors hover:bg-(--color-glass-hover)"
      style={{ borderLeftColor: borderColor }}
      onClick={handleClick}
    >
      <div className="flex h-8 items-center justify-between gap-2 py-3 pr-2 pl-2">
        <FaviconImg source={article.source} />
        <a
          href={article.url || "#"}
          target="_blank"
          rel="noopener noreferrer"
          className="min-w-0 flex-1 truncate text-[10px] leading-[1.3] text-(--color-text-primary) transition-colors hover:text-(--color-brand) hover:underline"
          onClick={(e) => e.stopPropagation()}
          title={article.title}
        >
          {article.title}
        </a>
        <span className="shrink-0 font-mono text-[9px] whitespace-nowrap text-(--color-text-muted) uppercase tabular-nums">
          {article.source}
        </span>
        <span className={cn("shrink-0 rounded px-1.5 py-0.5 text-[8px] font-semibold tabular-nums", impact.className)}>
          {impact.label}
        </span>
        <span className="shrink-0 font-mono text-[9px] whitespace-nowrap text-(--color-text-dim) tabular-nums">
          {timeStr}
        </span>
      </div>
      {expanded && hasBody && (
        <p className="px-2 pb-2 text-[10px] leading-relaxed text-(--color-text-dim)">
          {bodyText.length > 250 ? bodyText.slice(0, 247) + "..." : bodyText}
        </p>
      )}
    </div>
  );
}

/** Sentiment + bias only — used in Col 2 */
export function MarketPulsePanel({ pair = "EURUSD" }: { pair?: string }) {
  const activePair = useDashboardStore((s) => s.activePair);
  const demoMode = useAppStore((s) => s.demoMode);
  const displayPair = pair ?? activePair;
  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(displayPair, !demoMode);

  const pairData = sentiment?.pairs?.[displayPair] ?? sentiment?.pairs?.[displayPair.toUpperCase()];
  const recommendedPosition = pairData?.recommended_position ?? 0;
  const articleCount = pairData?.article_count ?? 0;
  const cacheAge = pairData?.cache_age_hours;
  const positionConfidence = pairData?.position_confidence ?? null;
  const isError = sentiment?.status === "error";
  const isNoData = sentiment?.status === "no_articles" || (articleCount === 0 && !sentLoading);

  if (sentLoading) {
    return (
      <div className="flex animate-pulse flex-col gap-3">
        <div className="h-8 rounded bg-(--color-glass-hover)" />
        <div className="h-16 rounded bg-(--color-glass-hover)" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex flex-col items-center gap-2 py-4">
        <span className="text-[10px] text-(--color-accent-danger)">Sentiment unavailable</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <SentimentOverview sentiment={sentiment} activePair={displayPair} />

      <div className="border-t border-(--color-glass-border) pt-2">
        <div className="mb-2 flex items-center justify-between">
          <span className="font-mono text-[10px] font-semibold text-(--color-text-secondary)">
            {displayPair}
          </span>
          {pairData?.currencies_affected && pairData.currencies_affected.length > 0 && (
            <span className="text-[8px] text-(--color-text-dim)">
              affects: {pairData.currencies_affected.join(", ")}
            </span>
          )}
        </div>

        {isNoData ? (
          <div className="flex flex-col items-center gap-1 py-3">
            <span className="text-[10px] text-(--color-text-muted)">No news data available</span>
            <span className="text-[9px] text-(--color-text-muted) opacity-60">
              RSS feeds will populate over time
            </span>
          </div>
        ) : (
          <>
            <BullBearBar
              position={recommendedPosition}
              articleCount={articleCount}
              confidence={positionConfidence}
              vaderContribution={pairData?.vader_contribution ?? null}
              llmContribution={pairData?.llm_contribution ?? null}
            />
            <SentimentScores
              llm={pairData?.llm_sentiment ?? null}
              vader={pairData?.vader_sentiment ?? null}
              confidence={pairData?.llm_confidence ?? null}
            />
            <div className="mt-2 flex items-center justify-between">
              <span className="font-mono text-[9px] text-(--color-text-muted) tabular-nums">
                {articleCount} articles
                {articleCount > 0 && articleCount < 3 && (
                  <span className="ml-1 text-(--color-accent-warning)">— limited</span>
                )}
              </span>
              {cacheAge != null && (
                <span className="text-[9px] text-(--color-text-muted)">
                  {formatStaleness(cacheAge)}
                </span>
              )}
            </div>
          </>
        )}
        {sentiment?.backend === "vader" && !sentiment?.llm_available && !isNoData && (
          <span className="mt-2 block text-[9px] text-(--color-accent-warning)">
            LLM unavailable — using VADER fallback.
          </span>
        )}
      </div>
    </div>
  );
}

/** News articles only — used in Col 3 */
export function NewsArticlesPanel({ pair = "EURUSD" }: { pair?: string }) {
  const activePair = useDashboardStore((s) => s.activePair);
  const demoMode = useAppStore((s) => s.demoMode);
  const displayPair = pair ?? activePair;
  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(displayPair, !demoMode);
  const { data: newsStatus } = useNewsStatus();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({ exact: true });

  const articles = (sentiment?.top_articles ?? []) as (ArticleItem & { relevance_tier?: number })[];
  const tierCounts = sentiment?.article_count_by_tier;

  const tiers = [
    {
      key: "exact" as const,
      label: `${displayPair} articles`,
      filter: (a: (typeof articles)[number]) => a.relevance_tier === 1,
    },
    {
      key: "partial" as const,
      label: "Related currency news",
      filter: (a: (typeof articles)[number]) => a.relevance_tier === 2,
    },
    {
      key: "other" as const,
      label: "Other / untagged",
      filter: (a: (typeof articles)[number]) => a.relevance_tier === 0,
    },
  ];

  const toggle = (key: string) => setExpanded((p) => ({ ...p, [key]: !p[key] }));

  if (sentLoading) {
    return (
      <div className="flex animate-pulse flex-col gap-2">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i} className="h-8 rounded bg-(--color-glass-hover)" />
        ))}
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-0">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[10px] font-medium tracking-[0.1em] text-(--color-text-muted) uppercase">
          Top Articles
        </span>
        <span className="font-mono text-[9px] text-(--color-text-muted)">
          {newsStatus?.cached_articles ?? 0} cached
        </span>
      </div>
      {articles.length > 0 ? (
        <div className="flex max-h-[480px] flex-col gap-1 overflow-y-auto">
          {tiers.map(({ key, label, filter }) => {
            const tierArticles = articles.filter(filter);
            if (tierArticles.length === 0) return null;
            const count = tierCounts?.[key] ?? tierArticles.length;
            const isExpanded = expanded[key];
            return (
              <div key={key}>
                <button
                  onClick={() => toggle(key)}
                  className="flex w-full items-center gap-1 py-1 text-left text-(--color-text-muted) hover:text-(--color-text-secondary)"
                >
                  {isExpanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                  <span className="text-[9px] font-medium tracking-[0.06em] uppercase">
                    {label}
                  </span>
                  <span className="font-mono text-[8px] text-(--color-text-dim) tabular-nums">
                    ({count})
                  </span>
                </button>
                {isExpanded &&
                  tierArticles.map((a, i) => <ArticleRow key={`${a.title}-${i}`} article={a} />)}
              </div>
            );
          })}
        </div>
      ) : (
        <span className="text-[10px] text-(--color-text-muted)">
          No articles available. RSS feeds will populate over time.
        </span>
      )}
    </div>
  );
}
