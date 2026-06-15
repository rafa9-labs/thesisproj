import { useState, useMemo, useCallback } from "react";
import {
  useLiveSentiment,
  useNewsArticles,
  useNewsEvents,
  useNewsStatus,
  usePairs,
} from "@/api/queries";
import { useNewsWebSocket } from "@/hooks/useNewsWebSocket";
import type { NewsArticleFull, LiveSentimentArticle } from "@/api/schemas";
import { BullBearBar } from "@/components/shared/BullBearBar";
import {
  TrendingUp,
  Activity,
  BarChart3,
  ExternalLink,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

const PAIRS_FALLBACK = [
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "AUDUSD",
  "USDCAD",
  "NZDUSD",
  "USDCHF",
] as const;

function formatRelativeTime(timestamp: string) {
  const diff = Date.now() - new Date(timestamp).getTime();
  const mins = Math.floor(diff / 60000);
  const hours = Math.floor(mins / 60);
  const days = Math.floor(hours / 24);
  if (days > 0) return `${days}d`;
  if (hours > 0) return `${hours}h`;
  if (mins > 0) return `${mins}m`;
  return "now";
}

function formatTimestamp(timestamp: string) {
  try {
    const d = new Date(timestamp);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "--:--";
  }
}

function truncateAtSentence(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  const cut = text.slice(0, maxLen);
  const lastPeriod = cut.lastIndexOf(".");
  if (lastPeriod > maxLen * 0.6) return text.slice(0, lastPeriod + 1);
  const lastSpace = cut.lastIndexOf(" ");
  if (lastSpace > maxLen * 0.6) return cut.slice(0, lastSpace) + "\u2026";
  return cut + "\u2026";
}

/* ── Scored Article Row (Left Panel) ── */

function ScoredArticleRow({ article }: { article: LiveSentimentArticle }) {
  const [expanded, setExpanded] = useState(false);
  const score = article.sentiment_score;
  const isBullish = score > 0.05;
  const isBearish = score < -0.05;
  const borderColor = isBullish
    ? "var(--color-accent-success)"
    : isBearish
      ? "var(--color-accent-danger)"
      : "var(--color-glass-border)";
  const hasBody = !!(article.body || article.summary);
  const bodyText = article.body || article.summary || "";
  const hlBody = article.highlighted_body || null;
  const llmSentiment = article.llm_sentiment ?? null;
  const llmConfidence = article.llm_confidence ?? null;

  return (
    <div
      className="flex flex-col border-l-2 transition-colors hover:bg-(--color-glass-hover)"
      style={{ borderLeftColor: borderColor }}
    >
      <div className="flex items-start gap-3 px-3 py-2.5">
        {/* Content */}
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <a
            href={article.url || "#"}
            target="_blank"
            rel="noopener noreferrer"
            className="line-clamp-2 text-[11px] leading-snug font-medium text-(--color-text-primary) transition-colors hover:text-(--color-brand) hover:underline"
            title={article.url ? "Open source" : undefined}
            onClick={(e) => e.stopPropagation()}
          >
            {article.title}
          </a>

          {!expanded && hasBody && (
            <p className="line-clamp-2 text-[10px] leading-relaxed text-(--color-text-dim)">
              {bodyText}
            </p>
          )}

          <div className="flex items-center gap-3">
            {llmSentiment != null && (
              <span
                className="font-mono text-[10px] font-semibold tabular-nums"
                style={{
                  color:
                    llmSentiment >= 0
                      ? "var(--color-accent-success)"
                      : "var(--color-accent-danger)",
                }}
              >
                LLM: {llmSentiment > 0 ? "+" : ""}
                {llmSentiment.toFixed(2)}
              </span>
            )}
            {llmConfidence != null && (
              <span className="font-mono text-[10px] text-(--color-text-dim) tabular-nums">
                CONF: {(llmConfidence * 100).toFixed(0)}%
              </span>
            )}
            <span className="text-[9px] text-(--color-text-muted)">
              {article.source} &middot; {formatRelativeTime(article.timestamp)}
            </span>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex shrink-0 items-center gap-1">
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex h-6 w-6 items-center justify-center rounded text-slate-500 transition-colors hover:text-cyan-400"
              title="Open article in new tab"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink size={13} />
            </a>
          )}
          {hasBody && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex h-6 w-6 items-center justify-center rounded text-slate-500 transition-colors hover:text-cyan-400"
              title={expanded ? "Collapse" : "Expand"}
            >
              {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
          )}
        </div>
      </div>

      {/* Expanded text with sentiment highlighting */}
      {expanded && hasBody && (
        <div className="px-3 pb-3">
          <div className="mt-2 rounded-md bg-slate-900/60 p-4">
            {hlBody ? (
              <p
                className="text-[11px] leading-relaxed text-slate-300"
                dangerouslySetInnerHTML={{ __html: hlBody }}
              />
            ) : (
              <p className="text-[11px] leading-relaxed text-slate-300">
                {truncateAtSentence(bodyText, 800)}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Macro Feed Row (Right Panel Top) ── */

function MacroFeedRow({ article }: { article: NewsArticleFull }) {
  const [expanded, setExpanded] = useState(false);
  const hasBody = !!(article.body || article.summary);
  const bodyText = article.body || article.summary || "";
  const hlBody = article.highlighted_body || null;

  return (
    <div className="flex flex-col border-b border-slate-800 transition-colors hover:bg-slate-900/30">
      <div className="flex items-center gap-2 px-3 py-2">
        <span className="shrink-0 font-mono text-[9px] text-slate-500 tabular-nums">
          {formatTimestamp(article.timestamp)}
        </span>
        <a
          href={article.url || "#"}
          target="_blank"
          rel="noopener noreferrer"
          className="min-w-0 flex-1 truncate text-[11px] leading-snug text-slate-300 transition-colors hover:text-cyan-400 hover:underline"
          onClick={(e) => e.stopPropagation()}
        >
          {article.title}
        </a>
        <span className="shrink-0 rounded border border-slate-700 px-1.5 py-0.5 text-[8px] font-medium text-slate-500 uppercase">
          {article.source}
        </span>
        <div className="flex shrink-0 items-center gap-0.5">
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex h-5 w-5 items-center justify-center rounded text-slate-500 hover:text-cyan-400"
              title="Open article"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink size={11} />
            </a>
          )}
          {hasBody && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex h-5 w-5 items-center justify-center rounded text-slate-500 hover:text-cyan-400"
              title={expanded ? "Collapse" : "Expand"}
            >
              {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
          )}
        </div>
      </div>
      {expanded && hasBody && (
        <div className="px-3 pb-2">
          <div className="rounded-md bg-slate-900/60 p-3">
            {hlBody ? (
              <p
                className="text-[10px] leading-relaxed text-slate-300"
                dangerouslySetInnerHTML={{ __html: hlBody }}
              />
            ) : (
              <p className="text-[10px] leading-relaxed text-slate-400">
                {truncateAtSentence(bodyText, 600)}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Economic Calendar Widget (Right Panel Bottom) ── */

function EventIndicator({ impact, label }: { impact: string; label: string }) {
  const colorMap: Record<string, string> = {
    high: "var(--color-event-high)",
    medium: "var(--color-event-medium)",
    low: "var(--color-event-low)",
  };
  return (
    <span className="flex items-center gap-1.5 text-[10px] text-slate-300">
      <span
        className="inline-block h-2 w-2 shrink-0 rounded-full"
        style={{ backgroundColor: colorMap[impact] ?? "var(--color-text-muted)" }}
      />
      {label}
    </span>
  );
}

function EconomicCalendarWidget({
  eventsData,
}: {
  eventsData: { time: number; event: string; currency: string; impact: string }[] | undefined;
}) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-2">
      <div className="mb-1.5 text-[9px] font-medium tracking-[0.08em] text-slate-500 uppercase">
        Upcoming ({eventsData?.length ?? 0} events)
      </div>
      {eventsData && eventsData.length > 0 ? (
        eventsData.slice(0, 5).map((ev, i) => (
          <div
            key={i}
            className="flex items-center gap-2 border-b border-slate-800 py-1.5 transition-colors hover:bg-slate-900/30"
          >
            <span className="shrink-0 font-mono text-[9px] text-slate-500 tabular-nums">
              {new Date(ev.time * 1000).toLocaleDateString([], {
                month: "short",
                day: "numeric",
              })}
            </span>
            <EventIndicator impact={ev.impact} label={ev.event} />
            <span className="ml-auto shrink-0 font-mono text-[9px] text-slate-500 uppercase">
              {ev.currency}
            </span>
          </div>
        ))
      ) : (
        <div className="py-4 text-center text-[10px] text-slate-500">
          No upcoming economic events
        </div>
      )}
    </div>
  );
}

/* ── Main Page ── */

export function NewsPage() {
  const [pair, setPair] = useState("EURUSD");

  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(pair);
  const { data: articlesData, isLoading: articlesLoading } = useNewsArticles(undefined, 30);
  const { data: newsStatus } = useNewsStatus();
  const { data: apiPairs } = usePairs();
  useNewsWebSocket(pair);

  const availablePairs = useMemo(() => {
    const symbols = (apiPairs ?? []).map((p) => p.pair?.symbol ?? "").filter((s) => s !== "");
    return symbols.length > 0 ? symbols.slice(0, 12) : [...PAIRS_FALLBACK];
  }, [apiPairs]);

  const [now] = useState(() => Math.floor(Date.now() / 1000));
  const futureEnd = now + 90 * 86400;
  const { data: eventsData } = useNewsEvents(now, futureEnd, "high,medium");

  const pairData = sentiment?.pairs?.[pair];
  const recommendedPosition = pairData?.recommended_position ?? 0;
  const articleCount = pairData?.article_count ?? 0;
  const llmAvailable = sentiment?.llm_available ?? false;

  /* Deduplicate scored articles by URL */
  const scoredArticles = useMemo(() => {
    const seen = new Set<string>();
    const filtered: LiveSentimentArticle[] = [];
    for (const a of sentiment?.top_articles ?? []) {
      const tags = (a.pair_tags ?? []).map((t: string) => t.toUpperCase());
      if (!tags.includes(pair.toUpperCase())) continue;
      const urlKey = (a.url || "").trim().toLowerCase();
      if (urlKey && seen.has(urlKey)) continue;
      if (urlKey) seen.add(urlKey);
      filtered.push(a);
    }
    return filtered.slice(0, 15);
  }, [sentiment?.top_articles, pair]);

  /* All articles deduped by URL */
  const allArticles = useMemo(() => {
    const seen = new Set<string>();
    return (articlesData?.articles ?? []).filter((a) => {
      const urlKey = (a.url || "").trim().toLowerCase();
      if (urlKey && seen.has(urlKey)) return false;
      if (urlKey) seen.add(urlKey);
      return true;
    });
  }, [articlesData]);

  const pairUpper = pair.toUpperCase();
  const baseCurrency = pairUpper.slice(0, 3);
  const quoteCurrency = pairUpper.slice(3);

  /* Global Macro articles: exclude pair-specific ones */
  const globalArticles = useMemo(
    () =>
      allArticles.filter((a) => {
        const tags = a.pair_tags.map((t) => t.toUpperCase());
        return (
          !tags.includes(pairUpper) && !tags.includes(baseCurrency) && !tags.includes(quoteCurrency)
        );
      }),
    [allArticles, pairUpper, baseCurrency, quoteCurrency],
  );

  const handlePairChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setPair(e.target.value);
  }, []);

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* ── Trading Context Band ── */}
      <div className="flex shrink-0 items-center gap-4 border-b border-(--color-glass-border) bg-(--color-glass) px-6 py-2">
        <div className="flex items-center gap-2">
          <Activity size={12} className="text-(--color-accent-success)" />
          <span className="text-[10px] font-medium text-(--color-text-primary)">{pair}</span>
          <span
            className="font-mono text-[10px] tabular-nums"
            style={{
              color:
                recommendedPosition >= 0.05
                  ? "var(--color-accent-success)"
                  : recommendedPosition <= -0.05
                    ? "var(--color-accent-danger)"
                    : "var(--color-text-muted)",
            }}
          >
            {recommendedPosition > 0 ? "+" : ""}
            {recommendedPosition.toFixed(2)}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-(--color-text-muted)">
          <span className="font-mono">{articleCount} articles</span>
          <span className="opacity-30">|</span>
          <span>{availablePairs.length} pairs tracked</span>
          <span className="opacity-30">|</span>
          {llmAvailable ? (
            <span className="text-(--color-accent-success)">LLM scoring active</span>
          ) : (
            <span className="text-(--color-accent-warning)">LLM offline</span>
          )}
        </div>
        <div className="ml-auto flex items-center gap-2 text-[10px]">
          <BarChart3 size={11} className="text-(--color-text-dim)" />
          <span className="text-(--color-text-dim)">
            News &rarr; Features &rarr; ML Model &rarr; Signals
          </span>
        </div>
      </div>

      {/* ── Main Content Grid ── */}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="grid flex-1 grid-cols-1 overflow-hidden p-6 xl:grid-cols-12 xl:gap-8">
          {/* ── COL A: Scored Pair Feed (span-6) ── */}
          <div className="flex flex-col overflow-hidden rounded-lg border border-(--color-glass-border) bg-(--color-glass) xl:col-span-6">
            <div className="shrink-0 border-b border-(--color-glass-border) px-4 py-3">
              <div className="flex items-center gap-3">
                <select
                  value={pair}
                  onChange={handlePairChange}
                  aria-label="Select news pair"
                  className="h-7 shrink-0 rounded border border-(--color-glass-border) bg-(--color-elevated) px-2.5 font-mono text-[11px] text-(--color-text-primary) transition focus:outline-none"
                >
                  {availablePairs.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
                <span className="font-mono text-[10px] text-(--color-text-muted) tabular-nums">
                  {articleCount} articles
                </span>
                <span
                  className="ml-auto font-mono text-[12px] font-semibold tabular-nums"
                  style={{
                    color:
                      recommendedPosition >= 0.05
                        ? "var(--color-accent-success)"
                        : recommendedPosition <= -0.05
                          ? "var(--color-accent-danger)"
                          : "var(--color-text-muted)",
                  }}
                >
                  {recommendedPosition > 0 ? "+" : ""}
                  {recommendedPosition.toFixed(2)}{" "}
                  {recommendedPosition >= 0.05
                    ? "Bullish"
                    : recommendedPosition <= -0.05
                      ? "Bearish"
                      : "Neutral"}
                </span>
              </div>
              <div className="mt-2">
                <BullBearBar
                  position={recommendedPosition}
                  articleCount={articleCount}
                  confidence={pairData?.position_confidence ?? null}
                />
              </div>
            </div>

            <div className="flex-1 overflow-y-auto [scrollbar-width:thin]">
              {sentLoading ? (
                <div className="flex flex-col gap-1">
                  {Array.from({ length: 5 }, (_, i) => (
                    <div
                      key={i}
                      className="mx-3 my-1 h-14 animate-skeleton rounded bg-(--color-glass-hover)"
                    />
                  ))}
                </div>
              ) : scoredArticles.length > 0 ? (
                scoredArticles.map((a, i) => (
                  <ScoredArticleRow key={a.url || `${a.title}-${i}`} article={a} />
                ))
              ) : (
                <div className="flex flex-col items-center gap-2 py-12">
                  <TrendingUp
                    size={28}
                    strokeWidth={1}
                    className="text-(--color-text-muted) opacity-40"
                  />
                  <span className="text-[11px] text-(--color-text-muted)">
                    No scored articles for {pair}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* ── COL B: Global Macro + Economic Calendar (span-6) ── */}
          <div className="flex flex-col overflow-hidden rounded-lg border border-slate-800 bg-slate-900/40 xl:col-span-6">
            <div className="flex h-full flex-col gap-0 divide-y divide-slate-800">
              {/* Top 70%: Global Macro Feed */}
              <div className="flex flex-col" style={{ flex: "70 0 0px" }}>
                <div className="shrink-0 border-b border-slate-800 px-4 py-2.5">
                  <span className="text-[10px] font-semibold tracking-[0.08em] text-slate-300 uppercase">
                    Global Macro
                  </span>
                  <span className="ml-2 font-mono text-[9px] text-slate-500">
                    {globalArticles.length} articles
                  </span>
                </div>
                <div className="flex-1 overflow-y-auto [scrollbar-width:thin]">
                  {articlesLoading ? (
                    <div className="flex flex-col gap-0">
                      {Array.from({ length: 8 }, (_, i) => (
                        <div
                          key={i}
                          className="mx-3 my-1 h-9 animate-skeleton rounded bg-slate-800"
                        />
                      ))}
                    </div>
                  ) : globalArticles.length > 0 ? (
                    globalArticles.map((a, i) => (
                      <MacroFeedRow key={a.url || `${a.title}-${a.source}-${i}`} article={a} />
                    ))
                  ) : (
                    <div className="flex flex-col items-center gap-2 py-12">
                      <TrendingUp size={28} strokeWidth={1} className="text-slate-600" />
                      <span className="text-[11px] text-slate-500">No global macro articles</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Bottom 30%: Economic Calendar */}
              <div className="flex flex-col" style={{ flex: "30 0 0px" }}>
                <div className="shrink-0 bg-slate-900/60 px-4 py-2.5">
                  <span className="text-[10px] font-semibold tracking-[0.08em] text-slate-300 uppercase">
                    Economic Calendar
                  </span>
                </div>
                <div className="flex-1 overflow-y-auto [scrollbar-width:thin]">
                  <EconomicCalendarWidget eventsData={eventsData} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Status Footer ── */}
      {newsStatus && (
        <div className="flex h-8 shrink-0 items-center gap-4 border-t border-(--color-glass-border) bg-(--color-app) px-6 text-[10px] text-(--color-text-muted)">
          <span>Sentiment: {newsStatus.sentiment_backend.toUpperCase()}</span>
          <span className="font-mono">{newsStatus.cached_articles} cached</span>
          <span>finBERT: {newsStatus.finbert_available ? "Ready" : "N/A"}</span>
          {sentiment && !llmAvailable && (
            <span className="text-(--color-accent-warning)">
              LLM unavailable — using VADER fallback
            </span>
          )}
          {llmAvailable && <span className="text-(--color-accent-success)">LLM active</span>}
        </div>
      )}
    </div>
  );
}
