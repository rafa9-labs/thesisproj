import { useState, useMemo } from "react";
import {
  useLiveSentiment,
  useNewsArticles,
  useNewsEvents,
  useNewsStatus,
  usePairs,
} from "@/api/queries";
import type { NewsArticleFull, LiveSentimentArticle } from "@/api/schemas";
import { BullBearBar } from "@/components/shared/BullBearBar";
import { SENTIMENT_THRESHOLDS } from "@/lib/sentiment-thresholds";
import {
  TrendingUp,
  Activity,
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

/* ── Helpers ── */

/* ── Title/Body Deduplication ── */

function stripTitleFromBody(title: string, body: string): string {
  if (!title || !body) return body;
  const normalizedTitle = title.trim().replace(/\.+$/, "");
  const normalizedBodyStart = body.trim().slice(0, normalizedTitle.length);
  if (normalizedBodyStart.toLowerCase() === normalizedTitle.toLowerCase()) {
    const remainder = body.trim().slice(normalizedTitle.length);
    return remainder.replace(/^[\s\.,;:!?]+/, "").trim() || body;
  }
  return body;
}

/* ── Contextual Phrase Highlighting ── */

// ── Multi-word phrase patterns (matched as complete context blocks) ──
const phrasePatterns = [
  // Bullish phrases
  { regex: /(?:dollar|euro|yen|pound|franc|loonie|aussie|kiwi)\s+(?:rall(?:y|ied|ies)|surge[ds]?|soar[eds]?|climb[edins]*|jump[eds]*|gain[eds]*|strengthen[eds]*|advance[ds]?|rise[s]?|rose)\s*\w*/gi, bullish: true },
  { regex: /(?:sharply|significantly|strongly|firmly)\s+(?:higher|rall(?:y|ied)|surge[ds]?|gain[eds]*|rose?)/gi, bullish: true },
  { regex: /\b(?:strong|solid|robust|resilient)\s+(?:growth|demand|recovery|data|jobs|employment|GDP|earnings)\b/gi, bullish: true },
  { regex: /\b(?:beat|exceeded|topped|outpaced)\s+(?:expectations|forecasts?|estimates?)\b/gi, bullish: true },
  { regex: /\b(?:upward|positive|bullish|optimistic)\s+(?:trend|momentum|outlook|sentiment|bias)\b/gi, bullish: true },
  { regex: /\b(?:dovish|easing|accommodative)\s+(?:tone|stance|bias|outlook|policy)\b/gi, bullish: true },
  { regex: /\b(?:record|all-time|multi-year|fresh)\s+highs?\b/gi, bullish: true },
  { regex: /\b(?:robust|rosy|upbeat)\s+(?:outlook|forecast|guidance)\b/gi, bullish: true },
  // ── Single-word bullish
  { regex: /\brall(?:y|ied|ies)\b/gi, bullish: true },
  { regex: /\bsurge[ds]?\b(?!\s*(?:protect|risk))?/gi, bullish: true },
  { regex: /\bsoar[eds]?\b/gi, bullish: true },
  { regex: /\brebound[sedin]*\b/gi, bullish: true },
  { regex: /\bstrengthen[eds]*\b/gi, bullish: true },
  { regex: /\boutperform[sedin]*\b/gi, bullish: true },
  { regex: /\bbullish\b/gi, bullish: true },
  { regex: /\boptimis(?:m|tic)\b/gi, bullish: true },
  { regex: /\b(?:gains?|upside)\b/gi, bullish: true },
  // Bearish phrases
  { regex: /(?:dollar|euro|yen|pound|franc|loonie|aussie|kiwi)\s+(?:plung[eds]?|tumble[ds]?|slump[sed]*|decline[ds]?|drop[ped]*|fall[sen]*|weaken[eds]*|sli[dp]\s*\w*)/gi, bullish: false },
  { regex: /(?:sharply|significantly|steeply|heavily)\s+(?:lower|fell|dropped|declined|plunged)/gi, bullish: false },
  { regex: /\b(?:weak|sluggish|tepid|soft|dismal)\s+(?:growth|demand|recovery|data|jobs|employment|GDP|earnings)\b/gi, bullish: false },
  { regex: /\b(?:missed|fell short of|trailed|lagged)\s+(?:expectations|forecasts?|estimates?)\b/gi, bullish: false },
  { regex: /\b(?:downward|negative|bearish|pessimistic)\s+(?:trend|momentum|outlook|sentiment|bias)\b/gi, bullish: false },
  { regex: /\b(?:hawkish|tightening|aggressive)\s+(?:tone|stance|bias|outlook|policy|rate hike)\b/gi, bullish: false },
  { regex: /\b(?:risk-off|risk off|sell-off|selloff)\s*(?:sentiment|mode|environment)?\b/gi, bullish: false },
  { regex: /\b(?:recession|downturn|contraction|stagflation)\s+(?:fears?|worries?|risks?|concerns?)\b/gi, bullish: false },
  // ── Single-word bearish
  { regex: /\bplung[eds]?\b/gi, bullish: false },
  { regex: /\btumble[ds]?\b/gi, bullish: false },
  { regex: /\bslump[se]?d?\b/gi, bullish: false },
  { regex: /\bdecline[ds]?\b/gi, bullish: false },
  { regex: /\bdrop[ped]*\b/gi, bullish: false },
  { regex: /\b(sell-off|selloff)\b/gi, bullish: false },
  { regex: /\bdownturn\b/gi, bullish: false },
  { regex: /\brecession\b/gi, bullish: false },
  { regex: /\bweaken[eds]*\b/gi, bullish: false },
  { regex: /\bdeteriorat[edsin]*\b/gi, bullish: false },
  { regex: /\bunderperform[edsin]*\b/gi, bullish: false },
  { regex: /\b(?:downside|bearish|pessimis(?:m|tic))\b/gi, bullish: false },
  { regex: /\b(?:losses?|turmoil|crisis|volatil(?:e|ity))\b/gi, bullish: false },
];

interface PhraseMatch {
  start: number;
  end: number;
  bullish: boolean;
  text: string;
}

function findPhraseMatches(text: string): PhraseMatch[] {
  const matches: PhraseMatch[] = [];

  for (const { regex, bullish } of phrasePatterns) {
    // Reset regex for global matching
    regex.lastIndex = 0;
    let m;
    while ((m = regex.exec(text)) !== null) {
      if (m[0].length === 0) break;
      matches.push({ start: m.index, end: m.index + m[0].length, bullish, text: m[0] });
    }
  }

  return matches.sort((a, b) => a.start - b.start);
}

/** Merge overlapping matches into unified spans. */
function mergeOverlapping(matches: PhraseMatch[]): PhraseMatch[] {
  if (matches.length <= 1) return matches;
  const result: PhraseMatch[] = [matches[0]];
  for (let i = 1; i < matches.length; i++) {
    const prev = result[result.length - 1];
    if (matches[i].start <= prev.end) {
      const longer = matches[i].end > prev.end
        ? { ...matches[i], start: prev.start }
        : prev;
      result[result.length - 1] = longer;
    } else {
      result.push(matches[i]);
    }
  }
  return result;
}

/** Merge nearby (<= 40 chars apart) matches into phrase blocks for unified highlighting. */
function mergeNearby(matches: PhraseMatch[], sourceText: string, maxGap: number = 40): PhraseMatch[] {
  if (matches.length <= 1) return matches;

  // First pass: merge overlapping
  const merged = mergeOverlapping(matches);

  // Second pass: merge adjacent matches within maxGap
  const blocks: PhraseMatch[] = [merged[0]];
  for (let i = 1; i < merged.length; i++) {
    const prev = blocks[blocks.length - 1];
    const gap = merged[i].start - prev.end;
    if (gap <= maxGap && merged[i].bullish === prev.bullish) {
      blocks[blocks.length - 1] = {
        start: prev.start,
        end: merged[i].end,
        bullish: prev.bullish,
        text: sourceText.slice(prev.start, merged[i].end),
      };
    } else {
      blocks.push(merged[i]);
    }
  }
  return blocks;
}

function highlightPhrases(text: string): string {
  const matches = findPhraseMatches(text);
  if (matches.length === 0) return escapeHtml(text);

  const blocks = mergeNearby(matches, text, 40);
  const parts: string[] = [];
  let cursor = 0;

  for (const m of blocks) {
    if (m.start > cursor) {
      parts.push(escapeHtml(text.slice(cursor, m.start)));
    }
    const spanClass = m.bullish
      ? "text-emerald-400 font-medium"
      : "text-rose-400 font-medium";
    parts.push(`<span class="${spanClass}">${escapeHtml(m.text)}</span>`);
    cursor = m.end;
  }

  if (cursor < text.length) {
    parts.push(escapeHtml(text.slice(cursor)));
  }

  return parts.join("");
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/* ── Universal body parser ── */

/** Strip all HTML tags from text, returning clean plaintext. */
function stripHtml(text: string): string {
  return text.replace(/<[^>]*>?/gm, "");
}

/** Split raw article text into clean paragraph segments. */
function formatNewsBody(text: string): string[] {
  return stripHtml(text)
    .split(/\n+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 1);
}

/** Render raw article text as properly spaced, highlighted paragraphs. */
function NewsParagraphs({ text, preHighlighted }: { text: string; preHighlighted?: boolean }) {
  if (preHighlighted) {
    const paragraphs = text.split(/\n+/).filter((s) => s.trim().length > 1);
    if (paragraphs.length === 0) {
      return (
        <p className="text-sm text-slate-400 leading-relaxed italic">
          No body text available.
        </p>
      );
    }
    return (
      <div className="flex flex-col gap-2.5">
        {paragraphs.map((p, i) => (
          <p key={i} className="text-sm text-slate-400 leading-relaxed" dangerouslySetInnerHTML={{ __html: p }} />
        ))}
      </div>
    );
  }

  const paragraphs = formatNewsBody(text);
  if (paragraphs.length === 0) {
    return (
      <p className="text-sm text-slate-400 leading-relaxed italic">
        No body text available.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-4 text-sm text-slate-400 leading-relaxed">
      {paragraphs.map((para, i) => (
        <p
          key={i}
          dangerouslySetInnerHTML={{ __html: highlightPhrases(para) }}
        />
      ))}
    </div>
  );
}

/* ── Common Sentiment Badge ── */

function sentimentLabel(score: number): string {
  if (score > SENTIMENT_THRESHOLDS.BADGE_BULLISH) return "Bullish";
  if (score < SENTIMENT_THRESHOLDS.BADGE_BEARISH) return "Bearish";
  return "Neutral";
}

function SentimentBadge({ score }: { score: number }) {
  const label = sentimentLabel(score);
  const colorClass =
    score > SENTIMENT_THRESHOLDS.BADGE_BULLISH
      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
      : score < SENTIMENT_THRESHOLDS.BADGE_BEARISH
        ? "bg-rose-500/20 text-rose-400 border border-rose-500/50"
        : "bg-slate-800 text-slate-400 border border-slate-700";
  return (
    <span
      className={`shrink-0 self-start rounded-full px-2 py-0.5 font-mono text-[9px] font-semibold tabular-nums leading-none ${colorClass}`}
    >
      {score > 0 ? "+" : ""}{score.toFixed(2)} {label}
    </span>
  );
}

/* ── Scored Article Row (Left Panel) ── */

function ScoredArticleRow({ article }: { article: LiveSentimentArticle }) {
  const [expanded, setExpanded] = useState(false);
  const score = article.sentiment_score;
  const isBullish = score > SENTIMENT_THRESHOLDS.BULLISH;
  const isBearish = score < SENTIMENT_THRESHOLDS.BEARISH;
  const impactScore = article.market_impact_score ?? score;
  const borderColor = isBullish
    ? "var(--color-accent-success)"
    : isBearish
      ? "var(--color-accent-danger)"
      : "var(--color-glass-border)";
  const hasBody = !!(article.body || article.summary);
  const bodyText = article.body || article.summary || "";
  const hlBody = article.highlighted_body || null;

  return (
    <div
      className="flex flex-col border-l-2 transition-colors hover:bg-(--color-glass-hover)"
      style={{ borderLeftColor: borderColor }}
    >
      <div className="flex items-start gap-3 px-3 py-2.5">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex items-start gap-2">
            <a
              href={article.url || "#"}
              target="_blank"
              rel="noopener noreferrer"
              className="min-w-0 flex-1 truncate text-sm font-semibold text-slate-200 transition-colors hover:text-(--color-brand) hover:underline"
              title={article.url ? "Open source" : undefined}
              onClick={(e) => e.stopPropagation()}
            >
              {article.title}
            </a>
            <SentimentBadge score={impactScore} />
          </div>

          {!expanded && hasBody && (
            <p className="line-clamp-2 text-xs text-slate-400 leading-snug">
              {stripTitleFromBody(article.title, bodyText)}
            </p>
          )}

          <div className="flex items-center gap-3">
            {article.llm_sentiment != null && (
              <span
                className="font-mono text-[10px] font-semibold tabular-nums"
                style={{
                  color:
                    article.llm_sentiment >= 0
                      ? "var(--color-accent-success)"
                      : "var(--color-accent-danger)",
                }}
              >
                LLM: {article.llm_sentiment > 0 ? "+" : ""}
                {article.llm_sentiment.toFixed(2)}
              </span>
            )}
            {article.llm_confidence != null && (
              <span className="font-mono text-[10px] text-slate-500 tabular-nums">
                CONF: {(article.llm_confidence * 100).toFixed(0)}%
              </span>
            )}
            <span className="text-xs text-slate-500">
              {article.source} &middot; {formatRelativeTime(article.timestamp)}
            </span>
          </div>
        </div>

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

      {expanded && hasBody && (
        <div className="pl-4 border-l-2 border-slate-700 ml-2 mt-3 max-h-64 overflow-y-auto">
          <NewsParagraphs text={hlBody ?? stripTitleFromBody(article.title, bodyText)} preHighlighted={!!hlBody} />
        </div>
      )}
    </div>
  );
}

/* ── Macro Feed Row (Right Panel) ── */

function MacroFeedRow({ article }: { article: NewsArticleFull }) {
  const [expanded, setExpanded] = useState(false);
  const hasBody = !!(article.body || article.summary);
  const bodyText = article.body || article.summary || "";
  const hlBody = article.highlighted_body || null;
  const score = article.sentiment_score ?? 0;
  const isBullish = score > SENTIMENT_THRESHOLDS.BULLISH;
  const isBearish = score < SENTIMENT_THRESHOLDS.BEARISH;
  const borderColor = isBullish
    ? "var(--color-accent-success)"
    : isBearish
      ? "var(--color-accent-danger)"
      : "var(--color-glass-border)";

  return (
    <div
      className="flex flex-col border-l-2 transition-colors hover:bg-(--color-glass-hover)"
      style={{ borderLeftColor: borderColor }}
    >
      <div className="flex items-start gap-3 px-3 py-2.5">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex items-start gap-2">
            <a
              href={article.url || "#"}
              target="_blank"
              rel="noopener noreferrer"
              className="min-w-0 flex-1 truncate text-sm font-semibold text-slate-200 transition-colors hover:text-(--color-brand) hover:underline"
              onClick={(e) => e.stopPropagation()}
            >
              {article.title}
            </a>
            <SentimentBadge score={score} />
          </div>

          {!expanded && hasBody && (
            <p className="line-clamp-2 text-xs text-slate-400 leading-snug">
              {stripTitleFromBody(article.title, bodyText)}
            </p>
          )}

          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">
              {article.source} &middot; {formatRelativeTime(article.timestamp)}
            </span>
          </div>
        </div>

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

      {expanded && hasBody && (
        <div className="pl-4 border-l-2 border-slate-700 ml-2 mt-3 max-h-64 overflow-y-auto">
          <NewsParagraphs text={hlBody ?? stripTitleFromBody(article.title, bodyText)} preHighlighted={!!hlBody} />
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
  const [econExpanded, setEconExpanded] = useState(false);

  const toggleEconCalendar = () => setEconExpanded((p) => !p);

  const { data: sentiment, isLoading: sentLoading } = useLiveSentiment(pair);
  const { data: articlesData, isLoading: articlesLoading } = useNewsArticles(undefined, 30);
  const { data: newsStatus } = useNewsStatus();
  const { data: apiPairs } = usePairs();

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

  const globalArticles = useMemo(
    () =>
      allArticles.filter((a) => {
        const tags = (a.pair_tags ?? []).map((t) => t.toUpperCase());
        return (
          !tags.includes(pairUpper) && !tags.includes(baseCurrency) && !tags.includes(quoteCurrency)
        );
      }),
    [allArticles, pairUpper, baseCurrency, quoteCurrency],
  );

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* ── Trading Context Band ── */}
      <div className="flex shrink-0 flex-col gap-2 rounded-lg border border-(--color-glass-border) bg-(--color-glass) px-4 py-2.5 mx-6 mt-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-(--color-accent-success)" />
            <span
              className="font-mono text-[11px] font-semibold tabular-nums"
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
            <span className="text-[10px] text-(--color-text-muted)">
              {recommendedPosition >= 0.05
                ? "Bullish"
                : recommendedPosition <= -0.05
                  ? "Bearish"
                  : "Neutral"}
            </span>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-(--color-text-muted)">
            <span className="font-mono">{articleCount} articles</span>
            <span className="opacity-30">|</span>
            <span>{availablePairs.length} pairs</span>
            <span className="opacity-30">|</span>
            {llmAvailable ? (
              <span className="text-(--color-accent-success)">LLM active</span>
            ) : (
              <span className="text-(--color-accent-warning)">LLM offline</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5 overflow-x-auto [scrollbar-width:none]">
          {availablePairs.map((p) => (
            <button
              key={p}
              onClick={() => setPair(p)}
              className={`shrink-0 rounded-md px-2.5 py-1 font-mono text-[10px] font-medium transition-all ${
                p === pair
                  ? "bg-(--color-brand)/15 text-(--color-brand) ring-1 ring-(--color-brand)/30"
                  : "text-(--color-text-muted) hover:bg-(--color-glass-hover) hover:text-(--color-text-secondary)"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* ── Main Content Grid ── */}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="grid flex-1 grid-cols-1 overflow-hidden p-6 lg:grid-cols-2 gap-4">
          {/* ── COL A: Scored Pair Feed ── */}
          <div className="flex flex-col overflow-hidden rounded-lg border border-(--color-glass-border) bg-(--color-glass)">
            <div className="shrink-0 border-b border-(--color-glass-border) px-4 py-3">
              <div className="flex items-center gap-3">
                <span className="text-[11px] font-semibold text-(--color-text-primary)">
                  {pair}
                </span>
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

          {/* ── COL B: Global Macro ── */}
          <div className="flex flex-col overflow-hidden rounded-lg border border-(--color-glass-border) bg-(--color-glass)">
            <div className="shrink-0 border-b border-(--color-glass-border) px-4 py-2.5">
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
                      className="mx-3 my-1 h-9 animate-skeleton rounded bg-(--color-glass-hover)"
                    />
                  ))}
                </div>
              ) : globalArticles.length > 0 ? (
                globalArticles.map((a, i) => (
                  <MacroFeedRow key={a.url || `${a.title}-${a.source}-${i}`} article={a} />
                ))
              ) : (
                <div className="flex flex-col items-center gap-2 py-12">
                  <TrendingUp size={28} strokeWidth={1} className="text-(--color-text-muted) opacity-40" />
                  <span className="text-[11px] text-(--color-text-muted)">No global macro articles</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ── Economic Calendar ── */}
      {eventsData && eventsData.length > 0 && (
        <div className="shrink-0 mx-4 mb-3 rounded-lg border border-(--color-glass-border) bg-(--color-glass) overflow-hidden">
          <button
            type="button"
            onClick={toggleEconCalendar}
            className="flex w-full items-center justify-between px-4 py-2.5 transition-colors hover:bg-(--color-glass-hover)"
          >
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-semibold tracking-[0.08em] text-slate-300 uppercase">
                Economic Calendar
              </span>
              <span className="font-mono text-[10px] text-(--color-text-muted)">
                {eventsData.length} events
              </span>
            </div>
            {econExpanded ? <ChevronUp size={14} className="text-(--color-text-muted)" /> : <ChevronDown size={14} className="text-(--color-text-muted)" />}
          </button>
          {econExpanded && (
            <div className="max-h-64 overflow-y-auto [scrollbar-width:thin] border-t border-(--color-glass-border)">
              <EconomicCalendarWidget eventsData={eventsData} />
            </div>
          )}
        </div>
      )}

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
