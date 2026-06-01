import { useState, useMemo, useCallback } from "react";
import { useLiveSentiment, useNewsArticles, useNewsEvents, useNewsStatus } from "@/api/queries";
import type { NewsArticleFull } from "@/api/schemas";
import { ExternalLink, TrendingUp, ChevronDown, ChevronRight } from "lucide-react";

const PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"] as const;

function BullBearBar({ position }: { position: number }) {
  const clamped = Math.max(-1, Math.min(1, position));
  const pct = ((clamped + 1) / 2) * 100;
  const isLong = clamped > 0;
  const color = isLong ? "var(--color-accent-success)" : clamped < 0 ? "var(--color-accent-danger)" : "var(--color-text-muted)";
  return (
    <div className="flex items-center gap-3 flex-1 min-w-0">
      <span className="text-[9px] uppercase tracking-[0.08em] font-medium shrink-0" style={{ color: "var(--color-accent-danger)" }}>SHORT</span>
      <div className="relative h-1.5 flex-1 rounded-full overflow-hidden" style={{ backgroundColor: "var(--color-glass-hover)" }}>
        <div
          className="absolute top-0 h-full rounded-full transition-all duration-500"
          style={{ left: clamped >= 0 ? "50%" : `${pct}%`, width: `${Math.abs(clamped) * 50}%`, backgroundColor: color }}
        />
        <div className="absolute top-0 h-full w-px" style={{ left: "50%", backgroundColor: "var(--color-text-muted)" }} />
      </div>
      <span className="text-[9px] uppercase tracking-[0.08em] font-medium shrink-0" style={{ color: "var(--color-accent-success)" }}>LONG</span>
    </div>
  );
}


const BIAS_BADGES: Record<string, { label: string; bg: string; color: string }> = {
  long:  { label: "LONG",  bg: "rgba(8,153,129,0.12)",  color: "var(--color-accent-success)" },
  short: { label: "SHORT", bg: "rgba(242,54,69,0.12)",  color: "var(--color-accent-danger)" },
  neutral: { label: "NEUTRAL", bg: "rgba(120,123,134,0.08)", color: "var(--color-text-muted)" },
};

function BiasBadge({ bias }: { bias: string }) {
  const b = BIAS_BADGES[bias] ?? BIAS_BADGES.neutral;
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-px text-[9px] font-bold uppercase tracking-[0.06em] shrink-0"
      style={{ backgroundColor: b.bg, color: b.color, fontFamily: "var(--font-mono)", lineHeight: "14px" }}
    >
      {b.label}
    </span>
  );
}

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

function ArticleRow({ article }: { article: NewsArticleFull }) {
  return (
    <div className="flex items-start gap-3 py-2.5 px-3" style={{ borderBottom: "1px solid var(--color-glass-border)" }}>
      <div className="flex-1 min-w-0 flex flex-col gap-0.5">
        <div className="flex items-center gap-2">
          <BiasBadge bias={article.bias} />
          <span className="text-[11px] font-medium leading-snug truncate" style={{ color: "var(--color-text-primary)" }}>
            {article.title}
          </span>
          {article.url && (
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="shrink-0 opacity-40 hover:opacity-100 transition-opacity ml-auto"
              title={article.url}
            >
              <ExternalLink size={12} style={{ color: "var(--color-accent-info)" }} />
            </a>
          )}
        </div>
        {article.summary && (
          <p className="text-[10px] leading-relaxed line-clamp-2" style={{ color: "var(--color-text-secondary)" }}>
            {article.summary}
          </p>
        )}
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-[9px]" style={{ color: "var(--color-text-muted)" }}>
            {article.source} · {formatRelativeTime(article.timestamp)}
          </span>
        </div>
      </div>
    </div>
  );
}

interface GroupedArticles {
  label: string;
  pair: string;
  articles: NewsArticleFull[];
  avgScore: number;
}

function groupArticles(articles: NewsArticleFull[], pair: string): GroupedArticles[] {
  const groups: Record<string, NewsArticleFull[]> = {};
  for (const a of articles) {
    let bestTag = "";
    if (a.pair_tags.length === 0) {
      bestTag = "Other";
    } else {
      for (const tag of a.pair_tags) {
        if (tag.toUpperCase() === pair.toUpperCase()) { bestTag = tag; break; }
      }
      if (!bestTag) bestTag = a.pair_tags[0];
    }
    const key = bestTag.toUpperCase();
    if (!groups[key]) groups[key] = [];
    groups[key].push(a);
  }

  const result: GroupedArticles[] = [];
  const preferred = PAIRS.map((p) => p.toUpperCase());
  const sortedKeys = Object.keys(groups).sort((a, b) => {
    const ai = preferred.indexOf(a);
    const bi = preferred.indexOf(b);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return a.localeCompare(b);
  });

  for (const key of sortedKeys) {
    const arts = groups[key];
    const avg = arts.reduce((sum, a) => sum + a.sentiment_score, 0) / arts.length;
    result.push({ label: key, pair: key, articles: arts, avgScore: avg });
  }
  return result;
}

function GroupSection({ group }: { group: GroupedArticles }) {
  const [open, setOpen] = useState(true);
  const isBullish = group.avgScore > 0.05;
  const isBearish = group.avgScore < -0.05;
  const aggColor = isBullish ? "var(--color-accent-success)" : isBearish ? "var(--color-accent-danger)" : "var(--color-text-muted)";
  const aggLabel = isBullish ? "Bullish" : isBearish ? "Bearish" : "Neutral";

  return (
    <div className="rounded-lg border overflow-hidden" style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full px-4 py-2.5 text-left transition-colors hover:bg-[var(--color-glass-hover)]"
      >
        {open ? <ChevronDown size={13} style={{ color: "var(--color-text-muted)" }} /> : <ChevronRight size={13} style={{ color: "var(--color-text-muted)" }} />}
        <span className="text-[11px] font-semibold uppercase tracking-[0.06em]" style={{ color: "var(--color-text-primary)" }}>
          {group.label}
        </span>
        <span className="text-[10px] tabular-nums" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
          ({group.articles.length})
        </span>
        <span
          className="ml-auto text-[10px] font-medium tabular-nums"
          style={{ color: aggColor, fontFamily: "var(--font-mono)" }}
        >
          {aggLabel} {group.avgScore > 0 ? "+" : ""}{group.avgScore.toFixed(2)}
        </span>
      </button>
      {open && (
        <div>
          {group.articles.map((a, i) => (
            <ArticleRow key={`${a.title}-${a.source}-${i}`} article={a} />
          ))}
        </div>
      )}
    </div>
  );
}

function EventIndicator({ impact, label }: { impact: string; label: string }) {
  const colorMap: Record<string, string> = {
    high: "var(--color-event-high)",
    medium: "var(--color-event-medium)",
    low: "var(--color-event-low)",
  };
  const labelMap: Record<string, string> = { high: "High", medium: "Med", low: "Low" };
  return (
    <span className="flex items-center gap-1.5 text-[10px] shrink-0" style={{ color: "var(--color-text-secondary)" }}>
      <span className="inline-block w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: colorMap[impact] ?? "var(--color-text-muted)" }} />
      {label} <span style={{ color: "var(--color-text-muted)", fontSize: 9 }}>({labelMap[impact] ?? impact})</span>
    </span>
  );
}

export function NewsPage() {
  const [pair, setPair] = useState("EURUSD");
  const [calendarOpen, setCalendarOpen] = useState(false);

  const { data: sentiment } = useLiveSentiment(pair);
  const { data: articlesData, isLoading: articlesLoading } = useNewsArticles(undefined, 30);
  const { data: newsStatus } = useNewsStatus();

  const [now] = useState(() => Math.floor(Date.now() / 1000));
  const futureEnd = now + 90 * 86400;
  const { data: eventsData } = useNewsEvents(now, futureEnd, "high,medium");

  const pairData = sentiment?.pairs?.[pair];
  const recommendedPosition = pairData?.recommended_position ?? 0;
  const articleCount = pairData?.article_count ?? 0;

  const allArticles = useMemo(() => articlesData?.articles ?? [], [articlesData]);

  const filteredArticles = useMemo(() => {
    if (!pair || pair === "ALL") return allArticles;
    const upperPair = pair.toUpperCase();
    return allArticles.filter((a) => {
      if (a.pair_tags.length === 0) return true;
      const tags = a.pair_tags.map((t) => t.toUpperCase());
      const base = upperPair.slice(0, 3);
      const quote = upperPair.slice(3);
      return tags.includes(upperPair) || tags.includes(base) || tags.includes(quote);
    });
  }, [allArticles, pair]);

  const groups = useMemo(() => groupArticles(filteredArticles, pair), [filteredArticles, pair]);

  const articleCountByPair = useMemo(() => {
    const map: Record<string, number> = {};
    for (const p of PAIRS) {
      const upper = p.toUpperCase();
      map[p] = allArticles.filter((a) => {
        if (a.pair_tags.length === 0) return false;
        const tags = a.pair_tags.map((t) => t.toUpperCase());
        const base = upper.slice(0, 3);
        const quote = upper.slice(3);
        return tags.includes(upper) || tags.includes(base) || tags.includes(quote);
      }).length;
    }
    return map;
  }, [allArticles]);

  const handlePairChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setPair(e.target.value);
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto px-6 pt-6 pb-4">
        <div className="flex flex-col gap-5">
          {/* ── Top Bar ─────────────────────────────────── */}
          <div
            className="rounded-lg border p-4 flex items-center gap-4 flex-wrap"
            style={{
              borderColor: "var(--color-glass-border)",
              backgroundColor: "var(--color-glass)",
              backdropFilter: "blur(12px)",
            }}
          >
            <select
              value={pair}
              onChange={handlePairChange}
              className="rounded border px-2.5 text-[11px] transition focus:outline-none shrink-0"
              style={{
                height: 28,
                borderColor: "var(--color-glass-border)",
                backgroundColor: "var(--color-elevated)",
                color: "var(--color-text-primary)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {PAIRS.map((p) => (
                <option key={p} value={p}>{p} ({articleCountByPair[p] ?? 0})</option>
              ))}
            </select>
            <BullBearBar position={recommendedPosition} />
            <span className="text-[10px] tabular-nums shrink-0" style={{ color: "var(--color-text-muted)", fontFamily: "var(--font-mono)" }}>
              {articleCount} articles
            </span>
          </div>

          {/* ── Economic Calendar ───────────────────────── */}
          <div>
            <button
              onClick={() => setCalendarOpen(!calendarOpen)}
              className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-[0.1em] transition-colors hover:opacity-80"
              style={{ color: "var(--color-accent-warning)" }}
            >
              {calendarOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              Economic Calendar
            </button>
            {calendarOpen && (
              <div
                className="mt-2 rounded-lg border p-3 flex flex-wrap items-center gap-x-6 gap-y-1.5"
                style={{ borderColor: "var(--color-glass-border)", backgroundColor: "var(--color-glass)" }}
              >
                {eventsData && eventsData.length > 0 ? (
                  eventsData.slice(0, 14).map((ev, i) => (
                    <EventIndicator key={i} impact={ev.impact} label={ev.event} />
                  ))
                ) : (
                  <span className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>
                    No upcoming events. Events are preloaded for {new Date().getFullYear()}.
                  </span>
                )}
                <span className="text-[9px] shrink-0" style={{ color: "var(--color-text-muted)", marginLeft: "auto" }}>
                  {eventsData?.length ?? 0} upcoming
                </span>
              </div>
            )}
          </div>

          {/* ── Article Feed ────────────────────────────── */}
          {articlesLoading ? (
            <div className="flex flex-col gap-3">
              {Array.from({ length: 5 }, (_, i) => (
                <div key={i} className="rounded-lg h-16 animate-skeleton" style={{ backgroundColor: "var(--color-glass-hover)" }} />
              ))}
            </div>
          ) : groups.length > 0 ? (
            <div className="flex flex-col gap-3">
              {groups.map((g) => (
                <GroupSection key={g.pair} group={g} />
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 py-16">
              <TrendingUp size={36} strokeWidth={1} style={{ color: "var(--color-text-muted)", opacity: 0.4 }} />
              <span className="text-[12px]" style={{ color: "var(--color-text-muted)" }}>
                No articles found. Try a different pair or run a backtest with news features enabled.
              </span>
            </div>
          )}
        </div>
      </div>

      {/* ── Status Bar ─────────────────────────────────── */}
      {newsStatus && (
        <div
          className="flex items-center gap-4 px-6 text-[10px] shrink-0"
          style={{
            height: 32,
            borderTop: "1px solid var(--color-glass-border)",
            backgroundColor: "var(--color-app)",
            color: "var(--color-text-muted)",
          }}
        >
          <span>Sentiment: {newsStatus.sentiment_backend.toUpperCase()}</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>{newsStatus.cached_articles} cached</span>
          <span>finBERT: {newsStatus.finbert_available ? "Ready" : "N/A"}</span>
        </div>
      )}
    </div>
  );
}
