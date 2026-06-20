import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown } from "lucide-react";
import { useLiveSentiment } from "@/api/queries";
import { useDashboardStore } from "@/stores/useDashboardStore";
import { useAppStore } from "@/stores/useAppStore";
import { BullBearBar } from "@/components/shared/BullBearBar";
import { cn } from "@/lib/utils";
import { SENTIMENT_THRESHOLDS, BIAS_THRESHOLD } from "@/lib/sentiment-thresholds";
import apiClient from "@/api/client";
import type { LiveSentimentPairData, LiveSentimentArticle } from "@/api/schemas";

// ── helpers ──

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

// ── sub-components ──

/** A single pair row in the master list */
function PairRow({
  pair,
  data,
  isExpanded,
  onToggle,
}: {
  pair: string;
  data: LiveSentimentPairData;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const pos = data.recommended_position ?? 0;
  const count = data.article_count ?? 0;
  const scoreText = pos > 0 ? `+${pos.toFixed(2)}` : pos.toFixed(2);
  const scoreColor =
    pos > BIAS_THRESHOLD ? "text-emerald-400" : pos < -BIAS_THRESHOLD ? "text-red-400" : "text-slate-400";

  return (
    <button
      onClick={onToggle}
      className={cn(
        "flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left transition-colors hover:bg-white/5",
        isExpanded && "bg-white/[0.04]",
      )}
    >
      {/* Pair + scores */}
      <div className="flex w-32 shrink-0 items-center gap-2">
        <span className="font-mono text-xs font-semibold tabular-nums text-slate-200">
          {pair}
        </span>
      </div>

      {/* Sentiment bar */}
      <div className="min-w-0 flex-1">
        <BullBearBar position={pos} articleCount={count} compact />
      </div>

      {/* Blended score */}
      <span className={cn("w-14 text-right font-mono text-xs font-semibold tabular-nums", scoreColor)}>
        {scoreText}
      </span>

      {/* Article count */}
      <span
        className={cn(
          "w-8 text-right font-mono text-[10px] tabular-nums",
          count === 0 ? "text-amber-400" : count < 3 ? "text-slate-500" : "text-slate-400",
        )}
      >
        {count}
      </span>

      {/* Chevron */}
      <ChevronDown
        size={14}
        className={cn(
          "shrink-0 text-slate-500 transition-transform duration-200",
          isExpanded && "rotate-180",
        )}
      />
    </button>
  );
}

/** Top 5-8 articles for a pair, lazy-loaded via TanStack Query */
function ArticleDetail({ pair }: { pair: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["news-articles", pair, 7],
    queryFn: async () => {
      const { data } = await apiClient.get<{ articles: LiveSentimentArticle[] }>(
        "/news/articles",
        { params: { pair, days: 7 } },
      );
      return data;
    },
    enabled: true,
    staleTime: 60_000,
    refetchInterval: 5 * 60_000,
  });

  const articles = (data?.articles ?? []).slice(0, 8);

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 px-3 pb-3">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="h-8 animate-pulse rounded bg-white/5" />
        ))}
      </div>
    );
  }

  if (articles.length === 0) {
    return (
      <div className="px-3 pb-3">
        <span className="text-[10px] text-slate-500">No articles found for {pair}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-0 px-3 pb-3">
      {articles.map((a, i) => (
        <ArticleRow key={`${a.title}-${i}`} article={a} />
      ))}
    </div>
  );
}

/** Single article row — favicon, title, impact badge, timestamp */
function ArticleRow({ article }: { article: LiveSentimentArticle }) {
  const score = article.sentiment_score;
  const isBullish = score > SENTIMENT_THRESHOLDS.BULLISH;
  const isBearish = score < SENTIMENT_THRESHOLDS.BEARISH;
  const borderColor = isBullish
    ? "border-l-emerald-400"
    : isBearish
      ? "border-l-red-400"
      : "border-l-slate-600";
  const [timeStr] = useState(() => {
    const ts = new Date(article.timestamp).getTime();
    const minsDiff = Math.floor((Date.now() - ts) / 60000);
    const hoursDiff = Math.floor(minsDiff / 60);
    const daysDiff = Math.floor(hoursDiff / 24);
    return daysDiff > 0
      ? `${daysDiff}d`
      : hoursDiff > 0
        ? `${hoursDiff}h`
        : minsDiff > 0
          ? `${minsDiff}m`
          : "now";
  });

  return (
    <div
      className={cn(
        "flex items-center gap-2 border-l-2 py-1.5 pl-2 pr-1 transition-colors hover:bg-white/[0.03]",
        borderColor,
      )}
    >
      <FaviconImg source={article.source} />
      <a
        href={article.url || "#"}
        target="_blank"
        rel="noopener noreferrer"
        className="min-w-0 flex-1 truncate font-sans text-[10px] leading-[1.3] text-slate-300 transition-colors hover:text-cyan-400 hover:underline"
        title={article.title}
      >
        {article.title}
      </a>
      <span className="shrink-0 font-mono text-[9px] tabular-nums text-slate-500">
        {timeStr}
      </span>
    </div>
  );
}

// ── main widget ──

export function SentimentNewsWidget({ pair = "EURUSD" }: { pair?: string }) {
  const activePair = useDashboardStore((s) => s.activePair);
  const demoMode = useAppStore((s) => s.demoMode);
  const displayPair = pair ?? activePair;
  const { data: sentiment, isLoading } = useLiveSentiment(displayPair, !demoMode);

  const [expandedPair, setExpandedPair] = useState<string | null>(null);

  const pairData = sentiment?.pairs?.[displayPair] ?? sentiment?.pairs?.[displayPair.toUpperCase()];
  const allPairs = sentiment?.pairs ? Object.entries(sentiment.pairs) : [];
  const majorPairs = allPairs.filter(([p]) => p !== "OTHER");
  const otherEntry = allPairs.find(([p]) => p === "OTHER");

  const handleToggle = useCallback((p: string) => {
    setExpandedPair((prev) => (prev === p ? null : p));
  }, []);

  if (isLoading) {
    return (
      <div className="flex flex-col gap-3 p-4">
        {Array.from({ length: 5 }, (_, i) => (
          <div key={i} className="h-10 animate-pulse rounded bg-white/5" />
        ))}
      </div>
    );
  }

  if (!sentiment || allPairs.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-8">
        <span className="text-[10px] text-slate-500">No sentiment data available</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-3 pt-6 pb-4">
        <span className="text-[10px] font-medium tracking-[0.12em] text-slate-400 uppercase">
          Market Sentiment
        </span>
        {pairData && (
          <span className="font-mono text-[9px] tabular-nums text-slate-500">
            {pairData.article_count ?? 0} articles
          </span>
        )}
      </div>

      {/* Master list */}
      <div className="flex flex-col">
        {majorPairs.map(([p, data]) => (
          <div key={p}>
            <PairRow
              pair={p}
              data={data}
              isExpanded={expandedPair === p}
              onToggle={() => handleToggle(p)}
            />

            {/* Expandable article detail */}
            <div
              className={cn(
                "grid transition-all duration-200 ease-out",
                expandedPair === p ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
              )}
            >
              <div className="overflow-hidden">
                <div className="border-t border-white/[0.06] pt-2">
                  {expandedPair === p && <ArticleDetail pair={p} />}
                </div>
              </div>
            </div>
          </div>
        ))}

        {otherEntry && (
          <>
            <div className="mx-3 my-1 border-t border-white/[0.05]" />
            <PairRow
              pair="OTHER"
              data={otherEntry[1]}
              isExpanded={expandedPair === "OTHER"}
              onToggle={() => handleToggle("OTHER")}
            />
            <div
              className={cn(
                "grid transition-all duration-200 ease-out",
                expandedPair === "OTHER" ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
              )}
            >
              <div className="overflow-hidden">
                {expandedPair === "OTHER" && <ArticleDetail pair="" />}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
