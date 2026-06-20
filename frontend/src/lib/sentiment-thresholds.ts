/**
 * Shared sentiment thresholds — single source of truth for all components.
 * Prevents visual inconsistency where different components classify the same
 * score differently (e.g., +0.10 shows green in NewsPage but neutral in Dashboard).
 */
export const SENTIMENT_THRESHOLDS = {
  /** Border / row color classification threshold */
  BULLISH: 0.15,
  BEARISH: -0.15,

  /** SentimentBadge label classification (Bullish/Neutral/Bearish) */
  BADGE_BULLISH: 0.3,
  BADGE_BEARISH: -0.3,

  /** Market impact tiers (via getImpactLabel) */
  IMPACT_HIGH: 0.6,
  IMPACT_MED: 0.3,
} as const;

export const BIAS_THRESHOLD = 0.05;
