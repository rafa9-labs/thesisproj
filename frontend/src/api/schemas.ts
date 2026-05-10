export type HpoIntensity = "light" | "quick" | "standard" | "deep";

export interface PairInfo {
  pair: {
    symbol: string;
    oanda_name: string;
    pip_value: number;
    lot_size: number;
    base_currency: string;
    quote_currency: string;
    typical_spread_bps: number;
  };
  timeframes: {
    timeframe: string;
    rows: number;
    start_date: string;
    end_date: string;
  }[];
}

export interface ModelInfo {
  name: string;
  display_name: string;
  category: "classical" | "deep" | "rl" | "ensemble";
  description: string;
}

export interface Metrics {
  model: string;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  total_return_pct: number | null;
  cagr: number | null;
  calmar_ratio: number | null;
  win_rate: number | null;
  total_trades: number | null;
  profit_factor: number | null;
  avg_trade: number | null;
  active_rate: number | null;
  directional_accuracy: number | null;
  precision_macro: number | null;
  f1_macro: number | null;
  equity_curve: EquityPoint[] | null;
  buy_hold_curve: EquityPoint[] | null;
  drawdown_curve: EquityPoint[] | null;
  monthly_results: MonthlyResult[] | null;
  trades: TradeRecord[] | null;
  hpo_param_importance: HpoParamImportance[] | null;
  hpo_trials: HpoTrial[] | null;
}

export interface BacktestRequest {
  pair: string;
  models: string[];
  start_date?: string;
  end_date?: string;
  trading_costs?: boolean;
  months?: number;
  repeats?: number;
  seed?: number;
  hpo_intensity?: HpoIntensity;
  config_overrides: Record<string, unknown>;
}

export interface QuickTestPreset {
  name: string;
  label: string;
  description: string;
  pair: string;
  timeframe: string;
  models: string[];
  months: number;
  hpo_intensity: HpoIntensity;
  seed: number;
  repeats: number;
  trading_costs: boolean;
}

export interface DateRangePreset {
  key: string;
  label: string;
  start_date: string;
  end_date: string;
}

export interface DateRangeResponse {
  symbol: string;
  timeframe: string;
  data_start: string;
  data_end: string;
  presets: DateRangePreset[];
}

export interface RuntimeEstimateRequest {
  models: string[];
  months: number;
  hpo_intensity: HpoIntensity;
}

export interface RuntimeEstimateResponse {
  models: string[];
  months: number;
  hpo_intensity: HpoIntensity;
  total_trials: number;
  estimated_minutes_low: number;
  estimated_minutes_high: number;
}

export interface JobSummary {
  job_id: string;
  type: string;
  status: "pending" | "running" | "completed" | "failed";
  pair: string | null;
  models: string[] | null;
  created_at: string;
}

export interface BacktestSummaryItem {
  job_id: string;
  created_at: string;
  pair: string;
  timeframe: string;
  models: string[];
  sharpe: number | null;
  total_return_pct: number | null;
  win_rate: number | null;
  max_drawdown_pct: number | null;
  total_trades: number | null;
  status: string;
}

export interface JobStatus {
  job_id: string;
  type: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  error: string | null;
  progress: Record<string, unknown> | null;
}

export interface TradeRecord {
  trade_id: number;
  entry_date: string;
  exit_date: string;
  direction: "BUY" | "SELL";
  entry_price: number;
  exit_price: number;
  pips: number;
  return_pct: number;
  duration_bars: number;
  barrier_hit?: string | null;
}

export interface MonthlyResult {
  month: string;
  return_pct: number;
  win_rate: number;
  trades: number;
  sharpe: number | null;
}

export interface EquityPoint {
  time: number;
  value: number;
}

export interface HpoParamImportance {
  param: string;
  importance: number;
}

export interface HpoTrial {
  trial_number: number;
  value: number;
  params: Record<string, unknown>;
}

export interface FullJobResults {
  job_id: string;
  pair: string;
  models: string[];
  config: Record<string, unknown>;
  metrics: Metrics[];
  monthly_results: MonthlyResult[] | null;
  trades: TradeRecord[] | null;
  equity_curve: EquityPoint[] | null;
  buy_hold_curve: EquityPoint[] | null;
  drawdown_curve: EquityPoint[] | null;
  hpo_param_importance: HpoParamImportance[] | null;
  hpo_trials: HpoTrial[] | null;
}

export interface JobResults {
  job_id: string;
  pair: string;
  models: string[];
  config: Record<string, unknown> | null;
  metrics: Metrics[];
}

export type WsEvent =
  | { event: "job_started"; job_id: string; pair: string; models: string[]; total_work: number }
  | { event: "model_training"; job_id: string; model: string; status: "starting" }
  | { event: "model_training"; job_id: string; model: string; status: "complete"; metrics: Partial<Metrics> }
  | { event: "model_phase"; job_id: string; model: string; phase: "hpo" | "simulation"; total_work?: number }
  | { event: "hpo_progress"; job_id: string; model: string; trial?: number; total_trials?: number; n_trials?: number; cv_blocks: number; completed_work: number; total_work: number; progress_pct: number }
  | { event: "month_progress"; job_id: string; model: string; month?: number; total_months?: number; period?: number; total_periods?: number; sharpe?: number; trades?: number; completed_work: number; total_work: number; progress_pct: number }
  | { event: "job_complete"; job_id: string; metrics: Partial<Metrics>[] }
  | { event: "job_failed"; job_id: string; error: string }
  | { event: "download_started"; job_id: string; pair: string }
  | { event: "download_complete"; job_id: string; pair: string }
  | { event: "download_failed"; job_id: string; error: string };

export interface HealthResponse {
  status: string;
  version: string;
  redis: string;
  db_rows: number;
}

export interface HeatmapCell {
  model: string;
  pair: string;
  sharpe: number | null;
  total_return_pct: number | null;
  win_rate: number | null;
  max_drawdown: number | null;
  total_trades: number | null;
  job_id: string | null;
}

export interface HeatmapResponse {
  models: string[];
  pairs: string[];
  cells: HeatmapCell[];
}

export interface NewsEvent {
  time: number;
  event: string;
  currency: string;
  impact: "high" | "medium" | "low";
}

export interface LicenseStatusResponse {
  plan: string;
  licensed: boolean;
  trial_active: boolean;
  trial_days_left: number;
  license_key: string;
  activation_id: string;
  expires_at: string;
  last_verified: string;
  machine_id: string;
  needs_activation: boolean;
  available_models: string[];
  locked_models: string[];
}

export interface SparklinePoint {
  t: number;
  v: number;
}

export interface LivePrice {
  symbol: string;
  bid: number | null;
  ask: number | null;
  mid: number | null;
  spread_pips: number | null;
  change_pct: number | null;
  timestamp: string;
  sparkline: SparklinePoint[];
}

export interface LivePricesResponse {
  prices: LivePrice[];
  source: "oanda" | "key_required" | "unavailable";
  message?: string;
  error?: string;
}

export interface OHLCBar {
  t: number;
  o: number;
  h: number;
  l: number;
  c: number;
  volume: number;
}

export interface CandlesResponse {
  pair: string;
  timeframe: string;
  candles: OHLCBar[];
}

export interface TradeChartMarker {
  trade_id: number;
  entry_time: number;
  exit_time: number;
  direction: "BUY" | "SELL";
  entry_price: number;
  exit_price: number;
  pnl_pct: number;
}

export interface TradeChartData {
  pair: string;
  timeframe: string;
  candles: OHLCBar[];
  trades: TradeChartMarker[];
  equity_curve: EquityPoint[];
}

export type HyperparamRange = {
  type: "float_range" | "int_range";
  low: number;
  high: number;
  step?: number;
  log_scale: boolean;
  default?: number;
};

export type HyperparamChoice = {
  type: "choice";
  values: (string | number)[];
  default?: string | number;
};

export type HyperparamFixed = {
  type: "fixed";
  value: string | number | boolean | null;
};

export type HyperparamSpec = HyperparamRange | HyperparamChoice | HyperparamFixed;

export interface ModelHyperparams {
  model: string;
  display_name: string;
  category: string;
  tunable: boolean;
  params: Record<string, HyperparamSpec>;
}

export interface ModelHyperparamsResponse {
  models: ModelHyperparams[];
}

export interface LiveSessionInfo {
  session_id: string;
  pair: string;
  model: string;
  timeframe: string;
  status: string;
  equity: number;
  position: string;
  signal_count: number;
  created_at: string;
}

export interface LiveSignalEvent {
  event: "signal" | "heartbeat" | "stopped" | "error";
  session_id?: string;
  time: number;
  direction?: string;
  confidence?: number;
  price?: number;
  equity?: number;
  pnl?: number;
  position?: string;
  message?: string;
}

export interface LiveSentimentPairData {
  vader_sentiment: number;
  vader_magnitude: number;
  blended_sentiment: number;
  article_count: number;
  last_updated: string;
  recommended_position: number;
  position_confidence: number;
  llm_sentiment?: number | null;
  llm_confidence?: number | null;
  llm_volatility?: number | null;
  llm_weight?: number;
  currencies_affected?: string[];
}

export interface LiveSentimentArticle {
  title: string;
  source: string;
  sentiment_score: number;
  timestamp: string;
}

export interface LiveSentimentResponse {
  pairs: Record<string, LiveSentimentPairData>;
  top_articles: LiveSentimentArticle[];
  backend: string;
  model: string;
  error?: string;
}
