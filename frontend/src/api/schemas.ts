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

export interface OverfittingCI {
  low: number | null;
  high: number | null;
  mean: number | null;
}

export interface OverfittingReportData {
  overfit_score: number;
  risk_level: "low" | "medium" | "high";
  risk_color: "green" | "yellow" | "red";
  train_oos_gap_pct: number;
  temporal_degradation_pct: number;
  sharpe_ci: OverfittingCI | null;
  return_ci: OverfittingCI | null;
  maxdd_ci: OverfittingCI | null;
  cv_sharpe_mean: number | null;
  cv_sharpe_std: number | null;
  cv_return_mean: number | null;
  cv_return_std: number | null;
  min_trl_trades: number;
  sufficient_trades: boolean;
  n_periods: number;
  n_signal_periods: number;
  signal_gap_pct: number;
  is_mean_sharpe: number | null;
  oos_mean_sharpe: number | null;
  dsr_min_sharpe: number | null;
  dsr_value: number | null;
  psr: number | null;
  interaction_effects: Array<{
    param: string;
    main_pct: number;
    interaction_pct: number;
    total_pct: number;
  }> | null;
}

export interface DeployedModelDetail {
  id: string;
  model_type: string;
  snapshot_path: string;
  best_sharpe: number | null;
  best_return: number | null;
  created_at: string;
  status: string;
  tags: string[];
  parent_job_id: string | null;
  missing_on_disk: boolean;
  win_rate: number | null;
  max_drawdown: number | null;
  total_trades: number | null;
  sortino: number | null;
  calmar_ratio: number | null;
  profit_factor: number | null;
  cagr: number | null;
  overfit_score: number | null;
  risk_level: string | null;
  train_start: string | null;
  train_end: string | null;
  feature_count: number | null;
  seed: number | null;
  calibrate_method: string | null;
  pair: string | null;
  timeframe: string | null;
  best_params: Record<string, unknown>;
  feature_names: string[];
  coverage_conf_thr: number | null;
  input_shape: number[] | null;
  pip_freeze: string | null;
  schema_version: number | null;
  directional_accuracy: number | null;
  active_rate: number | null;
  avg_trade: number | null;
  overfitting: OverfittingReportData | null;
}

export interface WalkForwardPeriod {
  period_start: string;
  period_end: string;
  train_start: string | null;
  train_end: string | null;
  test_sharpe: number | null;
  train_sharpe: number | null;
  strategy_return: number | null;
  bh_return: number | null;
  trades: number;
  signals_raw: number;
  signals_passed_gate: number;
  pct_sideways: number | null;
  pct_trend: number | null;
  pct_volatile: number | null;
  sharpe_gap_pct: number | null;
  return_gap_pct: number | null;
}

export interface FeatureImportanceEntry {
  feature: string;
  importance: number;
}

export interface PredictionHistogramBin {
  bin_start: number;
  bin_end: number;
  bin_center: number;
  count: number;
}

export interface ConfusionMatrixData {
  matrix: number[][] | null;
  labels: string[];
}

export interface ConfidenceBand {
  band_min: number;
  band_max: number;
  count: number;
  accuracy: number;
  mean_return: number;
}

export interface TrainingDiagnostics {
  feature_importance: FeatureImportanceEntry[] | null;
  prediction_histogram: PredictionHistogramBin[] | null;
  confusion_matrix: ConfusionMatrixData | null;
  confidence_bands: ConfidenceBand[] | null;
  importance_method: string | null;
  feature_families: Record<string, number> | null;
  vif_warnings: Array<{ feature: string; vif: number }> | null;
}

export interface DataStatusSingle {
  available: boolean;
  start: string | null;
  end: string | null;
  bars: number;
}

export interface DataStatusResponse {
  symbol: string;
  timeframes: Record<string, DataStatusSingle>;
  ready: boolean;
  missing: string[];
}

export interface DefinePairRequest {
  symbol: string;
  pip_value: number;
  decimal_places: number;
}

export interface DefinePairResponse {
  symbol: string;
  oanda_name: string;
  pip_value: number;
  lot_size: number;
  base_currency: string;
  quote_currency: string;
  typical_spread_bps: number;
}

export interface Metrics {
  model: string;
  metrics_version?: number;
  sharpe: number | null;
  sharpe_min_months?: number;
  sharpe_min_trades?: number;
  sortino: number | null;
  max_drawdown: number | null;
  total_return_pct: number | null;
  cagr: number | null;
  calmar_ratio: number | null;
  win_rate: number | null;
  positive_months_rate?: number | null;
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
  best_study: BestStudy | null;
  hpo_study_meta: HpoStudyMeta | null;
  hpo_learning_summary: HpoLearningSummary | null;
  hpo_sensitivity: HpoSensitivityEntry[] | null;
  overfitting: OverfittingReportData | null;
  walkforward_periods: WalkForwardPeriod[] | null;
  diagnostics: TrainingDiagnostics | null;
  summary_text: string | null;
  snapshot_path?: string | null;
}

export interface AnalysisSection {
  title: string;
  severity: "green" | "amber" | "red" | "info";
  detail: string;
  recommendation: string;
}

export interface LlmAnalysis {
  insight?: string;
  recommended_preset?: string;
  reason?: string;
  parameter_changes?: string[];
  predicted_improvement?: string;
  warning?: string | null;
  error?: string;
  raw_text?: string;
  dsr_analysis?: AnalysisSection;
  friction_analysis?: AnalysisSection;
  regime_analysis?: AnalysisSection;
}

export interface LlmAnalysisResponse {
  job_id: string;
  model: string;
  analysis: LlmAnalysis;
}

export interface BacktestRequest {
  pair: string;
  timeframe: string;
  models: string[];
  start_date?: string;
  end_date?: string;
  trading_costs?: boolean;
  months?: number;
  repeats?: number;
  seed?: number;
  hpo_intensity?: HpoIntensity;
  n_trials?: number;
  parent_job_id?: string;
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
  n_trials?: number;
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
  status: "pending" | "running" | "completed" | "failed" | "queued";
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
  error: string | null;
  metrics_version?: number;
  legacy?: boolean;
}

export interface JobStatus {
  job_id: string;
  type: string;
  status: "pending" | "running" | "completed" | "failed" | "queued";
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
  entry_price: number | null;
  exit_price: number | null;
  pips: number | null;
  return_pct: number;
  duration_bars: number;
  barrier_hit?: string | null;
}

export interface MonthlyResult {
  month: string;
  return_pct: number;
  win_rate: number;
  trades: number;
  wins?: number;
  sharpe: number | null;
  sharpe_legacy?: number | null;
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
  state?: string | null;
  duration_sec?: number | null;
  user_attrs?: Record<string, unknown>;
}

export interface BestStudy {
  best_trial: number;
  best_value: number | null;
  best_params: Record<string, unknown>;
}

export interface HpoStudyMeta {
  study_name?: string | null;
  direction?: string | null;
  n_trials: number;
  n_completed: number;
  n_pruned: number;
  n_failed: number;
  sampler_type?: string | null;
  total_duration_sec?: number | null;
}

export interface HpoLearningSummary {
  cliff_delta?: number | null;
  delta_interpretation?: string | null;
  startup_median_score?: number | null;
  post_startup_median_score?: number | null;
  share_beating_startup?: number | null;
  best_uplift_pct?: number | null;
  startup_trials: number;
  post_startup_trials: number;
}

export interface HpoSensitivityEntry {
  param: string;
  index: number;
  std_at_best?: number | null;
  range_at_best?: number | null;
  perturbation_direction?: string | null;
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
  | {
      event: "model_training";
      job_id: string;
      model: string;
      status: "complete";
      metrics: Partial<Metrics>;
    }
  | {
      event: "model_phase";
      job_id: string;
      model: string;
      phase: "hpo" | "simulation";
      total_work?: number;
    }
  | {
      event: "hpo_progress";
      job_id: string;
      model: string;
      trial?: number;
      total_trials?: number;
      n_trials?: number;
      cv_blocks: number;
      completed_work: number;
      total_work: number;
      progress_pct: number;
      elapsed_seconds?: number;
      eta_seconds?: number | null;
    }
  | {
      event: "hpo_trial_result";
      job_id: string;
      model: string;
      trial_number: number;
      score: number | null;
      params: Record<string, unknown>;
      best_score_so_far: number | null;
      trial_state: string;
    }
  | {
      event: "month_progress";
      job_id: string;
      model: string;
      month?: number;
      total_months?: number;
      period?: number;
      total_periods?: number;
      sharpe?: number;
      trades?: number;
      completed_work: number;
      total_work: number;
      progress_pct: number;
      elapsed_seconds?: number;
      eta_seconds?: number | null;
    }
  | {
      event: "oos_result";
      job_id: string;
      model: string;
      period: number;
      total_periods: number;
      equity: number | null;
      equity_bh: number | null;
      sharpe: number | null;
      sharpe_ann?: number | null;
      wins?: number;
      return_pct: number | null;
      trades: number | null;
      drawdown: number | null;
      win_rate: number | null;
      precision: number | null;
      f1: number | null;
      directional_accuracy: number | null;
      active_rate: number | null;
      flat?: boolean;
      train_sharpe?: number | null;
      train_sharpe_is_objective?: boolean;
      sharpe_gap_pct?: number | null;
      signals_raw?: number;
      signals_passed_gate?: number;
      signal_coverage?: number | null;
      profit_per_hit?: number | null;
      outperformance?: number | null;
    }
  | { event: "job_complete"; job_id: string; metrics: Partial<Metrics>[] }
  | { event: "job_failed"; job_id: string; error: string }
  | {
      event: "cycle_started";
      job_id: string;
      model: string;
      cycle_number: number;
      total_cycles: number;
    }
  | { event: "download_started"; job_id: string; pair: string }
  | { event: "download_complete"; job_id: string; pair: string }
  | { event: "download_failed"; job_id: string; error: string }
  | {
      event: "simulation_started";
      job_id: string;
      model: string;
      n_periods: number;
      bh_curve: { period: number; bh: number }[];
    };

export interface HpoTrialRow {
  trial_number: number;
  score: number | null;
  params: Record<string, unknown>;
  best_score_so_far: number | null;
  trial_state: string;
}

export interface OosPeriodResult {
  period: number;
  total_periods: number;
  model?: string;
  equity: number | null;
  equity_bh: number | null;
  sharpe: number | null;
  sharpe_ann?: number | null;
  wins?: number;
  return_pct: number | null;
  trades: number | null;
  drawdown: number | null;
  win_rate: number | null;
  precision: number | null;
  f1: number | null;
  flat?: boolean;
  train_sharpe?: number | null;
  train_sharpe_is_objective?: boolean;
  sharpe_gap_pct?: number | null;
  signals_raw?: number;
  signals_passed_gate?: number;
  directional_accuracy?: number | null;
  active_rate?: number | null;
  signal_coverage?: number | null;
  profit_per_hit?: number | null;
  outperformance?: number | null;
}

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
  tier: number;
  display_name: string;
  ui_control: string;
  description: string;
};

export type HyperparamChoice = {
  type: "choice";
  values: (string | number)[];
  default?: string | number;
  tier: number;
  display_name: string;
  ui_control: string;
  description: string;
};

export type HyperparamFixed = {
  type: "fixed";
  value: string | number | boolean | null;
  tier: number;
  display_name: string;
  ui_control: string;
  description: string;
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
  last_updated: string | null;
  next_update?: string | null;
  cache_age_hours?: number;
  recommended_position: number;
  position_confidence: number;
  llm_sentiment?: number | null;
  llm_confidence?: number | null;
  llm_volatility?: number | null;
  llm_weight?: number;
  vader_contribution?: number;
  llm_contribution?: number;
  article_count_by_tier?: { exact: number; partial: number; other: number };
  currencies_affected?: string[];
}

export interface LiveSentimentArticle {
  title: string;
  body: string;
  source: string;
  url: string;
  pair_tags: string[];
  sentiment_score: number;
  summary: string;
  bias: string;
  timestamp: string;
  relevance_tier?: number;
  llm_sentiment?: number | null;
  llm_confidence?: number | null;
  highlighted_body?: string | null;
}

export interface LiveSentimentResponse {
  pairs: Record<string, LiveSentimentPairData>;
  top_articles: LiveSentimentArticle[];
  article_count_by_tier?: { exact: number; partial: number; other: number };
  backend: string;
  model: string;
  error?: string;
  from_cache?: boolean;
  status?: string;
  llm_available?: boolean;
}

export interface NewsArticleFull {
  title: string;
  body: string;
  source: string;
  url: string;
  timestamp: string;
  pair_tags: string[];
  sentiment_score: number;
  summary: string;
  bias: string;
  highlighted_body?: string | null;
}

export interface NewsArticlesResponse {
  articles: NewsArticleFull[];
  total: number;
  pair: string;
}

export interface PaperSessionInfo {
  session_id: string;
  pair: string;
  model_type: string;
  timeframe: string;
  status: string;
  equity: number;
  position: string;
  unrealized_pnl: number;
  total_trades: number;
  signal_count: number;
  created_at: string;
}

export interface PaperTradeItem {
  trade_id: string;
  direction: string;
  size: number;
  entry_time: number;
  entry_price: number;
  exit_time: number | null;
  exit_price: number | null;
  pnl: number;
  exit_reason: string;
}

export interface PaperSummary {
  sharpe: number;
  sortino: number;
  total_return_pct: number;
  max_drawdown_pct: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  avg_trade_pnl: number;
  final_equity: number;
  signal_count: number;
}

export interface PaperMetricComparison {
  [key: string]: { paper: number; backtest: number | null; delta: number | null };
}

export interface PaperStopResult {
  session_id: string;
  status: string;
  summary: PaperSummary;
  comparison: PaperMetricComparison;
}

export interface PaperTradesResponse {
  session_id: string;
  trades: PaperTradeItem[];
  offset: number;
  limit: number;
}

export interface PaperSummaryResponse {
  session_id: string;
  summary: PaperSummary;
  comparison: PaperMetricComparison;
}

export interface DeployPaperRequest {
  pair: string;
  model_id?: string | null;
  model_type?: string;
  timeframe?: string;
  initial_equity?: number;
  position_sizing?: string;
  sizing_config?: Record<string, unknown>;
  live_news_blend_enabled?: boolean;
  live_news_blend_weight?: number;
}

export interface PaperSignalEvent {
  event: "signal" | "hold" | "trade_opened" | "trade_closed" | "heartbeat" | "stopped" | "error";
  direction?: string;
  confidence?: number;
  mid_price?: number;
  bid?: number;
  ask?: number;
  equity?: number;
  unrealized_pnl?: number;
  position?: string;
  time?: number;
  trade_id?: string;
  size?: number;
  entry_price?: number;
  exit_price?: number;
  pnl?: number;
  is_win?: boolean;
  exit_reason?: string;
  message?: string;
  sub_events?: PaperSignalEvent[];
  candle?: CandleBar;
  live_price?: number;
}

export interface CandleBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface LiveSessionInfo {
  session_id: string;
  pair: string;
  model_type: string;
  timeframe: string;
  mode: string;
  status: string;
  equity: number;
  position: string;
  unrealized_pnl: number;
  signal_count: number;
  killed: boolean;
  kill_reason: string;
}

export interface LiveJournalItem {
  trade_id: string;
  oanda_order_id: string;
  oanda_fill_id: string;
  direction: string;
  size: number;
  entry_price: number;
  exit_price: number | null;
  pnl: number;
  confidence: number;
  is_win: boolean;
  exit_reason: string;
  risk_blocked: boolean;
  risk_reason: string;
}

export interface LiveRiskState {
  equity_peak: number;
  current_equity: number;
  consecutive_losses: number;
  daily_trades: number;
  hourly_trades: number;
  total_trades: number;
  total_wins: number;
  paused: boolean;
  pause_reason: string;
  killed: boolean;
  kill_reason: string;
  kill_level: string;
  consecutive_api_errors: number;
  signal_age_sec: number;
}

export interface LiveStopResult {
  session_id: string;
  status: string;
  equity: number;
  events: Array<Record<string, unknown>>;
  journal_length: number;
  killed: boolean;
}

export interface LiveEmergencyResult {
  session_id: string;
  killed: boolean;
  close_results: Array<Record<string, unknown>>;
  stopped: boolean;
}

export interface DeployLiveRequest {
  pair: string;
  model_id?: string | null;
  model_type?: string;
  timeframe?: string;
  initial_equity?: number;
  position_sizing?: string;
  sizing_config?: Record<string, unknown>;
  mode?: string;
  risk_config?: Record<string, unknown>;
  live_news_blend_enabled?: boolean;
  live_news_blend_weight?: number;
}

export interface LiveSignalEvent {
  event:
    | "signal"
    | "hold"
    | "order_placed"
    | "risk_blocked"
    | "trade_closed"
    | "kill"
    | "heartbeat"
    | "stopped"
    | "error";
  direction?: string;
  confidence?: number;
  mid_price?: number;
  bid?: number;
  ask?: number;
  equity?: number;
  unrealized_pnl?: number;
  position?: string;
  time?: number;
  reason?: string;
  all_reasons?: string[];
  level?: string;
  instrument?: string;
  units?: number;
  price?: number;
  oanda_order_id?: string;
  oanda_fill_id?: string;
  pnl?: number;
  is_win?: boolean;
  error?: string;
  message?: string;
  sub_events?: LiveSignalEvent[];
  candle?: CandleBar;
  live_price?: number;
}

export interface SeedDemoTimeframe {
  timeframe: string;
  rows: number;
  start_date: string | null;
  end_date: string | null;
}

export interface SeedDemoResponse {
  status: string;
  pairs: Record<string, SeedDemoTimeframe[]>;
  total_candles: number;
}

// ════════════════════════════════════════════════════════════════════
// ════════════════════════════════════════════════════════════════════

export interface RegimeAssignmentSchema {
  models: string[];
  weights: number[];
}

export interface CommitteeConfigSchema {
  version: number;
  regimes: Record<string, RegimeAssignmentSchema>;
  fallback: RegimeAssignmentSchema;
  constraints?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  model_params?: Record<string, Record<string, unknown>>;
}

export interface RegimeMatrixEntry {
  regime: string;
  model: string;
  sharpe: number;
  trades: number;
  hit_rate: number;
}

export interface RegimeMatrixResponse {
  regimes: string[];
  models: string[];
  entries: RegimeMatrixEntry[];
  generated_at?: string;
}

export interface RegimeLabelPoint {
  timestamp: string;
  regime_id: number;
  regime_name: string;
}

export interface RegimeLabelsResponse {
  pair: string;
  timeframe: string;
  labels: RegimeLabelPoint[];
  count: number;
}

export interface CommitteeSnapshotInfo {
  version: string;
  created_at: string;
  models: string[];
}

export interface CommitteeSnapshotListResponse {
  snapshots: CommitteeSnapshotInfo[];
}

export interface SavedCommitteeOut {
  id: string;
  name: string;
  full_cycle_job_id: string | null;
  full_cycle_status: string | null;
  pair: string;
  timeframe: string;
  config_json: Record<string, unknown>;
  trust_score: number | null;
  avg_sharpe: number | null;
  avg_return: number | null;
  win_rate: number | null;
  max_drawdown: number | null;
  total_trades: number | null;
  sortino: number | null;
  regime_count: number;
  consensus_model_count: number;
  is_active: boolean;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface SavedCommitteeListResponse {
  committees: SavedCommitteeOut[];
  total: number;
}

// ════════════════════════════════════════════════════════════════════
export interface FactoryIterationAction {
  type: string;
  regime: string;
  model_add: string;
  model_remove: string;
  rationale?: string;
}

export interface FactoryIterationRecord {
  iteration: number;
  action: FactoryIterationAction;
  before_sharpe: number;
  after_sharpe: number;
  accepted: boolean;
  per_regime_delta: Record<string, number>;
  timestamp: string;
  rationale: string;
}

// ════════════════════════════════════════════════════════════════════
// Full Cycle — Racecar (B→C→D) + Factory (optimization) in one shot
// ════════════════════════════════════════════════════════════════════

export interface FullCycleRequest {
  models: string[];
  pair?: string;
  timeframe?: string;
  start_date?: string;
  end_date?: string;
  sweep_n_estimators?: number;
  sweep_max_depth?: number;
  skip_feature_sweep?: boolean;
  use_boruta_shap?: boolean;
  boruta_percentile?: number;
  boruta_max_iter?: number;
  enable_phase3?: boolean;
  enable_phase4?: boolean;
  enable_phase5?: boolean;
  enable_phase6?: boolean;
  debug_mode?: boolean;
  committee_top_k?: number;
  committee_weight_method?: string;
  committee_min_sharpe?: number;
  train_months?: number;
  test_months?: number;
  hpo_sampler?: string;
  cv_blocks?: number;
  cv_val_frac?: number;
  plateau_patience?: number;
  proposer?: string;
  llm_backend?: string;
  ucb_c?: number;
  max_iterations?: number;
  patience?: number;
  stopping_tolerance?: number;
  regime_sharpe_floor?: number;
  factory_proxy_months?: number;
  factory_proxy_folds?: number;
  hpo_trials?: Record<string, number>;
  hpo_startup_trials?: Record<string, number>;
}

export interface FullCycleStatusResponse {
  job_id: string;
  phase: string;
  phase_number: number;
  phase_progress: string;
  iteration: number;
  total_iterations: number;
  current_action: string;
  best_sharpe_so_far: number;
  started_at: string;
  error: string;
  surviving_models: string[];
  locked_features_count: number;
  hpo_model_scores: Record<string, number | null>;
  phase_timings: Record<string, number>;
  // Phase 1
  feature_names_locked?: string[];
  feature_names_pruned?: string[];
  pruned_count?: number;
  // Phase 2
  hpo_status?: Record<string, string>;
  // Phase 3
  committee_config?: Record<string, unknown>;
  // Phase 4
  fold_consistency_cv?: number | null;
  phase4_pbo?: number | null;
  phase4_dsr?: number | null;
  trust_score?: TrustScoreResult;
  regime_coverage?: Record<string, { sharpe: number; trades: number; folds_active: number; covered: boolean }>;
  seed_sharpes?: number[];
  seed_avg_sharpe?: number | null;
  seed_pass?: boolean | null;
  // Phase 5
  factory_accepted_count?: number;
  factory_last_delta?: number | null;
  factory_last_action?: string;
  factory_last_regime?: string;
  factory_last_accepted?: boolean | null;
  // Phase 4 WFO live progress
  wfo_fold_progress?: string;
  wfo_fold_sharpes?: number[];
  wfo_fold_trades?: number[];
  wfo_running_avg_sharpe?: number | null;
}

export interface TrustScoreResult {
  trust_score: number;
  action: "deploy" | "proceed" | "flag" | "reject";
  sub_scores: {
    pbo_contribution: number;
    dsr_contribution: number;
    coverage_contribution: number;
    floor_contribution: number;
  };
}

export interface HpoStatusEntry {
  model_type: string;
  status: string;
  best_score?: number | null;
  committee_size?: number;
}

export interface FullCycleResultsResponse {
  job_id: string;
  status: string;
  locked_features_count: number;
  pruned_features_count: number;
  top_importance_feature: string;
  locked_features_list?: string[];
  phase0_pruned: string[];
  phase0_survivors: string[];
  racecar_profile_matrix?: Record<string, unknown>;
  racecar_committee_config?: Record<string, unknown>;
  racecar_backtest?: {
    models?: string[];
    folds?: number;
    folds_detail?: Array<{
      fold_idx: number;
      sharpe: number;
      trades: number;
      return_val: number;
      drawdown: number;
      win_rate: number;
      active_rate: number;
    }>;
    avg_sharpe?: number;
    avg_trades?: number;
    avg_return?: number;
    avg_win_rate?: number;
    avg_active_rate?: number;
    avg_drawdown?: number;
    per_regime_summary?: Record<string, { sharpe: number; trades: number; folds_active: number }>;
    equity_curve?: Array<{ bar_index: number; value: number }>;
    diagnostics?: {
      prediction_histogram?: Array<{ bin_start: number; bin_end: number; bin_center: number; count: number }>;
      total_predictions?: number;
      active_signals?: number;
      active_rate?: number;
      vote_agreement?: Record<string, { count: number; pct: number; win_rate: number; avg_return: number }>;
      model_contributions?: Array<{ model: string; delta_sharpe: number; active_pct: number }>;
      model_agreement?: { models: string[]; kappa_matrix: number[][] };
    };
    trades?: Array<{
      entry_time: string;
      exit_time: string;
      direction: string;
      regime: string;
      return_pct: number;
      duration_bars: number;
    }>;
    fold_consistency_cv?: number;
    fold_consistency_pass?: boolean;
  };
  drawdown_curve?: Array<{ bar_index: number; value: number }>;
  buy_hold_curve?: Array<{ bar_index: number; value: number }>;
  monthly_returns?: Array<{ month: number; return_pct: number }>;
  phase3_fold_consistency_cv: number;
  phase3_regime_coverage?: Record<string, unknown>;
  phase3_seed_robustness_sharpe: number;
  phase3_seed_robustness_seeds: number;
  phase3_seed_robustness_pass: boolean;
  trust_score?: TrustScoreResult;
  pbo: number;
  dsr: number;
  hpo_status?: Record<string, string>;
  hpo_trial_summaries?: Record<string, { model_type: string; status: string; best_score?: number; committee_size?: number; consensus_pool_size?: number }>;
  hpo_model_params_count?: number;
  snapshot_dir?: string;
  final_fold_consistency_cv: number;
  final_fold_consistency_pass: boolean;
  final_regime_coverage?: Record<string, unknown>;
  final_seed_robustness_sharpe: number;
  final_seed_robustness_pass: boolean;
  final_full_wfo?: Record<string, unknown>;
  factory_best_sharpe: number;
  factory_total_iterations: number;
  factory_accepted_count: number;
  factory_best_config?: Record<string, unknown>;
  factory_history: FactoryIterationRecord[];
  factory_stop_reason: string;
  total_time_s: number;
  phase_timings?: Record<string, number>;
}

export interface FullCycleHistoryEntry {
  job_id: string;
  started_at: string;
  status: string;
  total_time_s: number;
  locked_features_count: number;
  survivors_count: number;
  survivors: string[];
  avg_sharpe: number;
  trust_score: number;
  factory_best_sharpe: number;
}

export interface FullCycleHistoryResponse {
  entries: FullCycleHistoryEntry[];
  total_runs: number;
}

export interface CancelFullCycleResponse {
  status: string;
}

export interface LogEntry {
  index: number;
  timestamp: string;
  level: string;
  message: string;
  phase?: string;
  phase_number?: number;
  phase_progress?: string;
  category?: string;
  metrics?: Record<string, unknown>;
}

export interface LogsResponse {
  entries: LogEntry[];
  next_index: number;
}

// ════════════════════════════════════════════════════════════════════
// Fast Loop Retrain (committee weight refit)
// ════════════════════════════════════════════════════════════════════

export interface RetrainRequest {
  lookback_bars?: number;
  oos_frac?: number;
}

export interface RetrainStartedResponse {
  session_id: string;
  status: string;
  started_at: string;
}

export interface RetrainStatus {
  session_id: string;
  status: "idle" | "running" | "complete" | "failed";
  progress: number;
  current_phase: string;
  started_at: string | null;
  completed_at: string | null;
  models_refitted: string[];
  models_skipped: string[];
  meta_labeler_refitted: boolean;
  meta_accuracy: number | null;
  elapsed_seconds: number | null;
  error: string | null;
}

export interface RetrainProgressEvent {
  event: "retrain_progress";
  phase: string;
  progress: number;
}

export interface RetrainCompleteEvent {
  event: "retrain_complete";
  models_refitted: string[];
  models_skipped: string[];
  meta_accuracy: number | null;
  elapsed_seconds: number | null;
}

export interface RetrainFailedEvent {
  event: "retrain_failed";
  error: string;
}
