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
  interaction_effects: Array<{ param: string; main_pct: number; interaction_pct: number; total_pct: number }> | null;
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
  overfitting: OverfittingReportData | null;
  walkforward_periods: WalkForwardPeriod[] | null;
  diagnostics: TrainingDiagnostics | null;
  summary_text: string | null;
  snapshot_path?: string | null;
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
}

export interface LlmAnalysisResponse {
  job_id: string;
  model: string;
  analysis: LlmAnalysis;
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
  | { event: "hpo_trial_result"; job_id: string; model: string; trial_number: number; score: number | null; params: Record<string, unknown>; best_score_so_far: number | null; trial_state: string }
  | { event: "month_progress"; job_id: string; model: string; month?: number; total_months?: number; period?: number; total_periods?: number; sharpe?: number; trades?: number; completed_work: number; total_work: number; progress_pct: number }
  | { event: "oos_result"; job_id: string; model: string; period: number; total_periods: number; equity: number | null; equity_bh: number | null; sharpe: number | null; return_pct: number | null; trades: number | null; drawdown: number | null; win_rate: number | null; precision: number | null; f1: number | null; directional_accuracy: number | null; active_rate: number | null; flat?: boolean; train_sharpe?: number | null; sharpe_gap_pct?: number | null; signals_raw?: number; signals_passed_gate?: number }
  | { event: "job_complete"; job_id: string; metrics: Partial<Metrics>[] }
  | { event: "job_failed"; job_id: string; error: string }
  | { event: "cycle_started"; job_id: string; model: string; cycle_number: number; total_cycles: number }
  | { event: "download_started"; job_id: string; pair: string }
  | { event: "download_complete"; job_id: string; pair: string }
  | { event: "download_failed"; job_id: string; error: string }
  | { event: "simulation_started"; job_id: string; model: string; n_periods: number; bh_curve: { period: number; bh: number }[] };

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
  return_pct: number | null;
  trades: number | null;
  drawdown: number | null;
  win_rate: number | null;
  precision: number | null;
  f1: number | null;
  flat?: boolean;
  train_sharpe?: number | null;
  sharpe_gap_pct?: number | null;
  signals_raw?: number;
  signals_passed_gate?: number;
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
  body: string;
  source: string;
  url: string;
  pair_tags: string[];
  sentiment_score: number;
  summary: string;
  bias: string;
  timestamp: string;
}

export interface LiveSentimentResponse {
  pairs: Record<string, LiveSentimentPairData>;
  top_articles: LiveSentimentArticle[];
  backend: string;
  model: string;
  error?: string;
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
}

export interface LiveSignalEvent {
  event: "signal" | "hold" | "order_placed" | "risk_blocked" | "trade_closed" | "kill" | "heartbeat" | "stopped" | "error";
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
// Committee (Racecar Phases A-E)
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

export interface CommitteeBacktestRequest {
  config: CommitteeConfigSchema;
  pair?: string;
  timeframe?: string;
  train_months?: number;
  test_months?: number;
  confidence_threshold?: number;
  seq_len?: number;
}

export interface CommitteeBacktestSubmitResponse {
  job_id: string;
  status: string;
}

export interface CommitteeFoldData {
  fold_idx: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  sharpe: number;
  trades: number;
  active_rate: number;
  win_rate: number;
  return_val: number;
  drawdown: number;
  regime_distribution?: Record<string, number>;
}

export interface CommitteeBacktestResultResponse {
  job_id: string;
  status: string;
  total_folds: number;
  avg_sharpe: number;
  avg_trades: number;
  models: string[];
  warnings: string[];
  folds: CommitteeFoldData[];
  execution_time_s: number;
}

export interface CommitteeSnapshotInfo {
  version: string;
  created_at: string;
  models: string[];
}

export interface CommitteeSnapshotListResponse {
  snapshots: CommitteeSnapshotInfo[];
}

// ════════════════════════════════════════════════════════════════════
// Racecar Auto-Optimize (B→C→D pipeline)
// ════════════════════════════════════════════════════════════════════

export interface RacecarAutoOptimizeRequest {
  models: string[];
  pair?: string;
  timeframe?: string;
  train_months?: number;
  test_months?: number;
  profile_trials?: number;
  committee_top_k?: number;
}

export interface RacecarJobStatus {
  job_id: string;
  phase: string;
  phase_progress: string;
  started_at: string;
  error: string;
}

export interface RacecarJobResults {
  job_id: string;
  status: string;
  profile_matrix?: Record<string, unknown>;
  committee_config?: Record<string, unknown>;
  backtest?: Record<string, unknown>;
  total_time_s: number;
}

// ════════════════════════════════════════════════════════════════════
// Factory — Iterative Committee Optimizer
// ════════════════════════════════════════════════════════════════════

export interface FactoryStartRequest {
  models: string[];
  proposer?: string;
  llm_backend?: string;
  max_iterations?: number;
  patience?: number;
  stopping_tolerance?: number;
  regime_sharpe_floor?: number;
  train_months?: number;
}

export interface FactoryIterationRecord {
  iteration: number;
  action_type: string;
  regime: string;
  model_add: string;
  model_remove: string;
  before_sharpe: number;
  after_sharpe: number;
  delta_sharpe: number;
  accepted: boolean;
  rationale: string;
}

export interface FactoryStatusResponse {
  job_id: string;
  phase: string;
  iteration: number;
  total_iterations: number;
  current_action: string;
  current_regime: string;
  before_sharpe: number;
  after_sharpe: number;
  delta_sharpe: number;
  accepted: boolean;
  best_sharpe_so_far: number;
  stopped: boolean;
  stop_reason: string;
  history: FactoryIterationRecord[];
}

export interface FactoryResultsResponse {
  job_id: string;
  status: string;
  best_sharpe: number;
  total_iterations: number;
  accepted_count: number;
  total_time_s: number;
  best_config?: Record<string, unknown>;
  history: FactoryIterationRecord[];
  stop_reason: string;
}
