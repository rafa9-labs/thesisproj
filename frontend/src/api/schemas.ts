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
  total_return: number | null;
  win_rate: number | null;
  total_trades: number | null;
  profit_factor: number | null;
  avg_trade: number | null;
}

export interface BacktestRequest {
  pair: string;
  models: string[];
  start_date?: string;
  end_date?: string;
  trading_costs?: boolean;
  months?: number;
  repeats?: number;
  config_overrides: Record<string, unknown>;
}

export interface JobSummary {
  job_id: string;
  type: string;
  status: "pending" | "running" | "completed" | "failed";
  pair: string | null;
  models: string[] | null;
  created_at: string;
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
  metrics: Metrics[];
  monthly_results: MonthlyResult[] | null;
  trades: TradeRecord[] | null;
  equity_curve: EquityPoint[] | null;
  buy_hold_curve: EquityPoint[] | null;
  drawdown_curve: EquityPoint[] | null;
  config: Record<string, unknown> | null;
  hpo_param_importance: HpoParamImportance[] | null;
  hpo_trials: HpoTrial[] | null;
}

export type WsEvent =
  | { event: "job_started"; job_id: string; pair: string; models: string[] }
  | { event: "model_training"; job_id: string; model: string; status: "starting" }
  | { event: "model_training"; job_id: string; model: string; status: "complete"; metrics: Partial<Metrics> }
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
