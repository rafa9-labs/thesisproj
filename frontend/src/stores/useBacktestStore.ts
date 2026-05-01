import { create } from "zustand";
import { DEFAULTS } from "@/lib/constants";
import type { BacktestRequest, HpoIntensity } from "@/api/schemas";

type Widen<T> = T extends boolean ? boolean : T extends string ? string : T extends number ? number : T;
type BacktestState = { -readonly [K in keyof typeof DEFAULTS]: Widen<(typeof DEFAULTS)[K]> } & {
  hpoIntensity: HpoIntensity;
};
type BacktestActions = {
  setField: <K extends keyof BacktestState>(key: K, value: BacktestState[K]) => void;
  toggleModel: (model: string) => void;
  resetToDefaults: () => void;
  applyPreset: (preset: { pair?: string; timeframe?: string; models?: string[]; months?: number; hpo_intensity?: HpoIntensity; seed?: number; start_date?: string; end_date?: string }) => void;
  toRequestPayload: () => BacktestRequest;
};

const DEFAULT_HPO_INTENSITY: HpoIntensity = "quick";

export const useBacktestStore = create<BacktestState & BacktestActions>()((set, get) => ({
  ...structuredClone(DEFAULTS) as BacktestState,
  hpoIntensity: DEFAULT_HPO_INTENSITY,

  setField: (key, value) => set({ [key]: value } as Partial<BacktestState>),

  toggleModel: (model) =>
    set((state) => {
      const current = state.selectedModels as string[];
      const next = current.includes(model)
        ? current.filter((m) => m !== model)
        : current.length >= 5
          ? current
          : [...current, model];
      return { selectedModels: next };
    }),

  resetToDefaults: () => set({ ...structuredClone(DEFAULTS), hpoIntensity: DEFAULT_HPO_INTENSITY } as Partial<BacktestState>),

  applyPreset: (preset) =>
    set((state) => {
      const updates: Partial<BacktestState> = {};
      if (preset.pair !== undefined) updates.pair = preset.pair;
      if (preset.timeframe !== undefined) updates.timeframe = preset.timeframe;
      if (preset.models !== undefined) updates.selectedModels = preset.models as string[];
      if (preset.months !== undefined) updates.testMonths = preset.months;
      if (preset.hpo_intensity !== undefined) updates.hpoIntensity = preset.hpo_intensity;
      if (preset.seed !== undefined) updates.seed = preset.seed;
      if (preset.start_date !== undefined) updates.startDate = preset.start_date;
      if (preset.end_date !== undefined) updates.endDate = preset.end_date;
      return updates;
    }),

  toRequestPayload: () => {
    const s = get();
    return {
      pair: s.pair,
      models: s.selectedModels,
      start_date: s.startDate || undefined,
      end_date: s.endDate || undefined,
      trading_costs: s.evalUseTradingCosts,
      months: s.testMonths,
      repeats: 1,
      seed: s.seed,
      hpo_intensity: s.hpoIntensity,
      config_overrides: {
        confidence_threshold: s.confidenceThreshold,
        target_active_rate: s.targetActiveRate,
        target_coverage: s.targetCoverage,
        calibrate_method: s.calibrateMethod,
        eval_use_trading_costs: s.evalUseTradingCosts,
        slip_norm_bps: s.slipNormBps,
        use_triple_barrier: s.useTripleBarrier,
        tb_pt_mult: s.tbPtMult,
        tb_sl_mult: s.tbSlMult,
        tb_neutral_zone: s.tbNeutralZone,
        tb_max_holding: s.tbMaxHolding,
        label_threshold: s.labelThreshold,
        lags: s.lags,
        lag_depth: s.lagDepth,
        use_fracdiff: s.useFracdiff,
        fracdiff_d: s.fracdiffD,
        use_adx: s.useAdx,
        use_atr: s.useAtr,
        use_bbands: s.useBbands,
        use_ema: s.useEma,
        use_sma: s.useSma,
        use_rsi: s.useRsi,
        use_macd: s.useMacd,
        use_stoch: s.useStoch,
        use_sar: s.useSar,
        use_donchian: s.useDonchian,
        use_mtf_ma: s.useMtfMa,
        use_crossover_bins: s.useCrossoverBins,
        use_ma_spread: s.useMaSpread,
        use_price_ma_z: s.usePriceMaZ,
        use_indicator_states: s.useIndicatorStates,
        use_mtf_alignment: s.useMtfAlignment,
        use_mtf_align: s.useMtfAlign,
        use_macd_atr_ratio: s.useMacdAtrRatio,
        use_triple_confirm: s.useTripleConfirm,
        use_trend_confirm: s.useTrendConfirm,
        use_vol_managed_mom: s.useVolManagedMom,
        use_vm_mom: s.useVmMom,
        use_squeeze_breakout: s.useSqueezeBreakout,
        use_squeeze_expansion: s.useSqueezeExpansion,
        use_atr_channel_breakout: s.useAtrChannelBreakout,
        use_ext_atr_low_adx: s.useExtAtrLowAdx,
        use_reentry_mom: s.useReentryMom,
        use_slope_diff: s.useSlopeDiff,
        use_rv_features: s.useRvFeatures,
        use_news: s.useNews,
        news_event_flags: s.newsEventFlags,
        news_sentiment_backend: s.newsSentimentBackend,
        logit_C: s.logitC,
        logit_solver: s.logitSolver,
        logit_penalty: s.logitPenalty,
        logit_max_iter: s.logitMaxIter,
        logit_tol: s.logitTol,
        sizing_method: s.sizingMethod,
        risk_fraction: s.riskFraction,
        kelly_fraction: s.kellyFraction,
        kelly_min_trades: s.kellyMinTrades,
        atr_risk_pct: s.atrRiskPct,
        atr_sl_mult: s.atrSlMult,
        initial_equity: s.initialEquity,
        max_leverage: s.maxLeverage,
        max_drawdown_pct: s.maxDrawdownPct,
        max_consecutive_losses: s.maxConsecutiveLosses,
        daily_loss_limit_pct: s.dailyLossLimitPct,
        trailing_method: s.trailingMethod,
        trailing_activation: s.trailingActivation,
      },
    };
  },
}));
