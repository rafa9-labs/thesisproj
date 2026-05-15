import { create } from "zustand";
import { DEFAULTS, STUDY_PRESETS, QUICK_START_CATEGORIES } from "@/lib/constants";
import type { BacktestRequest, HpoIntensity } from "@/api/schemas";

const _CUSTOM_PRESETS_KEY = "kodaquant-custom-presets";

function _loadCustomPresets(): Record<string, { name: string; subtitle: string; date: string }> {
  try {
    return JSON.parse(localStorage.getItem(_CUSTOM_PRESETS_KEY) || "{}");
  } catch { return {}; }
}
function _saveCustomPresets(data: Record<string, { name: string; subtitle: string; date: string }>) {
  try { localStorage.setItem(_CUSTOM_PRESETS_KEY, JSON.stringify(data)); } catch {}
}

type Widen<T> = T extends boolean ? boolean : T extends string ? string : T extends number ? number : T;
type BacktestState = { -readonly [K in keyof typeof DEFAULTS]: Widen<(typeof DEFAULTS)[K]> } & {
  hpoIntensity: HpoIntensity;
  activePreset: string | null;
  customPresets: Record<string, { name: string; subtitle: string; date: string }>;
};
type BacktestActions = {
  setField: <K extends keyof BacktestState>(key: K, value: BacktestState[K]) => void;
  toggleModel: (model: string) => void;
  resetToDefaults: () => void;
  applyPreset: (preset: { pair?: string; timeframe?: string; models?: string[]; months?: number; hpo_intensity?: HpoIntensity; seed?: number; start_date?: string; end_date?: string }) => void;
  applyStudyPreset: (presetKey: string) => void;
  applyQuickPreset: (presetKey: string) => void;
  saveCustomPreset: (name: string, subtitle: string) => void;
  removeCustomPreset: (key: string) => void;
  toRequestPayload: () => BacktestRequest;
};

const DEFAULT_HPO_INTENSITY: HpoIntensity = "quick";

export const useBacktestStore = create<BacktestState & BacktestActions>()((set, get) => ({
  ...structuredClone(DEFAULTS) as BacktestState,
  hpoIntensity: DEFAULT_HPO_INTENSITY,
  activePreset: null,
  customPresets: _loadCustomPresets(),

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
      return { ...updates, activePreset: null };
    }),

  applyStudyPreset: (presetKey) =>
    set((state) => {
      const p = STUDY_PRESETS[presetKey as keyof typeof STUDY_PRESETS];
      if (!p) return state;
      return {
        hpoIntensity: p.hpoIntensity,
        repeats: p.repeats,
        trainMonths: p.trainMonths,
        testMonths: p.testMonths,
        confidenceThreshold: p.confidenceThreshold,
        targetActiveRate: p.targetActiveRate,
        targetCoverage: p.targetCoverage,
        activePreset: presetKey,
      };
    }),

  applyQuickPreset: (presetKey) =>
    set((state) => {
      for (const cat of QUICK_START_CATEGORIES) {
        for (const opt of cat.options) {
          if (opt.key === presetKey) {
            return {
              selectedModels: opt.models,
              hpoIntensity: opt.hpoIntensity,
              nTrials: opt.nTrials,
              repeats: opt.repeats,
              trainMonths: opt.trainMonths,
              testMonths: opt.testMonths,
              confidenceThreshold: opt.confidenceThreshold,
              activePreset: null,
            };
          }
        }
      }
      return { selectedModels: [] };
    }),

  saveCustomPreset: (name, subtitle) =>
    set((state) => {
      const key = `custom-${Date.now()}`;
      const next = { ...state.customPresets, [key]: { name, subtitle, date: new Date().toISOString().slice(0, 10) } };
      _saveCustomPresets(next);
      return { customPresets: next };
    }),

  removeCustomPreset: (key) =>
    set((state) => {
      const next = { ...state.customPresets };
      delete next[key];
      _saveCustomPresets(next);
      return { customPresets: next };
    }),

  toRequestPayload: () => {
    const s = get();
    const configOverrides: Record<string, unknown> = {
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
      llm_sentiment_enabled: s.llmSentimentEnabled,
      llm_backend: s.llmBackend,
      llm_model: s.llmModel,
      llm_weight: s.llmWeight,
      llm_api_key: s.llmApiKey,
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
    };
    const stateKeys = Object.keys(s) as (keyof typeof s)[];
    for (const key of stateKeys) {
      if (String(key).includes("__")) {
        const val = s[key];
        if (val !== undefined && val !== null && val !== "") {
          configOverrides[String(key)] = val;
        }
      }
    }
    return {
      pair: s.pair,
      models: s.selectedModels,
      start_date: s.startDate || undefined,
      end_date: s.endDate || undefined,
      trading_costs: s.evalUseTradingCosts,
      months: s.testMonths,
      repeats: s.repeats ?? 1,
      seed: s.seed,
      hpo_intensity: s.hpoIntensity,
      config_overrides: configOverrides,
    };
  },
}));
