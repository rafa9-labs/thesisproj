import { useMemo } from "react";
import { useBacktestStore } from "@/stores/useBacktestStore";
import { usePairs } from "@/api/queries";

interface ValidationResult {
  warnings: string[];
  errors: string[];
  ok: boolean;
}

export function useValidation(): ValidationResult {
  const s = useBacktestStore();
  const { data: pairs } = usePairs();

  const selected = pairs?.find((p) => p.pair.symbol === s.pair);
  const tfData = selected?.timeframes.find((t) => t.timeframe === s.timeframe);
  const dataMin = tfData?.start_date?.slice(0, 10);
  const dataMax = tfData?.end_date?.slice(0, 10);

  return useMemo(() => {
    const warnings: string[] = [];
    const errors: string[] = [];

    // ── Models ──
    if (s.selectedModels.length === 0) {
      errors.push("Select at least one model to run a backtest");
    }
    if (s.selectedModels.length > 5) {
      errors.push("Maximum 5 models per backtest run");
    }

    // ── Walk-forward parameters ──
    if (s.trainMonths < 6) {
      errors.push(`Training window (${s.trainMonths}mo) is too short — minimum is 6 months`);
    }
    if (s.trainMonths > 60) {
      errors.push(`Training window (${s.trainMonths}mo) exceeds maximum of 60 months`);
    }
    if (s.testMonths < 1) {
      errors.push(`Test window (${s.testMonths}mo) must be at least 1 month`);
    }
    if (s.testMonths > 6) {
      errors.push(`Test window (${s.testMonths}mo) exceeds maximum of 6 months`);
    }
    if (s.repeats < 1) {
      errors.push("Walk-forward repeats must be at least 1");
    }
    if (s.repeats > 10) {
      errors.push(`Walk-forward repeats (${s.repeats}) exceeds maximum of 10`);
    }

    // ── HPO ──
    if (s.nTrials < 0) {
      errors.push("HPO trials cannot be negative");
    }
    if (s.nTrials > 500) {
      errors.push(`HPO trials (${s.nTrials}) exceeds maximum of 500`);
    }

    // ── Confidence threshold ──
    if (s.confidenceThreshold < 0 || s.confidenceThreshold > 1) {
      errors.push("Confidence threshold must be between 0 and 1");
    }

    // ── Triple barrier ──
    if (s.useTripleBarrier) {
      if (s.tbPtMult < s.tbSlMult) {
        warnings.push(
          `PT mult (${s.tbPtMult}) < SL mult (${s.tbSlMult}) — strategy risks more than it aims to gain`,
        );
      }
      if (s.tbNeutralZone > 1.0) {
        warnings.push("Neutral zone > 1.0 — many trades may expire at neutral");
      }
      if (s.tbMaxHolding < 12) {
        warnings.push("Max holding < 12 bars — may not give trades enough time");
      }
    }

    // ── Labels ──
    if (s.labelThreshold < 0.0002) {
      warnings.push("Label threshold < 0.0002 — may label noise as directional");
    }
    if (s.labelThreshold > 0.001) {
      warnings.push("Label threshold > 0.001 — very few directional labels");
    }

    // ── Features ──
    const featureCount = s.lags * s.lagDepth;
    if (featureCount > 100) {
      errors.push(`Lag feature count (${featureCount}) exceeds 100 — likely overfitting`);
    } else if (featureCount > 60) {
      warnings.push(`Lag feature count (${featureCount}) > 60 — consider reducing`);
    }

    if ((s.useSqueezeBreakout || s.useSqueezeExpansion) && !s.useBbands) {
      errors.push("Squeeze features require Bollinger Bands enabled");
    }
    if (s.useMacdAtrRatio && (!s.useMacd || !s.useAtr)) {
      errors.push("MACD/ATR ratio requires both MACD and ATR enabled");
    }
    if (s.usePriceMaZ && !s.useSma && !s.useEma) {
      errors.push("Price-MA Z-Score requires SMA or EMA enabled");
    }
    if (s.useMtfAlignment && !s.useMtfMa) {
      errors.push("MTF Alignment requires MTF MA enabled");
    }
    if (s.useTripleConfirm && !s.useRsi && !s.useMacd && !s.useSma && !s.useEma) {
      errors.push("Triple Confirm requires at least one of RSI, MACD, SMA, or EMA enabled");
    }
    if (s.useTrendConfirm && !s.useSma && !s.useEma) {
      errors.push("Trend Confirm requires SMA or EMA enabled");
    }
    if (s.fracdiffD < 0 || s.fracdiffD > 1) {
      errors.push("FracDiff d must be between 0.0 and 1.0");
    }

    // ── Dimensionality guard (KodaQuant v2.0 — user owns feature thesis) ──
    const enabledFeatureCount = [
      s.useAdx, s.useAtr, s.useBbands, s.useEma, s.useSma, s.useRsi,
      s.useMacd, s.useStoch, s.useSar, s.useDonchian,
      s.useFracdiff, s.useCrossoverBins, s.useMaSpread, s.usePriceMaZ,
      s.useIndicatorStates, s.useMtfMa, s.useMtfAlignment, s.useMtfAlign,
      s.useMacdAtrRatio, s.useTripleConfirm, s.useTrendConfirm,
      s.useVolManagedMom, s.useVmMom, s.useSqueezeBreakout,
      s.useSqueezeExpansion, s.useAtrChannelBreakout, s.useExtAtrLowAdx,
      s.useReentryMom, s.useSlopeDiff, s.useRvFeatures, s.useNews,
    ].filter(Boolean).length;
    if (enabledFeatureCount > 24) {
      errors.push(
        `${enabledFeatureCount} features enabled — severe overfitting risk (max 24 recommended)`,
      );
    } else if (enabledFeatureCount > 18) {
      warnings.push(
        `${enabledFeatureCount} features enabled — high dimensionality (consider reducing)`,
      );
    }

    // ── Coverage ──
    if (Math.abs(s.targetActiveRate - s.targetCoverage) > 0.01) {
      warnings.push("Target active rate and coverage differ by > 1%");
    }
    if (s.targetActiveRate > 0.3) {
      warnings.push("Target active rate > 30% is very high");
    }
    if (s.targetActiveRate < 0.05) {
      warnings.push("Target active rate < 5% is very low");
    }

    // ── Logistic model ──
    if (s.selectedModels.includes("logistic")) {
      const incompatible: Record<string, string[]> = {
        lbfgs: ["l1", "elasticnet"],
        "newton-cg": ["l1", "elasticnet"],
        sag: ["l1", "elasticnet"],
        liblinear: ["elasticnet", "none"],
      };
      const blocked = incompatible[s.logitSolver];
      if (blocked?.includes(s.logitPenalty)) {
        errors.push(`Solver '${s.logitSolver}' does not support penalty '${s.logitPenalty}'`);
      }
      if (s.logitC < 0.001) warnings.push("C < 0.001 — very strong regularization");
      if (s.logitC > 10000) warnings.push("C > 10000 — very weak regularization");
    }

    // ── LLM sentiment ──
    if (s.llmSentimentEnabled && s.llmBackend !== "ollama" && !s.llmApiKey) {
      errors.push(`LLM sentiment is enabled but no API key is set for ${s.llmBackend}`);
    }

    // ── Date range ──
    if (s.startDate && dataMin && s.startDate < dataMin) {
      errors.push(`Start date (${s.startDate}) is before available data (${dataMin})`);
    }
    if (s.endDate && dataMax && s.endDate > dataMax) {
      errors.push(`End date (${s.endDate}) is after available data (max: ${dataMax})`);
    }
    if (s.startDate && s.endDate && s.startDate > s.endDate) {
      errors.push("Start date must be before end date");
    }
    if (s.startDate && s.endDate && dataMin && dataMax) {
      const rangeDays =
        (new Date(s.endDate).getTime() - new Date(s.startDate).getTime()) / 86400000;
      const trainDays = s.trainMonths * 30;
      const testDays = s.testMonths * 30;
      if (rangeDays < trainDays + testDays) {
        warnings.push(
          `Selected range (${Math.round(rangeDays / 30)}mo) may be too short for ${s.trainMonths}mo train + ${s.testMonths}mo test`,
        );
      }
    }

    return { warnings, errors, ok: errors.length === 0 };
  }, [
    s.useTripleBarrier,
    s.tbPtMult,
    s.tbSlMult,
    s.tbNeutralZone,
    s.tbMaxHolding,
    s.labelThreshold,
    s.lags,
    s.lagDepth,
    s.useSqueezeBreakout,
    s.useSqueezeExpansion,
    s.useBbands,
    s.useMacdAtrRatio,
    s.useMacd,
    s.useAtr,
    s.usePriceMaZ,
    s.useSma,
    s.useEma,
    s.useMtfAlignment,
    s.useMtfMa,
    s.useTripleConfirm,
    s.useTrendConfirm,
    s.useRsi,
    s.fracdiffD,
    s.targetActiveRate,
    s.targetCoverage,
    s.selectedModels,
    s.logitSolver,
    s.logitPenalty,
    s.logitC,
    s.trainMonths,
    s.testMonths,
    s.repeats,
    s.nTrials,
    s.confidenceThreshold,
    s.llmSentimentEnabled,
    s.llmBackend,
    s.llmApiKey,
    s.startDate,
    s.endDate,
    s.useAdx,
    s.useDonchian,
    s.useStoch,
    s.useSar,
    s.useFracdiff,
    s.useCrossoverBins,
    s.useMaSpread,
    s.useIndicatorStates,
    s.useMtfAlign,
    s.useVmMom,
    s.useAtrChannelBreakout,
    s.useExtAtrLowAdx,
    s.useReentryMom,
    s.useSlopeDiff,
    s.useRvFeatures,
    s.useNews,
    s.useVolManagedMom,
    dataMin,
    dataMax,
  ]);
}
