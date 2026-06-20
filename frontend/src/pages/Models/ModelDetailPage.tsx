import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import type { DeployedModelDetail } from "@/api/schemas";
import {
  ArrowLeft, Play, Cpu, ListTree, Activity, Shield,
  Target, Gauge, AlertTriangle,
} from "lucide-react";

export function ModelDetailPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();

  const { data: model, isLoading } = useQuery({
    queryKey: ["deployed-model", modelId],
    queryFn: async () => {
      const { data } = await apiClient.get<DeployedModelDetail>(`/models/deployed/${modelId}`);
      return data;
    },
    enabled: !!modelId,
  });

  if (isLoading || !model) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 p-12">
        <span className="text-xs text-(--color-text-muted)">
          {isLoading ? "Loading..." : "Model not found"}
        </span>
      </div>
    );
  }

  const of = model.overfitting;

  return (
    <div className="flex flex-col gap-5 p-6">
      {/* ── Back nav ── */}
      <button
        onClick={() => navigate("/models")}
        className="flex items-center gap-2 text-[11px] text-(--color-text-muted) hover:text-(--color-text-secondary) transition-colors w-fit"
      >
        <ArrowLeft size={14} /> Back to Models
      </button>

      {/* ── Header ── */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="rounded-md bg-cyan-500/12 px-3 py-1 text-[11px] font-bold tracking-[0.06em] text-cyan-400 uppercase">
          {model.model_type}
        </span>
        <span className="font-mono text-[10px] text-(--color-text-dim)">{model.id.slice(0, 12)}...</span>
        {model.pair && (
          <span className="font-mono text-[10px] text-(--color-text-muted)">
            {model.pair}/{model.timeframe || "H1"}
          </span>
        )}
        {of && (
          <RiskBadge level={of.risk_level} score={of.overfit_score} />
        )}
        {model.status && (
          <span className={[
            "rounded-full px-2 py-0.5 text-[9px] font-semibold tracking-[0.06em] uppercase",
            model.status === "active"
              ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
              : "bg-(--color-glass) text-(--color-text-muted) border border-(--color-glass-border)",
          ].join(" ")}>
            {model.status}
          </span>
        )}
      </div>

      {/* ── Performance KPIs ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-2">
        <KpiCard label="Sharpe" value={fmtNum(model.best_sharpe)} ok={model.best_sharpe != null && model.best_sharpe >= 0} />
        <KpiCard label="Sortino" value={fmtNum(model.sortino)} ok={model.sortino != null && model.sortino >= 0} />
        <KpiCard label="Return" value={fmtPct(model.best_return, 1)} ok={model.best_return != null && model.best_return >= 0} />
        <KpiCard label="Max DD" value={fmtPct(model.max_drawdown)} ok={(model.max_drawdown ?? 0) > -10} />
        <KpiCard label="Win Rate" value={model.win_rate != null ? (model.win_rate * 100).toFixed(1) + "%" : "\u2014"} />
        <KpiCard label="Trades" value={model.total_trades != null ? String(model.total_trades) : "\u2014"} />
        <KpiCard label="Calmar" value={fmtNum(model.calmar_ratio)} ok={model.calmar_ratio != null && model.calmar_ratio >= 0} />
        <KpiCard label="CAGR" value={fmtPct(model.cagr, 1)} ok={model.cagr != null && model.cagr >= 0} />
        <KpiCard label="Profit Factor" value={fmtNum(model.profit_factor)} ok={model.profit_factor != null && model.profit_factor >= 1} />
        <KpiCard label="Directional" value={model.directional_accuracy != null ? (model.directional_accuracy * 100).toFixed(1) + "%" : "\u2014"} />
        <KpiCard label="Active Rate" value={model.active_rate != null ? (model.active_rate * 100).toFixed(1) + "%" : "\u2014"} />
        <KpiCard label="Avg Trade" value={fmtPct(model.avg_trade, 2)} ok={model.avg_trade != null && model.avg_trade >= 0} />
      </div>

      {/* ── Two-column main area ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* ── LEFT: Reliability & Overfitting ── */}
        <Section title="Reliability & Overfitting Assessment" icon={<Shield size={13} />}>
          {of ? (
            <div className="flex flex-col gap-4">
              {/* Overfit score gauge */}
              <OverfitGauge score={of.overfit_score} riskColor={of.risk_color} />

              {/* Gap + Degradation + CV stability */}
              <div className="grid grid-cols-2 gap-3">
                <StatRow
                  label="Train/OOS Gap"
                  value={of.train_oos_gap_pct.toFixed(1) + "%"}
                  sub={of.is_mean_sharpe != null && of.oos_mean_sharpe != null
                    ? `IS ${fmtNum(of.is_mean_sharpe)} / OOS ${fmtNum(of.oos_mean_sharpe)}`
                    : undefined}
                  warn={of.train_oos_gap_pct > 20}
                />
                <StatRow
                  label="Temporal Decay"
                  value={of.temporal_degradation_pct.toFixed(1) + "%"}
                  sub="1st vs 2nd half Sharpe"
                  warn={of.temporal_degradation_pct > 15}
                />
                <StatRow
                  label="CV Sharpe Mean"
                  value={of.cv_sharpe_mean != null ? fmtNum(of.cv_sharpe_mean) : "\u2014"}
                  sub={of.cv_sharpe_std != null ? `\u00B1${of.cv_sharpe_std.toFixed(2)} std` : undefined}
                  warn={of.cv_sharpe_std != null && of.cv_sharpe_std > 0.3}
                />
                <StatRow
                  label="Signal Gap"
                  value={of.signal_gap_pct.toFixed(1) + "%"}
                  sub={`${of.n_signal_periods}/${of.n_periods} active periods`}
                  warn={of.signal_gap_pct > 50}
                />
              </div>

              {/* DSR threshold */}
              {of.dsr_min_sharpe != null && (
                <div className="rounded-md border border-(--color-glass-border) bg-(--color-glass) p-3">
                  <div className="flex items-center gap-2 text-[10px] text-(--color-text-muted)">
                    <Target size={12} />
                    DSR Minimum Sharpe (95% confidence)
                  </div>
                  <div className="mt-1 flex items-baseline gap-2">
                    <span className="font-mono text-[18px] font-semibold text-amber-400">
                      {of.dsr_min_sharpe.toFixed(3)}
                    </span>
                    <span className="text-[10px] text-(--color-text-muted)">
                      {model.best_sharpe != null && model.best_sharpe >= of.dsr_min_sharpe
                        ? "Actual Sharpe exceeds threshold"
                        : "Actual Sharpe below significance threshold"}
                    </span>
                  </div>
                </div>
              )}

              {/* Bootstrap CIs */}
              <div className="grid grid-cols-3 gap-2">
                {of.sharpe_ci && <CiCard label="Sharpe (90% CI)" ci={of.sharpe_ci} />}
                {of.return_ci && <CiCard label="Return (90% CI)" ci={of.return_ci} fmt={(v) => fmtPct(v, 1)} />}
                {of.maxdd_ci && <CiCard label="MaxDD (90% CI)" ci={of.maxdd_ci} fmt={(v) => fmtPct(v, 1)} />}
              </div>

              {/* fANOVA interactions */}
              {of.interaction_effects && of.interaction_effects.length > 0 && (
                <details className="cursor-pointer group">
                  <summary className="text-[10px] text-(--color-text-muted) select-none hover:text-(--color-text-secondary)">
                    fANOVA Interaction Effects ({of.interaction_effects.length})
                  </summary>
                  <div className="mt-2 flex flex-col gap-1">
                    {of.interaction_effects.slice(0, 5).map((ix, i) => (
                      <div key={i} className="flex items-center justify-between rounded bg-(--color-glass) px-2 py-1 text-[10px]">
                        <span className="font-mono text-(--color-text-secondary) truncate">{ix.param}</span>
                        <span className="text-cyan-400 tabular-nums ml-2 shrink-0">
                          {ix.interaction_pct.toFixed(1)}% interaction
                        </span>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-[11px] text-(--color-text-muted)">
              <AlertTriangle size={14} />
              Overfitting analysis not available for this model. Run a new backtest with sufficient walk-forward periods.
            </div>
          )}
        </Section>

        {/* ── RIGHT: Training Configuration ── */}
        <Section title="Training Configuration" icon={<Activity size={13} />}>
          <div className="grid grid-cols-2 gap-3">
            <ConfigItem label="Seed" value={model.seed != null ? String(model.seed) : "\u2014"} />
            <ConfigItem label="Calibration" value={model.calibrate_method || "\u2014"} />
            <ConfigItem label="Feature Count" value={model.feature_count != null ? String(model.feature_count) : "\u2014"} />
            <ConfigItem label="Coverage Thr" value={model.coverage_conf_thr != null ? model.coverage_conf_thr.toFixed(3) : "\u2014"} />
            {model.input_shape && (
              <ConfigItem label="Input Shape" value={model.input_shape.join("\u00D7")} />
            )}
            {model.pair && (
              <ConfigItem label="Pair / TF" value={`${model.pair}/${model.timeframe || "H1"}`} />
            )}
            {(model.train_start || model.train_end) && (
              <ConfigItem
                label="Training Window"
                value={`${model.train_start?.slice(0, 10) || "?"} \u2192 ${model.train_end?.slice(0, 10) || "?"}`}
              />
            )}
            {model.schema_version != null && (
              <ConfigItem label="Schema" value={`v${model.schema_version}`} />
            )}
            {model.parent_job_id && (
              <ConfigItem label="Parent Job" value={model.parent_job_id.slice(0, 12) + "..."} />
            )}
          </div>

          {/* Walk-forward info */}
          {of && (
            <div className="mt-4 border-t border-(--color-glass-border) pt-3">
              <span className="text-[9px] font-semibold tracking-[0.08em] text-(--color-text-muted) uppercase">
                Walk-Forward Summary
              </span>
              <div className="mt-2 grid grid-cols-3 gap-2">
                <MiniStat label="Periods" value={String(of.n_periods)} />
                <MiniStat label="Active" value={String(of.n_signal_periods)} />
                <MiniStat label="Min Trades" value={String(of.min_trl_trades)} />
              </div>
            </div>
          )}
        </Section>
      </div>

      {/* ── Hyperparameters ── */}
      {(() => {
        const hp = model.best_params && typeof model.best_params === "object"
          ? Object.entries(model.best_params as Record<string, unknown>)
          : [];
        if (hp.length === 0) return null;
        return (
          <Section title="Hyperparameters" icon={<Cpu size={13} />}>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-x-4 gap-y-1.5">
              {hp.map(([k, v]) => (
                <div key={k} className="flex justify-between items-baseline gap-2 py-0.5">
                  <span className="text-[10px] text-(--color-text-muted) truncate">{k}</span>
                  <span className="font-mono text-[11px] text-(--color-text-secondary) shrink-0">
                    {typeof v === "number" ? (v % 1 === 0 ? v : v.toFixed(4)) : String(v)}
                  </span>
                </div>
              ))}
            </div>
          </Section>
        );
      })()}

      {/* ── Features ── */}
      {model.feature_names.length > 0 && (
        <Section title={`Features (${model.feature_names.length})`} icon={<ListTree size={13} />}>
          <FeatureGroups names={model.feature_names} />
        </Section>
      )}

      {/* ── Tags ── */}
      {model.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {model.tags.map((t) => (
            <span key={t} className="rounded bg-(--color-glass) px-2 py-0.5 font-mono text-[9px] text-(--color-text-muted)">
              {t}
            </span>
          ))}
        </div>
      )}

      {/* ── Environment ── */}
      {model.pip_freeze && (
        <Section title="Environment">
          <details className="cursor-pointer">
            <summary className="text-[10px] text-(--color-text-muted) select-none hover:text-(--color-text-secondary)">
              pip freeze (schema v{model.schema_version ?? 1})
            </summary>
            <pre className="mt-2 max-h-[160px] overflow-auto rounded border border-(--color-glass-border) bg-(--color-input-bg) p-3 font-mono text-[9px] text-(--color-text-muted)">
              {model.pip_freeze}
            </pre>
          </details>
        </Section>
      )}

      {/* ── Actions ── */}
      <div className="flex gap-3 pt-1">
        <button
          onClick={() => navigate(`/trading?modelId=${model.id}`)}
          className="flex items-center gap-2 rounded-md bg-cyan-600 px-4 py-2 text-[11px] font-semibold tracking-[0.06em] text-white uppercase hover:brightness-110 transition-all"
        >
          <Play size={12} /> Deploy
        </button>
        <button
          onClick={() => navigate(`/results/${model.parent_job_id}`)}
          disabled={!model.parent_job_id}
          className="rounded-md border border-(--color-glass-border) bg-(--color-glass) px-4 py-2 text-[11px] font-semibold tracking-[0.06em] text-(--color-text-secondary) uppercase disabled:opacity-40 hover:brightness-110 transition-all"
        >
          View Backtest
        </button>
      </div>
    </div>
  );
}

// ── Sub-components ──

function Section({ title, children, icon }: { title: string; children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-(--color-glass-border) bg-(--color-glass) p-4 backdrop-blur-[12px]">
      <div className="mb-3 flex items-center gap-2">
        {icon}
        <span className="text-[10px] font-semibold tracking-[0.08em] text-(--color-text-muted) uppercase">
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function KpiCard({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  const color = ok === undefined ? "text-(--color-text-secondary)" : ok ? "text-emerald-400" : "text-rose-400";
  return (
    <div className="rounded-sm border border-(--color-glass-border) bg-(--color-glass) p-3 backdrop-blur-[12px]">
      <div className={`font-mono text-[15px] font-semibold ${color}`}>{value}</div>
      <div className="mt-0.5 text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">{label}</div>
    </div>
  );
}

function ConfigItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] font-medium tracking-[0.04em] text-(--color-text-muted) uppercase">{label}</span>
      <span className="font-mono text-[12px] text-(--color-text-secondary)">{value}</span>
    </div>
  );
}

function StatRow({
  label, value, sub, warn,
}: {
  label: string;
  value: string;
  sub?: string;
  warn?: boolean;
}) {
  return (
    <div className="rounded-md border border-(--color-glass-border) bg-(--color-glass) p-2.5">
      <span className="text-[9px] font-medium tracking-[0.04em] text-(--color-text-muted) uppercase">{label}</span>
      <div className={`font-mono text-[14px] font-semibold mt-0.5 ${warn ? "text-amber-400" : "text-(--color-text-secondary)"}`}>
        {value}
      </div>
      {sub && <div className="text-[9px] text-(--color-text-dim) mt-0.5">{sub}</div>}
    </div>
  );
}

function CiCard({ label, ci, fmt }: { label: string; ci: { low: number | null; high: number | null; mean: number | null }; fmt?: (v: number) => string }) {
  const f = fmt ?? ((v: number) => v.toFixed(2));
  return (
    <div className="rounded-md border border-(--color-glass-border) bg-(--color-glass) p-2">
      <span className="text-[8px] font-medium tracking-[0.04em] text-(--color-text-muted) uppercase">{label}</span>
      <div className="font-mono text-[11px] text-(--color-text-secondary) mt-0.5">
        {ci.low != null && ci.high != null
          ? `${f(ci.low)} \u2013 ${f(ci.high)}`
          : "\u2014"}
      </div>
      {ci.mean != null && (
        <div className="font-mono text-[9px] text-(--color-text-dim)">
          mean {f(ci.mean)}
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[8px] text-(--color-text-dim)">{label}</span>
      <span className="font-mono text-[12px] text-(--color-text-secondary)">{value}</span>
    </div>
  );
}

function OverfitGauge({ score, riskColor }: { score: number; riskColor: string }) {
  const colorMap: Record<string, string> = { green: "bg-emerald-500", yellow: "bg-amber-500", red: "bg-rose-500" };
  const textMap: Record<string, string> = { green: "text-emerald-400", yellow: "text-amber-400", red: "text-rose-400" };
  const barColor = colorMap[riskColor] || "bg-(--color-glass-border)";
  const textColor = textMap[riskColor] || "text-(--color-text-muted)";

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <Gauge size={14} className={textColor} />
          <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
            Overfit Risk Score
          </span>
        </div>
        <span className={`font-mono text-[14px] font-bold ${textColor}`}>
          {score.toFixed(0)}/100
        </span>
      </div>
      <div className="h-2.5 w-full rounded-full bg-(--color-glass) border border-(--color-glass-border) overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${barColor}`}
          style={{ width: `${Math.min(score, 100)}%` }}
        />
      </div>
      <div className="flex justify-between mt-1">
        <span className="text-[8px] text-(--color-text-dim)">Low Risk</span>
        <span className="text-[8px] text-(--color-text-dim)">Medium</span>
        <span className="text-[8px] text-(--color-text-dim)">High Risk</span>
      </div>
    </div>
  );
}

function RiskBadge({ level, score }: { level: string; score: number }) {
  const colors: Record<string, string> = {
    low: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    medium: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    high: "bg-rose-500/10 text-rose-400 border-rose-500/20",
  };
  return (
    <span className={[
      "rounded-full border px-2 py-0.5 text-[9px] font-semibold tracking-[0.06em] uppercase",
      colors[level] || "bg-(--color-glass) text-(--color-text-muted) border-(--color-glass-border)",
    ].join(" ")}>
      {level} risk ({score.toFixed(0)})
    </span>
  );
}

// ── Helpers ──

const fmtPct = (v: number | null | undefined, decimals = 1) => {
  if (v == null) return "\u2014";
  return (v >= 0 ? "+" : "") + v.toFixed(decimals) + "%";
};

const fmtNum = (v: number | null | undefined, decimals = 3) => {
  if (v == null) return "\u2014";
  return (v >= 0 ? "+" : "") + v.toFixed(decimals);
};

const FEATURE_CATEGORIES: [RegExp, string][] = [
  [/price/i, "Price"],
  [/rsi|stoch|cci|momentum|macd|adx|aroon|dmi/i, "Momentum"],
  [/sma|ema|wma|boll|keltner|envelope|ma_/i, "Moving Averages"],
  [/atr|volatility|bb_/i, "Volatility"],
  [/volume|obv|adosc|mfi/i, "Volume"],
  [/sentiment|news|vader|finbert/i, "Sentiment"],
  [/regime/i, "Regime"],
  [/lag_|shift|return_|log_ret|pct/i, "Returns & Lags"],
  [/ratio|spread|corr/i, "Ratios"],
];

function FeatureGroups({ names }: { names: string[] }) {
  const byCategory = new Map<string, string[]>();
  for (const name of names) {
    let cat = "Other";
    for (const [re, label] of FEATURE_CATEGORIES) {
      if (re.test(name)) { cat = label; break; }
    }
    const existing = byCategory.get(cat);
    if (existing) existing.push(name);
    else byCategory.set(cat, [name]);
  }

  return (
    <div className="flex flex-col gap-3">
      {[...byCategory.entries()].map(([cat, ns]) => (
        <div key={cat}>
          <span className="text-[9px] font-semibold tracking-[0.08em] text-(--color-text-muted) uppercase">
            {cat} ({ns.length})
          </span>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {ns.map((n) => (
              <span key={n} className="rounded bg-(--color-glass) px-2 py-0.5 font-mono text-[9px] text-(--color-text-secondary)">
                {n}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
