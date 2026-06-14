import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { ArrowLeft, Power, Play } from "lucide-react";

interface ModelDetail {
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
  train_start: string | null;
  train_end: string | null;
  feature_count: number | null;
}

export function ModelDetailPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();

  const { data: all, isLoading } = useQuery({
    queryKey: ["deployed-models"],
    queryFn: async () => {
      const { data } = await apiClient.get<{ models: ModelDetail[] }>("/models/deployed");
      return data.models;
    },
  });

  const model = all?.find((m) => m.id === modelId);

  if (isLoading || !model) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 p-12">
        <span className="text-xs text-(--color-text-muted)">
          {isLoading ? "Loading..." : "Model not found"}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      <button
        onClick={() => navigate("/models")}
        className="flex items-center gap-2 text-[11px] text-(--color-text-muted) hover:text-(--color-text-primary)"
      >
        <ArrowLeft size={14} /> Back to Models
      </button>

      <div className="flex items-center gap-4">
        <span
          className="rounded px-3 py-1 text-[11px] font-bold tracking-[0.06em] uppercase"
          style={{
            backgroundColor: "rgba(0,229,255,0.12)",
            color: "var(--color-brand)",
          }}
        >
          {model.model_type}
        </span>
        {model.status === "active" && (
          <span className="rounded bg-[rgba(0,229,255,0.12)] px-2 py-0.5 text-[9px] font-bold tracking-[0.06em] text-(--color-brand) uppercase">
            ACTIVE
          </span>
        )}
        <span className="font-mono text-[10px] text-(--color-text-dim)">{model.id}</span>
      </div>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-4">
        <MetricBox
          label="Sharpe"
          value={
            model.best_sharpe != null
              ? (model.best_sharpe >= 0 ? "+" : "") + model.best_sharpe.toFixed(3)
              : "—"
          }
        />
        <MetricBox
          label="Return"
          value={
            model.best_return != null
              ? (model.best_return >= 0 ? "+" : "") + model.best_return.toFixed(2) + "%"
              : "—"
          }
        />
        <MetricBox
          label="Win Rate"
          value={model.win_rate != null ? (model.win_rate * 100).toFixed(1) + "%" : "—"}
        />
        <MetricBox
          label="Max DD"
          value={model.max_drawdown != null ? model.max_drawdown.toFixed(1) + "%" : "—"}
        />
        <MetricBox
          label="Total Trades"
          value={model.total_trades != null ? String(model.total_trades) : "—"}
        />
        <MetricBox
          label="Sortino"
          value={
            model.sortino != null ? (model.sortino >= 0 ? "+" : "") + model.sortino.toFixed(3) : "—"
          }
        />
        <MetricBox
          label="Features"
          value={model.feature_count != null ? String(model.feature_count) : "—"}
        />
        <MetricBox label="Status" value={model.status.toUpperCase()} />
      </div>

      {(model.train_start || model.train_end) && (
        <div className="rounded border border-(--color-glass-border) bg-(--color-glass) p-4">
          <span className="text-[10px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
            Training Range
          </span>
          <div className="mt-2 font-mono text-[12px] text-(--color-text-secondary)">
            {model.train_start ? model.train_start.slice(0, 10) : "Unknown"} →{" "}
            {model.train_end ? model.train_end.slice(0, 10) : "Unknown"}
          </div>
        </div>
      )}

      {model.tags.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-(--color-text-muted)">Tags:</span>
          {model.tags.map((t) => (
            <span
              key={t}
              className="rounded bg-(--color-elevated) px-2 py-0.5 font-mono text-[9px] text-(--color-text-dim)"
            >
              {t}
            </span>
          ))}
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={() =>
            apiClient
              .post("/trading/paper/start", {
                model_id: model.id,
                model_type: model.model_type,
                pair: "EURUSD",
                timeframe: "H1",
                initial_equity: 10000,
                position_sizing: "fixed",
              })
              .then(() => navigate("/trading"))
              .catch(console.error)
          }
          className="flex items-center gap-2 rounded bg-(--color-accent-success) px-4 py-2 text-[11px] font-semibold tracking-[0.06em] text-(--color-text-inverse) uppercase"
        >
          <Play size={12} /> Deploy Paper
        </button>
        <button
          onClick={() => navigate(`/results/${model.parent_job_id}`)}
          disabled={!model.parent_job_id}
          className="rounded border border-(--color-glass-border) bg-(--color-elevated) px-4 py-2 text-[11px] font-semibold tracking-[0.06em] text-(--color-text-secondary) uppercase"
          style={{ opacity: model.parent_job_id ? 1 : 0.4 }}
        >
          View Backtest
        </button>
      </div>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-(--color-glass-border) bg-(--color-glass) p-4">
      <div className="font-mono text-[18px] font-semibold text-(--color-text-primary)">{value}</div>
      <div className="mt-1 text-[9px] font-medium tracking-[0.06em] text-(--color-text-muted) uppercase">
        {label}
      </div>
    </div>
  );
}
