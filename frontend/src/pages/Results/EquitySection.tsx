import { forwardRef } from "react";
import { EquityCurveChart, type EquityCurveChartHandle } from "@/components/charts/EquityCurveChart";
import type { EquityPoint } from "@/api/schemas";

interface EquitySectionProps {
  equityCurve: EquityPoint[] | null;
  buyHoldCurve: EquityPoint[] | null;
  drawdownCurve: EquityPoint[] | null;
  chartRef?: React.Ref<EquityCurveChartHandle>;
}

export const EquitySection = forwardRef<EquityCurveChartHandle, Omit<EquitySectionProps, "chartRef">>(function EquitySection(
  { equityCurve, buyHoldCurve, drawdownCurve },
  ref,
) {
  const data = equityCurve ?? [];
  const buyHold = buyHoldCurve ?? undefined;
  const drawdown = drawdownCurve ?? undefined;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3
          className="text-xs font-semibold uppercase tracking-[0.08em]"
          style={{ color: "var(--color-text-secondary)" }}
        >
          Equity Curve
        </h3>
        <div className="flex items-center gap-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded"
              style={{ backgroundColor: "var(--color-accent-success)" }}
            />
            Equity
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-0.5 w-4 rounded"
              style={{ backgroundColor: "var(--color-text-muted)", borderStyle: "dashed" }}
            />
            Buy & Hold
          </span>
          <span className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-4 rounded-sm"
              style={{ backgroundColor: "rgba(242, 54, 69, 0.3)" }}
            />
            Drawdown
          </span>
        </div>
      </div>
      <EquityCurveChart
        ref={ref}
        data={data}
        buyHoldData={buyHold}
        drawdownData={drawdown}
        height={420}
      />
    </div>
  );
});
