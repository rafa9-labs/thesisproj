interface SignificanceMatrixProps {
  models: string[];
  pValues: number[][] | null;
}

function cellColor(p: number | null): string {
  if (p === null) return "var(--color-elevated)";
  if (p < 0.01) return "rgba(8,153,129,0.85)";
  if (p < 0.05) return "rgba(8,153,129,0.55)";
  if (p < 0.1) return "rgba(255,152,0,0.45)";
  return "rgba(242,54,69,0.35)";
}

function formatP(p: number | null, isDiag: boolean): string {
  if (isDiag) return "—";
  if (p === null) return "";
  if (p >= 0.1) return "ns";
  if (p < 0.01) return `**${p.toFixed(3)}`;
  if (p < 0.05) return `*${p.toFixed(3)}`;
  return p.toFixed(3);
}

export function SignificanceMatrix({ models = [], pValues }: SignificanceMatrixProps) {
  if (pValues === null) {
    return (
      <div className="flex flex-col gap-2">
        <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
          Significance Testing
        </h3>
        <div className="flex items-center justify-center rounded-sm border border-(--color-border) bg-(--color-surface) p-8">
          <div className="flex max-w-md flex-col items-center gap-2 text-center">
            <span className="font-mono text-sm text-(--color-text-muted)">
              Requires multiple repeats (repeats &gt; 1)
            </span>
            <span className="text-xs text-(--color-text-muted)">
              Run a backtest with repeats to generate paired t-test p-values between models. The
              matrix shows whether performance differences are statistically significant.
            </span>
          </div>
        </div>
      </div>
    );
  }

  const n = models.length;
  if (n < 2) return null;

  const cellSize = 64;
  const labelWidth = 100;
  const headerHeight = 60;
  const totalW = labelWidth + n * cellSize;
  const totalH = headerHeight + n * cellSize;

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold tracking-[0.08em] text-(--color-text-secondary) uppercase">
        Significance Testing
      </h3>
      <div className="overflow-auto rounded-sm border border-(--color-border) bg-(--color-surface) p-3">
        <svg width={totalW} height={totalH} viewBox={`0 0 ${totalW} ${totalH}`}>
          {models.map((model, col) => (
            <g key={`h-${model}`}>
              <text
                x={labelWidth + col * cellSize + cellSize / 2}
                y={headerHeight - 8}
                textAnchor="middle"
                fill="var(--color-text-muted)"
                fontSize={9}
                fontFamily="var(--font-mono)"
                transform={`rotate(-35, ${labelWidth + col * cellSize + cellSize / 2}, ${headerHeight - 8})`}
              >
                {model.length > 12 ? model.slice(0, 12) + "…" : model}
              </text>
            </g>
          ))}
          {models.map((model, row) => (
            <g key={`r-${model}`}>
              <text
                x={labelWidth - 6}
                y={headerHeight + row * cellSize + cellSize / 2 + 4}
                textAnchor="end"
                fill="var(--color-text-muted)"
                fontSize={9}
                fontFamily="var(--font-mono)"
              >
                {model.length > 14 ? model.slice(0, 14) + "…" : model}
              </text>
            </g>
          ))}
          {pValues.map((row, ri) =>
            row.map((p, ci) => {
              const isDiag = ri === ci;
              return (
                <g key={`${ri}-${ci}`}>
                  <rect
                    x={labelWidth + ci * cellSize + 2}
                    y={headerHeight + ri * cellSize + 2}
                    width={cellSize - 4}
                    height={cellSize - 4}
                    rx={4}
                    fill={isDiag ? "var(--color-border)" : cellColor(p)}
                  />
                  <text
                    x={labelWidth + ci * cellSize + cellSize / 2}
                    y={headerHeight + ri * cellSize + cellSize / 2 + 4}
                    textAnchor="middle"
                    fill={isDiag ? "var(--color-text-muted)" : "var(--color-text-primary)"}
                    fontSize={isDiag ? 11 : 9}
                    fontFamily="var(--font-mono)"
                    fontWeight={isDiag ? 400 : 600}
                  >
                    {formatP(p, isDiag)}
                  </text>
                </g>
              );
            }),
          )}
        </svg>
        <div className="mt-2 flex items-center gap-4 text-[10px] text-(--color-text-muted)">
          <span>ns = not significant (p&gt;0.1)</span>
          <span>* p&lt;0.05</span>
          <span>** p&lt;0.01</span>
        </div>
      </div>
    </div>
  );
}
