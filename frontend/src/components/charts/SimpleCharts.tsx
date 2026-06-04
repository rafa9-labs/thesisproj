interface BarChartProps {
  data: { label: string; value: number }[];
  height?: number;
  positiveColor?: string;
  negativeColor?: string;
}

export function SimpleBarChart({
  data,
  height = 300,
  positiveColor = "var(--color-accent-success)",
  negativeColor = "var(--color-accent-danger)",
}: BarChartProps) {
  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-sm border"
        style={{ height, backgroundColor: "var(--color-surface)", borderColor: "var(--color-border)", color: "var(--color-text-muted)" }}
      >
        No data
      </div>
    );
  }

  const max = Math.max(...data.map((d) => Math.abs(d.value)), 0.001);
  const barWidth = Math.max(100 / data.length - 2, 2);

  return (
    <div className="flex flex-col gap-2">
      <svg width="100%" height={height} viewBox={`0 0 ${data.length * 40} ${height}`}>
        {/* Zero line */}
        <line
          x1={0}
          y1={height / 2}
          x2={data.length * 40}
          y2={height / 2}
          stroke="var(--color-border)"
          strokeWidth={1}
        />
        {data.map((d, i) => {
          const barHeight = (Math.abs(d.value) / max) * (height / 2 - 20);
          const isPositive = d.value >= 0;
          const y = isPositive ? height / 2 - barHeight : height / 2;
          return (
            <g key={i}>
              <rect
                x={i * 40 + (40 - barWidth) / 2}
                y={y}
                width={barWidth}
                height={Math.max(barHeight, 1)}
                fill={isPositive ? positiveColor : negativeColor}
                rx={2}
              />
              {data.length <= 20 && (
                <text
                  x={i * 40 + 20}
                  y={height - 4}
                  textAnchor="middle"
                  fill="var(--color-text-muted)"
                  fontSize={9}
                  fontFamily="var(--font-mono)"
                >
                  {d.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function HorizontalBarChart({
  data,
  height,
  barColor = "var(--color-accent-success)",
}: {
  data: { label: string; value: number }[];
  height?: number;
  barColor?: string;
}) {
  if (data.length === 0) return null;

  const max = Math.max(...data.map((d) => d.value), 0.001);
  const rowHeight = 22;
  const svgHeight = height ?? data.length * rowHeight + 20;
  const labelWidth = 140;
  const chartWidth = 400;

  return (
    <svg width="100%" height={svgHeight} viewBox={`0 0 ${labelWidth + chartWidth} ${svgHeight}`}>
      {data.map((d, i) => {
        const barW = (d.value / max) * (chartWidth - 10);
        return (
          <g key={i}>
            <text
              x={labelWidth - 5}
              y={i * rowHeight + 14}
              textAnchor="end"
              fill="var(--color-text-secondary)"
              fontSize={10}
              fontFamily="var(--font-mono)"
            >
              {d.label.length > 18 ? d.label.slice(0, 18) + "…" : d.label}
            </text>
            <rect
              x={labelWidth}
              y={i * rowHeight + 3}
              width={Math.max(barW, 1)}
              height={14}
              fill={barColor}
              rx={2}
            />
          </g>
        );
      })}
    </svg>
  );
}
