import { useMemo } from "react";
import type { EquityPoint } from "@/api/schemas";

interface EquityThumbnailProps {
  data: EquityPoint[] | null;
  width?: number;
  height?: number;
}

export function EquityThumbnail({ data, width = 120, height = 36 }: EquityThumbnailProps) {
  const path = useMemo(() => {
    if (!data || data.length < 2) return null;
    const vals = data.map((d) => d.value);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min || 1;
    const step = width / (data.length - 1);
    return data
      .map((d, i) => {
        const x = i * step;
        const y = height - ((d.value - min) / range) * (height - 4) - 2;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [data, width, height]);

  if (!path) {
    return <div className="bg-(--color-elevated)" style={{ borderRadius: 4 }} />;
  }

  const finalVal = data && data.length > 0 ? data[data.length - 1].value : 0;
  const startVal = data && data.length > 0 ? data[0].value : 0;
  const color = finalVal >= startVal ? "var(--color-accent-success)" : "var(--color-accent-danger)";

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block" }}
    >
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={1.2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
