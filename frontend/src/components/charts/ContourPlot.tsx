import { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";

interface HpoTrial {
  trial_number: number;
  value: number | null;
  params: Record<string, unknown>;
}

interface Props {
  trials: HpoTrial[] | null;
}

export function ContourPlot({ trials }: Props) {
  const [xParam, setXParam] = useState<string>("");
  const [yParam, setYParam] = useState<string>("");

  // Extract numeric parameter names
  const numericParams = useMemo(() => {
    if (!trials || trials.length === 0) return [];
    const pset = new Set<string>();
    for (const t of trials) {
      if (!t.params) continue;
      for (const [k, v] of Object.entries(t.params)) {
        if (typeof v === "number") pset.add(k);
      }
    }
    return Array.from(pset).slice(0, 20);
  }, [trials]);

  const xParamKey = xParam || (numericParams.length >= 2 ? numericParams[0] : "");
  const yParamKey = yParam || (numericParams.length >= 2 ? (numericParams[1] || numericParams[0]) : "");

  const scatterData = useMemo(() => {
    if (!trials || !xParamKey || !yParamKey) return [];
    return trials
      .filter((t) => t.value != null && t.params?.[xParamKey] != null && t.params?.[yParamKey] != null)
      .map((t) => ({
        x: Number(t.params![xParamKey]),
        y: Number(t.params![yParamKey]),
        score: Number(t.value),
        trial: t.trial_number,
      }));
  }, [trials, xParamKey, yParamKey]);

  const colorByScore = (score: number) => {
    if (score > 1.0) return "rgba(34,197,94,0.7)";
    if (score > 0.5) return "rgba(234,179,8,0.7)";
    if (score > 0.0) return "rgba(249,115,22,0.6)";
    return "rgba(239,68,68,0.5)";
  };

  if (numericParams.length < 2) {
    return (
      <div className="flex items-center justify-center rounded-sm p-8" style={{ color: "var(--color-text-muted)" }}>
        <span className="text-xs">Need at least 2 numeric parameters for contour analysis.</span>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <select
          value={xParamKey}
          onChange={(e) => setXParam(e.target.value)}
          className="rounded px-2 py-1 text-[10px] border"
          style={{
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-elevated)",
            color: "var(--color-text-primary)",
            fontFamily: "var(--font-mono)",
            outline: "none",
          }}
        >
          {numericParams.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        <span style={{ color: "var(--color-text-muted)" }}>×</span>
        <select
          value={yParamKey}
          onChange={(e) => setYParam(e.target.value)}
          className="rounded px-2 py-1 text-[10px] border"
          style={{
            borderColor: "var(--color-border)",
            backgroundColor: "var(--color-elevated)",
            color: "var(--color-text-primary)",
            fontFamily: "var(--font-mono)",
            outline: "none",
          }}
        >
          {numericParams.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </div>
      <div style={{ width: "100%", height: 250 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 4, right: 8, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-glass-border)" />
            <XAxis
              dataKey="x"
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
              label={{ value: xParamKey, position: "bottom", offset: 0, style: { fill: "var(--color-text-muted)", fontSize: 9 } }}
            />
            <YAxis
              dataKey="y"
              tick={{ fill: "var(--color-text-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}
              label={{ value: yParamKey, angle: -90, position: "left", style: { fill: "var(--color-text-muted)", fontSize: 9 } }}
            />
            <ZAxis range={[30, 80]} />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-elevated)",
                border: "1px solid var(--color-glass-border)",
                borderRadius: 6,
                fontSize: 10,
                fontFamily: "var(--font-mono)",
              }}
              formatter={(v: number, name: string) => name === "score" ? [v.toFixed(4), "Score"] : [v, name]}
              labelFormatter={() => ""}
            />
            <Scatter data={scatterData} isAnimationActive={false}>
              {scatterData.map((d, i) => (
                <Cell key={i} fill={colorByScore(d.score)} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center gap-3 mt-1.5 justify-center">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: "rgba(34,197,94,0.7)" }} />
          <span className="text-[9px] font-mono" style={{ color: "var(--color-text-muted)" }}>&gt;1.0</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: "rgba(234,179,8,0.7)" }} />
          <span className="text-[9px] font-mono" style={{ color: "var(--color-text-muted)" }}>0.5–1.0</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: "rgba(249,115,22,0.6)" }} />
          <span className="text-[9px] font-mono" style={{ color: "var(--color-text-muted)" }}>0–0.5</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: "rgba(239,68,68,0.5)" }} />
          <span className="text-[9px] font-mono" style={{ color: "var(--color-text-muted)" }}>&lt;0</span>
        </div>
      </div>
    </div>
  );
}
