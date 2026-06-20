interface Props {
  modelAgreement: { models: string[]; kappa_matrix: number[][] } | undefined;
}

function kappaColor(kappa: number): string {
  if (kappa >= 0.8) return "rgba(242,54,69,0.55)";  // High agreement = bad (red)
  if (kappa >= 0.6) return "rgba(242,180,54,0.55)";  // Moderate = warning
  if (kappa >= 0.4) return "rgba(242,145,54,0.45)";  // Ok
  if (kappa >= 0.2) return "rgba(8,153,129,0.45)";   // Good diversity
  return "rgba(41,98,255,0.55)";  // Very diverse
}

function kappaText(kappa: number): string {
  if (kappa >= 0.8) return "var(--color-accent-danger)";
  if (kappa >= 0.6) return "var(--color-accent-warning)";
  return "var(--color-text-secondary)";
}

export function ModelAgreementMatrix({ modelAgreement }: Props) {
  if (!modelAgreement || !modelAgreement.models.length) {
    return <p className="text-[11px] text-(--color-text-dim)">No model agreement data</p>;
  }

  const { models, kappa_matrix } = modelAgreement;
  const n = models.length;

  return (
    <div>
      <div className="mb-2 flex items-center gap-3">
        <span className="text-[10px] font-semibold tracking-[0.06em] text-(--color-text-muted) uppercase">
          Pairwise Cohen's Kappa
        </span>
        <span className="font-mono text-[9px] text-(--color-text-dim)">
          κ &lt; 0.4 = healthy diversity
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="border-collapse text-[10px]">
          <thead>
            <tr>
              <th className="px-1 py-1" />
              {models.map((m) => (
                <th
                  key={m}
                  className="px-1.5 py-1 text-left font-mono text-[9px] tracking-[0.04em] text-(--color-text-muted) uppercase"
                  style={{ maxWidth: 80, overflow: "hidden", textOverflow: "ellipsis" }}
                >
                  {m.length > 10 ? m.slice(0, 8) + ".." : m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((mi, i) => (
              <tr key={mi}>
                <td
                  className="px-1.5 py-1 text-right font-mono text-[9px] text-(--color-text-muted) uppercase"
                  style={{ maxWidth: 80, overflow: "hidden", textOverflow: "ellipsis" }}
                >
                  {mi.length > 10 ? mi.slice(0, 8) + ".." : mi}
                </td>
                {kappa_matrix[i]?.map((k, j) => (
                  <td
                    key={j}
                    className="px-1.5 py-1 text-center font-mono text-[9px]"
                    style={{
                      backgroundColor: i === j ? "transparent" : kappaColor(k),
                      color: i === j ? "var(--color-text-dim)" : kappaText(k),
                      borderRadius: 2,
                    }}
                  >
                    {i === j ? "—" : k.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center gap-3 text-[9px] text-(--color-text-muted)">
        <span>Red = redundant</span>
        <span>Green/Blue = diverse</span>
      </div>
    </div>
  );
}
