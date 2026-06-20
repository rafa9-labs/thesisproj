export interface Tier3Param {
  displayName: string;
  value: string;
  description: string;
}

export function Tier3Accordion({ params }: { params: Tier3Param[] }) {
  if (params.length === 0) return null;

  return (
    <details className="group/adv">
      <summary className="cursor-pointer text-[10px] font-semibold tracking-[0.08em] text-(--color-text-muted)/60 uppercase hover:text-(--color-text-muted)/80 transition-colors">
        Advanced Parameters ({params.length})
      </summary>
      <div className="ml-1 mt-3 grid grid-cols-2 gap-4 md:grid-cols-4">
        {params.map((p) => (
          <div key={p.displayName} className="group/row relative min-w-0 flex flex-col gap-1" title={p.description}>
            <span className="min-w-0 truncate text-[10px] font-medium uppercase tracking-wider text-slate-500">
              {p.displayName}
            </span>
            <span className="min-w-0 truncate font-mono text-sm text-slate-200">
              {p.value}
            </span>
            {p.description && (
              <div className="pointer-events-none absolute right-0 bottom-full z-50 mb-1 hidden w-56 rounded-md border border-slate-700 bg-slate-800 p-2 text-[10px] leading-relaxed text-(--color-text-secondary) shadow-xl group-hover/row:block sm:left-0 sm:right-auto">
                {p.description}
              </div>
            )}
          </div>
        ))}
      </div>
    </details>
  );
}
