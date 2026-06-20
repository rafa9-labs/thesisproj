import { useHardware } from "@/api/queries";

export function GpuStatusCard() {
  const { data: hw, isLoading, isError } = useHardware();

  if (isLoading) {
    return (
      <span className="rounded-md border border-(--color-glass-border) bg-(--color-glass-hover) px-3 py-1.5 font-mono text-xs text-(--color-text-muted)">
        Detecting hardware…
      </span>
    );
  }

  if (isError || !hw) {
    return (
      <span className="rounded-md border border-(--color-glass-border) bg-(--color-glass-hover) px-3 py-1.5 font-mono text-xs text-(--color-accent-warning)">
        Could not detect hardware
      </span>
    );
  }

  const { gpu, budget } = hw;

  if (!gpu.available) {
    return (
      <div className="flex flex-col gap-1">
        <span className="rounded-md border border-(--color-glass-border) bg-(--color-glass-hover) px-3 py-1.5 font-mono text-xs text-(--color-text-secondary)">
          CPU Mode — No GPU detected
        </span>
        <span className="text-[9px] text-(--color-text-muted)">
          Using {hw.cpu.physical_cores} cores, batch size {budget.batch_size}
        </span>
      </div>
    );
  }

  const vramGb = (gpu.vram_mb / 1024).toFixed(1);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <span className="rounded-md border border-(--color-accent-success) bg-(color-mix(in srgb, var(--color-accent-success) 10%, transparent)) px-3 py-1.5 font-mono text-xs text-(--color-accent-success)">
          {gpu.name}
        </span>
        <span className="rounded-md border border-(--color-glass-border) bg-(--color-glass-hover) px-2 py-1 font-mono text-[10px] text-(--color-text-secondary)">
          {vramGb} GB
        </span>
        {gpu.tensor_cores && (
          <span className="rounded-md border border-(--color-brand) bg-(color-mix(in srgb, var(--color-brand) 10%, transparent)) px-2 py-1 text-[10px] font-semibold text-(--color-brand)">
            Tensor Cores
          </span>
        )}
      </div>
      <div className="flex items-center gap-3 text-[9px] text-(--color-text-muted)">
        <span>Compute {gpu.compute_capability}</span>
        <span>|</span>
        <span>BLAS threads: {budget.blas_threads}</span>
        <span>|</span>
        <span>CV folds: {budget.cv_n_jobs}</span>
        <span>|</span>
        <span>Batch: {budget.batch_size}</span>
        {budget.xla_enabled && (
          <>
            <span>|</span>
            <span className="text-(--color-brand)">XLA</span>
          </>
        )}
      </div>
    </div>
  );
}
