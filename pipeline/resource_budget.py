"""
Resource budget layer: consumes HardwareProfile, computes CPU/GPU budgets.

Pure math layer -- no I/O, no side effects.

Formulas:
  effective_cores = p_cores + int(e_cores * 0.50)   [hybrid-aware]
  target_cores    = max(2, int(effective * 0.60))     [60% equilibrium]
  blas_threads    = max(2, min(4, target // 2))        [per-op parallelism]
  cv_n_jobs       = max(1, target // blas_threads)     [parallel folds]

GPU:
  vram_limit_mb   = int(vram_mb * 0.60)
  batch_size      = vram-tier lookup (16..256)

Process priority: BELOW_NORMAL on Windows, nice(10) on Linux.
"""
import os
import platform as _platform
from dataclasses import dataclass
from typing import Optional

from pipeline.hardware_profile import HardwareProfile, get_hardware_profile


@dataclass(frozen=True)
class ResourceBudget:
    blas_threads: int
    cv_n_jobs: int
    train_n_jobs: int
    ram_limit_gb: float

    gpu_available: bool
    vram_limit_mb: int
    batch_size: int
    tf_intra_op_threads: int
    tf_inter_op_threads: int
    xla_enabled: bool

    fingerprint: str
    effective_cores: int
    target_cores: int

    @staticmethod
    def disabled():
        return ResourceBudget(
            blas_threads=2, cv_n_jobs=1, train_n_jobs=2,
            ram_limit_gb=4.0,
            gpu_available=False, vram_limit_mb=0, batch_size=32,
            tf_intra_op_threads=1, tf_inter_op_threads=1,
            xla_enabled=False, fingerprint="disabled",
            effective_cores=2, target_cores=2,
        )


def _batch_by_vram(vram_mb: int) -> int:
    vram_gb = vram_mb / 1024.0
    if vram_gb >= 24:   return 256
    elif vram_gb >= 16: return 192
    elif vram_gb >= 12: return 128
    elif vram_gb >= 8:  return 64
    elif vram_gb >= 4:  return 32
    return 16


def compute_budget(profile: HardwareProfile) -> ResourceBudget:

    if profile.is_hybrid:
        effective = profile.p_cores + int(profile.e_cores * 0.50)
    else:
        effective = profile.physical_cores

    target = max(2, int(effective * 0.60))

    blas = max(2, min(4, target // 2))
    cv = max(1, target // blas)

    ram = round(profile.ram_total_gb * 0.60, 1)

    intra = min(blas, 8)
    inter = max(2, blas // 4)

    if profile.gpu_available:
        vram_limit = int(profile.gpu_vram_mb * 0.60)
        batch = _batch_by_vram(profile.gpu_vram_mb)
        xla = (profile.gpu_compute_major >= 8)
    else:
        vram_limit = 0
        batch = 32
        xla = False

    return ResourceBudget(
        blas_threads=blas,
        cv_n_jobs=cv,
        train_n_jobs=blas,
        ram_limit_gb=ram,

        gpu_available=profile.gpu_available,
        vram_limit_mb=vram_limit,
        batch_size=batch,
        tf_intra_op_threads=intra,
        tf_inter_op_threads=inter,
        xla_enabled=xla,

        fingerprint=profile.fingerprint,
        effective_cores=effective,
        target_cores=target,
    )


_budget_cache: Optional[ResourceBudget] = None


def get_resource_budget() -> ResourceBudget:
    global _budget_cache
    if _budget_cache is None:
        try:
            profile = get_hardware_profile()
            _budget_cache = compute_budget(profile)
        except Exception:
            _budget_cache = ResourceBudget.disabled()
    return _budget_cache


def set_resource_budget(budget: ResourceBudget) -> None:
    global _budget_cache
    _budget_cache = budget


def apply_process_priority() -> None:
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(10)
    except Exception:
        try:
            os.nice(10)
        except Exception:
            pass
