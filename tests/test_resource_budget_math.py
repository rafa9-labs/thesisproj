"""Unit tests for resource budget computation formulas.

Pure math layer — no I/O, no side effects. Tests compute_budget() and
ResourceBudget.disabled() against the documented formulas.
"""
from __future__ import annotations

from pipeline.hardware_profile import HardwareProfile
from pipeline.resource_budget import compute_budget, ResourceBudget


def _make_profile(
    physical_cores=8,
    p_cores=0,
    e_cores=0,
    is_hybrid=False,
    ram_total_gb=32.0,
    gpu_available=False,
    gpu_vram_mb=0,
    gpu_compute_major=0,
    **kwargs,
) -> HardwareProfile:
    if p_cores == 0:
        p_cores = physical_cores
    return HardwareProfile(
        physical_cores=physical_cores,
        logical_cores=physical_cores * 2,
        vendor="GenuineIntel",
        model="Test CPU",
        is_hybrid=is_hybrid,
        p_cores=p_cores,
        e_cores=e_cores,
        l3_cache_mb=12.0,
        ram_total_gb=ram_total_gb,
        gpu_available=gpu_available,
        gpu_name="Test GPU" if gpu_available else "",
        gpu_vram_mb=gpu_vram_mb,
        gpu_compute_major=gpu_compute_major,
        gpu_compute_minor=0,
        gpu_has_tensor_cores=False,
    )


class TestComputeBudgetCPU:
    """CPU-only budget tests."""

    def test_8core_non_hybrid(self):
        profile = _make_profile(physical_cores=8)
        b = compute_budget(profile)
        assert b.effective_cores == 8
        assert b.target_cores == 4
        assert b.blas_threads == 2
        assert b.cv_n_jobs == 2
        assert b.gpu_available is False

    def test_16core_hybrid_8p8e(self):
        profile = _make_profile(
            physical_cores=16, p_cores=8, e_cores=8, is_hybrid=True,
        )
        b = compute_budget(profile)
        assert b.effective_cores == 12       # 8 + int(8*0.5)
        assert b.target_cores == 7            # max(2, int(12*0.6))
        assert b.blas_threads == 3            # max(2, min(4, 7//2))
        assert b.cv_n_jobs == 2              # max(1, 7//3)

    def test_2core_minimum(self):
        profile = _make_profile(physical_cores=2)
        b = compute_budget(profile)
        assert b.effective_cores == 2
        assert b.target_cores == 2
        assert b.blas_threads == 2
        assert b.cv_n_jobs == 1

    def test_ram_budget_60_percent(self):
        profile = _make_profile(physical_cores=8, ram_total_gb=64.0)
        b = compute_budget(profile)
        assert b.ram_limit_gb == 38.4


class TestComputeBudgetGPU:
    """GPU-aware budget tests."""

    def test_rtx3090_24gb(self):
        profile = _make_profile(
            physical_cores=8, gpu_available=True, gpu_vram_mb=24576,
            gpu_compute_major=8,
        )
        b = compute_budget(profile)
        assert b.gpu_available is True
        assert b.vram_limit_mb == 14745       # int(24576 * 0.60)
        assert b.batch_size == 256            # 24GB tier
        assert b.xla_enabled is True          # compute >= 8

    def test_gtx1060_6gb(self):
        profile = _make_profile(
            physical_cores=8, gpu_available=True, gpu_vram_mb=6144,
            gpu_compute_major=6,
        )
        b = compute_budget(profile)
        assert b.gpu_available is True
        assert b.vram_limit_mb == 3686        # int(6144 * 0.60)
        assert b.batch_size == 32             # 4-8GB tier
        assert b.xla_enabled is False         # compute < 8

    def test_no_gpu(self):
        profile = _make_profile(physical_cores=4, gpu_available=False)
        b = compute_budget(profile)
        assert b.gpu_available is False
        assert b.vram_limit_mb == 0
        assert b.batch_size == 32
        assert b.xla_enabled is False

    def test_gpu_vram_4gb(self):
        profile = _make_profile(
            physical_cores=4, gpu_available=True, gpu_vram_mb=4096,
            gpu_compute_major=7,
        )
        b = compute_budget(profile)
        assert b.batch_size == 32             # 4-8GB tier

    def test_gpu_vram_12gb(self):
        profile = _make_profile(
            physical_cores=4, gpu_available=True, gpu_vram_mb=12288,
            gpu_compute_major=7,
        )
        b = compute_budget(profile)
        assert b.batch_size == 128            # 12-16GB tier

    def test_gpu_vram_2gb_lowest(self):
        profile = _make_profile(
            physical_cores=4, gpu_available=True, gpu_vram_mb=2048,
            gpu_compute_major=5,
        )
        b = compute_budget(profile)
        assert b.batch_size == 16             # < 4GB tier


class TestDisabledBudget:
    """ResourceBudget.disabled() fallback."""

    def test_disabled_returns_safe_minimums(self):
        b = ResourceBudget.disabled()
        assert b.blas_threads == 2
        assert b.cv_n_jobs == 1
        assert b.train_n_jobs == 2
        assert b.ram_limit_gb == 4.0
        assert b.gpu_available is False
        assert b.vram_limit_mb == 0
        assert b.batch_size == 32
        assert b.tf_intra_op_threads == 1
        assert b.tf_inter_op_threads == 1
        assert b.xla_enabled is False
        assert b.fingerprint == "disabled"
        assert b.effective_cores == 2
        assert b.target_cores == 2


class TestTfThreadConfig:
    """TensorFlow thread configuration within budget."""

    def test_tf_intra_inter_from_blas(self):
        profile = _make_profile(physical_cores=8)
        b = compute_budget(profile)
        assert b.blas_threads == 2
        assert b.tf_intra_op_threads == 2     # min(blas, 8)
        assert b.tf_inter_op_threads == 2     # max(2, blas//4)

    def test_tf_intra_clamped_to_8(self):
        profile = _make_profile(physical_cores=32)
        b = compute_budget(profile)
        assert b.blas_threads == 4
        assert b.tf_intra_op_threads == 4     # min(4, 8)
        assert b.tf_inter_op_threads == 2     # max(2, 4//4)
