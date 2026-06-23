"""Hardware discovery -- probes GPU VRAM on server startup.

Called from FastAPI lifespan. Stores gpu_total_vram_mb in global settings
so the Gatekeeper can make VRAM-aware scheduling decisions.
"""
from __future__ import annotations

from typing import Optional


def discover_gpu_vram() -> int:
    """Probe total GPU VRAM (MB) via pynvml. Returns 0 if no GPU or probe fails."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_mb = int(info.total / (1024 * 1024))
        pynvml.nvmlShutdown()
        return vram_mb
    except Exception:
        return _try_tf_vram()


def _try_tf_vram() -> int:
    """Fallback: probe VRAM via TensorFlow's device listing."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return 0
        info = tf.config.experimental.get_device_details(gpus[0])
        mem_bytes = info.get("memory_size", 0)
        if mem_bytes:
            return int(mem_bytes / (1024 * 1024))
    except Exception:
        pass
    return 0


def _try_pipeline_profile() -> Optional[int]:
    """Fallback: use the existing hardware profile from pipeline."""
    try:
        from pipeline.hardware_profile import get_hardware_profile
        profile = get_hardware_profile()
        if profile.gpu_available and profile.gpu_vram_mb > 0:
            return profile.gpu_vram_mb
    except Exception:
        pass
    return None
