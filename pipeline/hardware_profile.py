"""
Hardware profiling layer: CPU topology, GPU detection, RAM sizing.

Runs once at process startup. Result is cached in a module-level singleton.
All heavy imports (psutil, subprocess) are deferred until first access.

Fingerprint: sha256(cpu_model | physical_cores | gpu_name | vram_mb)[:16]
  - Auto-invalidates cache when hardware changes.
  - Used by governor_cache.py for cross-run persistence.
"""
import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List

_PROFILE: Optional["HardwareProfile"] = None


@dataclass(frozen=True)
class HardwareProfile:
    physical_cores: int
    logical_cores: int
    vendor: str
    model: str
    is_hybrid: bool
    p_cores: int
    e_cores: int
    l3_cache_mb: float
    ram_total_gb: float

    gpu_available: bool
    gpu_name: str
    gpu_vram_mb: int
    gpu_compute_major: int
    gpu_compute_minor: int
    gpu_has_tensor_cores: bool

    fingerprint: str = field(init=False)

    def __post_init__(self):
        raw = f"{self.model}|{self.physical_cores}|{self.gpu_name}|{self.gpu_vram_mb}"
        object.__setattr__(self, "fingerprint",
                           hashlib.sha256(raw.encode()).hexdigest()[:16])


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def _detect_cpu() -> dict:
    try:
        import psutil
        phys = int(psutil.cpu_count(logical=False) or 1)
        logi = int(psutil.cpu_count(logical=True) or 1)
    except Exception:
        phys = os.cpu_count() or 1
        logi = phys

    model_str = platform.processor() or platform.machine() or "unknown"

    vendor = "unknown"
    model_lower = model_str.lower()
    if "intel" in model_lower:
        vendor = "intel"
    elif "amd" in model_lower:
        vendor = "amd"
    elif "apple" in model_lower or "arm" in model_lower:
        vendor = "apple"

    is_hybrid = (logi > phys) and vendor in {"intel", "apple"}
    if is_hybrid:
        p_cores = logi - phys
        e_cores = phys - p_cores
    else:
        p_cores = phys
        e_cores = 0

    l3 = _detect_l3_cache(vendor)
    ram = _detect_ram()

    return {
        "physical_cores": max(1, phys),
        "logical_cores": max(1, logi),
        "vendor": vendor,
        "model": model_str.strip(),
        "is_hybrid": is_hybrid,
        "p_cores": max(0, p_cores),
        "e_cores": max(0, e_cores),
        "l3_cache_mb": l3,
        "ram_total_gb": ram,
    }


def _detect_l3_cache(vendor: str) -> float:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "L3CacheSize"],
                stderr=subprocess.DEVNULL, timeout=5,
            )
            lines = out.decode("utf-8", errors="ignore").strip().splitlines()
            for line in lines:
                val = line.strip()
                if val.isdigit():
                    return float(val) / 1024.0
        else:
            for path in [
                "/sys/devices/system/cpu/cpu0/cache/index3/size",
                "/sys/devices/system/cpu/cpu0/cache/index2/size",
            ]:
                try:
                    with open(path) as f:
                        val = f.read().strip().upper()
                        if val.endswith("K"):
                            return float(val[:-1]) / 1024.0
                        elif val.endswith("M"):
                            return float(val[:-1])
                        elif val.isdigit():
                            return float(val) / 1024.0 / 1024.0
                except Exception:
                    continue
    except Exception:
        pass

    if vendor == "intel":
        return 30.0
    elif vendor == "amd":
        return 32.0
    return 8.0


def _detect_ram() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        return 8.0


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu() -> dict:
    gpu = _try_nvidia_smi()
    if gpu:
        return gpu
    gpu = _try_pynvml()
    if gpu:
        return gpu
    return {
        "gpu_available": False, "gpu_name": "none",
        "gpu_vram_mb": 0,
        "gpu_compute_major": 0, "gpu_compute_minor": 0,
        "gpu_has_tensor_cores": False,
    }


def _try_nvidia_smi() -> Optional[dict]:
    smi_candidates = ["nvidia-smi"]
    if platform.system() == "Windows":
        wsl_smi = "/mnt/c/Windows/System32/nvidia-smi.exe"
        if os.path.exists(wsl_smi):
            smi_candidates.insert(0, wsl_smi)

    for smi in smi_candidates:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=10,
            )
            lines = out.decode("utf-8", errors="ignore").strip().splitlines()
            if not lines:
                continue
            line = lines[0].strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name = parts[0]
            vram = int(parts[1].split()[0])
            cap_str = parts[2].strip()
            major_str, _, minor_str = cap_str.partition(".")
            major = int(major_str) if major_str.isdigit() else 0
            minor = int(minor_str) if minor_str.isdigit() else 0
            has_tc = (major >= 8) or (major == 7 and minor >= 5)
            return {
                "gpu_available": True,
                "gpu_name": name,
                "gpu_vram_mb": vram,
                "gpu_compute_major": major,
                "gpu_compute_minor": minor,
                "gpu_has_tensor_cores": has_tc,
            }
        except Exception:
            continue
    return None


def _try_pynvml() -> Optional[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram = int(info.total / (1024 * 1024))
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        except Exception:
            major, minor = 0, 0
        has_tc = (major >= 8) or (major == 7 and minor >= 5)
        pynvml.nvmlShutdown()
        return {
            "gpu_available": True,
            "gpu_name": str(name),
            "gpu_vram_mb": vram,
            "gpu_compute_major": int(major),
            "gpu_compute_minor": int(minor),
            "gpu_has_tensor_cores": has_tc,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_hardware_profile() -> HardwareProfile:
    global _PROFILE
    if _PROFILE is None:
        cpu = _detect_cpu()
        gpu = _detect_gpu()
        _PROFILE = HardwareProfile(
            physical_cores=cpu["physical_cores"],
            logical_cores=cpu["logical_cores"],
            vendor=cpu["vendor"],
            model=cpu["model"],
            is_hybrid=cpu["is_hybrid"],
            p_cores=cpu["p_cores"],
            e_cores=cpu["e_cores"],
            l3_cache_mb=cpu["l3_cache_mb"],
            ram_total_gb=cpu["ram_total_gb"],
            gpu_available=gpu["gpu_available"],
            gpu_name=gpu["gpu_name"],
            gpu_vram_mb=gpu["gpu_vram_mb"],
            gpu_compute_major=gpu["gpu_compute_major"],
            gpu_compute_minor=gpu["gpu_compute_minor"],
            gpu_has_tensor_cores=gpu["gpu_has_tensor_cores"],
        )
    return _PROFILE
