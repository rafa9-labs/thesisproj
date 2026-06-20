"""System hardware profile endpoint."""
from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/system", tags=["system"])


class CpuInfo(BaseModel):
    model: str
    physical_cores: int
    logical_cores: int
    is_hybrid: bool
    p_cores: int
    e_cores: int
    ram_total_gb: float


class GpuInfo(BaseModel):
    available: bool
    name: str
    vram_mb: int
    compute_capability: str
    tensor_cores: bool


class ComputeBudget(BaseModel):
    blas_threads: int
    cv_n_jobs: int
    batch_size: int
    xla_enabled: bool
    vram_limit_mb: int
    ram_limit_gb: float


class HardwareResponse(BaseModel):
    cpu: CpuInfo
    gpu: GpuInfo
    budget: ComputeBudget


@router.get("/hardware", response_model=HardwareResponse)
def get_hardware():
    from pipeline.hardware_profile import get_hardware_profile
    from pipeline.resource_budget import get_resource_budget

    profile = get_hardware_profile()
    budget = get_resource_budget()

    return HardwareResponse(
        cpu=CpuInfo(
            model=profile.model,
            physical_cores=profile.physical_cores,
            logical_cores=profile.logical_cores,
            is_hybrid=profile.is_hybrid,
            p_cores=profile.p_cores,
            e_cores=profile.e_cores,
            ram_total_gb=profile.ram_total_gb,
        ),
        gpu=GpuInfo(
            available=profile.gpu_available,
            name=profile.gpu_name,
            vram_mb=profile.gpu_vram_mb,
            compute_capability=f"{profile.gpu_compute_major}.{profile.gpu_compute_minor}",
            tensor_cores=profile.gpu_has_tensor_cores,
        ),
        budget=ComputeBudget(
            blas_threads=budget.blas_threads,
            cv_n_jobs=budget.cv_n_jobs,
            batch_size=budget.batch_size,
            xla_enabled=budget.xla_enabled,
            vram_limit_mb=budget.vram_limit_mb,
            ram_limit_gb=budget.ram_limit_gb,
        ),
    )
