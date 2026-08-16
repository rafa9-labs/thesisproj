"""FastAPI application configuration."""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_prefix": "API_", "extra": "ignore"}

    app_name: str = "KodaQuant API"
    version: str = "1.0.0"
    debug: bool = False

    host: str = "127.0.0.1"
    port: int = 8001

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    db_path: str = "data/forex.db"
    csv_data_dir: str = "csv_data"
    results_dir: str = "results"

    project_root: str = str(Path(__file__).resolve().parent.parent)

    model_check_interval: float = 2.0
    max_concurrent_backtests: int = 4
    gpu_enabled: bool = False
    max_concurrent_gpu: int = 4
    gpu_total_vram_mb: int = 0

    paddle_vendor_id: str = ""
    paddle_product_id: str = ""
    paddle_api_key: str = ""
    paddle_sandbox: bool = False
    app_secret: str = ""
    license_db_path: str = ""

    vast_enabled: bool = False
    vast_min_gpu_class: str = "RTX 3090"
    vast_min_vram_gb: float = 16.0
    vast_max_dph: float = 0.5
    vast_disk_gb: int = 60
    vast_image: str = "nvidia/cuda:12.2.0-base-ubuntu22.04"
    vast_repo_url: str = ""
    vast_remote_api_url: str = ""

    @property
    def db_full_path(self) -> str:
        fx_data_dir = os.environ.get("FX_DATA_DIR")
        if fx_data_dir:
            return os.path.join(fx_data_dir, "forex.db")
        if os.path.isabs(self.db_path):
            return self.db_path
        return os.path.join(self.project_root, self.db_path)

    @property
    def csv_data_full_path(self) -> str:
        csv_dir = os.environ.get("CSV_DATA_DIR")
        if csv_dir:
            return csv_dir
        return os.path.join(self.project_root, self.csv_data_dir)

    @property
    def results_full_path(self) -> str:
        fx_data_dir = os.environ.get("FX_DATA_DIR")
        if fx_data_dir:
            return os.path.join(fx_data_dir, "results")
        return os.path.join(self.project_root, self.results_dir)

    @property
    def sync_pairs(self) -> list[str]:
        raw = os.environ.get("SYNC_PAIRS", "EURUSD,GBPUSD,USDJPY")
        return [p.strip() for p in raw.split(",") if p.strip()]


settings = Settings()


def load_persisted_execution_settings() -> None:
    path = Path(os.environ.get("FX_EXEC_CONFIG_PATH", "fx_exec_config.json"))
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        settings.max_concurrent_backtests = data.get("max_concurrent_backtests", 4)
        settings.gpu_enabled = data.get("gpu_enabled", False)
        settings.max_concurrent_gpu = data.get("max_concurrent_gpu", 4)
        settings.vast_enabled = data.get("vast_enabled", False)
        settings.vast_min_gpu_class = data.get("vast_min_gpu_class", "RTX 3090")
        settings.vast_min_vram_gb = data.get("vast_min_vram_gb", 16.0)
        settings.vast_max_dph = data.get("vast_max_dph", 0.5)
        settings.vast_disk_gb = data.get("vast_disk_gb", 60)
        settings.vast_image = data.get("vast_image", settings.vast_image)
        settings.vast_repo_url = data.get("vast_repo_url", "")
        settings.vast_remote_api_url = data.get("vast_remote_api_url", "")
        print(f"[Config] Loaded execution settings from {path}", flush=True)
    except Exception:
        pass
