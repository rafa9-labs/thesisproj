"""FastAPI application configuration."""
from __future__ import annotations

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
    max_concurrent_backtests: int = 2

    paddle_vendor_id: str = ""
    paddle_product_id: str = ""
    paddle_api_key: str = ""
    paddle_sandbox: bool = False
    app_secret: str = ""
    license_db_path: str = ""

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


settings = Settings()
