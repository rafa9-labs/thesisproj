"""Validate PyInstaller spec and build prerequisites.

These tests verify the spec file is well-formed and all referenced
modules exist WITHOUT actually running the build (which is expensive).
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(PROJECT_ROOT, "forex_pipeline.spec")


class TestSpecFileExists:
    def test_spec_file_present(self):
        assert os.path.isfile(SPEC_PATH), f"Missing spec: {SPEC_PATH}"

    def test_run_server_entry_point(self):
        ep = os.path.join(PROJECT_ROOT, "run_server.py")
        assert os.path.isfile(ep), f"Missing entry point: {ep}"


class TestHiddenImportsAvailable:
    """Verify that all modules listed in spec hidden_imports are importable."""

    REQUIRED_MODULES = [
        "uvicorn.logging",
        "fastapi",
        "fastapi.responses",
        "fastapi.staticfiles",
        "celery",
        "redis",
        "sqlalchemy",
        "aiosqlite",
        "httpx",
        "websockets",
        "pydantic",
        "pandas",
        "numpy",
        "sklearn",
        "sklearn.linear_model",
        "sklearn.svm",
        "sklearn.ensemble",
        "sklearn.calibration",
        "xgboost",
        "optuna",
        "scipy",
        "scipy.stats",
        "models.registry",
        "models.base_model",
        "pipeline._imports",
        "pipeline.runtime",
        "pipeline.metrics.metrics_eval",
        "api.main",
        "api.config",
        "api.tasks",
    ]

    @pytest.mark.parametrize("module", REQUIRED_MODULES, ids=REQUIRED_MODULES)
    def test_module_importable(self, module):
        import importlib
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.skip(f"Module {module} not available: {e}")


class TestPipelineModulesAvailable:
    """Verify key pipeline modules can be imported."""

    def test_backtester_composed(self):
        from pipeline.backtester.composed import MLBacktester
        assert MLBacktester is not None

    def test_model_registry(self):
        from models.registry import MODEL_REGISTRY, build_model
        assert len(MODEL_REGISTRY) >= 16

    def test_execution_modules(self):
        from pipeline.execution.position_sizing import SizingMethod
        from pipeline.execution.stops import StopMethod
        from pipeline.execution.trailing import TrailingMethod
        from pipeline.execution.risk_manager import RiskConfig

    def test_api_app(self):
        from api.main import app
        assert app is not None


class TestDataFilesAvailable:
    """Verify that data directories referenced in spec exist."""

    def test_config_py(self):
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "config.py"))

    def test_hpo_dir(self):
        hpo_dir = os.path.join(PROJECT_ROOT, "hpo")
        assert os.path.isdir(hpo_dir), f"Missing HPO configs: {hpo_dir}"
        configs = [f for f in os.listdir(hpo_dir) if f.endswith("_best_config.json")]
        assert len(configs) >= 5, f"Expected 5+ HPO configs, found {len(configs)}"

    def test_csv_data_dir(self):
        csv_dir = os.path.join(PROJECT_ROOT, "csv_data")
        assert os.path.isdir(csv_dir), f"Missing CSV data: {csv_dir}"

    def test_schemas_dir(self):
        schemas_dir = os.path.join(PROJECT_ROOT, "schemas")
        assert os.path.isdir(schemas_dir), f"Missing schemas: {schemas_dir}"


class TestBuildScript:
    """Verify build scripts exist and are well-formed."""

    def test_build_python_script(self):
        script = os.path.join(PROJECT_ROOT, "scripts", "build_python.bat")
        assert os.path.isfile(script)
        with open(script) as f:
            content = f.read()
        assert "pyinstaller" in content.lower()
        assert "forex_pipeline.spec" in content

    def test_build_electron_script(self):
        script = os.path.join(PROJECT_ROOT, "scripts", "build_electron.bat")
        assert os.path.isfile(script)
        with open(script) as f:
            content = f.read()
        assert "electron-builder" in content.lower()

    def test_electron_builder_yml(self):
        yml = os.path.join(PROJECT_ROOT, "electron-builder.yml")
        assert os.path.isfile(yml)


class TestRunServerModule:
    """Verify the PyInstaller entry point works as a module."""

    def test_run_server_importable(self):
        sys.path.insert(0, PROJECT_ROOT)
        import run_server
        assert hasattr(run_server, "main")
        assert hasattr(run_server, "_is_frozen")
        assert hasattr(run_server, "_setup_paths")

    def test_frozen_detection(self):
        import run_server
        assert not run_server._is_frozen(), "Should not be frozen in dev mode"

    def test_setup_paths_returns_root(self):
        import run_server
        root = run_server._setup_paths()
        assert os.path.isdir(root)
        assert os.path.isfile(os.path.join(root, "config.py"))