"""ML pipeline tests — VRAM lock and TF logical device configuration.

Tests that apply_vram_lock correctly bounds GPU memory when
CUDA_VRAM_LIMIT_MB is set, and falls back to memory growth otherwise.
"""
from __future__ import annotations

import os
import sys

from unittest.mock import MagicMock, patch

import pytest


def _tf_available():
    try:
        import tensorflow as tf
        return True
    except Exception:
        return False


tf_mark = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")


@tf_mark
class TestVRAMLockLogicalDevice:

    def test_sets_logical_device_config_when_env_set(self, monkeypatch):
        monkeypatch.setenv("CUDA_VRAM_LIMIT_MB", "4096")
        mock_gpu = MagicMock()

        with patch("tensorflow.config.list_physical_devices", return_value=[mock_gpu]):
            with patch("tensorflow.config.set_logical_device_configuration") as mock_lcfg:
                from pipeline.runtime import apply_vram_lock
                apply_vram_lock()
                mock_lcfg.assert_called_once()

    def test_noop_when_no_gpu_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VRAM_LIMIT_MB", "4096")

        with patch("tensorflow.config.list_physical_devices", return_value=[]):
            with patch("tensorflow.config.experimental.set_memory_growth") as mock_growth:
                from pipeline.runtime import apply_vram_lock
                apply_vram_lock()
                mock_growth.assert_not_called()

    def test_falls_back_to_memory_growth_when_no_env(self, monkeypatch):
        monkeypatch.delenv("CUDA_VRAM_LIMIT_MB", raising=False)
        mock_gpu = MagicMock()

        with patch("tensorflow.config.list_physical_devices", return_value=[mock_gpu]):
            with patch("tensorflow.config.experimental.set_memory_growth") as mock_growth:
                from pipeline.runtime import apply_vram_lock
                apply_vram_lock()
                mock_growth.assert_called_once()

    def test_handle_tf_import_error(self, monkeypatch):
        monkeypatch.setenv("CUDA_VRAM_LIMIT_MB", "4096")
        monkeypatch.setitem(sys.modules, "tensorflow", None)
        from pipeline.runtime import apply_vram_lock
        apply_vram_lock()

    def test_logical_config_failure_falls_back(self, monkeypatch):
        monkeypatch.setenv("CUDA_VRAM_LIMIT_MB", "4096")
        mock_gpu = MagicMock()

        with patch("tensorflow.config.list_physical_devices", return_value=[mock_gpu]):
            cfg_mock = patch(
                "tensorflow.config.set_logical_device_configuration",
                side_effect=RuntimeError("logical config failed"),
            )
            growth_mock = patch(
                "tensorflow.config.experimental.set_memory_growth",
            )
            with cfg_mock, growth_mock as mock_growth:
                from pipeline.runtime import apply_vram_lock
                apply_vram_lock()
                mock_growth.assert_called_once()


class TestVRAMLockModuleLevelGuard:

    def test_vram_limit_env_blocks_memory_growth_at_module_load(self, monkeypatch):
        monkeypatch.setenv("CUDA_VRAM_LIMIT_MB", "4096")

        with patch("tensorflow.config.list_physical_devices", return_value=[MagicMock()]):
            with patch("tensorflow.config.experimental.set_memory_growth") as mock_growth:
                import importlib
                import pipeline.runtime
                importlib.reload(pipeline.runtime)
                mock_growth.assert_not_called()
