"""Unit tests for GPU VRAM hardware discovery.

Tests discover_gpu_vram() using sys.modules mocking since pynvml and tensorflow
are imported locally inside the discovery functions.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


class MockMemoryInfo:
    def __init__(self, total_mb):
        self.total = total_mb * 1024 * 1024


class TestDiscoverViaPynvml:

    def test_returns_vram_when_pynvml_works(self, monkeypatch):
        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4090"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MockMemoryInfo(24576)
        monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)

        from api.hardware import discover_gpu_vram
        vram = discover_gpu_vram()
        assert vram == 24576

    def test_returns_zero_when_pynvml_fails_all_fallbacks(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        monkeypatch.setattr("api.hardware._try_tf_vram", lambda: 0)
        monkeypatch.setattr("api.hardware._try_pipeline_profile", lambda: None)
        from api.hardware import discover_gpu_vram
        vram = discover_gpu_vram()
        assert vram == 0

    def test_falls_back_to_tf_when_pynvml_throws(self, monkeypatch):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = RuntimeError("No NVIDIA driver")
        monkeypatch.setitem(sys.modules, "pynvml", mock_pynvml)
        monkeypatch.setattr("api.hardware._try_tf_vram", lambda: 16384)

        from api.hardware import discover_gpu_vram
        vram = discover_gpu_vram()
        assert vram == 16384


class TestFallbackToTF:

    def test_tf_vram_from_device_details(self, monkeypatch):
        mock_tf = MagicMock()
        mock_gpu = MagicMock()
        mock_tf.config.list_physical_devices.return_value = [mock_gpu]
        mock_tf.config.experimental.get_device_details.return_value = {
            "memory_size": 12 * 1024 * 1024 * 1024,
        }
        monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)

        from api.hardware import _try_tf_vram
        vram = _try_tf_vram()
        assert vram == 12288

    def test_tf_no_gpus_returns_zero(self, monkeypatch):
        mock_tf = MagicMock()
        mock_tf.config.list_physical_devices.return_value = []
        monkeypatch.setitem(sys.modules, "tensorflow", mock_tf)

        from api.hardware import _try_tf_vram
        vram = _try_tf_vram()
        assert vram == 0

    def test_tf_import_fails_returns_zero(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tensorflow", None)
        from api.hardware import _try_tf_vram
        vram = _try_tf_vram()
        assert vram == 0
