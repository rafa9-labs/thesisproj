"""Shared fixtures for benchmark tests."""
import os
import pytest
import numpy as np


_N_ROWS = 500
_N_FEATURES = 12
_TIMESTEPS = 20


@pytest.fixture(scope="module")
def flat_synthetic():
    """Small 2D classification dataset for classical model benchmarks."""
    rng = np.random.default_rng(12345)
    X = rng.standard_normal((_N_ROWS, _N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 3, size=_N_ROWS).astype(np.int32)
    return X, y


@pytest.fixture(scope="module")
def seq_synthetic():
    """Small 3D + 2D dataset for deep/ensemble model benchmarks."""
    rng = np.random.default_rng(12345)
    X_seq = rng.standard_normal((_N_ROWS, _TIMESTEPS, _N_FEATURES)).astype(np.float32)
    X_flat = X_seq.mean(axis=1)
    y = rng.integers(0, 3, size=_N_ROWS).astype(np.int32)
    return X_seq, X_flat, y