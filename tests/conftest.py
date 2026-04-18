"""Shared fixtures for pipeline integrity tests."""
import pytest


@pytest.fixture(scope="session")
def pipeline_imports():
    """Import pipeline._imports once for the whole test session."""
    import pipeline._imports
    return pipeline._imports


@pytest.fixture(scope="session")
def ml_backtester_class():
    """Return the composed MLBacktester class."""
    from pipeline.backtester.composed import MLBacktester
    return MLBacktester


@pytest.fixture(scope="session")
def numpy_arr():
    """Small numpy array for metric tests."""
    import numpy as np
    return np.array