"""Shared fixtures for pipeline integrity tests."""
import gc
import os
import pytest

os.environ.setdefault("SKLEARN_JOBS", "1")


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


@pytest.fixture(autouse=True)
def _cleanup_between_tests():
    """Force garbage collection and clear TF session between tests.

    Prevents memory accumulation from NumPy arrays, TF sessions, and
    joblib worker pools that persist across test boundaries.
    """
    yield
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    except Exception:
        pass


@pytest.fixture(scope="session")
def _restrict_threading():
    """Limit thread pools across the entire test session.

    Sets OMP_NUM_THREADS and JOBLIB_NUM_CPUS to prevent joblib/mkl
    from spawning excessive worker threads in low-memory environments.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("JOBLIB_START_METHOD", "loky")
    yield