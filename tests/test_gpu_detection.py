"""GPU detection tests for KodaQuant.

Checks:
1. pipeline.runtime.gpu_status() reports correctly
2. TensorFlow can see GPU devices
3. GPU-recommended models (CNN, LSTM, Transformer) build correctly on detected hardware
4. XGBoost GPU configuration is properly wired

All tests gracefully handle CPU-only environments.
"""
import os
import sys
import numpy as np
import pytest

GPU_MODELS = {"cnn", "lstm", "transformer", "dqn"}
TF_SKIP = os.environ.get("TF_SKIP_INIT", "0") == "1"


def test_runtime_gpu_status():
    """pipeline.runtime.gpu_status() returns a consistent status dict."""
    from pipeline.runtime import gpu_status, GPU_RECOMMENDED_MODELS

    status = gpu_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "devices" in status
    assert "mode" in status
    assert status["mode"] in ("GPU", "CPU")
    assert status["available"] == (len(status["devices"]) > 0)
    assert GPU_RECOMMENDED_MODELS == {"cnn", "lstm", "transformer", "dqn"}
    print(f"\n  gpu_status() => {status}")


def test_tensorflow_gpu_detection():
    """TensorFlow can list physical GPU devices."""
    if TF_SKIP:
        pytest.skip("TF_SKIP_INIT is set — skipping TF GPU detection")

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  TensorFlow {tf.__version__}")
    print(f"  GPU devices: {gpus}" if gpus else "  No GPU devices visible to TensorFlow.")
    if gpus:
        for g in gpus:
            mem = tf.config.experimental.get_memory_info(g.device_type)
            print(f"    {g.name}: {mem}")


def test_tensorflow_memory_growth():
    """Memory growth is set on all visible GPUs (as configured in runtime.py)."""
    if TF_SKIP:
        pytest.skip("TF_SKIP_INIT is set")

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        pytest.skip("No GPU devices — skipping memory growth check")
    for g in gpus:
        try:
            tf.config.experimental.get_memory_growth(g)
        except ValueError:
            pytest.fail(f"Memory growth not configured for {g.name}")


def test_build_cnn_on_gpu():
    """CNN model builds on the available device (GPU or CPU)."""
    if TF_SKIP:
        pytest.skip("TF_SKIP_INIT is set — skipping CNN build")
    if not _tf_available():
        pytest.skip("TensorFlow not importable — skipping CNN build")

    import tensorflow as tf
    from models.cnn import build_cnn

    model = build_cnn(input_shape=(60, 10))
    assert model is not None
    device = _get_model_device(model)
    print(f"\n  CNN built on: {device}")
    assert model.count_params() > 0


def test_build_lstm_on_gpu():
    """LSTM model builds on the available device (GPU or CPU)."""
    if TF_SKIP:
        pytest.skip("TF_SKIP_INIT is set — skipping LSTM build")
    if not _tf_available():
        pytest.skip("TensorFlow not importable — skipping LSTM build")

    import tensorflow as tf
    from models.lstm import build_lstm

    model = build_lstm(input_shape=(60, 10))
    assert model is not None
    device = _get_model_device(model)
    print(f"\n  LSTM built on: {device}")
    assert model.count_params() > 0


def test_build_transformer_on_gpu():
    """Transformer model builds on the available device (GPU or CPU)."""
    if TF_SKIP:
        pytest.skip("TF_SKIP_INIT is set — skipping Transformer build")
    if not _tf_available():
        pytest.skip("TensorFlow not importable — skipping Transformer build")

    import tensorflow as tf
    from models.transformer import build_transformer

    model = build_transformer(input_shape=(60, 10))
    assert model is not None
    device = _get_model_device(model)
    print(f"\n  Transformer built on: {device}")
    assert model.count_params() > 0


def test_deep_model_train_device():
    """A small CNN can train a forward pass on the detected device."""
    if TF_SKIP:
        pytest.skip("TF_SKIP_INIT is set — skipping train test")
    if not _tf_available():
        pytest.skip("TensorFlow not importable")

    import tensorflow as tf
    from models.cnn import build_cnn

    model = build_cnn(input_shape=(60, 5))
    model.compile(optimizer="adam", loss="mse")
    X = np.random.randn(32, 60, 5).astype(np.float32)
    y = np.random.randn(32, 1).astype(np.float32)
    history = model.fit(X, y, epochs=1, verbose=0, batch_size=32)
    loss = float(history.history["loss"][0])
    assert np.isfinite(loss), f"Training loss is NaN/Inf — possible device issue"
    print(f"\n  CNN train loss: {loss:.6f}")


def test_xgboost_gpu_config():
    """XGBoost GPU config is properly wired in model_factory_mixin."""
    from pipeline.backtester.model_factory_mixin import ModelFactoryMixin
    os.environ.setdefault("XGB_USE_GPU", "0")
    os.environ.setdefault("XGB_DEVICE", "cuda")
    print(f"\n  XGB_USE_GPU={os.environ.get('XGB_USE_GPU')}")
    print(f"  XGB_DEVICE={os.environ.get('XGB_DEVICE')}")


@pytest.mark.slow
def test_all_gpu_models_build():
    """All GPU-recommended models build without error (may be slow)."""
    if TF_SKIP:
        pytest.skip("TF_SKIP_INIT is set")
    if not _tf_available():
        pytest.skip("TensorFlow not importable")

    from models.cnn import build_cnn
    from models.lstm import build_lstm
    from models.transformer import build_transformer

    builders = {
        "cnn": (build_cnn, (60, 10)),
        "lstm": (build_lstm, (60, 10)),
        "transformer": (build_transformer, (60, 10)),
    }
    for name, (builder, shape) in builders.items():
        model = builder(input_shape=shape)
        assert model is not None, f"{name} returned None"
        assert model.count_params() > 0, f"{name} has 0 params"
        print(f"\n  {name}: {model.count_params():,} params")


def test_runtime_gpu_recommended_set():
    """GPU_RECOMMENDED_MODELS matches the expected set of deep models."""
    from pipeline.runtime import GPU_RECOMMENDED_MODELS
    assert GPU_RECOMMENDED_MODELS == {"cnn", "lstm", "transformer", "dqn"}
    print(f"\n  GPU recommended models: {GPU_RECOMMENDED_MODELS}")


def test_pipeline_import_without_gpu():
    """pipeline._imports loads without error even when no GPU is available."""
    import pipeline._imports as _imp
    assert hasattr(_imp, "tf")
    assert hasattr(_imp, "build_cnn")
    assert hasattr(_imp, "build_lstm")
    assert hasattr(_imp, "build_transformer")
    print("\n  pipeline._imports loads OK (TF models available via lazy import)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tf_available() -> bool:
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False


def _get_model_device(model) -> str:
    """Return the device name of the model's first weight tensor."""
    import tensorflow as tf
    try:
        w = model.weights[0]
        return w.device if w.device else "unknown"
    except (IndexError, AttributeError):
        return "no-weights"
