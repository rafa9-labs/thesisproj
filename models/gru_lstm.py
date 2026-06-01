"""GRU-LSTM Hybrid — stacks GRU then LSTM layers for forex classification.

Literature:
  - Nature Scientific Reports (2025): Islam & Hossain — hybrid GRU-LSTM
    "outperforms standalone models and statistical techniques in predicting
    FOREX currency prices."
  - ScienceDirect (2020): "Foreign Exchange Currency Rate Prediction
    using a GRU-LSTM Hybrid Network" — beats standalone GRU, LSTM, and
    SMA on GBP/USD and USD/CAD.

Accepts BOTH prefixed and unprefixed keys:
  - gru_lstm_units / gru_lstm_gru_units
  - gru_lstm_lstm_units
  - gru_lstm_dense_units / gru_lstm_dense_units
  - gru_lstm_dropout_rate / gru_lstm_dropout_rate
  - gru_lstm_learning_rate / gru_lstm_learning_rate
  - gru_lstm_clipnorm / gru_lstm_clipnorm
  - gru_lstm_use_early_stopping / gru_lstm_use_early_stopping
  - gru_lstm_patience / gru_lstm_patience
"""
import tensorflow as tf
from tensorflow import keras

Sequential    = keras.Sequential
GRU           = keras.layers.GRU
LSTM          = keras.layers.LSTM
Dense         = keras.layers.Dense
Dropout       = keras.layers.Dropout
Input         = keras.Input

Adam          = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping
l2            = keras.regularizers.l2


def build_gru_lstm(input_shape, config=None):
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"input_shape must be (timesteps, features), got: {input_shape}")
    if config is None:
        config = {}

    P = "gru_lstm_"

    def g(key, default):
        return config.get(f"{P}{key}", config.get(key, default))

    gru_units     = max(1, int(g("gru_units", 64)))
    lstm_units    = max(1, int(g("lstm_units", 64)))
    dense_units   = max(1, int(g("dense_units", 64)))
    dropout_rate  = float(min(max(0.0, g("dropout_rate", 0.3)), 0.5))
    learning_rate = float(max(1e-6, g("learning_rate", 1e-3)))
    clipnorm_val  = float(g("clipnorm", 0.0))

    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(GRU(units=gru_units, return_sequences=True))

    model.add(LSTM(units=lstm_units, return_sequences=False))

    reg_val = float(g("l2", 0.0))
    model.add(Dense(dense_units,
                    activation="gelu",
                    kernel_regularizer=(l2(reg_val) if reg_val > 0 else None)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation="softmax", dtype="float32"))

    use_es   = bool(g("use_early_stopping", False))
    patience = int(g("patience", 15))
    if use_es:
        model.early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-4
        )
    else:
        model.early_stop_callback = None

    optimizer = Adam(learning_rate=learning_rate,
                     clipnorm=clipnorm_val if clipnorm_val > 0 else None)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    if config.get("verbose_build", False):
        print(
            f"[BUILD-GRU-LSTM] input_shape={input_shape} | gru_units={gru_units} "
            f"| lstm_units={lstm_units} | dense={dense_units} | dr={dropout_rate} "
            f"| lr={learning_rate} | clipnorm={clipnorm_val if clipnorm_val > 0 else 'off'} "
            f"| early_stop={use_es} (patience={patience if use_es else 'NA'})"
        )
    return model
