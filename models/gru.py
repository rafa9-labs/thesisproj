"""GRU classifier — same interface as LSTM for drop-in comparison.

Literature:
  - Springer Digital Finance (2020): GRU vs LSTM across 4 FX pairs —
    GRU simpler, faster, statistically competitive or superior.
  - ScienceDirect (2020): "Foreign Exchange Currency Rate Prediction
    using a GRU-LSTM Hybrid Network"
  - Springer (2024): "Forex Prediction Using Deep Learning: CNN, LSTM,
    GRU and Hybrid Models"

Accepts BOTH prefixed and unprefixed keys:
  - gru_units / units
  - gru_dense_units / dense_units
  - gru_dropout_rate / dropout_rate
  - gru_learning_rate / learning_rate
  - gru_num_layers / num_layers
  - gru_bidirectional / bidirectional
  - gru_use_early_stopping / use_early_stopping
  - gru_patience / patience
  - gru_clipnorm / clipnorm
"""
import tensorflow as tf
from tensorflow import keras

Sequential    = keras.Sequential
GRU           = keras.layers.GRU
Dense         = keras.layers.Dense
Dropout       = keras.layers.Dropout
Bidirectional = keras.layers.Bidirectional
Input         = keras.Input

Adam          = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping
l2            = keras.regularizers.l2


def build_gru(input_shape, config=None):
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"input_shape must be (timesteps, features), got: {input_shape}")
    if config is None:
        config = {}

    def g(pref, plain, default):
        return config.get(pref, config.get(plain, default))

    gru_units      = max(1, int(g("gru_units", "units", 64)))
    dense_units    = max(1, int(g("gru_dense_units", "dense_units", 64)))
    dropout_rate   = float(min(max(0.0, g("gru_dropout_rate", "dropout_rate", 0.3)), 0.5))
    learning_rate  = float(max(1e-6, g("gru_learning_rate", "learning_rate", 1e-3)))
    num_layers     = max(1, int(g("gru_num_layers", "num_layers", 1)))
    bidirectional  = bool(g("gru_bidirectional", "bidirectional", False))
    clipnorm_val   = float(g("gru_clipnorm", "clipnorm", 0.0))

    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(num_layers):
        return_seq = (i < num_layers - 1)

        gru_kwargs = dict(
            units=gru_units,
            return_sequences=return_seq,
        )

        gru_layer = GRU(**gru_kwargs)
        if bidirectional:
            gru_layer = Bidirectional(gru_layer)

        model.add(gru_layer)

    reg_val = float(g("gru_l2", "l2", 0.0))
    model.add(Dense(dense_units,
                    activation="gelu",
                    kernel_regularizer=(l2(reg_val) if reg_val > 0 else None)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation="softmax", dtype="float32"))

    use_es   = bool(g("gru_use_early_stopping", "use_early_stopping", False))
    patience = int(g("gru_patience", "patience", 15))
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
            f"[BUILD-GRU] input_shape={input_shape} | layers={num_layers} | units={gru_units} "
            f"| bidirectional={bidirectional} | dense={dense_units} | dr={dropout_rate} "
            f"| lr={learning_rate} | clipnorm={clipnorm_val if clipnorm_val > 0 else 'off'} "
            f"| early_stop={use_es} (patience={patience if use_es else 'NA'})"
        )
    return model
