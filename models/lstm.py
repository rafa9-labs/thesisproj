import tensorflow as tf
from tensorflow import keras

# Aliases so the rest of your code (Sequential, LSTM, etc.) stays the same
Sequential    = keras.Sequential
LSTM          = keras.layers.LSTM
Dense         = keras.layers.Dense
Dropout       = keras.layers.Dropout
Bidirectional = keras.layers.Bidirectional
Input         = keras.Input

Adam          = keras.optimizers.Adam
EarlyStopping = keras.callbacks.EarlyStopping
l2            = keras.regularizers.l2


def build_lstm(input_shape, config=None):
    """
    Build a classification LSTM compatible with our fast-CV + final-refit flow.

    Accepts BOTH prefixed and unprefixed keys, e.g.:
      - lstm_units / units
      - lstm_dense_units / dense_units
      - lstm_dropout_rate / dropout_rate
      - lstm_learning_rate / learning_rate
      - lstm_num_layers / num_layers
      - lstm_bidirectional / bidirectional
      - lstm_use_early_stopping / use_early_stopping
      - lstm_patience / patience
      - lstm_clipnorm / clipnorm
    """

    # tf.random.set_seed(42)

    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"input_shape must be (timesteps, features), got: {input_shape}")
    if config is None:
        config = {}

    def g(pref, plain, default):
        return config.get(pref, config.get(plain, default))

        # Hyperparameters (accept prefixed or plain)
    lstm_units    = max(1, int(g("lstm_units", "units", 64)))
    dense_units   = max(1, int(g("lstm_dense_units", "dense_units", 64)))
    dropout_rate  = float(min(max(0.0, g("lstm_dropout_rate", "dropout_rate", 0.3)), 0.5))
    learning_rate = float(max(1e-6, g("lstm_learning_rate", "learning_rate", 1e-3)))
    num_layers    = max(1, int(g("lstm_num_layers", "num_layers", 1)))
    bidirectional = bool(g("lstm_bidirectional", "bidirectional", False))
    clipnorm_val  = float(g("lstm_clipnorm", "clipnorm", 0.0))

    # Build stacked LSTM(s)
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(num_layers):
        return_seq = (i < num_layers - 1)

        lstm_kwargs = dict(
            units=lstm_units,
            return_sequences=return_seq,
        )

        lstm_layer = LSTM(**lstm_kwargs)
        if bidirectional:
            lstm_layer = Bidirectional(lstm_layer)

        model.add(lstm_layer)
        
    # Head
    reg_val = float(g("lstm_l2", "l2", 0.0))
    model.add(Dense(dense_units,
                    activation="gelu",
                    kernel_regularizer=(l2(reg_val) if reg_val > 0 else None)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation="softmax", dtype="float32"))  # fp32 logits for AMP

    # Early stopping
    use_es   = bool(g("lstm_use_early_stopping", "use_early_stopping", False))
    patience = int(g("lstm_patience", "patience", 15))
    if use_es:
        model.early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-4
        )
    else:
        model.early_stop_callback = None

    # Compile
    optimizer = Adam(learning_rate=learning_rate,
                     clipnorm=clipnorm_val if clipnorm_val > 0 else None)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    if config.get("verbose_build", False):
        print(
            f"[BUILD-LSTM] input_shape={input_shape} | layers={num_layers} | units={lstm_units} "
            f"| bidirectional={bidirectional} | dense={dense_units} | dr={dropout_rate} "
            f"| lr={learning_rate} | clipnorm={clipnorm_val if clipnorm_val > 0 else 'off'} "
            f"| early_stop={use_es} (patience={patience if use_es else 'NA'})"
        )
    return model
