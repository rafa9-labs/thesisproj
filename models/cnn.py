import tensorflow as tf
from tensorflow import keras

Sequential              = keras.Sequential
Input                   = keras.Input
Conv1D                  = keras.layers.Conv1D
BatchNormalization      = keras.layers.BatchNormalization
GlobalAveragePooling1D  = keras.layers.GlobalAveragePooling1D
Dense                   = keras.layers.Dense
Dropout                 = keras.layers.Dropout
EarlyStopping = keras.callbacks.EarlyStopping
Adam          = keras.optimizers.Adam

def build_cnn(input_shape, config=None):
    """
    1D CNN classifier compatible with our CV/refit flow.

    Accepts BOTH prefixed and unprefixed keys, e.g.:
      - cnn_filters1 / filters1
      - cnn_filters2 / filters2
      - cnn_kernel_size / kernel_size
      - cnn_dense_units / dense_units
      - cnn_dropout_rate / dropout_rate
      - cnn_learning_rate / learning_rate
      - cnn_use_early_stopping / use_early_stopping
      - cnn_patience / patience
      - cnn_padding_same / padding_same
      - cnn_clipnorm / clipnorm
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"`input_shape` must be (timesteps, features), got: {input_shape}")
    if config is None:
        config = {}

    def g(pref, plain, default):
        return config.get(pref, config.get(plain, default))

    # Params (accept prefixed or plain)
    filters1      = max(1, int(g("cnn_filters1", "filters1", 32)))
    filters2      = max(1, int(g("cnn_filters2", "filters2", 64)))
    kernel_size   = max(1, int(g("cnn_kernel_size", "kernel_size", 3)))
    dense_units   = max(1, int(g("cnn_dense_units", "dense_units", 64)))
    dropout_rate  = float(min(max(0.0, g("cnn_dropout_rate", "dropout_rate", 0.3)), 0.8))
    learning_rate = float(max(1e-6, g("cnn_learning_rate", "learning_rate", 1e-3)))
    padding_same  = bool(g("cnn_padding_same", "padding_same", True))
    clipnorm_val  = float(g("cnn_clipnorm", "clipnorm", 0.0))

    # Guard for 'valid' padding with short windows
    timesteps, _ = input_shape
    if not padding_same and kernel_size > timesteps:
        print(f"[BUILD-CNN] kernel_size {kernel_size} > timesteps {timesteps} with 'valid' padding; clamping.")
        kernel_size = timesteps

    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=filters1, kernel_size=kernel_size, activation='relu',
               padding='same' if padding_same else 'valid'),
        BatchNormalization(),
        Conv1D(filters=filters2, kernel_size=kernel_size, activation='relu',
               padding='same' if padding_same else 'valid'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax', dtype='float32'),  # keep fp32 logits for AMP
    ])

    # EarlyStopping handle
    use_es   = bool(g("cnn_use_early_stopping", "use_early_stopping", False))
    patience = int(g("cnn_patience", "patience", 10))
    if use_es:
        model.early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, min_delta=1e-4
        )
    else:
        model.early_stop_callback = None

    # Optimizer (optional gradient clipping; default off)
    optimizer = Adam(learning_rate=learning_rate,
                     clipnorm=clipnorm_val if clipnorm_val > 0 else None)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if config.get("verbose_build", False):
        print(
            f"[BUILD-CNN] f1={filters1}, f2={filters2}, k={kernel_size}, dense={dense_units}, "
            f"drop={dropout_rate}, lr={learning_rate}, padding={'same' if padding_same else 'valid'}, "
            f"clipnorm={clipnorm_val if clipnorm_val > 0 else 'off'}, ES={use_es}"
        )
    return model
