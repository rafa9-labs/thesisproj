import tensorflow as tf
from tensorflow import keras

# Layer / model aliases
Input                   = keras.Input
Dense                   = keras.layers.Dense
Dropout                 = keras.layers.Dropout
LayerNormalization      = keras.layers.LayerNormalization
MultiHeadAttention      = keras.layers.MultiHeadAttention
GlobalAveragePooling1D  = keras.layers.GlobalAveragePooling1D
Add                     = keras.layers.Add
Lambda                  = keras.layers.Lambda

Model = keras.Model

Adam = keras.optimizers.Adam

# AdamW may live in different places depending on TF version
try:
    AdamW = keras.optimizers.AdamW
except AttributeError:  # fallback for older TF
    from tensorflow.keras.optimizers.experimental import AdamW  # type: ignore

EarlyStopping = keras.callbacks.EarlyStopping



def _robust_sparse_ce(num_classes=None):
    """
    Robust sparse-categorical CE that tolerates y_true shaped as (N,), (N,1) or (N,T).
    If rank>1, we take the last element along the last axis (label at window end).
    """
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    @tf.function
    def loss_fn(y_true, y_pred):
        # y_pred: (N, C)
        c = tf.shape(y_pred)[-1] if num_classes is None else tf.convert_to_tensor(num_classes, dtype=tf.int32)

        y_true = tf.convert_to_tensor(y_true)
        # If labels have an extra dimension (N, 1) or (N, T), use the last step
        y_true = tf.cond(
            tf.rank(y_true) > 1,
            lambda: y_true[..., -1],
            lambda: y_true
        )
        # Flatten to (N,)
        y_true = tf.reshape(y_true, (-1,))
        y_true = tf.cast(y_true, tf.int32)

        # One-hot for stable broadcasting regardless of y_true rank/dtype
        y_true_oh = tf.one_hot(y_true, depth=c)
        return ce(y_true_oh, y_pred)

    return loss_fn


def build_transformer(input_shape, config=None):
    """
    Time-series classifier Transformer (no XLA).
    Keys (prefixed/unprefixed both accepted):
      - transformer_num_blocks     / num_blocks
      - transformer_num_heads      / num_heads
      - transformer_d_model        / d_model
      - transformer_d_multiple     / d_multiple
      - transformer_d_multiple_v2  / d_multiple_v2  (→ d_model = num_heads * d_multiple)
      - transformer_ff_dim         / ff_dim
      - transformer_dropout_rate   / dropout_rate
      - transformer_dense_units    / dense_units
      - transformer_learning_rate  / learning_rate
      - transformer_clipnorm       / clipnorm
      - transformer_use_early_stopping / use_early_stopping
      - transformer_patience       / patience
      - transformer_use_adamw      / use_adamw
      - transformer_weight_decay   / weight_decay
      - transformer_loss           / loss  ("robust_sparse_ce" | "sparse_cce")
    """
    if not (isinstance(input_shape, tuple) and len(input_shape) == 2):
        raise ValueError(f"`input_shape` must be (timesteps, features), got: {input_shape}")
    if config is None:
        config = {}

    def g(pref_key, plain_key, default):
        return config.get(pref_key, config.get(plain_key, default))

    # ----- hyperparams -----
    num_blocks  = int(g("transformer_num_blocks", "num_blocks", 1))
    num_heads   = int(g("transformer_num_heads", "num_heads", 2))

    d_multiple  = g("transformer_d_multiple_v2", "d_multiple", None)
    if d_multiple is not None:
        d_model = int(num_heads) * int(d_multiple)
    else:
        d_model = int(g("transformer_d_model", "d_model", 64))

    # FFN width: default to ~4×d_model (canonical Transformer convention)
    ff_dim      = int(g("transformer_ff_dim", "ff_dim", 4 * d_model))
    drop_rate   = float(min(max(0.0, g("transformer_dropout_rate", "dropout_rate", 0.1)), 0.5))
    dense_units = int(g("transformer_dense_units", "dense_units", ff_dim))
    learning_rate = float(max(1e-6, g("transformer_learning_rate", "learning_rate", 1e-3)))
    # Mild gradient clipping by default for stability on noisy FX series
    clipnorm_val  = float(g("transformer_clipnorm", "clipnorm", 1.0))

    patience      = int(g("transformer_patience", "patience", 15))
    use_es        = bool(g("transformer_use_early_stopping", "use_early_stopping", True))

    use_adamw     = bool(g("transformer_use_adamw", "use_adamw", False))
    weight_decay  = float(g("transformer_weight_decay", "weight_decay", 1e-4))
    loss_choice   = str(g("transformer_loss", "loss", "robust_sparse_ce")).lower()

    # Ensure divisibility
    if d_model % num_heads != 0:
        new_d_model = max(num_heads, int(round(d_model / num_heads) * num_heads))
        print(f"[BUILD-TRANS] Adjusting d_model {d_model} → {new_d_model} to be divisible by heads={num_heads}")
        d_model = new_d_model
    key_dim = d_model // num_heads

    # ----- model -----
    timesteps, feat_dim = input_shape
    inputs = Input(shape=input_shape)

    x = inputs
    if feat_dim != d_model:
        x = Dense(d_model)(x)

    # Sinusoidal positional encodings (dtype-safe)
    def _sinusoid_positions(T, D):
        import numpy as _np
        pos = _np.arange(T)[:, None]
        i   = _np.arange(D)[None, :]
        angle = pos / _np.power(10000, (2*(i//2))/D)
        pe = _np.zeros((T, D), dtype="float32")
        pe[:, 0::2] = _np.sin(angle[:, 0::2])
        pe[:, 1::2] = _np.cos(angle[:, 1::2])
        return tf.constant(pe)

    pos = _sinusoid_positions(timesteps, d_model)  # (T, D)
    pos = tf.expand_dims(pos, axis=0)              # (1, T, D) -> explicit broadcast to (B, T, D)
    if getattr(x, "dtype", None) is not None and pos.dtype != x.dtype:
        pos = tf.cast(pos, x.dtype)
    x = Add()([x, pos])

    def add_residual(a, b):
        if a.dtype != b.dtype:
            a = tf.cast(a, b.dtype)
        return Add()([a, b])

    for _ in range(num_blocks):
        # MHA block (PreNorm)
        h = LayerNormalization(epsilon=1e-6)(x)
        h = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=drop_rate)(h, h, use_causal_mask=True)
        h = Dropout(drop_rate)(h)
        x = add_residual(x, h)

        # FFN block
        h = LayerNormalization(epsilon=1e-6)(x)
        h = Dense(ff_dim, activation="gelu")(h)
        h = Dropout(drop_rate)(h)
        h = Dense(d_model)(h)
        x = add_residual(x, h)

    pooling = str(g("transformer_pooling", "pooling", "last")).lower()
    h = LayerNormalization(epsilon=1e-6)(x)
    if pooling == "last":
        # take last time step (causal readout)
        h = Lambda(lambda t: t[:, -1, :])(h)
    else:
        # default: GAP over time
        h = GlobalAveragePooling1D()(h)
    h = Dropout(drop_rate)(h)
    h = Dense(dense_units, activation="gelu")(h)
    # Keep classifier in float32 for stable numerics
    outputs = Dense(3, activation="softmax", dtype="float32")(h)

    model = Model(inputs=inputs, outputs=outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # --- Optimizer ---
    opt_name = str(g("transformer_optimizer", "optimizer", "adam")).lower()
    if use_adamw and (opt_name == "adamw") and (AdamW is not None):
        optimizer = AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm_val if clipnorm_val > 0 else None,
        )
    else:
        # Fallback (or explicit adam choice)
        optimizer = Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm_val if clipnorm_val > 0 else None,
        )


    # --- Loss: robust by default ---
    if loss_choice == "sparse_cce":
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        # robust_sparse_ce (default)
        loss_obj = _robust_sparse_ce(num_classes=3)

    model.compile(
        optimizer=optimizer,
        loss=loss_obj,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        jit_compile=False  # keep XLA off
    )

    # Early stopping (optional)
    if use_es:
        model.early_stop_callback = EarlyStopping(
            monitor="val_loss", mode="min", patience=patience,
            restore_best_weights=True, min_delta=1e-4, verbose=0
        )
    else:
        model.early_stop_callback = None

    if config.get("verbose_build", False):
        print(
            f"[BUILD-TRANS] blocks={num_blocks} d_model={d_model} heads={num_heads} "
            f"ff_dim={ff_dim} drop={drop_rate} lr={learning_rate} "
            f"clipnorm={'off' if clipnorm_val <= 0 else clipnorm_val} "
            f"ES={bool(model.early_stop_callback)} XLA=OFF"
        )
    return model
