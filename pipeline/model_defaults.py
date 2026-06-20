"""ModelDefaultsConfig — single source of truth for all model hyperparameters.

KodaQuant v3.0 Three-Tier Architecture:
  Tier 1 (HPO Toggle): User can fix a value OR define a search range.
                       Optuna samples within the range when HPO is enabled.
  Tier 2 (User Dropdown): User selects from options. Never sampled by HPO.
  Tier 3 (Hidden/Fixed): Hardcoded safe defaults. Shown in UI accordion.

Contract:
  - SEARCH_SPACE (config.py) is auto-derived from Tier 1 params only.
  - FIXED_DEFAULTS (config.py) is auto-derived from Tier 3 params only.
  - Sampler reads SEARCH_SPACE + FIXED_DEFAULTS; never hardcodes defaults.
  - Registry builders ingest from this config via key lookup.
  - API /models/hyperparams serves tiered metadata to frontend.

Discrepancies resolved:
  - Logistic solver: unified to lbfgs (was saga in registry)
  - Logistic max_iter: unified to 1000 (was 2000 in registry, 500 in sampler)
  - Logistic tol: unified to 1e-4 (was 1e-3 in registry)
  - RF class_weight: unified to balanced_subsample (was "balanced" in sampler)
  - LSTM clipnorm: unified to 1.0 (was 0.0 in build_lstm)
  - Transformer pooling: unified to cls (was "last" in build_transformer)
  - AdaptiveRegime train_lstm_on_trend_only: unified to True (was False in __init__)
"""

from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal[1, 2, 3]
ParamType = Literal["float_log", "float_linear", "int", "choice", "bool", "fixed"]
UIControl = Literal["slider", "dropdown", "toggle", "hidden"]


@dataclass(frozen=True)
class ParamDef:
    """Single hyperparameter definition.

    key:        Full prefixed internal key used by registry (e.g. "logit_C").
    hpo_key:    Short key used in SEARCH_SPACE (e.g. "C"). Same as key for ensembles.
    display_name: Human-readable name for the frontend (e.g. "C", "max_depth").
    tier:       1=HPO toggle, 2=user dropdown, 3=hidden/fixed.
    default:    Authoritative default value — single source of truth.
    type:       How the value is sampled/rendered.
    range:      (min, max, step/log) for slider, [list] for dropdown, None for fixed.
    description: Tooltip text for the frontend.
    ui_control: How the frontend renders this control.
    """
    key: str
    hpo_key: str
    display_name: str
    tier: Tier
    default: Any
    type: ParamType = "fixed"
    range: tuple | list | None = None
    description: str = ""
    ui_control: UIControl = "hidden"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build per-model ParamDef lists
# ═══════════════════════════════════════════════════════════════════════════

def _p(
    key: str, hpo_key: str, display: str, tier: Tier, default: Any,
    type: ParamType = "fixed", range: tuple | list | None = None,
    description: str = "", ui_control: UIControl = "hidden",
) -> ParamDef:
    return ParamDef(
        key=key, hpo_key=hpo_key, display_name=display, tier=tier,
        default=default, type=type, range=range,
        description=description, ui_control=ui_control,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-Model Parameter Definitions
# ═══════════════════════════════════════════════════════════════════════════

MODEL_PARAMS: dict[str, list[ParamDef]] = {}

# ── Logistic Regression ───────────────────────────────────────────────────

MODEL_PARAMS["logistic"] = [
    _p("logit_C", "C", "C", 1, 1.0, "float_log", (0.01, 100.0, True),
       "Inverse regularization strength. Smaller values = stronger regularization, reducing overfitting. Log-scale search covers the optimal range for FX signal sparsity.",
       "slider"),
    _p("logit_class_weight", "class_weight", "Class Weight", 2, "balanced", "choice",
       ["None", "balanced"],
       "Handle class imbalance by weighting minority classes. Balanced is standard for FX where signals are sparse.",
       "dropdown"),
    _p("logit_solver", "solver", "Solver", 3, "lbfgs", "fixed",
       description="Optimization algorithm. L-BFGS is fastest and most stable for multinomial logistic regression on FX data."),
    _p("logit_penalty", "penalty", "Penalty", 3, "l2", "fixed",
       description="Regularization norm. L2 (Ridge) is standard; L1 produces sparse feature selection at the cost of solver constraints."),
    _p("logit_max_iter", "max_iter", "Max Iterations", 3, 1000, "fixed",
       description="Maximum solver iterations. 1000 is sufficient for convergence on FX datasets; increasing beyond this rarely helps."),
    _p("logit_tol", "tol", "Tolerance", 3, 0.0001, "fixed",
       description="Convergence tolerance. 1e-4 provides adequate precision without unnecessary compute."),
]

# ── SVM ───────────────────────────────────────────────────────────────────

MODEL_PARAMS["svm"] = [
    _p("svm_C", "C", "C", 1, 1.0, "float_log", (0.01, 100.0, True),
       "Inverse regularization strength. Controls the margin width vs. training error tradeoff. Log-scale covers the full practical range for FX.",
       "slider"),
    _p("svm_gamma", "gamma", "Gamma", 1, 0.001, "choice",
       [0.0001, 0.001, 0.01, 0.05],
       "RBF kernel coefficient. Lower values = smoother decision boundaries (higher bias). Capped at 0.05 to prevent cubic-time training hangs.",
       "slider"),
    _p("svm_kernel", "kernel", "Kernel", 3, "rbf", "fixed",
       description="Kernel function. RBF is the standard for financial time series; linear and polynomial kernels add unnecessary complexity for FX."),
    _p("svm_class_weight", "class_weight", "Class Weight", 3, "balanced", "fixed",
       description="Automatically adjust weights inversely proportional to class frequencies."),
    _p("svm_max_iter", "max_iter", "Max Iterations", 3, 100000, "fixed",
       description="Hard iteration cap. 100k prevents infinite loops on difficult convergence; the solver typically converges well before this."),
    _p("svm_tol", "tol", "Tolerance", 3, 0.01, "fixed",
       description="Stopping criterion tolerance. 0.01 is sufficient for SVM on noisy FX data."),
    _p("svm_cache_size", "cache_size", "Cache (MB)", 3, 2048, "fixed",
       description="Kernel cache size in MB. 2GB handles large kernel matrices; capped at 1GB for multi-worker safety."),
]

# ── Random Forest ─────────────────────────────────────────────────────────

MODEL_PARAMS["random_forest"] = [
    _p("rf_n_estimators", "n_estimators", "N Estimators", 1, 300, "int",
       (200, 800, 100),
       "Number of trees. 200-800 covers the sweet spot: below 200 is underfit, above 800 brings diminishing returns and slower training.",
       "slider"),
    _p("rf_max_depth", "max_depth", "Max Depth", 1, 12, "int",
       (8, 20, 1),
       "Maximum tree depth. Limit this to prevent individual trees from memorizing noise. 8-20 is the practical range for FX with ~50-100 features.",
       "slider"),
    _p("rf_min_samples_leaf", "min_samples_leaf", "Min Samples Leaf", 1, 10, "int",
       (1, 10, 1),
       "Minimum samples per leaf. Higher values force broader splits and reduce overfitting. 1-10 is the effective range for walk-forward FX.",
       "slider"),
    _p("rf_max_features", "max_features", "Max Features", 1, "sqrt", "choice",
       ["sqrt", 0.33, 0.5],
       "Fraction of features considered per split. sqrt is the classic default; 0.33-0.5 limits tree correlation. None (all features) excluded to prevent overfit.",
       "slider"),
    _p("rf_class_weight", "class_weight", "Class Weight", 3, "balanced_subsample", "fixed",
       description="Adjust weights per bootstrap sample. Best for FX where signal/noise ratio varies across training windows."),
    _p("rf_bootstrap", "bootstrap", "Bootstrap", 3, True, "fixed",
       description="Use bootstrap samples for each tree. Required for out-of-bag scoring and ensemble diversity."),
    _p("rf_oob_score", "oob_score", "OOB Score", 3, True, "fixed",
       description="Use out-of-bag samples to estimate generalization error without a separate validation set."),
]

# ── Decision Tree ─────────────────────────────────────────────────────────

MODEL_PARAMS["decision_tree"] = [
    _p("dt_max_depth", "max_depth", "Max Depth", 1, 12, "int",
       (3, 15, 1),
       "Maximum tree depth. Decision trees are prone to overfitting on financial data; limit depth to control variance.",
       "slider"),
    _p("dt_min_samples_leaf", "min_samples_leaf", "Min Samples Leaf", 1, 10, "int",
       (1, 20, 1),
       "Minimum samples per leaf. Higher values increase bias but prevent memorization of noise.",
       "slider"),
    _p("dt_max_features", "max_features", "Max Features", 1, "sqrt", "choice",
       ["sqrt", "log2", None],
       "Features considered per split. sqrt is the default; log2 is more aggressive pruning; None uses all features.",
       "slider"),
    _p("dt_ccp_alpha", "ccp_alpha", "CCP Alpha", 1, 0.0001, "float_linear",
       (0.0, 0.01),
       "Cost-complexity pruning. Higher values prune more aggressively. 0.0-0.01 covers the effective range for FX.",
       "slider"),
    _p("dt_class_weight", "class_weight", "Class Weight", 3, "balanced", "fixed",
       description="Automatically adjust weights to handle class imbalance in sparse FX signals."),
]

# ── XGBoost ───────────────────────────────────────────────────────────────

MODEL_PARAMS["xgboost"] = [
    _p("xgb_n_estimators", "n_estimators", "N Estimators", 1, 400, "int",
       (200, 800, 100),
       "Number of boosting rounds. 200-800 is optimal for FX: below 200 is underfit, above 800 plateaus with walk-forward data.",
       "slider"),
    _p("xgb_max_depth", "max_depth", "Max Depth", 1, 6, "int",
       (3, 8, 1),
       "Maximum tree depth. XGBoost is aggressive — keep this low (3-8) to prevent memorizing training set noise.",
       "slider"),
    _p("xgb_learning_rate", "learning_rate", "Learning Rate", 1, 0.1, "float_log",
       (0.005, 0.3, True),
       "Step size shrinkage. Lower values = more conservative boosting. Log-scale search covers the full practical range.",
       "slider"),
    _p("xgb_subsample", "subsample", "Subsample", 1, 0.8, "float_linear",
       (0.6, 1.0),
       "Fraction of training rows per tree. <1.0 adds stochasticity and prevents overfitting.",
       "slider"),
    _p("xgb_colsample_bytree", "colsample_bytree", "Colsample By Tree", 1, 0.8, "float_linear",
       (0.6, 1.0),
       "Fraction of features per tree. Lower values reduce correlation between trees and improve ensemble diversity.",
       "slider"),
    _p("xgb_gamma", "gamma", "Gamma", 3, 0.0, "fixed",
       description="Minimum loss reduction for a split. 0.0 = no constraint; tuning this is a micro-optimization on FX data."),
    _p("xgb_min_child_weight", "min_child_weight", "Min Child Weight", 3, 1.0, "fixed",
       description="Minimum sum of instance weight in a child. 1.0 is the safe default."),
    _p("xgb_reg_lambda", "reg_lambda", "Reg Lambda", 3, 1.0, "fixed",
       description="L2 regularization on weights. Default 1.0 is sufficient for FX."),
    _p("xgb_reg_alpha", "reg_alpha", "Reg Alpha", 3, 0.0, "fixed",
       description="L1 regularization on weights. 0.0 is standard; non-zero adds sparsity."),
    _p("xgb_objective", "objective", "Objective", 3, "multi:softprob", "fixed",
       description="Multi-class softmax probability. Standard for 3-class FX (short/flat/long)."),
]

# ── LightGBM ──────────────────────────────────────────────────────────────

MODEL_PARAMS["lightgbm"] = [
    _p("lgbm_n_estimators", "n_estimators", "N Estimators", 1, 400, "int",
       (200, 800, 100),
       "Number of boosting rounds. Histogram-based boosting converges faster than XGBoost; 200-800 is sufficient.",
       "slider"),
    _p("lgbm_max_depth", "max_depth", "Max Depth", 1, 6, "int",
       (3, 8, 1),
       "Maximum tree depth. LightGBM grows leaf-wise — keep this low to prevent overfitting on FX noise.",
       "slider"),
    _p("lgbm_num_leaves", "num_leaves", "Num Leaves", 1, 31, "choice",
       [15, 31, 63, 127],
       "Maximum leaves per tree. Controls model complexity. 31 is the standard starting point.",
       "slider"),
    _p("lgbm_learning_rate", "learning_rate", "Learning Rate", 1, 0.1, "float_log",
       (0.01, 0.3, True),
       "Step size shrinkage. Log-scale covers the practical range for gradient boosting.",
       "slider"),
    _p("lgbm_subsample", "subsample", "Subsample", 1, 0.8, "float_linear",
       (0.6, 1.0),
       "Fraction of training rows per tree. <1.0 adds bagging for regularization.",
       "slider"),
    _p("lgbm_colsample_bytree", "colsample_bytree", "Colsample By Tree", 1, 0.8, "float_linear",
       (0.6, 1.0),
       "Fraction of features per tree. Controls tree diversity.",
       "slider"),
    _p("lgbm_reg_lambda", "reg_lambda", "Reg Lambda", 1, 1.0, "float_linear",
       (0.0, 10.0),
       "L2 regularization. Higher values penalize complex trees more aggressively.",
       "slider"),
    _p("lgbm_boosting_type", "boosting_type", "Boosting Type", 3, "gbdt", "fixed",
       description="GBDT (gradient boosting decision tree) is the standard for regression/classification tasks."),
    _p("lgbm_min_child_samples", "min_child_samples", "Min Child Samples", 3, 20, "fixed",
       description="Minimum data in a leaf. 20 prevents splits on tiny, noisy subsets."),
    _p("lgbm_class_weight", "class_weight", "Class Weight", 3, "balanced", "fixed",
       description="Auto-weight classes inversely proportional to frequency."),
]

# ── CatBoost ──────────────────────────────────────────────────────────────

MODEL_PARAMS["catboost"] = [
    _p("cb_iterations", "iterations", "Iterations", 1, 400, "int",
       (200, 800, 100),
       "Number of boosting iterations. Ordered boosting converges more reliably; 200-800 is optimal.",
       "slider"),
    _p("cb_depth", "depth", "Depth", 1, 6, "int",
       (3, 8, 1),
       "Tree depth. CatBoost uses symmetric trees — keep depth conservative for FX data.",
       "slider"),
    _p("cb_learning_rate", "learning_rate", "Learning Rate", 1, 0.1, "float_log",
       (0.01, 0.3, True),
       "Step size. Log-scale covers the practical range for ordered boosting.",
       "slider"),
    _p("cb_subsample", "subsample", "Subsample", 1, 0.8, "float_linear",
       (0.6, 1.0),
       "Fraction of training rows. <1.0 enables Bayesian bootstrap for built-in regularization.",
       "slider"),
    _p("cb_l2_leaf_reg", "l2_leaf_reg", "L2 Leaf Reg", 1, 3.0, "float_linear",
       (1.0, 10.0),
       "L2 regularization on leaf values. CatBoost defaults to 3.0 (higher than XGBoost/LightGBM).",
       "slider"),
    _p("cb_border_count", "border_count", "Border Count", 3, 128, "fixed",
       description="Buckets for numeric features. 128 provides sufficient resolution for FX price data."),
    _p("cb_loss_function", "loss_function", "Loss Function", 3, "MultiClass", "fixed",
       description="Multi-class classification. Standard for short/flat/long FX signals."),
]

# ── LSTM ──────────────────────────────────────────────────────────────────

MODEL_PARAMS["lstm"] = [
    _p("lstm_learning_rate", "learning_rate", "Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.005, True),
       "Adam optimizer learning rate. Log-scale search covers the stable range for sequence models on FX.",
       "slider"),
    _p("lstm_dropout_rate", "dropout_rate", "Dropout", 1, 0.3, "float_linear",
       (0.2, 0.5),
       "Dropout rate after dense layer. 0.2-0.5 provides meaningful regularization without destroying signal.",
       "slider"),
    _p("lstm_units", "units", "Units", 1, 64, "choice",
       [32, 64, 128],
       "Hidden units per LSTM layer. 32-128 balances capacity vs training speed on walk-forward windows.",
       "slider"),
    _p("lstm_num_layers", "num_layers", "Layers", 2, 1, "choice",
       [1, 2],
       "Stacked LSTM layers. 1 layer is standard; 2 layers add capacity for complex temporal patterns.",
       "dropdown"),
    _p("lstm_dense_units", "dense_units", "Dense Units", 3, 64, "fixed",
       description="Hidden units in the classification head. Per TF best practices for FX sequence models."),
    _p("lstm_bidirectional", "bidirectional", "Bidirectional", 3, False, "fixed",
       description="Process sequences forward and backward. False avoids look-ahead bias in walk-forward backtesting."),
    _p("lstm_clipnorm", "clipnorm", "Clipnorm", 3, 1.0, "fixed",
       description="Global gradient norm clipping. 1.0 prevents gradient explosions on volatile FX windows."),
    _p("lstm_use_early_stopping", "use_early_stopping", "Early Stopping", 3, True, "fixed",
       description="Halt training when validation loss plateaus. Critical for walk-forward to prevent over-epoching."),
    _p("lstm_patience", "patience", "Patience", 3, 15, "fixed",
       description="Epochs to wait before early stop. 15 provides sufficient runway for Adam to converge on FX data."),
]

# ── CNN ───────────────────────────────────────────────────────────────────

MODEL_PARAMS["cnn"] = [
    _p("cnn_learning_rate", "learning_rate", "Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.005, True),
       "Adam optimizer learning rate. CNN converges faster than LSTM; narrower range reflects this.",
       "slider"),
    _p("cnn_dropout_rate", "dropout_rate", "Dropout", 1, 0.3, "float_linear",
       (0.2, 0.5),
       "Dropout rate after the dense layer. Controls overfitting in the classification head.",
       "slider"),
    _p("cnn_filters1", "filters1", "Filters Layer 1", 1, 32, "choice",
       [32, 64, 96],
       "Convolution filters in first layer. More filters capture more patterns at the cost of training time.",
       "slider"),
    _p("cnn_filters2", "filters2", "Filters Layer 2", 1, 64, "choice",
       [32, 64, 96],
       "Convolution filters in second layer. Typically larger than Layer 1 for hierarchical feature learning.",
       "slider"),
    _p("cnn_kernel_size", "kernel_size", "Kernel Size", 2, 3, "choice",
       [3, 5],
       "Convolution kernel width (in bars). 3-bar and 5-bar windows capture short-term FX patterns.",
       "dropdown"),
    _p("cnn_dense_units", "dense_units", "Dense Units", 3, 64, "fixed",
       description="Hidden units in classification head. Standard for 1D CNN on sequence data."),
    _p("cnn_padding_same", "padding_same", "Padding", 3, True, "fixed",
       description="Same-padding preserves sequence length through convolution. Required for short FX windows."),
    _p("cnn_clipnorm", "clipnorm", "Clipnorm", 3, 1.0, "fixed",
       description="Global gradient norm clipping. 1.0 prevents gradient explosions on volatile data."),
    _p("cnn_use_early_stopping", "use_early_stopping", "Early Stopping", 3, True, "fixed",
       description="Halt training on plateau. CNNs converge faster than LSTMs; early stopping prevents overfitting."),
    _p("cnn_patience", "patience", "Patience", 3, 10, "fixed",
       description="Epochs to wait before early stop. 10 is sufficient given CNN convergence speed on FX."),
]

# ── Transformer ───────────────────────────────────────────────────────────

MODEL_PARAMS["transformer"] = [
    _p("transformer_learning_rate", "learning_rate", "Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.005, True),
       "Adam learning rate. Transformers are sensitive to learning rate; log-scale search is essential.",
       "slider"),
    _p("transformer_dropout_rate", "dropout_rate", "Dropout", 1, 0.1, "float_linear",
       (0.1, 0.4),
       "Dropout rate in attention and feed-forward layers. Transformers benefit from lower dropout (less capacity loss).",
       "slider"),
    _p("transformer_d_model", "d_model", "Model Dim", 1, 64, "choice",
       [32, 64, 128],
       "Embedding dimension. Controls the transformer's representational capacity. 128 is heavy for FX; prefer 32-64.",
       "slider"),
    _p("transformer_num_heads", "num_heads", "Attention Heads", 2, 4, "choice",
       [4, 8],
       "Multi-head attention heads. Must divide d_model evenly. 4 heads works well with d_model=32/64.",
       "dropdown"),
    _p("transformer_num_blocks", "num_blocks", "Blocks", 3, 1, "fixed",
       description="Transformer encoder blocks. 1 block is standard for FX sequence classification."),
    _p("transformer_ff_multiple", "ff_multiple", "FF Multiple", 3, 2, "fixed",
       description="Feed-forward expansion ratio relative to d_model. 2x is the standard transformer architecture."),
    _p("transformer_dense_units", "dense_units", "Dense Units", 3, 128, "fixed",
       description="Classification head hidden units. 128 provides sufficient capacity after global pooling."),
    _p("transformer_pooling", "pooling", "Pooling", 3, "cls", "fixed",
       description="Sequence aggregation method. CLS-token pooling is the standard transformer pattern for classification."),
    _p("transformer_clipnorm", "clipnorm", "Clipnorm", 3, 1.0, "fixed",
       description="Global gradient norm clipping. 1.0 stabilizes transformer training on noisy FX data."),
    _p("transformer_use_early_stopping", "use_early_stopping", "Early Stopping", 3, True, "fixed",
       description="Halt training on plateau. Essential for transformers which can quickly overfit FX windows."),
    _p("transformer_patience", "patience", "Patience", 3, 15, "fixed",
       description="Epochs to wait before early stop. 15 provides sufficient AdamW convergence runway."),
]

# ── GRU ───────────────────────────────────────────────────────────────────

MODEL_PARAMS["gru"] = [
    _p("gru_learning_rate", "learning_rate", "Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.005, True),
       "Adam learning rate. GRU trains faster than LSTM; narrower effective range.",
       "slider"),
    _p("gru_dropout_rate", "dropout_rate", "Dropout", 1, 0.3, "float_linear",
       (0.2, 0.5),
       "Dropout rate after dense layer. 0.2-0.5 provides meaningful regularization.",
       "slider"),
    _p("gru_units", "units", "Units", 1, 64, "choice",
       [32, 64, 128],
       "Hidden units per GRU layer. 32-128 balances capacity vs. training speed.",
       "slider"),
    _p("gru_num_layers", "num_layers", "Layers", 2, 1, "choice",
       [1, 2],
       "Stacked GRU layers. 1 layer is standard; 2 layers add temporal hierarchy for complex patterns.",
       "dropdown"),
    _p("gru_dense_units", "dense_units", "Dense Units", 3, 64, "fixed",
       description="Classification head hidden units. Standard for GRU sequence models."),
    _p("gru_bidirectional", "bidirectional", "Bidirectional", 3, False, "fixed",
       description="Process sequences forward and backward. False avoids look-ahead bias in walk-forward."),
    _p("gru_clipnorm", "clipnorm", "Clipnorm", 3, 1.0, "fixed",
       description="Global gradient norm clipping. 1.0 prevents gradient explosions on volatile FX windows."),
    _p("gru_use_early_stopping", "use_early_stopping", "Early Stopping", 3, True, "fixed",
       description="Halt training on plateau. GRU converges efficiently; early stopping prevents over-epoching."),
    _p("gru_patience", "patience", "Patience", 3, 15, "fixed",
       description="Epochs to wait before early stop. Matches LSTM for fair comparison."),
]

# ── GRU-LSTM Hybrid ───────────────────────────────────────────────────────

MODEL_PARAMS["gru_lstm"] = [
    _p("gru_lstm_learning_rate", "learning_rate", "Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.005, True),
       "Adam learning rate. Hybrid model balances GRU speed + LSTM memory; moderate learning rate range.",
       "slider"),
    _p("gru_lstm_dropout_rate", "dropout_rate", "Dropout", 1, 0.3, "float_linear",
       (0.2, 0.5),
       "Dropout rate after dense layer. Regularizes the combined GRU-LSTM feature space.",
       "slider"),
    _p("gru_lstm_gru_units", "gru_units", "GRU Units", 1, 64, "choice",
       [32, 64, 128],
       "Hidden units in the GRU stage. First in the stack; captures local temporal dynamics.",
       "slider"),
    _p("gru_lstm_lstm_units", "lstm_units", "LSTM Units", 1, 64, "choice",
       [32, 64, 128],
       "Hidden units in the LSTM stage. Second in the stack; captures longer-range dependencies.",
       "slider"),
    _p("gru_lstm_dense_units", "dense_units", "Dense Units", 3, 64, "fixed",
       description="Classification head hidden units. Standard for hybrid sequence models."),
    _p("gru_lstm_clipnorm", "clipnorm", "Clipnorm", 3, 1.0, "fixed",
       description="Global gradient norm clipping. 1.0 prevents gradient explosions."),
    _p("gru_lstm_use_early_stopping", "use_early_stopping", "Early Stopping", 3, True, "fixed",
       description="Halt training on plateau."),
    _p("gru_lstm_patience", "patience", "Patience", 3, 15, "fixed",
       description="Epochs to wait before early stop. Matches other sequence models."),
]

# ── Ensemble: Adaptive Regime (pruned from 11 to 3 T1 params) ─────────────

MODEL_PARAMS["ensemble_adaptive_regime"] = [
    _p("lstm_learning_rate", "lstm_learning_rate", "LSTM Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.01, True),
       "LSTM sub-model learning rate. The most impactful ensemble parameter — controls the trend regime expert.",
       "slider"),
    _p("rf_max_depth", "rf_max_depth", "RF Max Depth", 1, 12, "choice",
       [8, 12, 16, 20],
       "Random Forest sub-model depth. Controls the range regime expert's complexity.",
       "slider"),
    _p("logit_C", "logit_C", "Logistic C", 1, 1.0, "float_log",
       (0.01, 100.0, True),
       "Meta-logistic regularization. Controls how the ensemble weights sub-model outputs.",
       "slider"),
    _p("lstm_units", "lstm_units", "LSTM Units", 2, 64, "choice",
       [32, 64, 128],
       "LSTM hidden units. Structural choice that defines the trend expert's capacity.",
       "dropdown"),
    _p("rf_n_estimators", "rf_n_estimators", "RF N Estimators", 2, 300, "int",
       (100, 500, 100),
       "Number of RF trees. Structural capacity choice for the range expert.",
       "dropdown"),
    _p("adx_thresh", "adx_thresh", "ADX Threshold", 2, 25, "int",
       (15, 30, 1),
       "ADX threshold for regime detection. 25 is the Wilder standard; lower = more trend days classified.",
       "dropdown"),
    _p("vol_thresh", "vol_thresh", "Vol Threshold", 2, 0.002, "float_linear",
       (0.002, 0.02),
       "Volatility threshold for regime switching. Lower = more sensitive to volatility changes.",
       "dropdown"),
    _p("logit_solver", "logit_solver", "Logistic Solver", 3, "lbfgs", "fixed",
       description="Meta-learner solver. L-BFGS is fastest on the small ensemble output space."),
    _p("ensemble_method", "ensemble_method", "Ensemble Method", 3, "hard", "fixed",
       description="Hard voting selects the single best regime expert per sample. Soft explored; hard is more robust."),
]

# ── Ensemble: CNN-LSTM-XGBoost (pruned from 9 to 4 T1 params) ────────────

MODEL_PARAMS["ensemble_cnn_lstm_xgboost"] = [
    _p("cnn_learning_rate", "cnn_learning_rate", "CNN Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.005, True),
       "CNN sub-model learning rate. Controls how fast the spatial pattern expert adapts.",
       "slider"),
    _p("lstm_learning_rate", "lstm_learning_rate", "LSTM Learning Rate", 1, 0.001, "float_log",
       (1e-4, 0.005, True),
       "LSTM sub-model learning rate. Controls the sequential memory expert's adaptation speed.",
       "slider"),
    _p("xgb_learning_rate", "xgb_learning_rate", "XGB Learning Rate", 1, 0.1, "float_log",
       (0.005, 0.2, True),
       "XGBoost sub-model learning rate. Controls the tree ensemble expert's step size.",
       "slider"),
    _p("xgb_max_depth", "xgb_max_depth", "XGB Max Depth", 1, 6, "int",
       (3, 12, 1),
       "XGBoost tree depth. The most impactful structural parameter for the tree expert.",
       "slider"),
    _p("cnn_filters1", "cnn_filters1", "CNN Filters 1", 3, 64, "fixed",
       description="First conv layer filters. Fixed at 64 to prevent ensemble dimensionality explosion."),
    _p("cnn_filters2", "cnn_filters2", "CNN Filters 2", 3, 64, "fixed",
       description="Second conv layer filters. Fixed at 64 to match capacity across sub-models."),
    _p("cnn_kernel_size", "cnn_kernel_size", "CNN Kernel", 3, 3, "fixed",
       description="Convolution kernel width. Fixed at 3 bars; captures local FX microstructure."),
    _p("lstm_units", "lstm_units", "LSTM Units", 3, 64, "fixed",
       description="LSTM hidden units. Fixed at 64 to balance ensemble sub-model capacities."),
    _p("xgb_n_estimators", "xgb_n_estimators", "XGB N Estimators", 3, 400, "fixed",
       description="Number of boosting rounds. Fixed at 400 to maintain ensemble training speed."),
    _p("use_logit_meta", "use_logit_meta", "Use Logistic Meta", 3, True, "fixed",
       description="Use logistic regression as meta-learner on top of sub-model outputs."),
]

# ── Ensemble: Meta Ensemble (Signal Committee) ────────────────────────────

MODEL_PARAMS["meta_ensemble"] = [
    _p("meta_combination_method", "meta_combination_method", "Voting Method", 2, "majority", "choice",
       ["majority", "soft", "weighted"],
       "Signal combination strategy. Majority = equal vote per model. Soft = average probabilities. Weighted = user-defined model weights.",
       "dropdown"),
    _p("meta_sub_models", "meta_sub_models", "Sub-Models", 3, ["logistic", "xgboost"], "fixed",
       description="Member models selected by the user in the Models tab. Not tuned by HPO."),
    _p("meta_weights", "meta_weights", "Weights", 3, [], "fixed",
       description="Per-model voting weights for weighted combination. User-defined in frontend."),
]

# ── Ensemble: Stacking Ensemble ───────────────────────────────────────────

MODEL_PARAMS["stacking_ensemble"] = [
    _p("stack_cv", "stack_cv", "Stack CV", 2, 5, "choice",
       [3, 5, 8],
       "Cross-validation folds for out-of-fold meta-features. More folds = more training data per sub-model but slower.",
       "dropdown"),
    _p("stack_method", "stack_method", "Stack Method", 2, "auto", "choice",
       ["auto", "predict_proba"],
       "Meta-learner training data. Auto uses class labels; predict_proba uses probability vectors (more information).",
       "dropdown"),
    _p("stack_sub_models", "stack_sub_models", "Sub-Models", 3, ["logistic", "xgboost", "lightgbm"], "fixed",
       description="Member models selected by the user in the Models tab."),
]

# ── DQN (RL Agent — excluded from Optuna) ─────────────────────────────────

MODEL_PARAMS["dqn"] = [
    _p("dqn_gamma", "gamma", "Gamma", 3, 0.99, "fixed",
       description="Discount factor for future rewards. 0.99 weights near-term and far-term rewards nearly equally."),
    _p("dqn_epsilon", "epsilon", "Epsilon", 3, 1.0, "fixed",
       description="Initial exploration rate. 1.0 = pure exploration at start."),
    _p("dqn_epsilon_min", "epsilon_min", "Epsilon Min", 3, 0.1, "fixed",
       description="Minimum exploration rate after decay. 0.1 maintains some exploration throughout training."),
    _p("dqn_epsilon_decay", "epsilon_decay", "Epsilon Decay", 3, 0.995, "fixed",
       description="Multiplicative decay per step. 0.995 provides gradual transition from explore to exploit."),
    _p("dqn_learning_rate", "learning_rate", "Learning Rate", 3, 0.001, "fixed",
       description="Adam learning rate for the dueling network."),
    _p("dqn_batch_size", "batch_size", "Batch Size", 3, 32, "fixed",
       description="Replay buffer sample size per training step."),
    _p("dqn_buffer_size", "buffer_size", "Buffer Size", 3, 10000, "fixed",
       description="Experience replay memory capacity."),
    _p("dqn_target_update_freq", "target_update_freq", "Target Update Freq", 3, 10, "fixed",
       description="Steps between target network sync. 10 balances stability and learning speed."),
    _p("dqn_episodes", "episodes", "Episodes", 3, 2, "fixed",
       description="Training episodes per fold. 2 is sufficient for walk-forward; more can overfit."),
]

# ── Regime Classifier (internal/auxiliary) ─────────────────────────────────

MODEL_PARAMS["regime_classifier"] = [
    _p("regime_n_estimators", "n_estimators", "N Estimators", 3, 100, "fixed",
       description="Number of trees in the regime detection forest."),
    _p("regime_max_depth", "max_depth", "Max Depth", 3, 8, "fixed",
       description="Maximum tree depth for regime classification."),
    _p("regime_min_samples_leaf", "min_samples_leaf", "Min Samples Leaf", 3, 50, "fixed",
       description="Minimum samples per leaf. High value ensures regime labels are statistically meaningful."),
    _p("regime_class_weight", "class_weight", "Class Weight", 3, "balanced_subsample", "fixed",
       description="Handle regime class imbalance with per-bootstrap weighting."),
]


# ═══════════════════════════════════════════════════════════════════════════
# Derived dictionaries (imported by config.py, sampler.py, api/routers)
# ═══════════════════════════════════════════════════════════════════════════

def build_search_space() -> dict:
    """Auto-generate SEARCH_SPACE from all Tier 1 ParamDef entries.

    Returns {model_key: {hpo_key: range_spec}} matching the current
    config.SEARCH_SPACE format so the sampler can consume it unchanged.
    """
    space: dict[str, dict] = {}
    for model_key, params in MODEL_PARAMS.items():
        t1 = {p.hpo_key: p.range for p in params if p.tier == 1 and p.range is not None}
        if t1:
            space[model_key] = t1
    return space


def build_fixed_defaults() -> dict:
    """Auto-generate fixed defaults for Tier 3 params (hidden/computational).

    Returns {model_key: {prefixed_key: default_value}} consumed by the
    sampler to inject Tier 3 values into the param dict before model build.
    """
    fixed: dict[str, dict] = {}
    for model_key, params in MODEL_PARAMS.items():
        t3 = {p.key: p.default for p in params if p.tier == 3}
        if t3:
            fixed[model_key] = t3
    return fixed


def get_defaults(model_key: str) -> dict[str, Any]:
    """Return {prefixed_key: default_value} for ALL tiers of a model.

    Used by registry builders to look up authoritative defaults.
    """
    return {p.key: p.default for p in MODEL_PARAMS.get(model_key, [])}
