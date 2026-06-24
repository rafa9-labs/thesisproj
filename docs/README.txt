# FX MLBacktester — Short User Guide (Supervisor-Friendly)

This project is a bar-based FX ML trading pipeline with:
- leak-free (causal) feature/label construction
- cost-aware execution (spread/slippage) on an executed after-cost equity stream
- one-bar decision → execution delay
- walk-forward monthly refit + evaluation

------------------------------------------------------------
Quick test:
- In MLBacktesterNoWFO.py set:
  MODEL_LIST = ["logistic", "xgboost"]
  SEEDS = [33333]; REPEATS = 1
  N_REAL_MONTHS = 6
- Run: python MLBacktesterNoWFO.py
------------------------------------------------------------

1) Run it

Main entrypoint:
    python MLBacktesterNoWFO.py

This runs the “FINAL EXPERIMENT” real-trading simulation loop (month-by-month) for the models you select.

------------------------------------------------------------

2) Where you set the inputs (exact locations / dict names)

A) Select which models to run
File: MLBacktesterNoWFO.py
Function: main()
Section: “2) Choose models (edit this list as you like)”

Edit:
    MODEL_LIST = [
        "logistic",
        "xgboost",
        ...
    ]

Supported model strings (examples):
- Classical: logistic, svm, decision_tree, random_forest, xgboost, lightgbm, catboost
- Deep: cnn, lstm, transformer, gru, gru_lstm
- Ensembles: ensemble_cnn_lstm_xgboost, ensemble_adaptive_regime, meta_ensemble, stacking_ensemble
- RL: dqn


B) Select repeats + seeds (reproducibility)
File: MLBacktesterNoWFO.py
Function: main()
Section: “FINAL EXPERIMENT: fixed per-run seeds”

Edit:
    SEEDS = [11111, 22222, 33333]
    REPEATS = 3   # must match len(SEEDS)

Meaning:
- The pipeline is executed once per seed.
- Each repeat is fully isolated and written under:
    <RUN_DIR>/repetition_1, repetition_2, ...


C) Select how many walk-forward test months + end date
File: MLBacktesterNoWFO.py
Function: main()
Same block as SEEDS/REPEATS

Edit:
    N_REAL_MONTHS = 36
    END_DATE = "2025-12-01 00:00:00"

Meaning:
- END_DATE anchors the end of available data for split generation.
- N_REAL_MONTHS is how many sequential test months are executed in the real trading simulation loop.


D) Feature + gating configuration (user JSON)
File: configs/feature_config.json
Loaded in MLBacktesterNoWFO.py main() by:
    with open("configs/feature_config.json","r") as f:
        features_config = json.load(f)

This JSON is where you define:
- which indicators/features are enabled
- indicator windows (periods)
- gating settings (confidence_threshold OR target_active_rate/target_coverage)
- triple-barrier label settings (if enabled)


E) Research defaults that override JSON (for fairness)
File: MLBacktesterNoWFO.py (global dict)
Dict name:
    CLASS_DEFAULTS["features"]
    CLASS_DEFAULTS["cv"]

Enforced in MLBacktesterNoWFO.py main() by:
    for _k, _v in CLASS_DEFAULTS["features"].items():
        features_config[_k] = deepcopy(_v) if isinstance(_v,(dict,list)) else _v

Interpretation:
- JSON is read, but key “experiment policy” parameters are locked via CLASS_DEFAULTS.
- This is intentional to keep comparisons fair across models.

Related names in MLBacktesterNoWFO.py:
    DEFAULT_FEATURES = deepcopy(CLASS_DEFAULTS["features"])
    DEFAULT_CV       = deepcopy(CLASS_DEFAULTS["cv"])


F) Train/test months (default training span per model family)
File: utilsNoWFO.py
Dict names:
    TRAIN_TEST_MONTHS
    TRAIN_TEST_MONTHS_DEBUG

Meaning:
- Default training span and test span are defined per model.
- Used by tuning/WFO split logic when explicit months are not provided.

Debug shortcut:
File: tuningNoWFO.py
Variable:
    TRAIN_TEST_DEBUG_MODE = False
Set True for shorter train spans via TRAIN_TEST_MONTHS_DEBUG.

------------------------------------------------------------

3) The few parameters supervisors actually care about

A) Feature expansion (size/complexity)
Usually set in configs/feature_config.json (unless locked by CLASS_DEFAULTS["features"]):
- indicator_windows: indicator periods (RSI length, BBands window, etc.)
- lag_depth: how many lags per selected feature
- roll_windows: rolling windows for rollmean/rollstd/rollslope expansion
- lags / lags_range: number of raw return lags (returns_lagk)

More lags/windows -> more features -> slower and more RAM.

B) Trading “gate” (trade vs no-trade)
You generally use ONE of these approaches:

1) Confidence threshold mode:
- confidence_threshold

2) Coverage / active-rate mode (recommended for stable trade frequency):
- target_active_rate and/or target_coverage

Important rule:
If target_active_rate or target_coverage is > 0, the system treats it as “coverage intent”
(even if you didn’t explicitly set a gating mode).

C) Labels
- use_triple_barrier + tb_* parameters control triple-barrier style event labeling.

------------------------------------------------------------

4) Outputs (where to look)

Each run creates a results directory and writes per-model artifacts under it (CSV + plots).
Typical layout:
- csv/     monthly results tables
- graphs/  equity curves and comparisons
- heatmaps/ summary visuals (if enabled)

All metrics are computed from the executed after-cost equity stream under the one-bar delay.

------------------------------------------------------------

5) Example “safe” runs (supervisor can reproduce quickly)

Minimal (fast sanity test)
In MLBacktesterNoWFO.py main():
    MODEL_LIST = ["logistic", "xgboost"]
    SEEDS = [33333]
    REPEATS = 1
    N_REAL_MONTHS = 6

Run:
    python MLBacktesterNoWFO.py


Deep + ensemble path
    MODEL_LIST = ["xgboost", "cnn", "ensemble_cnn_lstm_xgboost"]
    SEEDS = [33333]
    REPEATS = 1
    N_REAL_MONTHS = 6

Optional: DQN (separate RL path)
    MODEL_LIST = ["dqn"]
    SEEDS = [33333]
    REPEATS = 1
    N_REAL_MONTHS = 6

------------------------------------------------------------

6) OS / environment variables (what they are for)

These are optional knobs set in the shell before running:

Performance / threading:
- MLB_THREADS=<n>            default CPU core budget per model fit
- BLAS_THREADS_PER_TRIAL=<n> hard override for BLAS/OpenMP thread counts
- CV_JOBS=<n>                cores used inside CV evaluation loops

GPU/CPU selection:
- TF_FORCE_CPU=1             disables GPU usage (forces CPU)

Output and plotting:
- RESULTS_RUN_DIR=/path      force results folder location
- SKIP_PLOTS=1               skip plotting (faster/headless)

CV debug verbosity:
- CV_DEBUG=1                 verbose CV debug prints
- CV_TABLE_MODE=off|compact|verbose|full  control fold table verbosity

------------------------------------------------------------

One-line summary:
Edit MODEL_LIST, SEEDS/REPEATS, N_REAL_MONTHS/END_DATE in MLBacktesterNoWFO.py,
edit features/gating in configs/feature_config.json,
run python MLBacktesterNoWFO.py,
and read results from the generated run directory.
