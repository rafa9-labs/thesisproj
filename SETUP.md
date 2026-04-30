Now I have a comprehensive understanding of the project. Let me create the guide:
FX ML Backtester — Setup & Run Guide
Prerequisites
#	Prerequisite	Description
1	Python 3.10+	Core runtime. Must be on PATH.
2	pip	Python package manager (ships with Python).
3	Git	Version control (to clone the repo).
4	WSL2 + Ubuntu 22.04	Required for GPU-accelerated runs. The .bat launchers delegate to WSL for CUDA/TensorFlow GPU support.
5	NVIDIA GPU + CUDA	Optional but recommended for deep models (CNN, LSTM, Transformer, DQN). CPU-only works but is 5–10x slower.
6	Docker Desktop	Required only for the full stack mode (docker-compose). Not needed for Streamlit UI or CLI-only usage.
7	Redis	Required only for the API/Celery worker stack. Not needed for Streamlit UI or CLI-only usage.
8	Node.js + npm	Required only for the React frontend (Sprint 8+). Not needed for Streamlit UI.
9	OANDA API key	Optional. Only needed if downloading fresh FX data via CSVDownloadOanda.py. Sample CSVs are bundled.
---
Commands (in order)
Step 1 — Clone the repository
git clone https://github.com/rafa9-labs/thesisproj.git
cd thesisproj
Description: Clones the forex pipeline repo and enters the project directory.
---
Step 2 — Create & activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
Description: Creates an isolated Python venv so packages don't pollute your system Python. Activate it before every session. (If using WSL for GPU, the venv is at .venv-wsl/ instead.)
---
Step 3 — Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_freeze.txt
Description: 
- requirements.txt — High-level dependencies (FastAPI, Streamlit, pandas, scikit-learn, etc.)
- requirements_freeze.txt — Pinned versions for reproducibility (TensorFlow 2.20, XGBoost 3.1, Optuna 4.5, CUDA toolkit libs, etc.)
Note: requirements_freeze.txt contains the full ML stack including TensorFlow with CUDA 12, XGBoost, Optuna, and all deep learning deps. Install it for full model support; requirements.txt alone covers the API/web layer.
---
Step 4 — Create the .env file
copy .env.example .env
Description: Copies the environment template. Edit .env to set optional values:
- OANDA_ACCESS_TOKEN / OANDA_ACCOUNT_ID — Only if downloading data
- TF_FORCE_CPU=1 — Force CPU mode (no GPU)
- LOG_MODE=COMPACT — Logging verbosity (COMPACT / DEBUG / QUIET)
- SKIP_PLOTS=1 — Disable chart generation for faster runs
- SIZING_METHOD=fixed — Position sizing model (fixed / kelly / atr)
---
Step 5 — Verify data files exist
ls csv_data\
Description: Confirms the bundled EURUSD CSV files are present:
- EURUSD_10_years_H1_OANDA.csv — Hourly bars
- EURUSD_10_years_H4_OANDA.csv — 4-hour bars
- EURUSD_10_years_M30_OANDA.csv — 30-minute bars
These are the default data source. No download needed unless you want different pairs/timeframes.
---
Step 6 — Verify config files exist
ls configs\
Description: Confirms required JSON configs:
- feature_config.json — Feature engineering settings (indicators, lags, sessions)
- dqn_grid_config.json — DQN hyperparameter grid
These are bundled and should not need manual editing for standard runs.
---
Step 7 — Quick import health check
python -m tests.smoke_import
Description: Validates that all pipeline modules, models, and UI components import without errors. Takes seconds. Run this first to catch missing packages.
---
Step 8 — Run smoke test (1 model, 1 trial, 1 month)
.\run_smoke.bat
Or equivalently:
$env:SMOKE_TEST="1"; $env:MODEL_LIST="logistic"; $env:N_MONTHS="1"; python -m pipeline.main_cli
Description: Runs a minimal end-to-end walk-forward backtest with the fastest model (logistic regression). Completes in ~2–5 minutes. Confirms the full pipeline works: data loading → feature engineering → HPO → training → prediction → execution → metrics.
---
Step 9 — Run smoke test on GPU (all deep models)
.\run_smoke_gpu.bat
Or for a specific model:
.\run_smoke_gpu.bat lstm
Description: Same as Step 8 but runs through WSL2 with CUDA GPU acceleration. Covers CNN, LSTM, Transformer, and DQN. Each model gets 1 trial and 1 month. Takes 5–15 min depending on GPU. Requires WSL2 + NVIDIA drivers.
---
Step 10 — Launch the Streamlit Web UI
.\launch_ui.bat
Or natively (no WSL):
streamlit run app.py --server.headless true --server.port 8501
Description: Starts the Streamlit web interface at http://localhost:8501. Use the UI to:
- Configure models, features, and execution parameters (6 tabs)
- Run backtests visually
- View results dashboard (equity curves, metrics, feature importance)
- Export results
---
Step 11 — Run model comparison & leaderboard
.\run_comparison.bat
Mode	Command	Description
Smoke	.\run_comparison.bat	All 8 models, 1 trial, 1 month (~10–30 min)
Full	.\run_comparison.bat full	All 8 models, full HPO trials, 3 months (hours)
Quick	.\run_comparison.bat quick	Logistic + XGBoost only
Analyze	.\run_comparison.bat analyze	Re-render leaderboard from existing results (no new runs)
GPU	.\run_comparison.bat gpu	All models on GPU via WSL
Description: Runs multiple models through the pipeline, then generates a statistical comparison leaderboard with significance testing. Results are saved to results/ and can be re-analyzed anytime with analyze mode.
---
Step 12 — Run the test suite
.\run_all_tests.bat
Or natively:
python -m pytest tests\ -v --tb=short
Description: Runs 16+ test files covering pipeline integrity, metrics, model registry, schemas, walk-forward correctness, execution patches, risk management, feature cache, and API endpoints. The .bat version runs via WSL for GPU test support. Takes 20–60 minutes.
---
Step 13 — Full stack (Docker) — optional, for API + frontend mode
.\start.ps1 docker
Description: Launches the full production stack via Docker Compose:
- Redis (port 6379) — Celery broker/result backend
- FastAPI (port 8000) — REST API with /docs Swagger UI
- Celery worker — Async backtest job execution
- Frontend (port 5173) — React dev server
Use this for the commercial API/frontend mode, not for interactive Streamlit usage.
---
Step 14 — Full stack (Native) — alternative to Docker
.\start.ps1 native
Description: Same stack as Step 13 but runs processes natively in your venv instead of Docker. Requires Redis running separately.
---
Quick-Start Summary (minimum to get running)
git clone https://github.com/rafa9-labs/thesisproj.git
cd thesisproj
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_freeze.txt
copy .env.example .env
python -m tests.smoke_import          # verify imports
.\run_smoke.bat                       # quick backtest test
streamlit run app.py                  # launch web UI