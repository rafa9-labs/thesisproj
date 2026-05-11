@echo off
REM Run smoke tests via WSL with GPU acceleration (RTX 3090)
REM Usage: run_smoke_gpu [model]
REM   model: logistic, xgboost, cnn, lstm, transformer, ensemble_cnn_lstm_xgboost, ensemble_adaptive_regime, dqn
REM   If no model specified, runs ALL models (warning: slow even on GPU for DQN).

echo ============================================
echo   KodaQuant - GPU Smoke Test Runner
echo ============================================
echo.

if "%~1"=="" (
    echo Running ALL models via WSL GPU...
    wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_smoke_gpu.sh all
) else (
    echo Running model: %1 via WSL GPU...
    wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_smoke_gpu.sh %1
)

echo.
echo ============================================
echo   Smoke test complete.
echo ============================================
pause