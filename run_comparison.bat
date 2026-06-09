@echo off
REM ═══════════════════════════════════════════════════════════════════
REM  Model Comparison Runner — run all models and generate leaderboard
REM
REM  Usage:
REM    run_comparison              — run all 8 models (smoke mode: 1 trial, 1 month)
REM    run_comparison full         — run all 8 models (full mode: all trials, 3 months)
REM    run_comparison quick        — run logistic + xgboost only
REM    run_comparison analyze      — just analyze existing results (no new runs)
REM    run_comparison gpu          — run on GPU via WSL
REM ═══════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion

set "MODE=%~1"
if "%MODE%"=="" set "MODE=smoke"

if "%MODE%"=="analyze" goto :analyze
if "%MODE%"=="analyse" goto :analyze

echo ════════════════════════════════════════════════════════════
echo   Model Comparison Runner — %MODE% mode
echo ════════════════════════════════════════════════════════════
echo.

if "%MODE%"=="gpu" goto :gpu

REM ── Set models based on mode ──
if "%MODE%"=="quick" (
    set "MODELS=logistic,xgboost"
    set "N_MONTHS=1"
    set "N_TRIALS_ENV=1"
) else if "%MODE%"=="full" (
    set "MODELS=logistic,xgboost,lightgbm,catboost,cnn,lstm,transformer,gru,gru_lstm,dqn,ensemble_cnn_lstm_xgboost,ensemble_adaptive_regime,meta_ensemble,stacking_ensemble"
    set "N_MONTHS=3"
    set "N_TRIALS_ENV=0"
) else (
    REM smoke mode
    set "MODELS=logistic,xgboost,lightgbm,catboost,cnn,lstm,transformer,gru,gru_lstm,dqn,ensemble_cnn_lstm_xgboost,ensemble_adaptive_regime,meta_ensemble,stacking_ensemble"
    set "N_MONTHS=1"
    set "N_TRIALS_ENV=1"
)

echo Models: %MODELS%
echo Months: %N_MONTHS%
echo.

REM ── Run pipeline ──
set "SMOKE_TEST=%N_TRIALS_ENV%"
set "MODEL_LIST=%MODELS%"
set "N_MONTHS=%N_MONTHS%"
set "REPEATS=1"
set "SEEDS=33333"
set "SKIP_PLOTS=1"

echo Running backtests...
echo.
python -m pipeline.main_cli

echo.
echo ── Generating Leaderboard ──
python -m pipeline.model_comparison --significance --export

goto :done

:gpu
echo Running on GPU via WSL...
wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_smoke_gpu.sh
echo.
echo ── Generating Leaderboard ──
python -m pipeline.model_comparison --significance --export
goto :done

:analyze
echo Analyzing existing results...
echo.
python -m pipeline.model_comparison --list
echo.
python -m pipeline.model_comparison --significance --export

:done
echo.
echo ═════════════════════════════════════════════════════════════
echo   Done! Run 'run_comparison analyze' to view results anytime.
echo ═════════════════════════════════════════════════════════════
endlocal