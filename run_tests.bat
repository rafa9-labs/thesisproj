@echo off
REM =============================================================
REM run_tests.bat — Tiered Windows test runner with resource control
REM
REM Usage:
REM   .\run_tests.bat              (T1 + T2: unit + classical models, ~2min)
REM   .\run_tests.bat t1           (Unit tests only, ~30s)
REM   .\run_tests.bat t2           (Classical models only, ~2min)
REM   .\run_tests.bat t3           (Deep models: CNN/LSTM/Transformer/DQN, ~10min)
REM   .\run_tests.bat t4           (Integration: news, LLM, full system, ~5min)
REM   .\run_tests.bat t12          (T1 + T2 combined)
REM   .\run_tests.bat t1234        (All tiers sequentially)
REM   .\run_tests.bat --all        (All tiers sequentially)
REM   .\run_tests.bat -k test_name (Run specific test)
REM =============================================================

setlocal enabledelayedexpansion

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set TF_CPP_MIN_LOG_LEVEL=2

if not exist "test_results" mkdir test_results

for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value ^| findstr "="') do set DT=%%a
set TIMESTAMP=%DT:~0,8%_%DT:~8,6%

set COMMON=--tb=short -p no:cacheprovider

if "%~1"=="" (
    call :run_t1
    call :run_t2
    goto :summary
)
if "%~1"=="t1" (
    call :run_t1
    goto :summary
)
if "%~1"=="t2" (
    call :run_t2
    goto :summary
)
if "%~1"=="t3" (
    call :run_t3
    goto :summary
)
if "%~1"=="t4" (
    call :run_t4
    goto :summary
)
if "%~1"=="t12" (
    call :run_t1
    call :run_t2
    goto :summary
)
if "%~1"=="--all" (
    call :run_t1
    call :run_t2
    call :run_t3
    call :run_t4
    goto :summary
)
if "%~1"=="t1234" (
    call :run_t1
    call :run_t2
    call :run_t3
    call :run_t4
    goto :summary
)
REM Custom args (e.g. -k test_name)
echo Running custom: python -m pytest tests/ %* %COMMON%
python -m pytest tests/ %* %COMMON%
goto :eof

:run_t1
echo.
echo  ================================================================
echo   TIER 1: Unit Tests (fast, low memory, ~30s)
echo  ================================================================
echo.
set T1_LOG=test_results\t1_%TIMESTAMP%.log
python -m pytest tests/ %COMMON% --ignore=tests/benchmarks --ignore=tests/test_electron_build.py --ignore=tests/test_models_train_predict.py --ignore=tests/test_api.py --ignore=tests/test_systematic_model_features.py --ignore=tests/test_full_system.py --ignore=tests/test_news.py --ignore=tests/test_llm_sentiment.py --ignore=tests/test_pipeline_validation.py --ignore=tests/test_pipeline_integration.py -m "not deep" 2>&1 | tee %T1_LOG%
set T1_ERR=%errorlevel%
echo  T1 exit code: %T1_ERR%
echo.
goto :eof

:run_t2
echo.
echo  ================================================================
echo   TIER 2: Classical Models (logistic, svm, rf, dt, xgboost, ~2min)
echo  ================================================================
echo.
set T2_LOG=test_results\t2_%TIMESTAMP%.log
python -m pytest tests/test_models_train_predict.py %COMMON% -k "Logistic or SVM or RandomForest or DecisionTree or XGBoost or BuildModelEdgeCases" 2>&1 | tee %T2_LOG%
set T2_ERR=%errorlevel%
echo  T2 exit code: %T2_ERR%
echo.
goto :eof

:run_t3
echo.
echo  ================================================================
echo   TIER 3: Deep Models (CNN, LSTM, Transformer, DQN, ensembles)
echo  ================================================================
echo.
set T3_LOG=test_results\t3_%TIMESTAMP%.log
python -m pytest tests/test_models_train_predict.py %COMMON% -k "CNN or LSTM or Transformer or DQN or EnsembleAdaptiveRegime or EnsembleCNNLSTMXGBoost" tests/benchmarks/ -m "slow or deep" --timeout=300 2>&1 | tee %T3_LOG%
set T3_ERR=%errorlevel%
echo  T3 exit code: %T3_ERR%
echo.
goto :eof

:run_t4
echo.
echo  ================================================================
echo   TIER 4: Integration (news, LLM, full system, pipeline, API)
echo  ================================================================
echo.
set T4_LOG=test_results\t4_%TIMESTAMP%.log
python -m pytest tests/ %COMMON% --ignore=tests/benchmarks --ignore=tests/test_electron_build.py --ignore=tests/test_models_train_predict.py tests/test_full_system.py tests/test_news.py tests/test_llm_sentiment.py tests/test_pipeline_validation.py tests/test_pipeline_integration.py tests/test_systematic_model_features.py tests/test_api.py 2>&1 | tee %T4_LOG%
set T4_ERR=%errorlevel%
echo  T4 exit code: %T4_ERR%
echo.
goto :eof

:summary
echo.
echo  ================================================================
echo   TEST SUMMARY
echo  ================================================================
echo   Logs saved in test_results\
echo   T1 (unit)=%T1_ERR%  T2 (classical)=%T2_ERR%  T3 (deep)=%T3_ERR%  T4 (integration)=%T4_ERR%
echo  ================================================================
echo.