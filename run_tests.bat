@echo off
REM =============================================================
REM run_tests.bat — Native Windows pytest runner
REM
REM Usage:
REM   .\run_tests.bat            (all tests, excl. slow)
REM   .\run_tests.bat --all     (including slow/benchmark)
REM   .\run_tests.bat -k test_name  (specific test)
REM =============================================================

setlocal enabledelayedexpansion

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

for /f "tokens=2 delims==" %%a in ('findstr /r "^version" frontend\package.json 2^>nul') do (
    set APP_VERSION=%%~a
)

if not exist "test_results" mkdir test_results

for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do (
    set TIMESTAMP=%%a%%b%%c_%%d%%e%%f
)

set LOG_FILE=test_results\test_run_!TIMESTAMP!.log

echo.
echo  ================================================================
echo   Running pytest (native Windows)...
echo   Log: !LOG_FILE!
echo  ================================================================
echo.

if "%~1"=="--all" (
    echo   Mode: ALL tests (including slow/benchmark)
    python -m pytest tests/ -v --tb=short -s 2>&1 | tee !LOG_FILE!
) else if "%~1"=="" (
    echo   Mode: Default (excluding slow/api marks)
    python -m pytest tests/ -v --tb=short -s -m "not slow and not api" 2>&1 | tee !LOG_FILE!
) else (
    echo   Mode: Custom args: %*
    python -m pytest tests/ -v --tb=short -s %* 2>&1 | tee !LOG_FILE!
)

echo.
echo  ================================================================
echo   Test run complete. Log: !LOG_FILE!
echo  ================================================================
echo.