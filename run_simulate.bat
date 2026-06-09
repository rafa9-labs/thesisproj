@echo off
REM =====================================================================
REM  KodaQuant — Automated Committee Pipeline Simulation
REM  One-click runner: tests all 12 complexity levels from import smoke
REM  through full _run_full_cycle() integration.
REM
REM  Results are saved to:
REM    results\simulations\simulate_YYYYMMDD-HHMMSS.log   (human-readable)
REM    results\simulations\simulate_YYYYMMDD-HHMMSS.json  (machine-readable)
REM
REM  Usage:
REM    run_simulate.bat                  — all 12 levels
REM    run_simulate.bat --levels 1,2,3    — component smoke only (fast)
REM    run_simulate.bat --levels 11,12    — real UI integration only
REM =====================================================================

cd /d "%~dp0"

echo.
echo ================================================================
echo   KodaQuant -- Committee Pipeline Simulation
echo   All results logged to: results\simulations\
echo ================================================================
echo.

python scripts\simulate_committee.py %*

echo.
echo ================================================================
echo   Simulation finished. Check results\simulations\ for logs.
echo ================================================================
