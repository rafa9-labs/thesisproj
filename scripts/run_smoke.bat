@echo off
REM ── Smoke test wrapper (works in cmd.exe AND PowerShell) ──
REM Usage:  run_smoke.bat          (output to console + smoke_output.txt)
REM         run_smoke.bat clean    (delete old output first)

if "%1"=="clean" del /f smoke_output*.txt 2>nul

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python -u tests/smoke_full.py 2>&1 | tee smoke_output5.txt