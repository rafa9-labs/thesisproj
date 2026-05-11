@echo off
REM =============================================================
REM run_all_tests.bat — Windows launcher for the full test suite
REM
REM Double-click this file or run from PowerShell:
REM   .\run_all_tests.bat
REM
REM Output saved to: test_results\test_run_YYYYMMDD_HHMMSS.log
REM =============================================================

echo.
echo  ================================================================
echo   Running full phase gate test suite via WSL...
echo   This may take 20-60 minutes depending on GPU.
echo  ================================================================
echo.

wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_all_tests.sh

echo.
echo  Done! Check test_results\ folder for detailed logs.
echo.
pause