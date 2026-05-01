@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  LOCAL CI - Run all checks before pushing
echo ============================================
echo.

echo [1/6] Python Lint (ruff)...
ruff check .
if %errorlevel% neq 0 (
    echo FAILED: ruff check
    exit /b 1
)
echo PASSED: ruff check
echo.

echo [2/6] Python Tests (excluding slow)...
python -m pytest tests/ -m "not slow" -x -q --tb=short
if %errorlevel% neq 0 (
    echo FAILED: pytest
    exit /b 1
)
echo PASSED: pytest
echo.

echo [3/6] Frontend Lint...
cd frontend
call npm run lint
if %errorlevel% neq 0 (
    echo FAILED: eslint
    cd ..
    exit /b 1
)
echo PASSED: eslint
echo.

echo [4/6] Frontend Typecheck...
call npm run typecheck
if %errorlevel% neq 0 (
    echo FAILED: tsc
    cd ..
    exit /b 1
)
echo PASSED: tsc
echo.

echo [5/6] Frontend Unit Tests...
call npm run test
if %errorlevel% neq 0 (
    echo FAILED: vitest
    cd ..
    exit /b 1
)
echo PASSED: vitest
echo.

echo [6/6] Frontend Production Build...
call npm run build
if %errorlevel% neq 0 (
    echo FAILED: vite build
    cd ..
    exit /b 1
)
echo PASSED: vite build
cd ..

echo.
echo ============================================
echo  ALL CHECKS PASSED
echo ============================================