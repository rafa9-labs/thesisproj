@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  FX ML Backtester - Development Launcher
echo ============================================
echo.

cd /d "%~dp0.."

echo [1/3] Checking Redis...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo [START] Starting Redis...
    start "Redis" redis-server
    timeout /t 2 /nobreak >nul
) else (
    echo [OK] Redis already running.
)

echo [2/3] Starting Celery worker...
start "Celery" cmd /c "celery -A api.tasks.celery_app worker --loglevel=info --pool=solo -Q celery"

echo [3/3] Starting FastAPI backend...
start "FastAPI" cmd /c "uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload"

echo.
echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

echo Starting React frontend...
cd frontend
npm run dev

echo.
echo All services shut down.