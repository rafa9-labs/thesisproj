@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  FX ML Backtester - Python Backend Build
echo ============================================
echo.

cd /d "%~dp0.."

if not exist "frontend\dist" (
    echo [WARN] Frontend not built. Run 'cd frontend && npm run build' first.
    echo [WARN] Continuing without frontend — server will be API-only.
)

echo [1/4] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller pyinstaller-hooks-contrib
)

echo [2/4] Checking TensorFlow...
python -c "import tensorflow" 2>nul
if errorlevel 1 (
    echo [WARN] TensorFlow not installed — deep models will be unavailable in bundle.
)

echo [3/4] Building Python backend...
pyinstaller forex_pipeline.spec --noconfirm --clean

if errorlevel 1 (
    echo [ERROR] PyInstaller build failed!
    exit /b 1
)

echo [4/4] Verifying output...
if exist "dist\fx_backend\fx_backend.exe" (
    echo [OK] Build successful: dist\fx_backend\fx_backend.exe
    for /f "tokens=*" %%a in ('dir /s /b "dist\fx_backend\*.pyd" 2^>nul ^| find /c /v ""') do set PYD_COUNT=%%a
    echo       Python extensions: !PYD_COUNT! .pyd files
    for /f "tokens=3" %%a in ('dir /s "dist\fx_backend" ^| findstr /c:"File(s)"') do set BUILD_SIZE=%%a
    echo       Bundle size: !BUILD_SIZE! bytes
) else (
    echo [ERROR] fx_backend.exe not found in dist!
    exit /b 1
)

echo.
echo Build complete. To test: dist\fx_backend\fx_backend.exe --port 8001