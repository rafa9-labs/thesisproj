@echo off
setlocal enabledelayedexpansion

echo =============================================
echo  FX ML Backtester - Full Electron Build
echo =============================================
echo.

cd /d "%~dp0.."

echo [1/6] Checking prerequisites...
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    exit /b 1
)
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Add to PATH.
    exit /b 1
)
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller pyinstaller-hooks-contrib
)

echo [2/6] Building React frontend...
cd frontend
call npm run build
if errorlevel 1 (
    echo [ERROR] React build failed!
    exit /b 1
)
cd ..
echo       React build complete.

echo [3/6] Compiling Electron TypeScript...
call npx tsc -p electron/tsconfig.json
if errorlevel 1 (
    echo [ERROR] Electron TypeScript compilation failed!
    exit /b 1
)
echo       Electron TS compile complete.

echo [4/6] Building Python backend (PyInstaller)...
call scripts\build_python.bat
if errorlevel 1 (
    echo [ERROR] Python backend build failed!
    exit /b 1
)

echo [5/6] Generating app icons...
if not exist "build\icon.ico" (
    echo [INFO] Generating placeholder icons...
    node scripts\generate_icon.mjs
    if errorlevel 1 (
        echo [WARN] Icon generation failed, using fallback.
    )
)

echo [6/6] Packaging with electron-builder...
cd frontend
call npx electron-builder --config ../electron-builder.yml --win
if errorlevel 1 (
    echo [ERROR] electron-builder failed!
    cd ..
    exit /b 1
)
cd ..

echo.
echo =============================================
echo  BUILD SUCCESSFUL!
echo =============================================
echo.
echo Output: release\

for %%f in (release\FX*Setup*.exe) do (
    echo   Installer: %%f
    for %%a in ("%%f") do echo   Size: %%~za bytes
)
echo.
echo To install: Run the .exe installer in release\