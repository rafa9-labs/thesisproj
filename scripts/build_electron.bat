@echo off
setlocal enabledelayedexpansion

echo =============================================
echo  FX ML Backtester - Full Electron Build
echo =============================================
echo.

cd /d "%~dp0.."
set "PROJECT_ROOT=%cd%"

echo [1/7] Checking prerequisites...
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    exit /b 1
)
echo       Node.js found.
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Add to PATH.
    exit /b 1
)
echo       Python found.

pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller pyinstaller-hooks-contrib
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller.
        exit /b 1
    )
)
echo       PyInstaller found.

echo [2/7] Building React frontend...
cd /d "%PROJECT_ROOT%\frontend"
call npm run build
if errorlevel 1 (
    echo [ERROR] React build failed!
    exit /b 1
)
cd /d "%PROJECT_ROOT%"
echo       React build complete.

echo [3/7] Compiling Electron TypeScript...
call npx tsc -p electron/tsconfig.json
if errorlevel 1 (
    echo [ERROR] Electron TypeScript compilation failed!
    exit /b 1
)
if not exist "frontend\electron-dist\main.js" (
    echo [ERROR] Electron main.js not found in frontend\electron-dist\!
    exit /b 1
)
echo       Electron TS compile complete. Output: frontend\electron-dist\

echo [4/7] Building Python backend (PyInstaller)...
call scripts\build_python.bat
if errorlevel 1 (
    echo [ERROR] Python backend build failed!
    exit /b 1
)
if not exist "dist\fx_backend\fx_backend.exe" (
    echo [ERROR] fx_backend.exe not found in dist\fx_backend\!
    exit /b 1
)
echo       Python backend build complete.

echo [5/7] Verifying Python backend bundle size...
for /f "tokens=3" %%a in ('dir /s "dist\fx_backend" ^| findstr /c:"File(s)"') do set BUILD_SIZE=%%a
echo       Bundle size: !BUILD_SIZE! bytes

echo [6/7] Generating app icons...
if not exist "build\icon.ico" (
    echo [INFO] Generating placeholder icons...
    node scripts\generate_icon.mjs
    if errorlevel 1 (
        echo [WARN] Icon generation failed, using fallback.
    )
) else (
    echo       Icons already exist.
)

echo [7/7] Packaging with electron-builder...
cd /d "%PROJECT_ROOT%\frontend"
call npx electron-builder --config ..\electron-builder.yml --win
if errorlevel 1 (
    echo [ERROR] electron-builder failed!
    cd /d "%PROJECT_ROOT%"
    exit /b 1
)
cd /d "%PROJECT_ROOT%"

echo.
echo =============================================
echo  BUILD SUCCESSFUL!
echo =============================================
echo.

if exist "release\" (
    echo Output: release\
    for %%f in (release\FX*Setup*.exe) do (
        echo   Installer: %%f
        for %%a in ("%%f") do echo   Size: %%~za bytes
    )
    if not exist "release\FX*Setup*.exe" (
        echo   (No installer .exe found - check release\ directory)
    )
    for /f "tokens=3" %%a in ('dir /s "release\win-unpacked" 2^>nul ^| findstr /c:"File(s)"') do set UNPACKED_SIZE=%%a
    echo   Unpacked size: !UNPACKED_SIZE! bytes
)
echo.
echo To install: Run the .exe installer in release\
echo To test unpacked: release\win-unpacked\FX ML Backtester.exe