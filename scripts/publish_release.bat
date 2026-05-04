@echo off
REM KodaQuant — Build + Publish Release to GitHub
REM
REM Usage:
REM   scripts\publish_release.bat [version]
REM
REM If version is not provided, reads from frontend/package.json
REM
REM Prereqs:
REM   - Python 3.10+ with PyInstaller
REM   - Node.js 18+ with pnpm
REM   - GitHub CLI (gh) authenticated
REM   - GITHUB_TOKEN or gh auth login

setlocal enabledelayedexpansion

REM Read version from package.json if not provided
if "%~1"=="" (
    for /f "tokens=2 delims=:, " %%a in ('findstr /C:"\"version\"" frontend\package.json') do (
        set "VER=%%~a"
    )
) else (
    set "VER=%~1"
)

echo ============================================
echo  KodaQuant Release Builder v%VER%
echo ============================================
echo.

REM Step 1: Set version across all files
echo [1/5] Setting version %VER%...
python scripts\set_version.py %VER%
if errorlevel 1 (
    echo ERROR: set_version.py failed
    exit /b 1
)
echo.

REM Step 2: Build Python backend
echo [2/5] Building Python backend...
call scripts\build_python.bat
if errorlevel 1 (
    echo ERROR: Python build failed
    exit /b 1
)
echo.

REM Step 3: Build Electron app
echo [3/5] Building Electron app...
call scripts\build_electron.bat
if errorlevel 1 (
    echo ERROR: Electron build failed
    exit /b 1
)
echo.

REM Step 4: Create git tag
echo [4/5] Creating git tag v%VER%...
git tag -a v%VER% -m "Release v%VER%"
if errorlevel 1 (
    echo WARNING: git tag failed (may already exist)
)
echo.

REM Step 5: Push tag and trigger release
echo [5/5] Pushing to GitHub...
git push origin feature/phase2-api-bridge --tags
if errorlevel 1 (
    echo ERROR: git push failed
    exit /b 1
)
echo.

echo ============================================
echo  Release v%VER% complete!
echo  Installer: release\KodaQuant-Setup-%VER%.exe
echo ============================================

endlocal