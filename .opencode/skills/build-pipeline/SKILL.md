---
name: build-pipeline
description: Run the full PyInstaller + electron-builder production build pipeline. Verifies both artifacts exist with correct sizes, checks hidden imports, rebuilds if needed, and validates the Electron app launches successfully.
---

# Skill: /build-pipeline

**Trigger:** User types `/build-pipeline` or asks to build/rebuild the desktop app.

**Objective:** Execute the complete dual-build pipeline (Python backend + Electron installer) and validate artifacts.

**Protocol:**

1. **Pre-flight checks:**
   - Kill any running `fx_backend` or `FX ML Backtester` processes.
   - Verify `forex_pipeline.spec` exists and is readable.
   - Verify `electron-builder.yml` exists.
   - Check `frontend/electron-dist/` has all 8 expected JS files (main, python, preload, tray, utils, splash, updater, ipc).
   - Check `build/icon.ico` or `frontend/public/favicon.ico` exists for the installer icon.

2. **Python backend build (PyInstaller):**
   ```powershell
   pyinstaller forex_pipeline.spec --noconfirm --clean 2>&1
   ```
   - Monitor for `Build complete!` in output.
   - Check `dist/fx_backend/fx_backend.exe` exists and is >100MB.
   - If build fails: inspect the error, check for missing hidden imports in the spec file, fix, and retry (max 3 attempts).

3. **Electron frontend build (Vite + TypeScript):**
   ```powershell
   cd frontend; npm run build 2>&1
   ```
   - Then compile Electron TypeScript:
   ```powershell
   cd frontend; npx tsc --project ..\electron\tsconfig.json --outDir electron-dist 2>&1
   ```
   - Verify all expected JS files exist in `frontend/electron-dist/`.

4. **Electron installer build (electron-builder):**
   ```powershell
   cd frontend; npx electron-builder --config ..\electron-builder.yml --win 2>&1
   ```
   - Must be run from `frontend/` directory (electron-builder.yml references `../dist/fx_backend`).
   - Check `release/FX ML Backtester-Setup-1.0.0.exe` exists and is >400MB.
   - Check `release/win-unpacked/` directory exists with app contents.

5. **Validation:**
   - Run `python -m pytest tests/test_build_validation.py -x -q` to verify 46 hidden import checks pass.
   - Optionally launch `release/win-unpacked/FX ML Backtester.exe` and verify splash screen appears and backend starts (health check on port).

6. **Output format:**
   ```
   ## Build Pipeline Results

   | Step | Status | Duration | Size | Notes |
   |------|--------|----------|------|-------|
   | PyInstaller | PASS | 45s | 1.66GB | Build complete |
   | Vite/TS Build | PASS | 12s | - | 8 JS files |
   | Electron Builder | PASS | 90s | 514MB | NSIS installer created |
   | Build Validation Tests | PASS | 9s | - | 46/46 passed |

   **Verdict: PASS** - All artifacts built successfully.

   Artifacts:
   - `dist/fx_backend/fx_backend.exe` (1.66GB)
   - `release/FX ML Backtester-Setup-1.0.0.exe` (514MB)
   - `release/win-unpacked/FX ML Backtester.exe`
   ```

7. **Failure handling:**
   - If PyInstaller fails: check `forex_pipeline.spec` hidden imports, check for new imports added since last build.
   - If electron-builder fails: check `electron-builder.yml` paths (must be run from `frontend/`), check icon paths.
   - If validation tests fail: inspect the specific hidden import that failed and add it to the spec file.
   - Max 3 retry cycles before asking for human intervention.

**Important:**
- NEVER run the installer (.exe) on the build machine — just verify it exists and has correct size.
- If `dist/fx_backend/` already exists and user says "quick rebuild", skip to step 4 only.
- Always push any spec/yml changes to git after a successful build.