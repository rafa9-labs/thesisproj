# Sprint 13: Beta & Launch — Execution Plan

> **Created**: 2026-05-11
> **Status**: DEFERRED (work on S16-S22 first, launch after app is complete)
> **Re-scoped from**: 6-8h naive → ~25-35h production-quality

## What "Beta Launch" Means

Sending the app to 10-20 testers. They double-click an installer, run backtests, hit bugs. We need:
1. Crash visibility (Sentry working everywhere)
2. A way for them to report issues (in-app feedback)
3. Logs on disk (for debugging their reports)
4. Known bugs visible (so they don't report duplicates)
5. The installer works end-to-end

---

## S13.1 — Closed Beta

| # | Task | Executor | Est | How |
|---|------|----------|-----|-----|
| 1.1a | Add Python Sentry SDK (`sentry-sdk`) to `requirements.txt`, init in `api/main.py` lifespan | AI | 1h | Mirror `electron/sentry.ts` pattern. Wrap with `SENTRY_DSN` env var guard. Instrument `run_server.py` for PyInstaller path |
| 1.1b | Add `@sentry/react` to React `ErrorBoundary.componentDidCatch` | AI | 0.5h | Install `@sentry/react`, call `captureException` in ErrorBoundary |
| 1.1c | Add crash-reporting opt-in UI toggle | AI | 1h | Settings page toggle "Crash Reporting" (on by default). DSN bundled at build time via env var injected into `electron-builder.yml` |
| 1.1d | Persist logs to disk in `logs/` directory | AI | 1h | Python: add `RotatingFileHandler` to `logging_config.py`. Electron: write `python.ts` captured stdout/stderr to log file |
| 1.1e | Build in-app feedback form | AI | 2h | New `FeedbackDialog.tsx`. Fields: title, description, attach screenshot, attach log file. POST to `api/routers/feedback.py` |
| 1.1f | Known issues panel in About/Settings | AI | 1h | Static JSON list at `frontend/src/data/knownIssues.json`. Display in AboutDialog |
| 1.1g | Recruit beta testers | Human | 2h | Post on r/algotrading, Forex Factory, Discord. Google Form for signups |
| 1.1h | Beta build + distribute | AI + Human | 1h | Build installer, upload to GitHub Releases, send links to testers |

## S13.2 — Performance Optimization

| # | Task | Executor | Est | How |
|---|------|----------|-----|-----|
| 2.1a | Instrument startup timing | AI | 1h | `performance.now()` marks in `electron/main.ts`. Log time-to-splash, time-to-healthy, time-to-UI-ready |
| 2.1b | Bundle size analysis | AI | 1h | Identify top-10 largest files in dist. Report findings — actual TF slimming is separate |
| 2.1c | FastAPI startup profiling | AI | 1h | Profile `lifespan` — data store init, model registry scan, router registration |
| 2.1d | Memory profiling (deferred) | — | — | Not blocker for first beta |

## S13.3 — Launch Preparation (Human-executable)

| # | Task | Executor |
|---|------|----------|
| 3.1 | Product Hunt listing draft | Human (AI can draft copy) |
| 3.2 | Social media announcements | Human |
| 3.3 | Email list setup | Human |
| 3.4 | Demo video recording | Human |
| 3.5 | Press kit | Human |

## S13.4 — Post-Launch Monitoring (Ongoing)

| # | Task |
|---|------|
| 4.1 | Monitor Paddle sales dashboard |
| 4.2 | Monitor Sentry error rates |
| 4.3 | Support channel (Discord + email) |
| 4.4 | Weekly metrics review |

## Build Pipeline Fixes (must-do before beta build)

| # | Task | Est |
|---|------|-----|
| B1 | Fix hardcoded branch in `publish_release.bat` line 68 (`feature/phase2-api-bridge` → `main`) | 0.1h |
| B2 | Add SENTRY_DSN to `electron-builder.yml` extraMetadata | 0.2h |
| B3 | Verify installer on clean Windows VM | 1h (human) |

## Completion Criteria

| # | Criterion | Must/Should |
|---|-----------|-------------|
| C1 | Sentry receives crash reports from Electron + Python + React | Must |
| C2 | Users can submit feedback without leaving the app | Must |
| C3 | Logs persist to `logs/` directory and survive restarts | Must |
| C4 | Known issues are visible in-app | Must |
| C5 | Startup time is measured and logged (target <5s) | Must |
| C6 | 10-20 testers have installer and launch successfully | Must |
| C7 | `publish_release.bat` pushes to `main` branch | Must |
| C8 | Bundle size is analyzed | Should |
| C9 | Telemetry pipeline exists | Should |
| C10 | Paddle webhook endpoint works | Defer to S22 |

## AI-Executable Sequence

```
Phase A (6h code):
  1. fix-publish-branch     → edit scripts/publish_release.bat
  2. python-sentry          → requirements.txt + api/main.py + run_server.py
  3. react-sentry           → ErrorBoundary + package.json
  4. log-persistence        → logging_config.py + python.ts
  5. feedback-form          → FeedbackDialog.tsx + api/routers/feedback.py
  6. known-issues           → knownIssues.json + AboutDialog
  7. startup-profiling      → electron/main.ts + splash.ts timing
  8. bundle-analysis        → build script size report

Phase B (concurrent human tasks):
  - Recruit testers (forums/Reddit/Discord)
  - Set up Google Form for signups
  - Test installer on clean machine

Phase C (1h code, pre-release):
  - Build + upload to GitHub Releases
  - Send download links
```
