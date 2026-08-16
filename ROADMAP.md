# KodaQuant Roadmap

See `README.md` for the high-level sprint table. This file tracks per-sprint details.

## Sprint 15: KodaQuant Branding ✅ COMPLETE

- All stale "FX ML Backtester" references replaced with KodaQuant
- API config reports `KodaQuant API`
- `run_server.py` and PyInstaller spec use KodaQuant branding
- Icon assets generated (`build/icon.{png,ico,svg}`, installer BMPs, favicons)

## Sprint 16: Overfitting & Transparency ✅ DONE

- Deflated Sharpe Ratio (Bailey & López de Prado 2014) with selection-bias,
  non-normality and sample-length corrections
- Probability of Backtest Overfitting via real CSCV (configs × time matrix)
- PSR, HAC Sharpe, trust score, diagnostics engine (histograms, confusion matrices)
- Three-tier hyperparameter architecture (`pipeline/models/model_defaults.py`)

## Sprint 17: UI Polish & Search — IN PROGRESS

- Models API wiring (registry + descriptions + hyperparams metadata)
- Deployed-models endpoints, forward test, live prediction bridge

## Sprint 19: Ensemble Models ✅ DONE

- CNN-LSTM-XGBoost ensemble, adaptive regime ensemble, stacking, meta ensemble,
  regime classifier

## Sprint 21: Live Trading (OANDA) — IN PROGRESS

- Live committee runner, paper/live/committee engines, risk gates,
  VRAM gatekeeper, process manager lifecycle
