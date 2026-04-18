# FX ML Backtester — Commercial Product Roadmap

> **Last Updated**: 2026-04-08
> **Strategy**: Option C — Web App (Streamlit) + Python SDK (pip-installable)
> **Current Phase**: Phase 3 (Simplification) → Phase 4 (Streamlit Web App)
> **Revenue Target**: £500-2K/month within 3 months of launch

---

## 1. Product Vision

**FX ML Backtester** is a professional-grade, walk-forward FX backtesting platform that lets traders and quants compare 10+ ML model families — from XGBoost to Deep Reinforcement Learning — with realistic cost-aware execution.

**Tagline**: *"The first backtesting engine built for ML model comparison."*

### Core Differentiators
| Feature | Us | Competitors |
|---------|----|-----------| 
| Walk-forward validation | ✅ Monthly refit | ❌ Single train/test split |
| Execution delay (1-bar) | ✅ Anti-look-ahead | ❌ Same-bar execution |
| After-cost equity curves | ✅ Spread + slippage | 🟡 Sometimes |
| Multi-model comparison | ✅ 10+ models at once | ❌ One model at a time |
| Triple-barrier labeling | ✅ Advanced | ❌ Basic fixed horizon |
| Regime-adaptive ensembles | ✅ MoE + stacking | ❌ Not available |
| Deep RL (Dueling DQN) | ✅ Built-in | ❌ Not available |

---

## 2. Delivery Model — Option C (Both)

### Layer 1: Web App (Primary — Streamlit MVP)
**Target**: Non-technical traders, students, fund managers
**URL**: Streamlit Cloud → custom domain later
**Monetization**: Freemium SaaS

```
User Journey:
1. Visit URL → Sign up (email/password)
2. Select instrument (EURUSD, GBPUSD...)
3. Select timeframe (M5, M15, M30, H1, H4, D1)
4. Choose models to compare (checkboxes)
5. Configure settings (months, spread, slippage, seeds)
6. Click "Run Backtest" → progress bar
7. View results: equity curves, metrics table, rankings
8. Export: CSV, PNG, PDF report
```

### Layer 2: Python SDK (Secondary — pip-installable)
**Target**: Quant developers, researchers
**Distribution**: PyPI
**Monetization**: Open-source core + premium models

```python
pip install fxbacktester

from fxbacktester import MLBacktester, Config
config = Config(models=["xgboost", "cnn", "lstm"], months=24)
bt = MLBacktester(config)
results = bt.run()
results.metrics_table()
results.equity_curve("xgboost")
```

---

## 3. Commercial Phases

### Phase 3: Pipeline Simplification ✅ MOSTLY COMPLETE
**Branch**: `refactor/phase3-simplification` (current)

| Step | Description | Status |
|------|-------------|--------|
| 3.1 | Extract `compute_full_evaluation_metrics` → `pipeline/metrics_eval.py` | ✅ |
| 3.2 | Fix circular imports, lazy TF init | ✅ |
| 3.3 | Feature disk cache (Parquet) | ⬜ |
| 3.4 | Simplify Optuna search space | ⬜ |
| 3.5 | Replace magic numbers with named constants | ⬜ |

### Phase 4: Streamlit Web App 🔄 NEXT
**Branch**: `feature/phase4-streamlit-ui`

| Step | Description | Status | Est. |
|------|-------------|--------|------|
| 4.1 | Cherry-pick `app.py` + `src/ui/` from `init-proj` | ⬜ | 1h |
| 4.2 | Create adapter: `pipeline/` backend → Streamlit frontend | ⬜ | 3h |
| 4.3 | Build model selection sidebar (checkboxes, dropdowns) | ⬜ | 2h |
| 4.4 | Build results dashboard (equity curves, metrics table) | ⬜ | 3h |
| 4.5 | Add progress bar + cancellation for long runs | ⬜ | 2h |
| 4.6 | Add export buttons (CSV, PNG) | ⬜ | 1h |
| 4.7 | Test end-to-end: select → run → view → export | ⬜ | 2h |
| 4.8 | Deploy to Streamlit Cloud | ⬜ | 30min |

### Phase 5: User Accounts + Monetization
**Branch**: `feature/phase5-auth-billing`

| Step | Description | Status |
|------|-------------|--------|
| 5.1 | Add Streamlit authenticator (stauth) | ⬜ |
| 5.2 | User registration/login flow | ⬜ |
| 5.3 | Save backtest results per user (SQLite) | ⬜ |
| 5.4 | Free tier limits (3 backtests/day, 3 models max) | ⬜ |
| 5.5 | Stripe integration for Pro tier | ⬜ |
| 5.6 | Pro tier: unlimited backtests, all models, export | ⬜ |

### Phase 6: Python SDK (pip-installable)
**Branch**: `feature/phase6-pypi-package`

| Step | Description | Status |
|------|-------------|--------|
| 6.1 | Create `pyproject.toml` + `setup.py` | ⬜ |
| 6.2 | Define public API (`fxbacktester` package) | ⬜ |
| 6.3 | Write 5 example Jupyter notebooks | ⬜ |
| 6.4 | Write API documentation (Sphinx/MkDocs) | ⬜ |
| 6.5 | Publish to PyPI | ⬜ |
| 6.6 | Gate premium models behind license key | ⬜ |

### Phase 7: Scale (React + FastAPI)
| Step | Description | Status |
|------|-------------|--------|
| 7.1 | FastAPI backend wrapping `pipeline/` | ⬜ |
| 7.2 | React/Next.js frontend (based on pomodoro patterns) | ⬜ |
| 7.3 | PostgreSQL + Redis for user data/sessions | ⬜ |
| 7.4 | Docker + cloud deployment (AWS/GCP) | ⬜ |
| 7.5 | API rate limiting + usage analytics | ⬜ |

---

## 4. Pricing Tiers

| Tier | Price | Features |
|------|-------|----------|
| **Free** | £0 | 3 backtests/day, 3 models, basic metrics, no export |
| **Pro** | £19/mo | Unlimited backtests, all models, full metrics, CSV/PNG export |
| **Team** | £49/mo | Pro + shared configs, team dashboard, priority support |
| **API** | £99/mo | REST API access, custom data, webhooks, white-label |

---

## 5. Architecture — `init-proj` Disposition

| `init-proj` Component | Decision | Reason |
|----------------------|----------|--------|
| `app.py` | ✅ **Cherry-pick** | Working Streamlit entry point |
| `src/ui/*` | ✅ **Cherry-pick** | Dashboard, controls, state modules |
| `src/features/*` | 🔄 **Evaluate later** | Vectorized indicators — may replace mixin code |
| `src/execution/*` | 🔄 **Evaluate later** | Risk management — useful for Pro tier |
| `src/models/*` | ❌ **Skip** | We already have model wrappers + registry |
| `src/core/config.py` | ❌ **Skip** | We have our own `config.py` |
| `PHASE*.md`, design docs | 📝 **Archive** | Reference only, keep branch alive |

---

## 6. Technical Stack

| Layer | Current | Target |
|-------|---------|--------|
| Backend | `pipeline/` (Python) | Same — battle-tested |
| Frontend v1 | None | Streamlit (fast to market) |
| Frontend v2 | — | React/Next.js (scale) |
| Auth | None | stauth → Supabase Auth |
| Database | File system | SQLite → PostgreSQL |
| Payments | None | Stripe |
| Hosting | Local | Streamlit Cloud → AWS/GCP |
| Package | None | PyPI (`fxbacktester`) |

---

## 7. Key Files Reference

| File | Role |
|------|------|
| `pipeline/backtester/composed.py` | MLBacktester class (entry point for UI) |
| `pipeline/main_cli.py` | CLI runner (reference for UI integration) |
| `config.py` | Configuration (used by both CLI and UI) |
| `models/registry.py` | Model registry (UI model selection) |
| `pipeline/metrics_eval.py` | Evaluation metrics (UI results display) |

---

## 8. Validation Milestones

### Phase 4 Complete When:
- [ ] Streamlit app launches locally (`streamlit run app.py`)
- [ ] User can select models and configure backtest
- [ ] Backtest runs with progress feedback
- [ ] Results display: equity curves + metrics table
- [ ] Export works (CSV + PNG)
- [ ] Deployed to Streamlit Cloud URL

### Phase 5 Complete When:
- [ ] User registration/login works
- [ ] Results persist across sessions
- [ ] Free tier limits enforced
- [ ] Stripe checkout works for Pro tier