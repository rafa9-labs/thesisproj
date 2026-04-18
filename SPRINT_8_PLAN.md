# Sprint 8: React Frontend — Complete Implementation Plan

> **Estimate**: 26 hours across 7 phases
> **Stack**: Vite + TypeScript + React 18 + TailwindCSS + shadcn/ui
> **Target**: Professional quant-grade desktop UI (Electron-ready)
> **Design References**: TradingView, QuantConnect, LuxAlgo, QuantTekel, Bloomberg Terminal

---

## Table of Contents

1. [Design Foundation](#1-design-foundation)
2. [Architecture](#2-architecture)
3. [Use Cases](#3-use-cases)
4. [Component Specification](#4-component-specification)
5. [Implementation Phases](#5-implementation-phases)
6. [API Contract](#6-api-contract)
7. [State Management](#7-state-management)
8. [Professional Safeguards](#8-professional-safeguards)
9. [Accessibility & i18n](#9-accessibility--i18n)
10. [Testing Strategy](#10-testing-strategy)
11. [Performance Budgets](#11-performance-budgets)
12. [Dependencies](#12-dependencies)
13. [File Structure](#13-file-structure)
14. [Completion Criteria](#14-completion-criteria)

---

## 1. Design Foundation

### 1.1 Color Palette

Synthesized from TradingView (Mirage theme), QuantConnect (dark IDE), LuxAlgo (premium SaaS), and QuantTekel (institutional prop).

| Token | Hex | Usage | Source |
|-------|-----|-------|--------|
| `--bg-app` | `#131722` | App background | TradingView Mirage |
| `--bg-surface` | `#1E222D` | Cards, panels, inputs | TradingView |
| `--bg-elevated` | `#2A2E39` | Hover states, active elements | TradingView |
| `--bg-overlay` | `#1E222DE6` | Modal backdrops (90% opacity) | Derived |
| `--border-default` | `#363A45` | Standard borders | TradingView |
| `--border-subtle` | `#2A2E39` | Inner dividers | TradingView |
| `--border-active` | `#2962FF` | Active/focused border | TradingView Dodger Blue |
| `--text-primary` | `#EDEFF5` | Headings, primary content | LuxAlgo off-white |
| `--text-secondary` | `#80899F` | Labels, captions, helper text | LuxAlgo blue-gray |
| `--text-muted` | `#787B86` | Disabled, placeholder | TradingView |
| `--text-inverse` | `#131722` | Text on accent backgrounds | Derived |
| `--accent-primary` | `#2962FF` | Primary actions, selection | TradingView |
| `--accent-success` | `#089981` | Bullish, positive metrics, "go" | TradingView teal-green |
| `--accent-danger` | `#F23645` | Bearish, negative, errors | TradingView |
| `--accent-warning` | `#FF9800` | Warnings, caution states | Standard amber |
| `--accent-info` | `#2196F3` | Informational, tooltips | QuantConnect |
| `--accent-classical` | `#06B6D4` | Classical/shallow model accent | Cyan-500 |
| `--accent-deep` | `#7C3AED` | Deep learning model accent | Purple-600 |
| `--accent-rl` | `#F59E0B` | RL model accent | Amber-500 |
| `--accent-ensemble` | `#EC4899` | Ensemble model accent | Pink-500 |
| `--accent-news` | `#44AAFF` | News/sentiment overlay | Custom |
| `--chart-line` | `#00D4AA` | Primary chart line (equity) | Current pipeline |
| `--chart-buyhold` | `#555555` | Buy-and-hold reference | Current pipeline |
| `--chart-drawdown` | `rgba(255,80,80,0.4)` | Drawdown shading | Current pipeline |

### 1.2 Typography Scale

| Role | Font | Size | Weight | Line Height | Tracking |
|------|------|------|--------|-------------|----------|
| **Metric value** | JetBrains Mono | 24px | 700 | 1.2 | — |
| **Metric label** | Inter | 11px | 600 | 1.0 | +0.08em (UPPERCASE) |
| **Navigation item** | Inter | 11px | 600 | 1.0 | +0.1em (UPPERCASE) |
| **Section header** | Inter | 16px | 600 | 1.3 | — |
| **Card title** | Inter | 14px | 600 | 1.3 | — |
| **Body text** | Inter | 13px | 400 | 1.5 | — |
| **Body small** | Inter | 12px | 400 | 1.4 | — |
| **Engine param** | JetBrains Mono | 12px | 400 | 1.4 | — |
| **Price/tick** | JetBrains Mono | 14px | 400 | 1.3 | — |
| **Button label** | Inter | 12px | 700 | 1.0 | +0.05em (UPPERCASE) |
| **Terminal log** | JetBrains Mono | 12px | 400 | 1.4 | — |
| **Tooltip text** | Inter | 11px | 400 | 1.4 | — |

### 1.3 Spacing System

| Token | Value | Usage |
|-------|-------|-------|
| `--space-1` | 4px | Tight padding, inline gaps |
| `--space-2` | 8px | Standard component gaps |
| `--space-3` | 12px | Card internal padding |
| `--space-4` | 16px | Section padding, form groups |
| `--space-5` | 20px | Panel padding |
| `--space-6` | 24px | Page margins |
| `--space-8` | 32px | Section gaps |
| `--space-10` | 40px | Major section dividers |

### 1.4 Layout Constants

| Token | Value | Usage |
|-------|-------|-------|
| `--sidebar-collapsed` | 56px | Icon-only sidebar width |
| `--sidebar-expanded` | 200px | Icon + label sidebar width |
| `--header-height` | 48px | Global header height |
| `--status-bar-height` | 24px | Bottom status bar height |
| `--terminal-height` | 200px | Terminal panel height (expanded) |
| `--min-window-width` | 1280px | Electron minimum window width |
| `--min-window-height` | 800px | Electron minimum window height |

### 1.5 Component Styling Reference

#### Metric Cards (from TradingView + QuantConnect)

```
┌─────────────────────────┐
│  SHARPE RATIO           │  ← 11px Inter UPPERCASE tracking-wide text-secondary
│  1.47                   │  ← 24px JetBrains Mono 700 text-primary
│  ▲ +0.12 vs baseline    │  ← 12px Inter text-success / text-danger
└─────────────────────────┘
Background: bg-surface
Border: 1px border-default
Border-radius: 8px
Padding: space-4
Hover: border shifts to border-active
```

#### Model Selection Cards (from QuantConnect + LuxAlgo)

```
┌──────────────────────────────┐
│  ── Classical ────────────── │  ← Category header with accent color line
│                              │
│  ┌──────────┐ ┌──────────┐  │
│  │ ● XGBoost│ │ Logistic │  │  ← Cards with category-colored left border
│  │  The     │ │  The     │  │
│  │ Workhorse│ │ Baseline │  │  ← Description truncated at 2 lines
│  └──────────┘ └──────────┘  │
│                              │
│  ── Deep Learning ────────── │  ← Purple accent line
│  ┌──────────┐ ┌──────────┐  │
│  │ ● LSTM   │ │  CNN     │  │
│  │  Time    │ │  Pattern │  │
│  │ Traveler │ │  Scanner │  │
│  └──────────┘ └──────────┘  │
└──────────────────────────────┘

Selected card: accent-{category} left border (3px), bg-elevated background, subtle glow
Unselected: border-default, bg-surface
Status dot: Green = available, Yellow = requires GPU, Red = unavailable
```

#### Data Tables (from QuantConnect AG Grid Alpine Dark)

```
┌─ Model ────────┬─ Sharpe ──┬─ Sortino ──┬─ Max DD ───┬─ Win Rate ──┬─ Trades ──┐
│ XGBoost        │      1.47 │       2.31 │    -8.50%  │      55.2%  │       320 │  ← JetBrains Mono for numbers
│ Logistic       │      0.98 │       1.42 │   -11.20%  │      52.1%  │       280 │  ← Red for negative DD
│ CNN            │      1.12 │       1.89 │    -9.80%  │      54.0%  │       295 │
└────────────────┴───────────┴────────────┴────────────┴─────────────┴───────────┘
Font: 13px JetBrains Mono for numeric, 13px Inter for text
Row height: 42px
Header height: 48px with sort arrows
Border: border-subtle between rows
Alternating: bg-app / bg-surface every other row
Sortable: click column header
Filterable: column header dropdown
```

### 1.6 Animation & Motion

| Element | Animation | Duration | Easing |
|---------|-----------|----------|--------|
| Sidebar expand/collapse | Width transition | 200ms | ease-out |
| Tab content switch | Fade in | 150ms | ease-in |
| Card hover | Border color + elevation | 150ms | ease-out |
| Progress bar | Smooth fill | 300ms | ease-in-out |
| Toast notification | Slide in from top-right | 200ms | ease-out |
| Terminal panel | Slide up from bottom | 200ms | ease-out |
| Skeleton loading | Pulse opacity 0.4 → 0.8 | 1.5s | ease-in-out (infinite) |
| Metric value update | Counter roll | 300ms | ease-out |
| Chart crosshair | Immediate (no animation) | 0ms | — |

---

## 2. Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REACT FRONTEND (Sprint 8)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Tier 1: Config Store (Zustand) — Low Velocity               │   │
│  │  ├─ useBacktestStore: 50+ form params (models, pair, etc.)   │   │
│  │  ├─ useJobStore: active jobs, history, selected job          │   │
│  │  └─ useSettingsStore: verbose mode, theme, API URL           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Tier 2: WebSocket Bypass (useRef + Imperative) — High Vel.  │   │
│  │  ├─ chartRef.current.update(tick) — 60fps chart updates      │   │
│  │  ├─ progressRef.current = value — progress bar bypass        │   │
│  │  └─ terminalRef.current.append(log) — terminal streaming     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Tier 3: Heavy Data (TanStack Query + Web Workers)            │   │
│  │  ├─ useQuery('results', fetchResults) — cached API calls     │   │
│  │  ├─ dataParser.worker.ts — off-thread JSON parse (120K rows) │   │
│  │  └─ Virtual scrolling in AG Grid — only render visible rows  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Rendering Layer                                              │   │
│  │  ├─ TradingView Lightweight Charts (Canvas, not SVG)         │   │
│  │  ├─ AG Grid (virtual DOM, handles 120K+ rows)               │   │
│  │  ├─ Monaco Editor (code editing, JSON config)                │   │
│  │  └─ React components (forms, cards, navigation)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP REST + WebSocket
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (Sprint 7 — COMPLETE)              │
│                    localhost:8000/api/v1/                             │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow: Backtest Execution

```
User clicks "DEPLOY BACKTEST"
        │
        ▼
┌─ Zustand Store packages config ─────────────────────────────────┐
│  { pair, models, config_overrides: { ...all 50+ params } }      │
└────────────────────┬────────────────────────────────────────────┘
                     │ POST /api/v1/backtest
                     ▼
┌─ FastAPI returns job_id ────────────────────────────────────────┐
│  { job_id: "abc-123", status: "pending", models: [...] }        │
└────────────────────┬────────────────────────────────────────────┘
                     │ WS /api/v1/backtest/{job_id}/ws
                     ▼
┌─ WebSocket events stream in ────────────────────────────────────┐
│  event: "job_started"     → Terminal: "Backtest started..."     │
│  event: "model_training"  → Progress: "Training Logistic..."    │
│       status: "starting"  → Terminal: "Starting model..."       │
│  event: "model_training"  → KPI: update partial metrics        │
│       status: "complete"  → Terminal: "Complete. Sharpe: 1.47"  │
│  event: "job_complete"    → Navigate to Results page           │
│       metrics: [...]      → Fetch full results via REST         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Flow: Results Loading

```
job_complete event received
        │
        ├─► GET /api/v1/backtest/{job_id}/results
        │       │
        │       ▼ Response: 5MB+ JSON (equity curve + 120K trades)
        │       │
        │       ▼ Web Worker parses JSON off main thread
        │       │  dataParser.worker.ts:
        │       │    1. Parse JSON string → JS objects
        │       │    2. Extract equity curve → { time, value }[]
        │       │    3. Extract trade log → TradeRow[]
        │       │    4. PostMessage back to main thread
        │       │
        │       ▼ Main thread receives parsed data
        │       │
        │       ├─► EquityCurveChart: chartRef.current.setData(curve)
        │       ├─► TradeLogTable: AG Grid receives row data
        │       └─► MetricsGrid: Zustand store updated
        │
        └─► TanStack Query caches result
                Next visit to Results page → instant load from cache
```

---

## 3. Use Cases

### 3.1 Researcher Use Cases

| UC# | Use Case | Actor | Flow |
|-----|----------|-------|------|
| UC-01 | **Quick Smoke Test** | Researcher | Select EURUSD H1 → Choose logistic → DEPLOY BACKTEST → See results in <2min |
| UC-02 | **Full Walk-Forward** | Researcher | Select pair + timeframe → Choose XGBoost → Set 3 months walk-forward → Configure triple barrier → Run overnight → Review results |
| UC-03 | **Model Comparison** | Researcher | Select multiple models (3-5) → Same pair/timeframe → Run comparison → View leaderboard sorted by Sharpe → Drill into individual model results |
| UC-04 | **Feature Engineering Study** | Researcher | Enable/disable indicator groups → Run same model multiple times with different feature sets → Compare which indicators improve Sharpe |
| UC-05 | **Execution Model Tuning** | Researcher | Select model → Switch from fixed to Kelly sizing → Add Chandelier trailing stop → Set 10% DD circuit breaker → Compare net results vs baseline |
| UC-06 | **News Sentiment Impact** | Researcher | Enable VADER sentiment + event flags → Run backtest → View equity curve with NFP/FOMC markers → Assess whether news features improve the model |
| UC-07 | **Previous Run Review** | Researcher | Navigate to Dashboard → Select previous run from history → View full results without re-running |
| UC-08 | **Export for Analysis** | Researcher | Run backtest → Export metrics CSV → Export equity curve PNG → Export config JSON → Import into external analysis tools |
| UC-09 | **Hyperparameter Exploration** | Researcher | Set HPO trials to 50 → Choose Optuna direction "maximize" → Run → View optimization trace showing which trials improved Sharpe → Review best config |
| UC-10 | **Cross-Pair Analysis** | Researcher | Run same model on 3 pairs → Compare leaderboards → Identify which pair the model performs best on → Check correlation between pair performances |

### 3.2 Account & Session Use Cases (Sprint 10 Prep)

| UC# | Use Case | Actor | Flow |
|-----|----------|-------|------|
| UC-11 | **First Launch** | New user | App launches → Welcome wizard → Set data directory → Configure OANDA API key → Choose default pair → Run sample backtest |
| UC-12 | **License Activation** | Licensed user | Settings → License tab → Enter Paddle license key → Validate online → Unlock all models + advanced execution |
| UC-13 | **Trial Mode** | Trial user | First launch → 14-day full access trial → Countdown shown in header → Day 15: restrict to 3 models + basic execution → License prompt |
| UC-14 | **Settings Persistence** | Any user | Configure preferences (theme, verbose mode, default pair) → Settings saved to encrypted local SQLite → Persist across sessions |
| UC-15 | **Data Source Management** | Any user | Settings → Data Sources → Enter OANDA API key → Test connection → Download new pairs → Verify data integrity |

### 3.3 Error & Edge Case Use Cases

| UC# | Use Case | Actor | Flow |
|-----|----------|-------|------|
| UC-16 | **Backend Crash During Run** | Researcher | Python crashes mid-backtest → Electron detects process death → Auto-restart Python → Show "Recovering..." toast → Reconnect WebSocket → Offer to resume or discard |
| UC-17 | **Large Dataset Handling** | Researcher | Load 10-year M30 EURUSD (175K rows) → Web Worker parses off-thread → Chart renders most recent 3000 candles → Scroll left loads historical chunks → UI stays responsive |
| UC-18 | **Validation Failure** | Researcher | Set PT mult < SL mult → Validation alerts appear immediately → "DEPLOY BACKTEST" button shows warning count → Click still allowed (warnings, not errors) → Hard errors block deployment |
| UC-19 | **WebSocket Disconnect** | Researcher | Network drops during backtest → Status bar shows red WS dot → Fallback to polling GET /status every 2s → Reconnect when available → Merge any missed updates |
| UC-20 | **GPU Unavailable for Deep Model** | Researcher | Select CNN model → GPU warning appears → "GPU not detected. CNN will train on CPU (10-50x slower). Continue?" → User confirms or switches model |
| UC-21 | **Invalid Config Import** | Researcher | Import config JSON from file → Schema validation runs → Invalid keys highlighted → Correct or reject → Valid config loads into form |
| UC-22 | **OANDA API Key Invalid** | Researcher | Enter OANDA key → Test connection fails → "Authentication failed. Check your API key." → Key not saved → Retry allowed |

---

## 4. Component Specification

### 4.1 Layout Components

#### AppShell

The root layout component. Manages sidebar, header, terminal panel, and status bar.

```
┌─────────────────────────────────────────────────────────────┐
│ ┌──┐  FX ML Backtester              [GPU ●][WS ●] [⚙ 🌙]  │ ← Header (48px)
│ └──┘                                                        │
├────┬────────────────────────────────────────────────────────┤
│ ◉  │                                                        │
│ D  │                                                        │
│ ◉  │                                                        │
│ B  │        <Outlet /> — React Router                       │
│ ◉  │        Current page content renders here               │
│ R  │                                                        │
│ ◉  │                                                        │
│ C  │                                                        │
│ ◉  │                                                        │
│ S  │                                                        │
│ ◉  │                                                        │
│ ⚙  │                                                        │
│    │────────────────────────────────────────────────────────│
│    │  ▸ Terminal [Collapse ▼]                                │ ← Terminal (200px, collapsible)
├────┴────────────────────────────────────────────────────────┤
│ ● Connected │ Job: abc-123 running │ Memory: 2.1GB │ v1.0  │ ← Status bar (24px)
└─────────────────────────────────────────────────────────────┘
```

**Sidebar Navigation Items** (6 pages):

| Icon | Label | Route | Description |
|------|-------|-------|-------------|
| LayoutDashboard | DASHBOARD | `/` | KPI overview, recent runs, model heatmap |
| FlaskConical | BACKTEST | `/backtest` | Configure and launch backtests |
| BarChart3 | RESULTS | `/results` | Charts, metrics, trade log, export |
| GitCompare | COMPARE | `/compare` | Multi-model leaderboard + significance |
| Newspaper | NEWS | `/news` | Sentiment dashboard, event calendar |
| Settings | SETTINGS | `/settings` | Config, GPU, data sources, license (S10) |

**Header Elements**:

| Position | Element | Data Source |
|----------|---------|-------------|
| Left | App logo + title "FX ML Backtester" | Static |
| Center | Active job status (if running) | WebSocket |
| Right | GPU status dot (green/yellow/red) | GET /api/v1/health → redis status |
| Right | WebSocket status dot (green/red) | WS connection state |
| Right | Verbose mode toggle (Pro/Apprentice) | useSettingsStore |
| Right | Theme toggle (always dark for v1) | useSettingsStore |

**Status Bar Elements**:

| Position | Element | Data Source |
|----------|---------|-------------|
| Left | Connection status ("Connected" / "Reconnecting...") | WS state |
| Center | Active job ID + progress | useJobStore |
| Center | Memory usage | navigator.performance? or Electron IPC |
| Right | Version number | package.json version |

**Terminal Panel**:

- Collapsible bottom panel (200px when expanded, 32px when collapsed)
- Shows real-time log output from WebSocket events
- Monospace font (JetBrains Mono 12px)
- Auto-scroll to bottom, with "scroll lock" toggle
- Color-coded log levels: INFO (text-secondary), SUCCESS (accent-success), WARNING (accent-warning), ERROR (accent-danger)
- Clear button
- Copy all button
- Connected to WebSocket `onmessage` — appends directly via ref, no React re-renders

### 4.2 Dashboard Page

```
┌─────────────────────────────────────────────────────────────┐
│  DASHBOARD                                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │TOTAL     │ │BEST      │ │AVG WIN   │ │EQUITY    │       │
│  │RUNS      │ │SHARPE    │ │RATE      │ │CURVE     │       │
│  │  47      │ │  1.47    │ │  54.2%   │ │ ▁▂▃▅▇▆▅ │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
│  ┌─ Recent Backtests ────────────────────────────────────┐  │
│  │ Status │ Pair   │ Model   │ Sharpe │ Return │ Date   │  │
│  │ ● Done │ EURUSD │ XGBoost │  1.47  │ +12.4% │ 2h ago │  │
│  │ ● Done │ GBPUSD │ LSTM    │  0.89  │  +6.2% │ 5h ago │  │
│  │ ● Done │ EURUSD │ CNN     │  1.12  │  +8.7% │ 1d ago │  │
│  │ ◐ Run  │ USDJPY │ XGBoost │   —    │   —    │ now    │  │
│  │ [Load] [Compare] [Delete]                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Model × Pair Performance ────────────────────────────┐  │
│  │          EURUSD  GBPUSD  USDJPY  AUDUSD  USDCAD       │  │
│  │ XGBoost   1.47    1.12    0.98    0.87    1.03        │  │
│  │ Logistic  0.98    0.76    0.65    0.71    0.82        │  │
│  │ CNN       1.12    0.95    0.88    0.79    0.91        │  │
│  │ LSTM      0.89    0.84    0.92    0.68    0.77        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**KpiGrid — 4 cards**:

| Card | Metric | Format | Source |
|------|--------|--------|--------|
| 1 | Total Runs | integer | GET /api/v1/backtest → jobs.length |
| 2 | Best Sharpe | 2 decimal | max of all completed job metrics |
| 3 | Avg Win Rate | percentage | avg of all completed job win_rates |
| 4 | Equity Sparkline | mini line chart | best-performing run's equity curve (last 30 points) |

**RecentJobsTable columns**:

| Column | Width | Sortable | Format |
|--------|-------|----------|--------|
| Status | 60px | Yes | StatusDot (green=completed, yellow=running, red=failed) |
| Pair | 80px | Yes | Text |
| Model(s) | 120px | No | Comma-separated text |
| Sharpe | 80px | Yes | JetBrains Mono, 2 decimal |
| Return | 80px | Yes | JetBrains Mono, percentage, color-coded |
| Date | 100px | Yes | Relative time ("2h ago") |
| Actions | 120px | No | [Load] [Compare] [Delete] buttons |

**ModelHeatmap (Model × Pair Sharpe grid)**:

- Columns: currency pairs (6)
- Rows: models (10)
- Cells: Sharpe ratio value, background color scaled green (high) → red (low)
- Cell click → navigate to that backtest result
- Null cells (no data) shown as dash with muted background

### 4.3 Backtest Configuration Page ("Flight Deck")

The most complex page. Organized as a vertical flow of sections, each a collapsible card.

```
┌─────────────────────────────────────────────────────────────┐
│  NEW BACKTEST                              [DEPLOY BACKTEST] │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ Asset Selection ─────────────────────────────────────┐  │
│  │  Pair: [EURUSD ▼]    Timeframe: [H1 ▼]                │  │
│  │  87,600 rows | 2014-01-01 → 2024-01-01 | OANDA       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Model Selection ─────────────────────────────────────┐  │
│  │                                                        │  │
│  │  ── Classical ──────────────────────────── [select all]│  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐         │  │
│  │  │   XGBoost  │ │  Logistic  │ │ Random For.│         │  │
│  │  │ Workhorse  │ │  Baseline  │ │   Forest   │         │  │
│  │  │ ✓ selected │ │            │ │            │         │  │
│  │  └────────────┘ └────────────┘ └────────────┘         │  │
│  │                                                        │  │
│  │  ── Deep Learning ──────────────────────── [select all]│  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐         │  │
│  │  │    CNN     │ │    LSTM    │ │ Transformer│         │  │
│  │  │  Pattern   │ │    Time    │ │ Attention  │         │  │
│  │  │  Scanner   │ │  Traveler  │ │  Engine    │         │  │
│  │  │ ⚠ GPU      │ │ ⚠ GPU      │ │ ⚠ GPU      │         │  │
│  │  └────────────┘ └────────────┘ └────────────┘         │  │
│  │                                                        │  │
│  │  ── Reinforcement Learning ─────────────── [select all]│  │
│  │  ┌────────────┐                                       │  │
│  │  │ Dueling DQN│                                       │  │
│  │  │ Autonomous │                                       │  │
│  │  │   Agent    │                                       │  │
│  │  │ ⚠ GPU      │                                       │  │
│  │  └────────────┘                                       │  │
│  │                                                        │  │
│  │  ── Ensemble ───────────────────────────── [select all]│  │
│  │  ┌────────────────────┐                               │  │
│  │  │ Adaptive Regime    │                               │  │
│  │  │    Shapeshifter    │                               │  │
│  │  │ ⚠ GPU              │                               │  │
│  │  └────────────────────┘                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Feature Engineering ─────────────────────────────────┐  │
│  │  Core Indicators (toggle grid, 3 columns)             │  │
│  │  ┌─ADX─┐ ┌─ATR─┐ ┌─EMA─┐ ┌─SMA─┐ ┌─RSI─┐ ┌─MACD─┐  │  │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └──────┘  │  │
│  │  ┌BBands┐ ┌Donch┐ ┌Stoch┐                             │  │
│  │  └──────┘ └─────┘ └─────┘                             │  │
│  │                                                        │  │
│  │  Feature Engineering                                   │  │
│  │  ┌─FracDiff── [ON] d: [0.40 ▓▓▓▓░░░░░░] ─┐          │  │
│  │  └─────────────────────────────────────────┘          │  │
│  │  ┌─Crossover Bins [ON]─┐ ┌─Price-MA Z [ON]─┐         │  │
│  │  └──────────────────────┘ └─────────────────┘         │  │
│  │                                                        │  │
│  │  Lag Features                                          │  │
│  │  Lags: [14 ▓▓▓▓░░░░░░]    Depth: [1 ▓░░]             │  │
│  │                                                        │  │
│  │  ▸ Advanced Toggles (16 features)              [▼]    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Labels & Triple Barrier ─────────────────────────────┐  │
│  │  Label Threshold: [0.0005]                             │  │
│  │  Triple Barrier: [ON]                                  │  │
│  │  PT Mult: [2.0]  SL Mult: [2.0]  Max Hold: [36]      │  │
│  │  Neutral Zone: [0.50]                                  │  │
│  │                                                        │  │
│  │  ⚠ 2 validation warnings                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Walk-Forward & HPO ──────────────────────────────────┐  │
│  │  HPO Trials: [10 ▓▓░░░]    Direction: [maximize ▼]    │  │
│  │  Seed: [42]                                            │  │
│  │  Train Months: [36 ▓▓▓░]  Test Months: [1 ▓]          │  │
│  │  Active Rate: [0.15 ▓▓░]  Coverage: [0.15 ▓▓░]       │  │
│  │  Confidence: [0.80 ▓▓▓▓]                              │  │
│  │  Trading Costs: [ON]   Slippage: [0.25 bps]           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Execution Models (Advanced) ─────────────────── [▼] ─┐  │
│  │  ▸ Position Sizing                                     │  │
│  │    Method: [fixed ▼]  Risk Fraction: [0.02]            │  │
│  │  ▸ Stop-Loss / Take-Profit                             │  │
│  │    SL Method: [atr ▼]  TP Method: [atr ▼]             │  │
│  │  ▸ Trailing Stops                                      │  │
│  │    Method: [none ▼]  Activation: [0.01]                │  │
│  │  ▸ Risk Management                                     │  │
│  │    Max DD: [15% ▓▓░]  Max Consec Losses: [5]          │  │
│  │    Daily Loss Limit: [3% ▓░]                           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ─────────────────────────────────────────────────────────  │
│  2 models selected │ EURUSD H1 │ 2 warnings │ 0 errors     │
│                                        [DEPLOY BACKTEST]    │
└─────────────────────────────────────────────────────────────┘
```

**ModelSelector — Card Specification**:

Each model card is a clickable tile with:
- Category accent border color (left 3px): classical=cyan, deep=purple, rl=amber, ensemble=pink
- Model name (14px Inter 600)
- Short description (12px Inter 400, truncated at 2 lines)
- Category icon in top-right corner
- Status dot: green (ready), yellow (requires GPU), red (unavailable)
- Selected state: elevated background, accent border glow, checkmark overlay
- Multi-select: clicking toggles selection, "Select All" per category
- Maximum 5 models per backtest run (frontend validation)

**Model Card Descriptors (Verbose Mode)**:

| Model | Name | Short | Apprentice Description |
|-------|------|-------|----------------------|
| `logistic` | Logistic Regression | The Baseline | Fast linear model. Use to establish baseline performance before deploying heavy neural networks. |
| `svm` | Support Vector Machine | The Kernel | Kernel-based classifier (RBF). Good for non-linear decision boundaries in feature space. |
| `random_forest` | Random Forest | The Forest | Ensemble of decision trees with bagging. Robust, handles mixed features well. |
| `decision_tree` | Decision Tree | The Tree | Single decision tree. Highly interpretable but prone to overfitting. Use for feature analysis. |
| `xgboost` | XGBoost | The Workhorse | Gradient-boosted trees. Exceptionally good at finding complex non-linear patterns in tabular data. |
| `cnn` | CNN | Pattern Scanner | 1D convolutional network that learns local price patterns across sliding windows. |
| `lstm` | LSTM | Time Traveler | Deep learning network designed for sequential time-series. Remembers price action across hundreds of bars. |
| `transformer` | Transformer | Attention Engine | Self-attention architecture that weighs the importance of every historical bar simultaneously. |
| `dqn` | Dueling DQN | Autonomous Agent | Reinforcement learning agent. Instead of predicting price, it learns a trading policy through trial and error. |
| `ensemble_adaptive_regime` | Adaptive Regime | The Shapeshifter | Dynamically shifts between models depending on market regime (trending vs ranging). |

### 4.4 Results Page

```
┌─────────────────────────────────────────────────────────────┐
│  RESULTS — EURUSD H1 — XGBoost                              │
│  [Zero Lookahead Bias Confirmed ✓]                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │SHARPE    │ │MAX DD    │ │RETURN    │ │WIN RATE  │       │
│  │  1.47    │ │  -8.5%   │ │ +12.4%   │ │  55.2%   │       │
│  │ ▲ +0.12  │ │ ▼ -2.1%  │ │ ▲ +3.2%  │ │ ▲ +1.3%  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │TRADES    │ │ACTIVE %  │ │F1 MACRO  │ │OUTPERF.  │       │
│  │  320     │ │  15.2%   │ │  0.543   │ │  +4.7%   │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                              │
│  [CSV ↓] [PNG ↓] [JSON ↓]                                   │
│                                                              │
│  ┌─ Equity Curve ────────────────────────────────────────┐  │
│  │                                                        │  │
│  │  ▁▂▃▅▆▇▅▃▄▅▇▇█▇▅▄▃▄▅▇█████▇▅▄▃▂▃▄▅▆▇█████████████  │  │
│  │  ─ ─ ─ ─ buy & hold ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │  │
│  │  ░░░░ drawdown area ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│  │                                                        │  │
│  │  |Train||Test||Train||Test||Train||Test|  ← WF overlay │  │
│  │                                                        │  │
│  │  Crosshair: O=1.0842 H=1.0851 L=1.0838 C=1.0847      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Monthly Breakdown ──────────────────────────────────┐   │
│  │  ┌─ Table (2/3) ──────────┐ ┌─ Chart (1/3) ───────┐ │   │
│  │  │ Month │ Ret  │ WR │ Tr │ │  ▓ ▓   ▓             │ │   │
│  │  │ Jan   │+1.2% │55% │ 28 │ │  ▓ ▓ ▓ ▓   ▓         │ │   │
│  │  │ Feb   │-0.8% │48% │ 25 │ │  ▓ ▓ ▓ ▓ ▓ ▓   ▓     │ │   │
│  │  │ Mar   │+2.1% │61% │ 32 │ │                     │ │   │
│  │  └─────────────────────────┘ └─────────────────────┘ │   │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ HPO Diagnostics ────────────────────────────────────┐   │
│  │  ┌─ Param Importance ──────┐ ┌─ Opt Trace ─────────┐ │   │
│  │  │ confidence_threshold ██ │ │  ·  ·    ·          │ │   │
│  │  │ lags             ████   │ │    ·  · ·    ·      │ │   │
│  │  │ tb_pt_mult       ███    │ │      ·   ·  ·   ·   │ │   │
│  │  │ fracdiff_d       ██     │ │  ──────────────────  │ │   │
│  │  └─────────────────────────┘ └─────────────────────┘ │   │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Trade Log ───────────────────────────────────────────┐  │
│  │  # │ Entry Date │ Exit Date │ Dir │ Pips │ Return │  │  │
│  │  1 │ 2024-01-15 │ 2024-01-17│ BUY │ +12  │ +0.12% │  │  │
│  │  2 │ 2024-01-22 │ 2024-01-23│ SELL│  -8  │ -0.08% │  │  │
│  │  ... (120,000 rows, virtual scroll)                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ▸ Best Configuration (JSON)                          [▼]  │
│  { "confidence_threshold": 0.85, "lags": 18, ... }          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**EquityCurveChart Specification**:

- Library: TradingView Lightweight Charts v4
- Height: 60% of viewport (approximately 480px on 800px window)
- Background: `#131722` (matching app background)
- Grid lines: `#2A2E39` (subtle)
- Primary line: equity curve, color `#089981` (teal-green), width 2
- Secondary line: buy-and-hold, color `#787B86` (muted), dashed, width 1
- Drawdown: filled area below zero, color `rgba(242,54,69,0.2)`, on right Y-axis
- Crosshair: enabled, with OHLC data displayed in top-left floating tooltip
- Walk-forward overlay: alternating shaded bands (blue tint for train, green tint for test)
- Event markers: vertical dashed lines for NFP/FOMC/CPI events (color-coded by impact)
- Mouse wheel: zoom in/out
- Drag: pan horizontally
- Default visible range: last 3000 candles (loads more on scroll left)

**TradeLogTable Specification**:

- Library: AG Grid Community (virtual scrolling)
- Font: JetBrains Mono 12px for numeric columns, Inter 12px for text
- Row height: 42px
- Header height: 48px with sort indicators
- Columns: #, Entry Date, Exit Date, Direction, Entry Price, Exit Price, Pips, Return %, Duration, Barrier Hit (if TB enabled)
- Direction column: colored pill (green BUY / red SELL)
- Return column: colored text (green positive / red negative)
- Sortable: all columns
- Filterable: direction, date range, return range
- Selectable: click row to highlight trade on equity curve chart
- Export: built-in CSV export button
- Pagination: infinite scroll (virtual), no page buttons
- Row count display: "Showing 320 of 320 trades" or "120,000 trades (virtual scroll)"

### 4.5 Model Comparison Page

```
┌─────────────────────────────────────────────────────────────┐
│  MODEL COMPARISON — EURUSD H1                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ Leaderboard ─────────────────────────────────────────┐  │
│  │ Rank │ Model    │ Sharpe │ Sortino │ MaxDD │ Win% │ PF │  │
│  │  #1  │ XGBoost  │  1.47  │   2.31  │ -8.5% │55.2% │2.1│  │
│  │  #2  │ CNN      │  1.12  │   1.89  │ -9.8% │54.0% │1.9│  │
│  │  #3  │ LSTM     │  0.89  │   1.52  │-11.2% │52.1% │1.6│  │
│  │  #4  │ Logistic │  0.76  │   1.21  │-14.3% │50.8% │1.3│  │
│  │                                                              │
│  │  Sort by: [Sharpe ▼]  Direction: [Desc ▼]                  │
│  └────────────────────────────────────────────────────────────┘
│                                                              │
│  ┌─ Equity Overlay ──────────────────────────────────────┐  │
│  │  ── XGBoost (green)                                    │  │
│  │  ── CNN (blue)                                         │  │
│  │  ── LSTM (purple)                                      │  │
│  │  ── Logistic (gray)                                    │  │
│  │  [Full equity curves overlaid on same chart]           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Significance Testing ────────────────────────────────┐  │
│  │          XGBoost  CNN     LSTM    Logistic             │  │
│  │  XGBoost    —     *0.03   ns      **0.008             │  │
│  │  CNN       *0.03    —     ns      *0.04               │  │
│  │  LSTM       ns      ns     —       ns                 │  │
│  │  Logistic **0.008 *0.04   ns       —                  │  │
│  │  ns = not significant (p>0.05)                         │  │
│  │  * p<0.05  ** p<0.01                                  │  │
│  │  [Heatmap: green=significant, red=not, diagonal=gray]  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**LeaderboardTable — Columns**:

| Column | Key | Format | Sortable |
|--------|-----|--------|----------|
| Rank | computed | #N badge | Auto (by sort metric) |
| Model | model | Text + category color dot | Yes |
| Sharpe | sharpe | 2 decimal, JetBrains Mono | Yes |
| Sortino | sortino | 2 decimal | Yes |
| Max Drawdown | max_drawdown | Percentage (negative = red) | Yes |
| Win Rate | win_rate | Percentage | Yes |
| Profit Factor | profit_factor | 2 decimal | Yes |
| Total Return | total_return | Percentage, color-coded | Yes |
| Total Trades | total_trades | Integer | Yes |

### 4.6 News & Sentiment Page

```
┌─────────────────────────────────────────────────────────────┐
│  NEWS & SENTIMENT                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ Sentiment Overview ─────────────────────────────────┐  │
│  │  ┌─ EURUSD ─────────┐ ┌─ GBPUSD ─────────┐          │  │
│  │  │ Sentiment: 0.32   │ │ Sentiment: -0.15  │          │  │
│  │  │ ▓▓▓▓▓▓▓▓░░ Bullish│ │ ░░░░░░░▓▓▓ Bearish│          │  │
│  │  │ Events: FOMC (2d) │ │ Events: None      │          │  │
│  │  └───────────────────┘ └───────────────────┘          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Upcoming Events ────────────────────────────────────┐  │
│  │ Date       │ Event          │ Impact │ Pair(s)       │  │
│  │ 2024-02-01 │ NFP            │ HIGH   │ EURUSD,GBPUSD │  │
│  │ 2024-02-03 │ FOMC Minutes   │ HIGH   │ All USD pairs │  │
│  │ 2024-02-05 │ CPI            │ MED    │ All USD pairs │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Sentiment Timeline ─────────────────────────────────┐  │
│  │  VADER compound score over last 30 days               │  │
│  │  ▁▂▃▅▇▆▅▃▄▅▇▇█▇▅▄▃▄▅▇█████▇▅▄▃▂▃▄▅▆▇██████████   │  │
│  │  Events marked as vertical lines on the chart         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Sentiment Configuration ────────────────────────────┐  │
│  │  Backend: [VADER ▼]  (or finBERT if installed)        │  │
│  │  Window: [6 bars ▼] [24 bars ▼]                       │  │
│  │  Event flags: [ON]                                     │  │
│  │  Sources: ☑ Reuters ☑ Bloomberg ☑ ForexLive ☑ Inv.com │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.7 Settings Page

```
┌─────────────────────────────────────────────────────────────┐
│  SETTINGS                                                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ General ────────────────────────────────────────────┐  │
│  │  Mode: ● Pro  ○ Apprentice                            │  │
│  │  Theme: ● Dark (only option for v1)                   │  │
│  │  API URL: [http://localhost:8000]                      │  │
│  │  Data Directory: [C:\Users\...\csv_data] [Browse]      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ GPU & Compute ──────────────────────────────────────┐  │
│  │  CUDA Available: ✓ Yes (NVIDIA RTX 3090, 24GB)       │  │
│  │  Thread Budget: [4 ▼]                                 │  │
│  │  Mixed Precision: [ON]                                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Data Sources ───────────────────────────────────────┐  │
│  │  OANDA API Key: [••••••••••••••••] [Test] [Save]      │  │
│  │  Connection Status: ✓ Valid                           │  │
│  │  Available Pairs: 6 | Downloaded: 6 of 18             │  │
│  │  [Download Missing] [Update All]                      │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ License (Sprint 10) ────────────────────────────────┐  │
│  │  Status: Trial (12 days remaining)                    │  │
│  │  License Key: [Enter key...] [Activate]               │  │
│  │  Machine ID: A4F2-8C1D-E7B3                           │  │
│  │  [Deactivate This Machine] [Transfer License]         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ Pipeline Configuration ─────────────────────────────┐  │
│  │  ┌─ Monaco Editor ────────────────────────────────┐  │  │
│  │  │  {                                               │  │  │
│  │  │    "vol_window_bars": 48,                        │  │  │
│  │  │    "high_vol_q": 0.75,                           │  │  │
│  │  │    "slip_norm_bps": 0.25,                        │  │  │
│  │  │    ...                                           │  │  │
│  │  │  }                                               │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │  [Reset to Defaults] [Apply]                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─ About ──────────────────────────────────────────────┐  │
│  │  FX ML Backtester v1.0.0                              │  │
│  │  Pipeline: 436 tests passing                          │  │
│  │  Python: 3.11.x | TensorFlow: 2.15 | PyTorch: 2.x    │  │
│  │  [Check for Updates]                                  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Phases

### Phase 8.1: Scaffold + Design System (Est: 3h)

**Goal**: Bootable dark-theme React app with layout shell and design tokens.

| Step | What | Files to Create/Modify | Verification |
|------|------|----------------------|--------------|
| 1.1 | `npm create vite@latest frontend -- --template react-ts` | `frontend/` scaffold | `npm run dev` boots |
| 1.2 | Install all deps (see Dependencies section) | `frontend/package.json` | No install errors |
| 1.3 | Configure Tailwind with design tokens as CSS custom properties | `frontend/tailwind.config.ts`, `frontend/postcss.config.js`, `frontend/src/styles/globals.css` | Tokens accessible as `var(--bg-app)` and Tailwind classes `bg-app` |
| 1.4 | Create `design-tokens.ts` with full palette, typography, spacing as typed constants | `frontend/src/design-tokens.ts` | Tokens importable in components |
| 1.5 | Add Google Fonts (Inter, JetBrains Mono) via `index.html` link tags | `frontend/index.html` | Fonts render correctly |
| 1.6 | Build `AppShell` — sidebar + header + content outlet + terminal + status bar | `frontend/src/components/layout/AppShell.tsx`, `Sidebar.tsx`, `Header.tsx`, `TerminalPanel.tsx`, `StatusBar.tsx` | App renders with sidebar, header, empty content, status bar |
| 1.7 | Build shared primitives: `MetricCard`, `StatusDot`, `ParamSlider`, `ParamToggle`, `ParamSelect`, `EmptyState`, `ZeroLeakageBadge` | `frontend/src/components/shared/*.tsx` | Each renders correctly in Storybook or test page |
| 1.8 | Configure Vite dev proxy to FastAPI backend | `frontend/vite.config.ts` | `/api/*` proxies to `localhost:8000` |

**Exit Criteria**: `npm run dev` shows dark-themed app with sidebar navigation, all 6 pages render (empty), fonts load, tokens work.

### Phase 8.2: API Client + State Management (Est: 3h)

**Goal**: All backend endpoints callable, Zustand stores functional, WebSocket connected.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 2.1 | Axios client: base URL, timeout 30s, error interceptor (toast on 5xx) | `frontend/src/api/client.ts` | Manual test with FastAPI running |
| 2.2 | TanStack Query provider + hooks: `usePairs()`, `useModels()`, `useJobStatus(id)`, `useJobResults(id)`, `useJobHistory()` | `frontend/src/api/queries.ts` | Console logs show fetched data |
| 2.3 | Zustand `useBacktestStore` — all 50+ form params with current defaults from `config.py` | `frontend/src/stores/useBacktestStore.ts` | Store initializes with correct defaults |
| 2.4 | Zustand `useJobStore` — activeJobs map, selectedJobId, jobHistory | `frontend/src/stores/useJobStore.ts` | Store initializes empty |
| 2.5 | Zustand `useSettingsStore` — verboseMode, theme, apiUrl, dataDir | `frontend/src/stores/useSettingsStore.ts` | Persisted to localStorage |
| 2.6 | WebSocket manager class: connect, disconnect, reconnect with exponential backoff, message dispatch | `frontend/src/api/websocket.ts` | Connection opens/closes cleanly |
| 2.7 | `useBacktestWebSocket` hook — connects WS for a job_id, dispatches events to stores/refs | `frontend/src/hooks/useBacktestWebSocket.ts` | Receives events from running backtest |
| 2.8 | Web Worker for off-thread JSON parsing | `frontend/src/workers/dataParser.worker.ts` | Parse 5MB JSON without blocking UI |
| 2.9 | `constants.ts` with all default values, valid ranges, model categories, validator rules | `frontend/src/lib/constants.ts` | Constants match `config.py` |
| 2.10 | `validation.ts` — TypeScript port of all 5 validators from `ui/validators.py` | `frontend/src/lib/validation.ts` | Same validation behavior as Streamlit |
| 2.11 | `formatters.ts` — formatMetric, formatPercent, formatPrice, formatDate | `frontend/src/lib/formatters.ts` | Correct formatting |

**Exit Criteria**: App boots, fetches models and pairs from FastAPI, Zustand store initialized, WS connects to active job.

### Phase 8.3: Dashboard Page (Est: 3h)

**Goal**: Landing page with KPI overview, recent runs table, model heatmap.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 3.1 | `DashboardPage` layout — KPI grid (top) + recent table (middle) + heatmap (bottom) | `frontend/src/pages/Dashboard/DashboardPage.tsx` | Page renders with placeholders |
| 3.2 | `KpiGrid` — 4 metric cards (total runs, best Sharpe, avg win rate, equity sparkline) | `frontend/src/pages/Dashboard/KpiGrid.tsx` | Cards render with "--" when no data |
| 3.3 | `RecentJobsTable` — AG Grid with status dots, actions, relative dates | `frontend/src/pages/Dashboard/RecentJobsTable.tsx` | Table renders, empty state shown |
| 3.4 | `ModelHeatmap` — 6×10 grid of Sharpe values, color-scaled | `frontend/src/pages/Dashboard/ModelHeatmap.tsx` | Grid renders with "--" cells |
| 3.5 | Wire to TanStack Query — auto-fetch job history on page load, populate all components | All dashboard files | Data flows from API to components |
| 3.6 | Empty state — when no runs exist, show welcome message + "Run Your First Backtest" CTA | `frontend/src/pages/Dashboard/DashboardPage.tsx` | Empty state renders correctly |

**Exit Criteria**: Dashboard shows real data when jobs exist, empty state when no jobs, heatmap populates.

### Phase 8.4: Backtest Configuration (Est: 5h)

**Goal**: Full "flight deck" for configuring and launching backtests.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 4.1 | `BacktestPage` layout — vertical flow of collapsible sections + sticky deploy bar at bottom | `frontend/src/pages/Backtest/BacktestPage.tsx` | Page renders with all sections |
| 4.2 | `AssetSelector` — pair dropdown (from `/pairs`), timeframe dropdown, data range display | `frontend/src/pages/Backtest/AssetSelector.tsx` | Selecting pair updates timeframes |
| 4.3 | `ModelSelector` — categorized card grid (4 categories), multi-select with "select all" per category, GPU warnings on deep models | `frontend/src/pages/Backtest/ModelSelector.tsx` | Clicking cards toggles selection, max 5 limit |
| 4.4 | `FeaturesPanel` — core indicators toggle grid (3-col), FracDiff + slider, lag controls, collapsible advanced toggles (16 features) | `frontend/src/pages/Backtest/FeaturesPanel.tsx` | Toggles update Zustand store |
| 4.5 | `LabelsPanel` — threshold input, triple barrier accordion with PT/SL/holding/neutral params | `frontend/src/pages/Backtest/LabelsPanel.tsx` | Conditional rendering when TB enabled |
| 4.6 | `HpoPanel` — HPO trials/direction/seed, walk-forward train/test months, coverage sliders, trading costs | `frontend/src/pages/Backtest/HpoPanel.tsx` | All sliders update store |
| 4.7 | `ExecutionPanel` — 4 accordions: position sizing, SL/TP, trailing stops, risk management. Each with method dropdown + conditional parameter inputs | `frontend/src/pages/Backtest/ExecutionPanel.tsx` | Methods switch, params show/hide correctly |
| 4.8 | `RunSummary` — modal dialog showing all selected config before deploy. Categorized sections. "DEPLOY BACKTEST" button with validation summary. | `frontend/src/pages/Backtest/RunSummary.tsx` | Dialog shows current config |
| 4.9 | `BacktestProgress` — top progress bar + per-model training status cards + log stream in terminal | `frontend/src/pages/Backtest/BacktestProgress.tsx` | Progress updates via WS |
| 4.10 | `useValidation` hook — runs all 5 validators on current store, returns `{ warnings, errors, ok }` | `frontend/src/hooks/useValidation.ts` | Warnings appear in real-time |
| 4.11 | Wire deploy: Zustand → `POST /api/v1/backtest` → get job_id → connect WS → navigate to results | Multiple files | Full end-to-end flow works |
| 4.12 | Verbose mode: when `useSettingsStore.verboseMode === true`, show descriptor text below each control | All config panels | Descriptions appear/disappear correctly |

**Exit Criteria**: Full config flow works: select pair, select models, configure features, click deploy, see progress, auto-navigate to results.

### Phase 8.5: Results & Charts (Est: 6h)

**Goal**: Professional results page with TradingView-style charts and dense data tables.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 5.1 | `ResultsPage` layout — metrics grid → export bar → equity chart → monthly → HPO → trade log → config viewer | `frontend/src/pages/Results/ResultsPage.tsx` | Page renders with all sections |
| 5.2 | `MetricsGrid` — 8 KPI cards in 2×4 grid, JetBrains Mono values, color-coded deltas, "Zero Lookahead Bias Confirmed" badge | `frontend/src/pages/Results/MetricsGrid.tsx` | All metrics display correctly |
| 5.3 | `EquityCurveChart` — TradingView Lightweight Charts integration. Equity line (teal), buy-hold (gray dashed), drawdown area (red). Crosshair with OHLC tooltip. | `frontend/src/components/charts/EquityCurveChart.tsx` | Chart renders with real equity data |
| 5.4 | `WalkForwardOverlay` — vertical shaded regions on chart (blue=train, green=test). Computed from walk-forward config. | `frontend/src/components/charts/WalkForwardOverlay.tsx` | Bands visible on chart |
| 5.5 | Data virtualization for chart — default last 3000 candles, fetch more on scroll left. Web Worker parses full dataset. | `EquityCurveChart.tsx` updates | Smooth scrolling through 120K candles |
| 5.6 | `MonthlySection` — formatted monthly table (AG Grid) + green/red bar chart side by side (2:1 ratio) | `frontend/src/pages/Results/MonthlySection.tsx` | Both render correctly |
| 5.7 | `MonthlyReturnsChart` — vertical bars, green positive, red negative | `frontend/src/components/charts/MonthlyReturnsChart.tsx` | Correct colors |
| 5.8 | `HpoDiagnostics` — param importance horizontal bars (left) + optimization trace scatter/line (right) | `frontend/src/pages/Results/HpoDiagnostics.tsx` | Both charts render |
| 5.9 | `ParamImportanceChart` — top 20 params, sorted descending, teal bars | `frontend/src/components/charts/ParamImportanceChart.tsx` | Renders correctly |
| 5.10 | `OptimizationTraceChart` — scatter dots (gray) + running-best line (teal) | `frontend/src/components/charts/OptimizationTraceChart.tsx` | Renders correctly |
| 5.11 | `TradeLogTable` — AG Grid with 120K+ rows, virtual scroll, sortable, filterable, monospace numbers | `frontend/src/pages/Results/TradeLogTable.tsx` | Scrolling through 120K rows smooth |
| 5.12 | Trade ↔ chart interaction — click trade row → chart scrolls to that point and highlights | `TradeLogTable.tsx` + `EquityCurveChart.tsx` coordination | Click scrolls chart |
| 5.13 | `ExportBar` — CSV download (Blob), PNG export (chart.toImage()), JSON export (config) | `frontend/src/components/shared/ExportBar.tsx` | All 3 downloads work |
| 5.14 | `ConfigViewer` — collapsible JSON tree with syntax highlighting | `frontend/src/pages/Results/ConfigViewer.tsx` | JSON renders with colors |
| 5.15 | `EquitySection` — compose chart + drawdown + event markers + walk-forward overlay | `frontend/src/pages/Results/EquitySection.tsx` | All overlays render together |
| 5.16 | Web Worker integration — off-thread JSON parse for large results payloads | `dataParser.worker.ts` + Results page | No UI freeze on 5MB+ results |

**Exit Criteria**: Full results page renders with real data, charts interactive, trade log scrollable, exports download correctly.

### Phase 8.6: Model Comparison (Est: 3h)

**Goal**: Institutional-grade leaderboard and comparison tools.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 6.1 | `ComparePage` layout — leaderboard table (top) + equity overlay (middle) + significance matrix (bottom) | `frontend/src/pages/Compare/ComparePage.tsx` | Page renders with all sections |
| 6.2 | `LeaderboardTable` — dense AG Grid with all metrics, sortable by any column, category color dots, monospace numbers | `frontend/src/pages/Compare/LeaderboardTable.tsx` | Sorting works, correct formatting |
| 6.3 | `EquityOverlay` — multiple equity curves on same TradingView chart, legend with model names + colors | `frontend/src/pages/Compare/EquityOverlay.tsx` | 3+ curves render simultaneously |
| 6.4 | `SignificanceMatrix` — paired t-test p-values as colored heatmap (green=significant, red=not) | `frontend/src/pages/Compare/SignificanceMatrix.tsx` | Matrix renders correctly |
| 6.5 | `CorrelationMatrix` — Pearson correlation heatmap for portfolio analysis (stretch goal) | `frontend/src/components/charts/CorrelationMatrix.tsx` | Renders correctly |
| 6.6 | Integration with `/compare` endpoint or post-processing of multiple job results | `ComparePage.tsx` | End-to-end comparison works |

**Exit Criteria**: Run comparison with 3+ models, leaderboard sorts, equity curves overlay, significance matrix renders.

### Phase 8.7: Settings + Terminal + Polish (Est: 3h)

**Goal**: Settings page, verbose mode, error handling, terminal panel.

| Step | What | Files | Verification |
|------|------|-------|--------------|
| 7.1 | `SettingsPage` — 6 sections (General, GPU, Data Sources, License stub, Pipeline Config, About) | `frontend/src/pages/Settings/SettingsPage.tsx` | All sections render |
| 7.2 | `VerboseToggle` — switches between Pro and Apprentice mode, persisted in settings store | `frontend/src/pages/Settings/VerboseToggle.tsx` | Mode persists across page nav |
| 7.3 | `PipelineConfig` — Monaco editor with JSON validation, "Reset to Defaults" button | `frontend/src/pages/Settings/PipelineConfig.tsx` | Editor loads, edits validate |
| 7.4 | `DataSourceManager` — OANDA API key input (masked), test connection button, pair download status | `frontend/src/pages/Settings/DataSourceManager.tsx` | Key saves, test works |
| 7.5 | `GpuSettings` — GPU status display, thread budget slider, mixed precision toggle | `frontend/src/pages/Settings/GpuSettings.tsx` | Status shows correctly |
| 7.6 | `TerminalPanel` — collapsible bottom panel, real-time log streaming, color-coded levels, auto-scroll | `frontend/src/components/layout/TerminalPanel.tsx` | Logs stream during backtest |
| 7.7 | Global error boundary — catches React render errors, shows friendly message with retry | `frontend/src/components/shared/ErrorBoundary.tsx` | Errors caught gracefully |
| 7.8 | Loading skeletons — pulse animation placeholders for all data-fetching components | Throughout | Skeletons show while loading |
| 7.9 | Toast notifications — success/error/warning/info toasts via shadcn/ui toast | Throughout | Toasts appear on relevant actions |
| 7.10 | `SentimentGauge` — semi-circular gauge for VADER/finBERT scores | `frontend/src/components/charts/SentimentGauge.tsx` | Gauge renders correctly |
| 7.11 | `NewsPage` — sentiment overview, upcoming events, timeline, configuration | `frontend/src/pages/News/` | Full news page renders |

**Exit Criteria**: Settings persist, verbose mode works, terminal streams logs, errors caught, toasts appear.

---

## 6. API Contract

### 6.1 REST Endpoints Used

| Method | Endpoint | Request | Response | Used By |
|--------|----------|---------|----------|---------|
| GET | `/api/v1/health` | — | `{ status, version, redis, db_rows }` | Header status bar |
| GET | `/api/v1/pairs` | — | `{ pairs: [{ pair, timeframes }] }` | AssetSelector |
| GET | `/api/v1/pairs/{symbol}/data-range` | — | `{ symbol, timeframes }` | AssetSelector detail |
| GET | `/api/v1/models` | — | `{ models: [{ name, display_name, category, description }] }` | ModelSelector |
| POST | `/api/v1/backtest` | `BacktestRequest` | `{ job_id, status, pair, models }` | Deploy button |
| GET | `/api/v1/backtest?limit=50` | — | `{ jobs: [...] }` | Dashboard, job history |
| GET | `/api/v1/backtest/{job_id}` | — | `BacktestStatusResponse` | Job polling fallback |
| GET | `/api/v1/backtest/{job_id}/results` | — | `BacktestResultsResponse` | Results page |
| DELETE | `/api/v1/backtest/{job_id}` | — | 204 | Job deletion |
| POST | `/api/v1/data/download` | `{ pair, years }` | `{ job_id, pair, status }` | Data source manager |
| WS | `/api/v1/backtest/{job_id}/ws` | — | Event stream | Progress, terminal |

### 6.2 BacktestRequest Shape

```typescript
interface BacktestRequest {
  pair: string;                          // "EURUSD"
  models: string[];                      // ["logistic", "xgboost"]
  start_date?: string;                   // ISO date
  end_date?: string;                     // ISO date
  trading_costs?: boolean;               // default true
  months?: number;                       // walk-forward test window
  repeats?: number;                      // default 1
  config_overrides: {
    // Core
    confidence_threshold?: number;
    target_active_rate?: number;
    target_coverage?: number;
    calibrate_method?: "sigmoid" | "isotonic";
    eval_use_trading_costs?: boolean;
    slip_norm_bps?: number;

    // Triple Barrier
    use_triple_barrier?: boolean;
    tb_pt_mult?: number;
    tb_sl_mult?: number;
    tb_neutral_zone?: number;
    tb_max_holding?: number;
    label_threshold?: number;

    // Features
    lags?: number;
    lag_depth?: number;
    use_fracdiff?: boolean;
    fracdiff_d?: number;
    use_adx?: boolean;
    use_atr?: boolean;
    use_bbands?: boolean;
    use_ema?: boolean;
    use_sma?: boolean;
    use_rsi?: boolean;
    use_macd?: boolean;
    use_stoch?: boolean;
    use_sar?: boolean;
    use_donchian?: boolean;
    use_mtf_ma?: boolean;
    use_crossover_bins?: boolean;
    use_ma_spread?: boolean;
    use_price_ma_z?: boolean;
    use_indicator_states?: boolean;
    use_mtf_alignment?: boolean;
    use_mtf_align?: boolean;
    use_macd_atr_ratio?: boolean;
    use_triple_confirm?: boolean;
    use_trend_confirm?: boolean;
    use_vol_managed_mom?: boolean;
    use_vm_mom?: boolean;
    use_squeeze_breakout?: boolean;
    use_squeeze_expansion?: boolean;
    use_atr_channel_breakout?: boolean;
    use_ext_atr_low_adx?: boolean;
    use_reentry_mom?: boolean;
    use_slope_diff?: boolean;
    use_rv_features?: boolean;

    // Logistic HPs
    logit_C?: number;
    logit_solver?: string;
    logit_penalty?: string;
    logit_max_iter?: number;
    logit_tol?: number;

    // Execution (Sprint 2 — needs API integration)
    sizing_method?: "fixed" | "kelly" | "fixed_fractional" | "atr";
    risk_fraction?: number;
    kelly_fraction?: number;
    kelly_min_trades?: number;
    atr_risk_pct?: number;
    atr_sl_mult?: number;
    initial_equity?: number;
    max_leverage?: number;

    // Risk Manager
    max_drawdown_pct?: number;
    max_consecutive_losses?: number;
    daily_loss_limit_pct?: number;

    // Trailing
    trailing_method?: "none" | "standard" | "atr" | "chandelier";
    trailing_activation?: number;
  };
}
```

### 6.3 WebSocket Event Types

```typescript
type WsEvent =
  | { event: "job_started"; job_id: string; pair: string; models: string[] }
  | { event: "model_training"; job_id: string; model: string; status: "starting" }
  | { event: "model_training"; job_id: string; model: string; status: "complete"; metrics: Partial<Metrics> }
  | { event: "job_complete"; job_id: string; metrics: Metrics[] }
  | { event: "job_failed"; job_id: string; error: string }
  | { event: "download_started"; job_id: string; pair: string }
  | { event: "download_complete"; job_id: string; pair: string }
  | { event: "download_failed"; job_id: string; error: string };

interface Metrics {
  model: string;
  sharpe: number | null;
  sortino: number | null;
  max_drawdown: number | null;
  total_return: number | null;
  win_rate: number | null;
  total_trades: number | null;
  profit_factor: number | null;
  avg_trade: number | null;
}
```

---

## 7. State Management

### 7.1 useBacktestStore (Zustand)

```typescript
interface BacktestStore {
  // Asset
  pair: string;                          // "EURUSD"
  timeframe: string;                     // "H1"

  // Models
  selectedModels: string[];              // ["logistic", "xgboost"]

  // Core
  confidenceThreshold: number;
  targetActiveRate: number;
  targetCoverage: number;
  calibrateMethod: "sigmoid" | "isotonic";
  evalUseTradingCosts: boolean;
  slipNormBps: number;

  // Triple Barrier
  useTripleBarrier: boolean;
  tbPtMult: number;
  tbSlMult: number;
  tbNeutralZone: number;
  tbMaxHolding: number;
  labelThreshold: number;

  // Features — core indicators
  useAdx: boolean;
  useAtr: boolean;
  useBbands: boolean;
  useEma: boolean;
  useSma: boolean;
  useRsi: boolean;
  useMacd: boolean;
  useStoch: boolean;
  useSar: boolean;
  useDonchian: boolean;

  // Features — engineering
  useFracdiff: boolean;
  fracdiffD: number;
  useCrossoverBins: boolean;
  useMaSpread: boolean;
  usePriceMaZ: boolean;
  useIndicatorStates: boolean;

  // Features — lags
  lags: number;
  lagDepth: number;

  // Features — advanced
  useMtfMa: boolean;
  useMtfAlignment: boolean;
  useMtfAlign: boolean;
  useMacdAtrRatio: boolean;
  useTripleConfirm: boolean;
  useTrendConfirm: boolean;
  useVolManagedMom: boolean;
  useVmMom: boolean;
  useSqueezeBreakout: boolean;
  useSqueezeExpansion: boolean;
  useAtrChannelBreakout: boolean;
  useExtAtrLowAdx: boolean;
  useReentryMom: boolean;
  useSlopeDiff: boolean;
  useRvFeatures: boolean;

  // Logistic HPs
  logitC: number;
  logitSolver: string;
  logitPenalty: string;
  logitMaxIter: number;
  logitTol: number;

  // HPO
  nTrials: number;
  optunaDirection: "maximize" | "minimize";
  seed: number;

  // Walk-forward
  trainMonths: number;
  testMonths: number;

  // Execution
  sizingMethod: "fixed" | "kelly" | "fixed_fractional" | "atr";
  riskFraction: number;
  kellyFraction: number;
  kellyMinTrades: number;
  atrRiskPct: number;
  atrSlMult: number;
  initialEquity: number;
  maxLeverage: number;

  // Risk
  maxDrawdownPct: number;
  maxConsecutiveLosses: number;
  dailyLossLimitPct: number;

  // Trailing
  trailingMethod: "none" | "standard" | "atr" | "chandelier";
  trailingActivation: number;

  // Actions
  setField: <K extends keyof BacktestStore>(key: K, value: BacktestStore[K]) => void;
  resetToDefaults: () => void;
  toRequestPayload: () => BacktestRequest;
}
```

### 7.2 useJobStore (Zustand)

```typescript
interface JobStore {
  activeJobs: Map<string, JobState>;
  selectedJobId: string | null;
  jobHistory: JobSummary[];

  startJob: (jobId: string, pair: string, models: string[]) => void;
  updateJobProgress: (jobId: string, event: WsEvent) => void;
  completeJob: (jobId: string, metrics: Metrics[]) => void;
  failJob: (jobId: string, error: string) => void;
  selectJob: (jobId: string) => void;
  deleteJob: (jobId: string) => void;
  loadHistory: () => Promise<void>;
}

interface JobState {
  jobId: string;
  pair: string;
  models: string[];
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  progressText: string;
  completedModels: string[];
  currentModel: string | null;
  metrics: Map<string, Partial<Metrics>>;
  error: string | null;
  createdAt: Date;
  completedAt: Date | null;
}
```

### 7.3 useSettingsStore (Zustand + localStorage)

```typescript
interface SettingsStore {
  verboseMode: boolean;                  // false = Pro, true = Apprentice
  theme: "dark";                         // Only dark for v1
  apiUrl: string;                        // "http://localhost:8000"
  dataDir: string;                       // path to csv_data
  oandaApiKey: string | null;            // masked
  threadBudget: number;                  // 1-8
  mixedPrecision: boolean;
  sidebarCollapsed: boolean;
  terminalCollapsed: boolean;

  setField: <K extends keyof SettingsStore>(key: K, value: SettingsStore[K]) => void;
  loadFromStorage: () => void;
  saveToStorage: () => void;
}
```

---

## 8. Professional Safeguards

### 8.1 Data Integrity Safeguards

| Safeguard | Implementation | Where |
|-----------|---------------|-------|
| **Input validation** | All 5 validators run on every config change (debounced 300ms). Hard errors block deploy, warnings shown but allow proceed. | `useValidation` hook |
| **Type safety** | TypeScript strict mode, Zod schemas for all API responses | `frontend/tsconfig.json`, `frontend/src/api/schemas.ts` |
| **API response validation** | TanStack Query `select` + Zod parse on every API response | `frontend/src/api/queries.ts` |
| **CSRF protection** | Same-origin policy via Vite proxy (dev) + FastAPI CORS whitelist (prod) | `vite.config.ts`, `api/main.py` |
| **XSS prevention** | React's built-in escaping, no dangerouslySetInnerHTML except Monaco editor (sandboxed) | Throughout |
| **Data sanitization** | All numeric inputs clamped to valid ranges before sending to API | `useBacktestStore.toRequestPayload()` |
| **Race condition protection** | AbortController on all API calls, cancel previous request when new one fires | `frontend/src/api/client.ts` |
| **Stale data detection** | TanStack Query `staleTime: 30s` for job status, `staleTime: 5min` for pairs/models, `cacheTime: 30min` | `frontend/src/api/queries.ts` |

### 8.2 Error Handling Strategy

| Error Type | User Experience | Recovery |
|------------|----------------|----------|
| **Network failure** | Red status dot + "Connection Lost" in status bar + toast "Unable to reach backend" | Auto-retry every 5s, reconnect WS when available |
| **API 4xx error** | Toast with specific error message from `detail` field | User corrects input and retries |
| **API 5xx error** | Toast "Server error. Please try again." + terminal shows error log | Auto-retry once after 2s |
| **WebSocket disconnect** | Status bar WS dot turns red, fallback to polling GET /status every 2s | Auto-reconnect with exponential backoff (1s, 2s, 4s, 8s, max 30s) |
| **WebSocket reconnect** | Green flash on status bar + toast "Reconnected" | Resume from last known state |
| **Backend crash** (Electron) | Modal: "The backend process has stopped. Restarting..." → Electron restarts Python | Auto-restart Python process |
| **JSON parse failure** | Log to terminal, skip malformed message | Continue with next message |
| **Chart render failure** | Graceful fallback: "Chart unavailable" placeholder with "Retry" button | User clicks retry |
| **Large data OOM** | Web Worker catches OOM → shows "Dataset too large to display. Try a shorter date range." | User adjusts parameters |
| **React render error** | Error boundary catches → shows "Something went wrong" with component stack + "Reload" button | User clicks reload |

### 8.3 Performance Safeguards

| Guard | Threshold | Action |
|-------|-----------|--------|
| **Main thread blocking** | > 100ms | Move computation to Web Worker |
| **WebSocket message rate** | > 100 msg/s | Throttle UI updates to 30fps, buffer excess messages |
| **Chart data points** | > 10,000 | Downsample to 3,000 visible, load more on scroll |
| **Trade log rows** | > 50,000 | Force AG Grid virtual scroll, disable full-height render |
| **API response size** | > 10MB | Show warning: "Large result set. Loading may take a moment." |
| **Zustand store size** | > 1MB | Move large data to TanStack Query cache, keep store lean |
| **Memory usage** | > 2GB (Electron) | Clear TanStack Query cache, show memory warning in status bar |

### 8.4 Security Considerations (Sprint 10 Prep)

| Concern | Frontend Mitigation | Full Solution (S10) |
|---------|-------------------|-------------------|
| **API key storage** | Never log, mask in UI (`••••••••`), store in settings store (not localStorage in production) | Encrypted SQLite via machine fingerprint key derivation |
| **Config tampering** | Validate all API responses with Zod schemas | Server-side validation on all endpoints |
| **Model IP theft** | N/A in frontend | AES-256 encrypted model files, decryption via license key + hardware ID |
| **License bypass** | Stub — license UI exists but not enforced | Paddle SDK integration, online verification, offline grace period |
| **Debug mode** | DevTools auto-open in dev, disabled in production build | Anti-debugging checks in Electron main process |
| **Source code exposure** | Vite build minifies + tree-shakes | PyInstaller with `--key` for Python, Webpack obfuscation for React |

---

## 9. Accessibility & i18n

### 9.1 Accessibility (WCAG 2.1 AA)

| Requirement | Implementation |
|-------------|---------------|
| **Keyboard navigation** | All interactive elements focusable, Tab order follows visual order, Enter/Space activate |
| **Focus indicators** | Visible focus ring (accent-primary 2px outline) on all interactive elements |
| **ARIA labels** | All icons have `aria-label`, charts have `aria-label` with summary, form fields have associated labels |
| **Color contrast** | Primary text on dark bg: 15.2:1 ratio (WCAG AAA). Muted text: 4.8:1 (AA) |
| **Screen reader** | KPI cards announce value + label + delta. Charts announce summary. Terminal announces new log lines |
| **Reduced motion** | `prefers-reduced-motion` media query disables all animations |
| **High contrast** | Border colors increase contrast in `prefers-contrast: more` |

### 9.2 i18n Preparation (Post-Launch)

All user-facing strings extracted to `frontend/src/lib/strings.ts` as constants. No inline strings in components. This enables future i18n via `react-i18next` without refactoring.

```typescript
export const STRINGS = {
  nav: {
    dashboard: "DASHBOARD",
    backtest: "BACKTEST",
    results: "RESULTS",
    compare: "COMPARE",
    news: "NEWS",
    settings: "SETTINGS",
  },
  actions: {
    deploy: "DEPLOY BACKTEST",
    terminate: "TERMINATE RUN",
    export: "EXPORT",
    load: "LOAD",
  },
  // ... all user-facing strings
};
```

---

## 10. Testing Strategy

### 10.1 Unit Tests (Vitest)

| Test Category | What | Tool | Target |
|---------------|------|------|--------|
| **Validation** | All 5 validators produce correct warnings/errors for edge cases | Vitest + jest-dom | 100% of validator rules |
| **Formatters** | formatMetric, formatPercent, formatPrice, formatDate | Vitest | All edge cases |
| **Store actions** | Zustand stores update correctly, toRequestPayload produces valid JSON | Vitest | All store actions |
| **Constants** | Default values match `config.py` PIPELINE_CONSTANTS | Vitest | All defaults |

### 10.2 Component Tests (Vitest + Testing Library)

| Test Category | What | Tool | Target |
|---------------|------|------|--------|
| **MetricCard** | Renders label, value, delta correctly | Testing Library | All formats |
| **ModelSelector** | Multi-select works, max 5 limit, category grouping | Testing Library | Selection state |
| **ParamSlider** | Updates store on change, clamps to range | Testing Library | Range enforcement |
| **ValidationAlert** | Shows warnings and errors with correct severity | Testing Library | Both severities |

### 10.3 Integration Tests (Playwright)

| Test Category | What | Tool | Target |
|---------------|------|------|--------|
| **Smoke test** | App boots, navigates to all 6 pages | Playwright | All pages render |
| **Config → Deploy** | Fill config form, click deploy, see progress | Playwright | Full flow |
| **Results display** | Results page shows charts, metrics, trade log | Playwright | All sections visible |
| **WebSocket reconnect** | Kill WS, verify fallback polling, restore WS | Playwright | Graceful degradation |

### 10.4 Performance Tests (Lighthouse)

| Metric | Budget | Tool |
|--------|--------|------|
| First Contentful Paint | < 1.5s | Lighthouse |
| Time to Interactive | < 3.0s | Lighthouse |
| Total Bundle Size | < 500KB gzipped | Vite build analyzer |
| Chart render (3000 candles) | < 200ms | Performance API |
| Trade log scroll (120K rows) | 60fps | Chrome DevTools |
| Memory usage (idle) | < 200MB | Chrome Task Manager |

---

## 11. Performance Budgets

| Metric | Budget | Current Baseline | Action if Exceeded |
|--------|--------|-----------------|-------------------|
| JS bundle (gzipped) | < 400KB | TBD | Code-split by route, lazy-load AG Grid and Monaco |
| CSS bundle (gzipped) | < 50KB | TBD | Purge unused Tailwind classes |
| First paint | < 1.5s | TBD | Preconnect fonts, inline critical CSS |
| Chart render (3000 candles) | < 200ms | TBD | TradingView Lightweight Charts is 35KB + Canvas |
| AG Grid render (10K visible rows) | < 100ms | TBD | Virtual scroll only renders visible rows |
| WebSocket message processing | < 5ms/msg | TBD | Imperative ref bypass, no React re-render |
| Zustand store update | < 1ms | TBD | Zustand is O(1) shallow compare |
| TanStack Query cache hit | < 5ms | TBD | In-memory cache, no network |
| Web Worker JSON parse (5MB) | < 500ms | TBD | Off main thread, no UI freeze |

---

## 12. Dependencies

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.22.0",
    "lightweight-charts": "^4.1.7",
    "@tanstack/react-query": "^5.17.0",
    "zustand": "^4.5.0",
    "axios": "^1.6.5",
    "ag-grid-react": "^31.0.0",
    "ag-grid-community": "^31.0.0",
    "@monaco-editor/react": "^4.6.0",
    "zod": "^3.22.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.0",
    "class-variance-authority": "^0.7.0",
    "lucide-react": "^0.312.0",
    "@radix-ui/react-accordion": "^1.1.2",
    "@radix-ui/react-dialog": "^1.0.5",
    "@radix-ui/react-dropdown-menu": "^2.0.6",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-slider": "^1.1.2",
    "@radix-ui/react-switch": "^1.0.3",
    "@radix-ui/react-tabs": "^1.0.4",
    "@radix-ui/react-toast": "^1.1.5",
    "@radix-ui/react-tooltip": "^1.0.7",
    "@radix-ui/react-progress": "^1.0.3",
    "@radix-ui/react-separator": "^1.0.3",
    "@radix-ui/react-scroll-area": "^1.0.5",
    "@radix-ui/react-collapsible": "^1.0.3"
  },
  "devDependencies": {
    "typescript": "^5.3.3",
    "@types/react": "^18.2.48",
    "@types/react-dom": "^18.2.18",
    "vite": "^5.0.12",
    "@vitejs/plugin-react": "^4.2.1",
    "tailwindcss": "^3.4.1",
    "postcss": "^8.4.33",
    "autoprefixer": "^10.4.17",
    "vitest": "^1.2.0",
    "@testing-library/react": "^14.1.2",
    "@testing-library/jest-dom": "^6.2.0",
    "@testing-library/user-event": "^14.5.2",
    "playwright": "^1.41.0",
    "eslint": "^8.56.0",
    "@typescript-eslint/eslint-plugin": "^6.19.0",
    "@typescript-eslint/parser": "^6.19.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "prettier": "^3.2.4",
    "prettier-plugin-tailwindcss": "^0.5.11"
  }
}
```

---

## 13. File Structure

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── tailwind.config.ts
├── postcss.config.js
├── .eslintrc.cjs
├── .prettierrc
├── public/
│   └── favicon.svg
│
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── design-tokens.ts
    │
    ├── api/
    │   ├── client.ts
    │   ├── queries.ts
    │   ├── schemas.ts
    │   └── websocket.ts
    │
    ├── stores/
    │   ├── useBacktestStore.ts
    │   ├── useJobStore.ts
    │   └── useSettingsStore.ts
    │
    ├── hooks/
    │   ├── useBacktestWebSocket.ts
    │   ├── useGPUStatus.ts
    │   └── useValidation.ts
    │
    ├── workers/
    │   └── dataParser.worker.ts
    │
    ├── lib/
    │   ├── constants.ts
    │   ├── validation.ts
    │   ├── formatters.ts
    │   └── strings.ts
    │
    ├── components/
    │   ├── layout/
    │   │   ├── AppShell.tsx
    │   │   ├── Sidebar.tsx
    │   │   ├── Header.tsx
    │   │   ├── StatusBar.tsx
    │   │   └── TerminalPanel.tsx
    │   │
    │   ├── shared/
    │   │   ├── MetricCard.tsx
    │   │   ├── ModelCard.tsx
    │   │   ├── StatusDot.tsx
    │   │   ├── ValidationAlert.tsx
    │   │   ├── ParamSlider.tsx
    │   │   ├── ParamToggle.tsx
    │   │   ├── ParamSelect.tsx
    │   │   ├── ExportBar.tsx
    │   │   ├── EmptyState.tsx
    │   │   ├── ZeroLeakageBadge.tsx
    │   │   └── ErrorBoundary.tsx
    │   │
    │   └── charts/
    │       ├── EquityCurveChart.tsx
    │       ├── WalkForwardOverlay.tsx
    │       ├── MonthlyReturnsChart.tsx
    │       ├── MonthlyHeatmap.tsx
    │       ├── ParamImportanceChart.tsx
    │       ├── OptimizationTraceChart.tsx
    │       ├── CorrelationMatrix.tsx
    │       └── SentimentGauge.tsx
    │
    ├── pages/
    │   ├── Dashboard/
    │   │   ├── DashboardPage.tsx
    │   │   ├── KpiGrid.tsx
    │   │   ├── RecentJobsTable.tsx
    │   │   └── ModelHeatmap.tsx
    │   │
    │   ├── Backtest/
    │   │   ├── BacktestPage.tsx
    │   │   ├── AssetSelector.tsx
    │   │   ├── ModelSelector.tsx
    │   │   ├── FeaturesPanel.tsx
    │   │   ├── ExecutionPanel.tsx
    │   │   ├── LabelsPanel.tsx
    │   │   ├── HpoPanel.tsx
    │   │   ├── RunSummary.tsx
    │   │   └── BacktestProgress.tsx
    │   │
    │   ├── Results/
    │   │   ├── ResultsPage.tsx
    │   │   ├── MetricsGrid.tsx
    │   │   ├── EquitySection.tsx
    │   │   ├── TradeLogTable.tsx
    │   │   ├── MonthlySection.tsx
    │   │   ├── HpoDiagnostics.tsx
    │   │   └── ConfigViewer.tsx
    │   │
    │   ├── Compare/
    │   │   ├── ComparePage.tsx
    │   │   ├── LeaderboardTable.tsx
    │   │   ├── EquityOverlay.tsx
    │   │   └── SignificanceMatrix.tsx
    │   │
    │   ├── News/
    │   │   ├── NewsPage.tsx
    │   │   ├── SentimentOverview.tsx
    │   │   ├── EventCalendar.tsx
    │   │   └── SentimentTimeline.tsx
    │   │
    │   └── Settings/
    │       ├── SettingsPage.tsx
    │       ├── GeneralSettings.tsx
    │       ├── GpuSettings.tsx
    │       ├── DataSourceManager.tsx
    │       ├── LicensePanel.tsx
    │       ├── PipelineConfig.tsx
    │       ├── VerboseToggle.tsx
    │       └── AboutPanel.tsx
    │
    └── styles/
        └── globals.css
```

**Total files**: ~65 source files + 10 config files = **~75 files**

---

## 14. Completion Criteria

### Phase Gate Checklist

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 1 | All 6 pages render without errors | Navigate to each page |
| 2 | Dashboard shows real data from FastAPI | Run a backtest via CLI, refresh dashboard |
| 3 | Full config → deploy → results flow works end-to-end | Click through entire flow |
| 4 | WebSocket progress streaming works during backtest | Deploy and watch progress bar |
| 5 | Charts render with real data (equity curve, monthly, HPO) | Run backtest, check results page |
| 6 | Trade log handles 120K rows smoothly (virtual scroll) | Load large result, scroll trade log |
| 7 | Model comparison shows leaderboard + equity overlay | Run comparison with 3+ models |
| 8 | All exports work (CSV, PNG, JSON) | Click each export button |
| 9 | Verbose mode toggles descriptions correctly | Switch mode, check controls |
| 10 | Terminal panel streams logs in real-time | Run backtest, watch terminal |
| 11 | All validation rules fire correctly | Trigger each validator edge case |
| 12 | Error handling: WS disconnect, API failure, backend crash | Kill processes, verify graceful handling |
| 13 | Keyboard navigation works on all pages | Tab through entire config form |
| 14 | Bundle size < 500KB gzipped | `npm run build`, check dist/ size |
| 15 | Lighthouse performance > 80 | Run Lighthouse audit |
| 16 | No console errors in production build | `npm run build && npm run preview` |

### Not in Scope for Sprint 8

These are deferred to later sprints:

| Feature | Sprint | Reason |
|---------|--------|--------|
| Electron wrapper | S9 | Separate codebase |
| License enforcement | S10 | Needs Paddle SDK |
| Auto-update | S11 | Needs Electron + GitHub Releases |
| Light theme | Post-launch | Dark-only for v1 |
| i18n translations | Post-launch | Strings extracted, no translations yet |
| Mobile responsive | Post-launch | Desktop-first for Electron |
| PDF tear sheet export | Post-launch | Nice-to-have |
| Paper trading mode | Post-launch | Needs live OANDA WebSocket |
| Portfolio optimization | Post-launch | Needs multi-model merge logic |
| Anomaly scrubbing filter | Post-launch | Needs pipeline modification |
