# QuantConnect UI/UX Design Pattern Analysis

> **Source**: quantconnect.com — Algorithm Lab IDE, Backtest Results, Strategies Explorer, Documentation  
> **Date**: 2026-04-18  
> **Purpose**: Reference design patterns for the Forex ML Pipeline's Electron + React frontend (Sprint 8+)

---

## 1. Layout Structure — IDE / Backtesting Interface

### Algorithm Lab (Terminal) Architecture
```
┌──────────────────────────────────────────────────────────────────┐
│  THIN HEADER (40px fixed) — Logo | Search | Nav Items | Sign In │
│  bg: #1B1B1F                                                     │
├────────────┬─────────────────────────────────────────────────────┤
│            │                                                      │
│  SIDEBAR   │              MAIN CONTENT AREA                       │
│  (64px     │   ┌─────────────────────────────────────────────┐   │
│  collapsed │   │  Project View / IDE / Backtest Results      │   │
│  ~200px    │   │                                              │   │
│  expanded) │   │  (fills remaining viewport)                  │   │
│            │   │                                              │   │
│  Icons:    │   └─────────────────────────────────────────────┘   │
│  • Home     │                                                     │
│  • Code     │                                                     │
│  • Org      │   ┌─────────────────────────────────────────────┐  │
│  • Learn    │   │  BOTTOM CONSOLE (collapsible)                │  │
│  • Data     │   │  Logs | Errors | Output                      │  │
│  • Strategy │   └─────────────────────────────────────────────┘  │
│  • Live     │                                                     │
│  • Support  │                                                     │
│            │                                                      │
├────────────┴─────────────────────────────────────────────────────┤
│  (No visible footer in IDE — maximizes workspace)                │
└──────────────────────────────────────────────────────────────────┘
```

### Key Layout Patterns
- **Full-viewport IDE**: `height: calc(-40px + 100vh)` — everything below the 40px header is workspace
- **Collapsible sidebar**: ~64px (icon-only) ↔ ~200px (icon + label), with chevron toggle button
- **Service label**: "Terminal" text next to logo in header (differentiates from "Web" on docs pages)
- **Zero-footer**: No footer in IDE — all pixels dedicated to workspace
- **Tab-based views**: Sidebar items swap entire main content via Knockout.js template binding
- **Split panel**: Code editor on left, backtest results on right when running

### For Our Pipeline
```css
/* QuantConnect layout reference */
.header-thin { height: 40px; background: #1B1B1F; position: fixed; }
.sidebar { width: 64px; /* collapsed */ transition: width 0.2s; }
.sidebar.expanded { width: 200px; }
.workspace { height: calc(100vh - 40px); overflow: hidden; }
```

---

## 2. Dashboard Design — Projects, Backtests, Results

### Projects Dashboard
- **Profile card** at top: user avatar + "Welcome" + organization badge + tier (FREE)
- **"New Project" CTA**: Prominent button to create algorithm
- **Template menu**: AI prompt textarea + categorized templates (Basic, Indicators, Machine Learning, etc.)
- **Project list**: Card grid with project name, language icon (Py/C#), last modified date
- **Active sessions**: Running coding environments shown in sidebar with "Requesting..." loading state

### Backtest Results Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│  RUNTIME STATISTICS BANNER (horizontal metrics strip)       │
│  Equity | Fees | Holdings | Net Profit | PSR | Return | ... │
│  (values update in real-time as backtest runs)               │
├─────────────────────────────────────────────────────────────┤
│  TABS: Overview | Orders | Trades | Insights | Logs | Code  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CHARTS (Overview tab):                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Strategy Equity  [equity curve + returns bars]      │    │
│  │  (resizable, draggable, zoom 1m/3m/1y/All)           │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │  Drawdown             │  │  Benchmark               │   │
│  └───────────────────────┘  └──────────────────────────┘   │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │  Exposure             │  │  Asset Sales Volume       │   │
│  └───────────────────────┘  └──────────────────────────┘   │
│                                                              │
│  CHART SELECT (sidebar toggle):                             │
│  ☑ Strategy Equity  ☑ Drawdown  ☑ Benchmark  ☐ Exposure   │
│  ☐ Portfolio Turnover  ☐ Performance                        │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  REPORT TAB: Download full PDF-style report                  │
│  SHARE: Make Public / Embed Code / Backtest URL              │
└─────────────────────────────────────────────────────────────┘
```

### Key Dashboard Patterns
- **Runtime statistics banner**: Horizontal strip of key metrics, updates live during backtest
- **Chart select sidebar**: Toggle individual charts on/off, reset zoom, resize
- **Linked zoom**: Adjusting zoom on one chart affects ALL charts (time-axis synced)
- **Resizable charts**: Drag bottom-right corner to resize; drag title bar to reorder
- **Tab structure**: Overview / Orders / Trades / Insights / Logs / Code / Report

---

## 3. Color Scheme — Dark Mode Palette & Accent Colors

### Primary Color Tokens
```css
/* === QUANTCONNECT DARK THEME (Chrome) === */

/* Navigation & Chrome */
--qc-navbar-bg:         #1B1B1F;   /* Near-black dark charcoal */
--qc-footer-bg:         #202024;   /* Footer dark */
--qc-sidebar-bg:        #242424;   /* Sidebar / CLI blocks */

/* Content Areas (AG Grid Alpine Dark) */
--ag-background-color:       #181D1F;  /* Main background */
--ag-header-background:      #222628;  /* Table headers */
--ag-odd-row-background:     #222628;  /* Alternating rows */
--ag-control-panel-bg:       #222628;  /* Side panels */
--ag-foreground-color:       #FFFFFF;  /* Primary text */
--ag-border-color:           #68686E;  /* Borders */
--ag-secondary-border:       rgba(88, 86, 82, 0.5);

/* Accent Colors */
--qc-brand-gold:        #F5AE29;   /* CTA buttons, highlights */
--qc-brand-blue:        #0072BC;   /* Links */
--qc-brand-green:       #35CD78;   /* Positive returns */
--qc-brand-red:         #E02525;   /* Negative returns, errors */
--qc-brand-purple:      #9240D0;   /* Premium features gradient */

/* AG Grid Alpine Active */
--ag-alpine-active:     #2196F3;   /* Selection, hover, focus */

/* Text Hierarchy */
--qc-text-primary:      #FFFFFF;   /* Headers, active items */
--qc-text-secondary:    #9D9D9D;   /* Nav links, inactive */
--qc-text-muted:        #A4A7B5;   /* Labels, captions */
--qc-text-tertiary:     #979BA3;   /* Subheadings */
--qc-text-dark:         #313131;   /* Dark-on-light text */

/* Status / Semantic */
--qc-success:           #5CB85C;   /* Green */
--qc-warning:           #F0AD4E;   /* Amber */
--qc-danger:            #D9534F;   /* Red */
--qc-info:              #5BC0DE;   /* Cyan */
```

### Homepage Marketing Palette (Light)
```css
--qc-home-bg:           #FFFFFF;
--qc-home-section-alt:  #F7F8FA;   /* Alternating sections */
--qc-home-section-dark: #202024;   /* Dark sections (LEAN, Build Alpha) */
--qc-home-text:         #313131;
--qc-home-subtitle:     #979BA3;
--qc-home-link:         #0072BC;
--qc-home-tab-active:   #A8AFBF;   /* Active tab background */
```

### Brand Gradient
```css
/* Premium / invest button gradient */
background: linear-gradient(143deg, rgb(43, 61, 199) 0%, rgb(146, 64, 208) 100%);
```

---

## 4. Data Tables — Backtest Results, Metrics, Trade Logs

### AG Grid Alpine Dark Theme
QuantConnect uses **AG Grid** with the **Alpine Dark** theme for all data tables:

```css
/* AG Grid Alpine Dark tokens */
--ag-grid-size: 6px;
--ag-row-height: calc(var(--ag-grid-size) * 7);  /* ~42px */
--ag-header-height: calc(var(--ag-grid-size) * 8); /* ~48px */
--ag-font-size: 13px;
--ag-font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
--ag-border-radius: 3px;
--ag-header-font-weight: 700;

/* Colors */
--ag-selected-row-background: rgba(33, 150, 243, 0.3);  /* Blue 30% */
--ag-row-hover-color: rgba(33, 150, 243, 0.1);           /* Blue 10% */
--ag-range-selection-bg: rgba(33, 150, 243, 0.2);        /* Blue 20% */

/* Skeleton loading effect */
--ag-row-loading-skeleton-color: rgba(202, 203, 204, 0.4);
```

### Orders Table Columns
| Column | Description |
|--------|-------------|
| Time | ET timestamp |
| Symbol | Asset ticker |
| Type | Market, Limit, Stop, etc. |
| Quantity | Shares/contracts |
| Fill Price | Execution price |
| Status | Filled, Partial, Cancelled |
| Tags | Custom order tags |

### Trades Table Pattern
- **Expandable rows**: Click trade to reveal underlying order fills
- **Pagination**: 10 rows per page with pagination controls
- **Download**: CSV and JSON export buttons per tab
- **Asset plot icon**: Small chart icon next to Symbol opens mini price chart

### Metrics Display
- **Runtime stats**: Horizontal banner of key-value pairs
- **Statistics table**: Grouped by category (Total Performance, Risk, Drawdown, etc.)
- **Color coding**: Green for positive returns, red for negative

---

## 5. Chart Integration — Equity Curves & Performance Charts

### Built-in Charts (Overview Tab)

| Chart | Type | Description |
|-------|------|-------------|
| **Strategy Equity** | Line + Bar | Equity curve (line) with periodic returns (bars) |
| **Capacity** | Line | Strategy capacity snapshots over time |
| **Drawdown** | Area (negative) | Peak-to-trough equity drawdown |
| **Benchmark** | Line | Benchmark closing price (default: SPY) |
| **Exposure** | Stacked Area | Long/short exposure ratios |
| **Asset Sales Volume** | Pie/Donut | Proportion of volume per security |
| **Portfolio Turnover** | Line | Turnover rate over time |
| **Portfolio Margin** | Stacked Area | Margin usage breakdown |
| **Performance** | Multi-line | CPU, RAM, securities count, data points/sec |
| **Asset Plot** | Candle + Annotations | Price with order events overlaid |

### Chart Interaction Patterns
- **Series toggle**: Click series name at top of chart to show/hide
- **Time zoom**: 1m / 3m / 1y / All buttons in top-right corner
- **Brush zoom**: Click and drag horizontally to select time range
- **Linked zoom**: Zoom affects ALL charts simultaneously
- **Resize**: Drag bottom-right corner
- **Reorder**: Drag chart by title bar
- **Scroll bar**: Horizontal scrollbar appears after zoom for panning
- **Refresh**: Reset button restores default zoom (drops streaming data)

### Asset Plot Order Event Annotations
| Event | Icon |
|-------|------|
| Submission | Gray circle |
| Update | Blue circle |
| Cancellation | Gray square |
| Fill (buy) | Green arrow |
| Fill (sell) | Red arrow |

### Chart Quotas (by tier)
| Tier | Max Series | Max Points/Series |
|------|-----------|-------------------|
| Free | 10 | 4,000 |
| Quant Researcher | 10 | 8,000 |
| Team | 25 | 16,000 |
| Trading Firm | 25 | 32,000 |
| Institution | 100 | 96,000 |

---

## 6. Code Editor Integration — Monaco/VS Code Embed

### Editor Setup
- **Editor**: Custom Monaco-based editor embedded in the Terminal
- **Languages**: Python (default) and C# with language selector
- **Theme name**: `chrome` (their default dark theme; also supports light)
- **Preloaded**: Editor JS loaded with `<link rel="preload">` for fast launch

### Editor Layout
```
┌──────────────────────────────────────────────┐
│ File Tabs: main.py | alpha.py | research.ipynb│
├──────────────────────────────────────────────┤
│                                               │
│  Monaco Editor (full panel)                   │
│  - Syntax highlighting                        │
│  - IntelliSense / autocomplete                │
│  - LEAN API documentation tooltips            │
│                                               │
├──────────────────────────────────────────────┤
│  Bottom Console (collapsible)                 │
│  [Build] [Run Backtest] [Deploy Live]         │
│  Console output / errors / logs               │
└──────────────────────────────────────────────┘
```

### Key Editor Features
- **Console panel**: Collapsible bottom panel with Build/Run/Deploy buttons
- **File browser**: Tree view in sidebar or dropdown
- **Cloud sessions**: Active coding environments tracked in sidebar
- **AI assistance**: Prompt-based project creation from templates
- **Code view in backtest results**: "Code" tab shows exact code used for a backtest

### For Our Pipeline (Monaco in Electron)
```typescript
// Monaco editor configuration reference
const editorConfig = {
  theme: 'vs-dark',           // or custom QC-like theme
  language: 'python',
  fontSize: 13,
  fontFamily: 'Consolas, "Courier New", monospace',
  minimap: { enabled: true },
  automaticLayout: true,       // Auto-resize
  scrollBeyondLastLine: false,
  renderLineHighlight: 'all',
};
```

---

## 7. Navigation — Sidebar, Top Bar, Breadcrumbs

### Top Navigation Bar (40px Thin)
```
┌─────────────────────────────────────────────────────────────────┐
│ [QC Logo] Terminal | [🔍 Search]  | Pricing Research Strategies │
│                                    | Data Docs Lab  [Sign In]   │
│ background: #1B1B1F                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Sidebar Navigation
```
┌──────────┐
│ 🏠 Projects│  (active: selected state with white bg)
│ 💻 Session │  (dynamic: shows active coding environments)
│ ⏳ Loading │  (dynamic: "Requesting..." for pending environments)
│ ───────── │
│ 🏢 Org     │
│ 📚 Learning│
│ 📊 Data    │
│ ♟ Strategy │
│ 📡 Live    │
│ 🎧 Support │
│            │
│ ◀ Collapse│  (toggle button at bottom)
└──────────┘
```

### Sidebar Design Tokens
- **Icon size**: ~22px (custom QC icon font: `qci-home`, `qci-environment`, etc.)
- **Active state**: White background, darker text, right caret indicator
- **Hover**: Light highlight background
- **Collapsed**: Icon-only mode at ~64px width
- **Expanded**: Icon + text at ~200px width

### Search
- **Global search**: Magnifying glass icon expands to full search bar
- **Keyboard shortcut**: Focus on click, type to search across projects/docs
- **Dropdown results**: Below search with ↑↓ navigation + Enter to select + Esc to close

### Mobile Navigation
- **Hamburger menu**: Three-line toggle in thin header
- **Full-screen overlay**: Slides in from left with categorized links
- **Sign-in section**: Separate card at bottom of mobile nav

---

## 8. Card-Based Layouts — Strategy & Algorithm Cards

### Strategy Cards (Strategy Explorer)
```
┌─────────────────────────────────────────┐
│  Strategy Name                [Clone]    │
│  Author: username            ⭐ 234      │
│                                          │
│  ┌──────────────────────────────────┐   │
│  │  Mini equity curve sparkline      │   │
│  │  (small preview chart)            │   │
│  └──────────────────────────────────┘   │
│                                          │
│  Return: +23.4%   Sharpe: 1.82          │
│  Drawdown: -8.2%   Trades: 1,247        │
│                                          │
│  Tags: [Momentum] [Equity] [Daily]       │
└─────────────────────────────────────────┘
```

### Project Cards
- **Grid layout**: Cards in responsive grid
- **Info displayed**: Project name, language badge (Py/C#), last modified, backtest count
- **Quick actions**: Clone, Delete, Open in IDE
- **Status indicators**: Running cloud sessions shown as active

### Template Cards (New Project Dialog)
```
┌──────────────────────────────────────────┐
│  Describe Your Strategy or Choose...      │
│  ┌────────────────────────────────────┐  │
│  │  AI Prompt textarea                 │  │
│  │  "Each week scan the QQQ universe..."│  │
│  └────────────────────────────────────┘  │
│                                           │
│  BASIC                                    │
│  • Use Default Template                   │
│                                           │
│  INDICATORS                               │
│  • Bollinger Bands  • EMA Cross          │
│  • MACD             • RSI Mean Reversion │
│                                           │
│  MACHINE LEARNING                         │
│  • Random Forest   • Deep Learning       │
│  • SVM              • Neural Network     │
│                                           │
│  TODAY'S IDEAS ✨                          │
│  • [AI-generated strategy suggestions]   │
└──────────────────────────────────────────┘
```

### Social Proof / KPI Cards
```
┌──────────────────────────────────────┐
│  AWARD WINNING QUANT ANALYTICS       │
│                                      │
│  488K      500K+      $45B     +7%   │
│  quant     backtests  volume   over   │
│  community per month  per month market│
└──────────────────────────────────────┘
```

---

## 9. Progress Indicators — Long-Running Backtests

### Backtest Progress Patterns

1. **Loading Spinner**
   - SVG-based spinner animation (86x86px)
   - Text: "Loading Algorithm Lab v3.0..."
   - Full-screen overlay during IDE initialization

2. **Streaming Results**
   - Backtest results page updates IN REAL-TIME as algorithm executes
   - Runtime statistics banner refreshes continuously
   - Charts render incrementally as data points arrive
   - Page can be closed/refreshed without interrupting backtest (server-side processing)

3. **Cloud Session Loading**
   - Sidebar shows "Requesting..." text with spinning indicator
   - Individual session items track loading state per project

4. **Report Generation**
   - "Report is being generated" message
   - May take a minute — user repeats download click to check
   - Background processing indicator

5. **Skeleton Loading** (AG Grid)
   ```css
   --ag-row-loading-skeleton-color: rgba(202, 203, 204, 0.4);
   /* Shimmer effect on table rows while data loads */
   ```

### Progress Design for Our Pipeline
```typescript
// Streaming backtest progress reference
interface BacktestProgress {
  status: 'queued' | 'running' | 'completed' | 'error';
  percentage: number;           // 0-100
  currentWindow: number;        // Walk-forward window index
  totalWindows: number;
  elapsedMs: number;
  etaMs: number;
  liveMetrics: {                // Updated every tick
    equity: number;
    returnPct: number;
    sharpe: number;
    tradesSoFar: number;
  };
}
```

---

## 10. Professional Quant Aesthetic — "Institutional" Feel

### What Makes QuantConnect Feel Professional

#### Typography
- **Inter** (primary UI font — clean, geometric, modern)
- **Roboto** (secondary — monospace-like clarity)
- **Open Sans** (documentation body text)
- **Droid Sans** (legacy — still in CSS stack)
- **System font stack**: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
- **Monospace**: `Consolas, "Courier New", monospace` for code blocks
- **Font weight**: 500 for headings, 700 for selected/active, 400 for body

#### Visual Hierarchy
1. **Minimal decoration**: No gradients on content areas (flat dark surfaces)
2. **Data density**: Packed metrics without clutter — 13px font, 42px row height
3. **Muted color palette**: Dark grays and subtle borders, NOT flashy colors
4. **Gold accent only for CTAs**: `#F5AE29` used sparingly (Sign Up button, highlights)
5. **Blue for interactions**: `#2196F3` for selection, hover, links, active states
6. **Green/Red only for data**: Positive returns = green, negative = red — no decorative use

#### Institutional Cues
- **"488K quant community"**: Social proof with specific numbers
- **"AS SEEN ON"**: FT, WSJ, Business Insider logos
- **Tier system**: Free → Quant Researcher → Team → Trading Firm → Institution
- **Organization concept**: Multi-user workspaces with permission controls
- **Volume metrics**: "$45B volume per month", "500K+ backtests/month"
- **LEAN Engine**: Open-source backbone mentioned prominently
- **Terminal Link**: Bloomberg integration mentioned as institutional feature

#### "No-Nonsense" Design Rules
1. **No illustrations in the IDE**: Pure functional workspace
2. **No animations in data tables**: Instant rendering, no fade-ins
3. **Dense information**: Multiple metrics visible at once
4. **Monospace for numbers**: Alignment of financial figures
5. **Minimal whitespace in tables**: Compact but readable
6. **Consistent icon set**: Custom QC icon font (qci-home, qci-environment, etc.)
7. **Server-side processing**: "Close the window without interrupting" — professional UX

#### Footer
```css
/* Institutional footer */
border-top: 1px solid #000;
background: #292929;
color: #BBB;
font-size: 11px;
padding: 20px;
/* Subtle inner shadow for depth */
box-shadow: inset 0 1px 0 rgba(255,255,255,0.3);
```

---

## Design Tokens Summary — For Implementation

### CSS Custom Properties Reference
```css
:root {
  /* === LAYOUT === */
  --header-height: 40px;
  --sidebar-width-collapsed: 64px;
  --sidebar-width-expanded: 200px;
  --border-radius: 3px;
  --workspace-height: calc(100vh - var(--header-height));

  /* === COLORS — DARK MODE === */
  --bg-primary: #181D1F;
  --bg-secondary: #222628;
  --bg-nav: #1B1B1F;
  --bg-footer: #292929;
  --bg-sidebar: #242424;

  --text-primary: #FFFFFF;
  --text-secondary: #9D9D9D;
  --text-muted: #A4A7B5;
  --text-dark: #313131;

  --border-primary: #68686E;
  --border-secondary: rgba(88, 86, 82, 0.5);

  --accent-blue: #2196F3;
  --accent-gold: #F5AE29;
  --accent-link: #0072BC;

  --success: #35CD78;
  --danger: #E02525;
  --warning: #F0AD4E;
  --info: #5BC0DE;

  /* === AG GRID === */
  --ag-grid-size: 6px;
  --ag-row-height: 42px;
  --ag-header-height: 48px;
  --ag-font-size: 13px;
  --ag-selected-row-bg: rgba(33, 150, 243, 0.3);
  --ag-hover-row-bg: rgba(33, 150, 243, 0.1);

  /* === TYPOGRAPHY === */
  --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: Consolas, 'Courier New', monospace;
  --font-size-sm: 11px;
  --font-size-base: 13px;
  --font-size-md: 14px;
  --font-size-lg: 16px;
  --font-size-xl: 20px;

  /* === SPACING === */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
}
```

---

## Component Mapping — QC → Our Pipeline

| QuantConnect Component | Our Pipeline Equivalent | Implementation |
|------------------------|------------------------|----------------|
| Algorithm Lab sidebar | Pipeline sidebar | React + CSS variables |
| AG Grid data tables | Backtest results tables | AG Grid React (AG Charts) |
| Equity curve charts | Walk-forward equity plots | Plotly.js (already in use) |
| Monaco code editor | Strategy config editor | @monaco-editor/react |
| Runtime stats banner | Pipeline metrics strip | React + CSS Grid |
| Strategy cards | Model comparison cards | React card components |
| Template dialog | Pipeline config wizard | React modal + forms |
| Progress spinner | Backtest progress | React + streaming API |
| Console panel | Log viewer | React + virtualized list |
| Dark theme toggle | Theme system | CSS custom properties |

---

## Key Takeaways for Sprint 8 (React Frontend)

1. **Start with dark mode**: QC defaults to dark — institutional quants prefer it
2. **40px header**: Thin, dense, maximizes workspace — not a marketing header
3. **AG Grid Alpine Dark**: The de facto standard for quant data tables
4. **Inter font**: Clean, modern, professional — use as primary
5. **Gold accent (#F5AE29)**: Use sparingly for CTAs only
6. **Blue interaction (#2196F3)**: Selection, hover, focus states
7. **Streaming results**: Backtest page updates live — this is expected behavior
8. **Data density**: 13px font, 42px rows — compact but readable
9. **Zero decoration**: No gradients, no illustrations in workspace — pure function
10. **Server-side processing narrative**: "Close the window, it keeps running" — essential UX
