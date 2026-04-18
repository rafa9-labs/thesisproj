# TradingView UI/UX Design Analysis

> Extracted programmatically from the live TradingView chart application + official TradingView Charting Library documentation.
> Purpose: Reference design document for Sprint 8 (React Frontend) + Sprint 9 (Electron Desktop Shell).

---

## 1. Layout Structure

### Overall Viewport Composition (Desktop 1920x1080)

```
+--------------------------------------------------------------------+
|  TV Logo | Symbol  Interval  Chart-Type  Indicators |  Profile     |  <- Header Bar (42px)
+-----+------------------------------------------+-------------------+
|     |                                          |                   |
| D   |                                          |   Watchlist       |
| r   |                                          |                   |
| a   |           MAIN CHART                     |   Details         |
| w   |           (Canvas)                       |   Panel           |
| i   |                                          |                   |
| n   |                                          |   (300px)         |
| g   |                                          |                   |
|     |  +-Legend------------------------------+ |                   |
| T   |  | EURUSD  O:1.1779 H:1.1849           | |                   |
| o   |  | 1D      L:1.1760 C:1.1764           | |                   |
| o   |  +-------------------------------------+ |                   |
| l   |                                          |                   |
| s   +------------------------------------------+                   |
|     |  Price Axis (64px) | Time Axis (28px)   |                   |
|52px |                                          |                   |
+-----+------------------------------------------+-------------------+
|  Zoom | Date Range | Screenshot | ...                              |  <- Bottom Toolbar (39px)
+--------------------------------------------------------------------+
```

### Exact Dimensions (1440x900 viewport)

| Zone | Position | Size | Notes |
|------|----------|------|-------|
| **Header Bar** | top:0, left:0 | full-width x 42px | Symbol, interval, chart type, indicators |
| **Left Drawing Toolbar** | top:42, left:0 | 52px x ~820px | Icon buttons, 38px each, scrollable |
| **Center Chart Area** | top:42, left:56 | 1034 x 819 | Main canvas + overlays |
| **Price Axis** | right of chart | 64px wide | Y-axis price labels |
| **Time Axis** | bottom of chart | 28px tall | X-axis date labels |
| **Right Widget Panel** | top:42, right edge | 346px (300px content) | Watchlist + Details tabs |
| **Bottom Chart Toolbar** | below chart | full-width x 39px | Zoom, date range, screenshot |

### Canvas Architecture

TradingView uses **multiple stacked canvases** for the chart area:

| Canvas | Size | Purpose |
|--------|------|---------|
| Main Chart | 970 x 791 | Candlesticks, indicators, drawings |
| Overlay | 970 x 791 | Crosshair, selection, hover states |
| Price Axis | 64 x 791 | Y-axis labels |
| Price Axis Overlay | 64 x 791 | Crosshair price line |
| Time Axis | 970 x 28 | X-axis date labels |
| Time Axis Overlay | 970 x 28 | Crosshair time line |
| Corner | 64 x 28 | Crosshair intersection point |
| Sub-chart (indicators) | 268 x 142 | RSI/MACD sub-panels |
| Sub-chart overlay | 268 x 142 | Sub-chart crosshair |

---

## 2. Color Palette

### 2.1 Official 7-Color Theme System (19 Shades Each)

TradingView's charting library uses a **7-color palette with 19 shades** per color. This is the backbone of their entire design system.

#### Color 1: Blue (Primary / Brand)
```
Shade 500 (base): #2962ff
Full range: #eaefff -> #040a1a (lightest -> darkest)
Key shades:
  - 100: #bbd9fb  (hover backgrounds)
  - 300: #5b9cf6  (secondary buttons)
  - 400: #3179f5  (active elements)
  - 500: #2962ff  (primary action color)
  - 600: #1e53e5  (pressed state)
  - 800: #143eb2  (deep accent)
  - 900: #0c3299  (darkest accent)
```

#### Color 2: Grey (UI Structure)
```
Shade 500 (base): #787b86
Full range: #f2f2f3 -> #0c0c0d
Key shades:
  - 100: #f2f2f2  (light background)
  - 200: #dbdbdb  (borders, dividers)
  - 300: #b8b8b8  (disabled text)
  - 500: #787b86  (secondary text) <-- MOST IMPORTANT
  - 700: #4a4a4a  (dark mode borders)
  - 850: #1f1f1f  (dark background)
  - 900: #0f0f0f  (deepest dark bg)
```

#### Color 3: Red (Bearish / Sell / Error)
```
Shade 500 (base): #f23645
Full range: #feebec -> #180507
Key shades:
  - 400: #f7525f  (sell hover)
  - 500: #f23645  (sell/bearish primary) <-- TRADING RED
  - 600: #cc2f3c  (pressed)
  - 700: #b22833  (dark mode sell)
```

#### Color 4: Green (Bullish / Buy / Success)
```
Shade 500 (base): #089981
Full range: #e6f5f2 -> #010f0d
Key shades:
  - 400: #39ad9a  (buy hover)
  - 500: #089981  (buy/bullish primary) <-- TRADINGVIEW GREEN
  - 600: #078a74  (pressed)
  - 700: #067a67  (dark mode buy)
```

#### Color 5: Orange (Warnings / Highlights)
```
Shade 500 (base): #ff9800
Range: #fff5e6 -> #1a0f00
```

#### Color 6: Purple (Premium / Special)
```
Shade 500 (base): #9c27b0
Range: #f5e9f7 -> #100412
```

#### Color 7: Yellow (Caution / Attention)
```
Shade 500 (base): #ffeb3b
Range: #fffdeb -> #1a1806
```

### 2.2 Dark Mode Palette (Computed from Live App)

```css
/* === DARK THEME EXACT VALUES === */

/* Background layers (from outermost to innermost) */
--bg-page:               #131722;     /* rgb(19, 23, 34) -- Main page background */
--bg-overlay:            oklch(0.2348 0.0041 264.49);  /* ~= #1c2131 -- Panel overlay */
--bg-panel:              #011127;     /* rgb(1, 11, 36) -- Deep dark panels */
--bg-toolbar:            #1c2131;     /* rgb(28, 33, 49) -- Toolbar background */
--bg-card:               #252c3f;     /* rgb(37, 44, 63) -- Card/elevated surfaces */
--bg-card-hover:         rgba(37, 44, 63, 0.5);  /* Hover state */
--bg-input:              #394259;     /* rgb(57, 66, 89) -- Input/button backgrounds */
--bg-accent:             #0d8ed6;     /* rgb(13, 142, 214) -- Active accent */

/* Text layers */
--text-primary:          #ffffff;     /* rgb(255, 255, 255) -- Primary text */
--text-secondary:        #edeff5;     /* rgb(237, 239, 245) -- Secondary text */
--text-tertiary:         #80899f;     /* rgb(128, 137, 159) -- Muted/hint text */
--text-legend:           rgba(255, 255, 255, 0.5);  /* Legend overlay text */

/* Trading colors */
--color-bullish:         #089981;     /* TradingView Green 500 */
--color-bearish:         #f23645;     /* Ripe Red 500 */
--color-bullish-bright:  #35cd78;     /* rgb(53, 205, 120) -- Bright green for indicators */
--color-signal:          #00f0ff;     /* rgb(0, 240, 255) -- Cyan for signals/alerts */

/* Borders */
--border-primary:        oklch(0.269 0 0);  /* ~= #2a2a2e -- Primary borders */
--border-secondary:      #394259;     /* rgb(57, 66, 89) -- Secondary borders */
--border-tertiary:       #80899f;     /* rgb(128, 137, 159) -- Active/selected borders */

/* Chart grid */
--grid-line:             rgba(255, 255, 255, 0.06);  /* Subtle grid lines */
--crosshair:             rgba(255, 255, 255, 0.3);   /* Crosshair lines */
```

### 2.3 Light Mode Palette (Derived)

```css
/* === LIGHT THEME === */
--bg-page:               #ffffff;
--bg-toolbar:            #f2f2f2;     /* cold-gray-100 */
--bg-card:               #ffffff;
--bg-card-hover:         #f9f9f9;     /* cold-gray-50 */
--bg-input:              #f2f2f2;

--text-primary:          #0f0f0f;     /* cold-gray-900 */
--text-secondary:        #4a4a4a;     /* cold-gray-700 */
--text-tertiary:         #787b86;     /* Grey 500 */

--border-primary:        #dbdbdb;     /* cold-gray-200 */
--border-secondary:      #ebebeb;     /* cold-gray-150 */
```

---

## 3. Typography

### Font Stack

| Usage | Font Family | Source |
|-------|-------------|--------|
| **Header / Navigation** | `Aeonik, "Aeonik Fallback"` | Custom brand font |
| **Chart Labels / Legend** | `-apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif` | System font stack |
| **Marketing Pages** | `ui-sans-serif, system-ui, sans-serif` | Generic system font |

### Font Sizes (From Live App)

| Size | Usage | Example |
|------|-------|---------|
| **16px** | Symbol name, interval selector, nav items | "Euro / U.S. Dollar", "1D" |
| **14px** | Chart legend (OHLCV), indicator values, toolbar items | "O 1.1779 H 1.1849..." |
| **13px** | Secondary buttons, market status, watchlist items | "Market closed", flag button |
| **12px** | Small labels, tertiary actions | Deny/Accept buttons |
| **11px** | Price axis labels, time axis labels | Y-axis prices |
| **10px** | Tiny annotations, sub-chart labels | Indicator sub-panel |

### Font Weights

| Weight | Usage |
|--------|-------|
| **400 (Regular)** | Almost everything -- body text, labels, buttons, prices |
| **700 (Bold)** | Rarely used; active tab indicators only |

### Line Heights

- Chart legend: `normal` (auto)
- Toolbar buttons: `18px` (ui-lib-typography-line-height)

### Key Typography Rules

1. **All-caps never used** -- labels are sentence case
2. **No letter-spacing adjustments** -- relies on font defaults
3. **Price values use same font as UI** -- no monospace for prices (!)
4. **Single weight dominance** -- regular (400) for 95% of text
5. **Color = hierarchy** -- opacity and gray shades create visual hierarchy, not font size

---

## 4. Chart Design

### Candlestick Styling

Based on TradingView's default overrides and extracted values:

```css
/* Candlestick (from overrides API defaults) */
candlestick.upColor:          #089981;    /* Bullish body */
candlestick.downColor:        #f23645;    /* Bearish body */
candlestick.borderUpColor:    #089981;    /* Bullish border */
candlestick.borderDownColor:  #f23645;    /* Bearish border */
candlestick.wickUpColor:      #089981;    /* Bullish wick */
candlestick.wickDownColor:    #f23645;    /* Bearish wick */

/* Bar chart alternative */
bar.upColor:                  #089981;
bar.downColor:                #f23645;
```

### Chart Grid

```css
grid.horizontal:   rgba(255, 255, 255, 0.06);  /* Very subtle */
grid.vertical:     rgba(255, 255, 255, 0.06);  /* Very subtle */
/* Light mode: rgba(0, 0, 0, 0.06) */
```

### Crosshair Behavior

- **Style**: Dotted/dashed horizontal and vertical lines
- **Color**: `rgba(255, 255, 255, 0.3)` (dark mode), semi-transparent
- **Price label**: Appears on the right price axis (floating pill)
- **Time label**: Appears on the bottom time axis (floating pill)
- **Crosshair mode**: "Cross" (default) or "Magnet" (snaps to OHLC)

### Chart Legend (OHLCV Overlay)

```
Position: Inside chart, top-left
Content:  Symbol  Interval  Source | O 1.1779  H 1.1849  L 1.1760  C 1.1764  (down) -0.00188 (-0.16%)
Font:     14px, weight 400, rgba(255,255,255,0.5)
Layout:   Horizontal, no-wrap (with wrappable fallback)
Behavior: Click to expand/collapse indicator visibility
```

### Sub-Charts (Indicators)

- **Height**: ~142px (for RSI, MACD type indicators)
- **Separator**: 1px line, subtle border color
- **Own legend**: Same style as main chart legend
- **Own price axis**: 64px right-side labels

### Price Axis

- **Width**: 64px
- **Background**: Same as chart area (transparent)
- **Text color**: `#787b86` (Grey 500)
- **Font size**: 11px
- **Ticks**: Decimal precision follows instrument (5 decimals for EURUSD)
- **Auto-scaling**: Yes, with zoom-dependent label density

### Time Axis

- **Height**: 28px
- **Format**: Adaptive (time for intraday, dates for daily+)
- **Font size**: 11px
- **Text color**: `#787b86`

---

## 5. Navigation Patterns

### 5.1 Top Header Bar

```
+------------------------------------------------------------------+
| [TV Logo] | [EURUSD v] [1D v] [Chart v] [Indicators] | ... |[U] |
|  180x40    |  Symbol   Interval  Chart    Indicators   |     |Menu|
+------------------------------------------------------------------+
```

- **Symbol selector**: Text button, 16px, clickable -> opens symbol search modal
- **Interval selector**: Text "1D" with dropdown arrow -> opens interval picker
- **Chart type**: Icon button -> opens chart type picker (candle, bar, line, etc.)
- **Indicators**: Icon -> opens indicator dialog
- **All items**: Transparent background, no border, 14-16px text

### 5.2 Left Drawing Toolbar (Vertical)

```
+------+
|  +   | <- Crosshair cursor
|  /   | <- Trendline (with dropdown arrow)
|  ::  | <- Fibonacci retracement
| <>   | <- Pattern tools
| /_\  | <- Position tools
|  O   | <- Shape tools
|  T   | <- Text/annotation
|  |   | <- (More tools...)
|      |
| ---  | <- Separator
| (M)  | <- Magnet mode (toggle)
| (L)  | <- Lock drawings (toggle)
| (E)  | <- Show/hide drawings
| (X)  | <- Delete drawings
+------+
```

- **Width**: 52px
- **Button size**: 28x28 icon, 38px clickable area
- **Dropdown arrow**: 11px wide, right side of button
- **Active state**: Filled icon background
- **Hover**: Subtle background highlight

### 5.3 Right Widget Panel

- **Tabs**: Watchlist, Details (collapsible)
- **Width**: 346px outer, 300px content
- **Background**: White (light), Dark panel color (dark)
- **Scrollable**: Yes, with custom thin scrollbar
- **Drag-to-resize**: Yes, left edge is a drag handle

### 5.4 Bottom Chart Toolbar

```
+-------------------------------------------------------+
| [-] [+] | [2024 v] [Jan v] [1 v] | [...] | [cam]     |
|  Zoom    |  Date Range Selector     | More  | Screenshot|
+-------------------------------------------------------+
```

- **Height**: 39px
- **Zoom**: Plus/minus buttons
- **Date range**: Dropdown selectors for year, month, day
- **Screenshot**: Camera icon

### 5.5 Workspaces / Layout Tabs

- Multiple chart layouts via top-right menu
- Layout save/load system
- Tab-based workspace switching (premium feature)

---

## 6. Data Density Patterns

### How TradingView Handles Dense Information

1. **Opacity-based hierarchy**: Legend text at 50% opacity (`rgba(255,255,255,0.5)`). Active/hover items brighter.

2. **Collapsible panels**: Right panel widgets collapse/expand. Watchlist items hidden until scrolled.

3. **Hover-to-reveal**: Drawing toolbar dropdowns only show on hover/click. Crosshair details appear on hover.

4. **Inline legends**: OHLCV data overlaid on the chart itself (no separate panel). Click to toggle visibility.

5. **Thin scrollbars**: Custom scrollbars, ~4-6px wide, auto-hide on inactivity.

6. **Adaptive label density**: Price axis and time axis auto-adjust label count based on zoom level.

7. **Tooltips everywhere**: Every button has `title` or `aria-label` for tooltip on hover.

8. **Color coding**: Green/red for buy/sell, grey for neutral, blue for interactive.

9. **Compact ticker rows**: In watchlist, each row is ~30-36px tall with symbol + price + change.

10. **Sub-chart stacking**: Indicators stack vertically below the main chart, each ~142px.

---

## 7. Component Patterns

### Buttons

```css
/* Default toolbar button */
background: transparent;
color: #edeff5;
font-size: 16px;
font-weight: 400;
font-family: Aeonik, "Aeonik Fallback";
border: none;
border-radius: 0px;
padding: 0px;
cursor: pointer;
height: 32px;

/* Active/selected state */
background: #1c2131;  /* rgb(28, 33, 49) */
color: #ffffff;

/* Hover state */
background: rgba(57, 66, 89, 0.5);
```

### Segmented Controls (Toggle Groups)

```css
/* Active segment */
background: #1c2131;     /* Dark surface */
color: #ffffff;
font-size: 16px;
padding: 6px 20px;
height: 36px;

/* Inactive segment */
background: #394259;     /* rgb(57, 66, 89) */
color: #ffffff;
```

### Pills / Badge Buttons

```css
/* Market status pill */
background: rgba(255, 255, 255, 0.5);  /* Semi-transparent */
color: rgba(255, 255, 255, 0.5);
border-radius: 9px;
font-size: 13px;
height: 18px;
width: 40px;
```

### Card/Panels

```css
/* Right panel widget */
background: #ffffff;  /* Light mode */
border-radius: 0px;   /* Sharp corners */
box-shadow: none;
overflow: hidden;
```

### Inputs (Symbol Search)

```css
/* Search input */
font-size: 16px;
background: #252c3f;
border: 1px solid #394259;
border-radius: 4px;
padding: 8px 12px;
color: #ffffff;
```

### Dividers/Separators

```css
/* Toolbar dividers */
background: #394259;  /* Grey border color */
width: 1px;
height: 24px;         /* Shorter than toolbar height */
margin: 0 4px;
```

### Dropdowns / Popups

```css
/* Popup menu (from official docs) */
--tv-color-popup-background: #252c3f;
--tv-color-popup-element-text: #edeff5;
--tv-color-popup-element-text-hover: #ffffff;
--tv-color-popup-element-background-hover: #394259;
--tv-color-popup-element-divider-background: #394259;
border-radius: 4px;
box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
```

### Toggle Buttons (Drawing Toolbar)

```css
/* Active toggle (e.g., Magnet mode) */
--tv-color-toolbar-toggle-button-background-active: #2962ff;  /* Blue 500 */
--tv-color-item-active-text: #2962ff;                         /* Blue 500 */
```

---

## 8. Dark Mode Specifics

### The Exact Dark Theme Shade Palette

TradingView's dark mode is **not pure black** -- it uses a **blue-tinted dark navy** palette:

| Role | Hex | RGB | Usage |
|------|-----|-----|-------|
| **Page bg** | `#131722` | 19, 23, 34 | Deepest background |
| **Panel bg** | `#1c2131` | 28, 33, 49 | Toolbar/panel surfaces |
| **Card bg** | `#252c3f` | 37, 44, 63 | Elevated cards, inputs |
| **Card hover** | `#394259` | 57, 66, 89 | Hover state bg |
| **Border** | `#2e2e2e` | -- | Subtle borders |
| **Chart bg** | `#ffffff` | 255, 255, 255 | Chart canvas (white by default) |

### Dark Mode Key Insight

**The chart canvas itself defaults to white** even in dark mode. The dark theme only affects the chrome/toolbar around the chart. Users can change the chart background separately in chart settings.

### Chart Background in Dark Mode (User Setting)

When users set the chart to dark:
```css
chart.bgColor:      #131722;  /* Matches page background */
grid.lineColor:     rgba(255, 255, 255, 0.06);
crosshair.color:    rgba(255, 255, 255, 0.3);
```

### Dark Mode Text Hierarchy

| Level | Color | Opacity | Usage |
|-------|-------|---------|-------|
| 1 | `#ffffff` | 100% | Primary text, active labels |
| 2 | `#edeff5` | 97% | Navigation, buttons |
| 3 | `rgba(255,255,255,0.5)` | 50% | Legend text, muted labels |
| 4 | `#80899f` | 100% | Tertiary text, hints |

---

## 9. Micro-Interactions

### Loading States

- **Chart loading**: Skeleton placeholder with subtle shimmer
- **Data loading**: Spinning circle loader (10px) in toolbar
- **Layout save**: `--tv-color-toolbar-save-layout-loader` animation

### Hover Effects

| Element | Effect |
|---------|--------|
| Toolbar button | Background: `rgba(57, 66, 89, 0.5)` |
| Toolbar button (active) | Background: `#1c2131` |
| Popup menu item | Background: `#394259` |
| Watchlist row | Subtle highlight, price updates flash green/red |
| Chart candle | Crosshair appears, OHLCV updates in legend |
| Drawing tool | Icon brightens, dropdown arrow appears |

### Animations

- **Chart panning**: Smooth 60fps canvas re-rendering
- **Price updates**: Flash animation (green pulse up, red pulse down)
- **Panel resize**: Smooth CSS transition
- **Modal open/close**: Fade + slight scale
- **Toast notifications**: Slide-in from bottom-right

### Cursor Changes

- **Crosshair mode**: Crosshair cursor (default on chart)
- **Drawing mode**: Custom cursor per tool (pencil, line, etc.)
- **Resize handles**: `nwse-resize` / `nesw-resize` cursors
- **Pan/scroll**: Grab/grabbing cursors

---

## 10. Mobile Responsiveness

### Breakpoint Strategy

TradingView uses adaptive breakpoints with CSS custom properties:

```css
/* Responsive spacing vars */
--v-rhythm-header-1-space-phone:    56px;
--v-rhythm-header-1-space-tablet:   80px;
--v-rhythm-header-1-space-laptop:   100px;
--v-rhythm-header-1-space-desktop:  120px;
```

### Desktop -> Mobile Adaptations

| Feature | Desktop | Mobile |
|---------|---------|--------|
| Left toolbar | 52px vertical | Hidden, hamburger menu |
| Right panel | 300px sidebar | Full-screen overlay |
| Header | Full toolbar | Condensed, scroll |
| Chart interactions | Mouse crosshair | Touch gestures |
| Drawing tools | Full panel | Bottom sheet |
| Watchlist | Side panel | Separate tab |
| Keyboard shortcuts | Full support | None |

### Size Scaling System

TradingView uses a `--ui-lib-size` variable (1-6) for responsive component sizing:

```css
/* Button heights by size tier */
xsmall:  28px;
small:   34px;
medium:  40px;  /* Default */
large:   48px;
xlarge:  56px;
```

---

## 11. Implementation Reference for Our Pipeline

### Recommended Color Tokens for Our App

```typescript
// design_tokens.ts

export const colors = {
  // Background layers (dark mode)
  bg: {
    page:      '#131722',
    panel:     '#1c2131',
    card:      '#252c3f',
    elevated:  '#394259',
    input:     '#394259',
    hover:     'rgba(37, 44, 63, 0.5)',
  },

  // Text
  text: {
    primary:   '#ffffff',
    secondary: '#edeff5',
    tertiary:  '#80899f',
    legend:    'rgba(255, 255, 255, 0.5)',
    disabled:  '#787b86',
  },

  // Trading
  trading: {
    bullish:   '#089981',
    bearish:   '#f23645',
    signal:    '#00f0ff',
    neutral:   '#787b86',
  },

  // Brand / Accent
  accent: {
    primary:   '#2962ff',   // Blue 500
    hover:     '#3179f5',   // Blue 400
    active:    '#1e53e5',   // Blue 600
    warning:   '#ff9800',   // Orange 500
    premium:   '#9c27b0',   // Purple 500
  },

  // Borders
  border: {
    primary:   '#2e2e2e',
    secondary: '#394259',
    tertiary:  '#80899f',
  },
} as const;

export const typography = {
  fontFamily: {
    brand: 'Aeonik, "Aeonik Fallback", sans-serif',
    ui: '-apple-system, BlinkMacSystemFont, "Trebuchet MS", Roboto, Ubuntu, sans-serif',
    mono: '"SF Mono", "Cascadia Code", "Consolas", monospace',
  },
  fontSize: {
    xs: '10px',
    sm: '11px',
    md: '13px',
    base: '14px',
    lg: '16px',
  },
  fontWeight: {
    regular: 400,
    bold: 700,
  },
} as const;

export const spacing = {
  toolbarWidth: '52px',
  headerHeight: '42px',
  bottomBarHeight: '39px',
  rightPanelWidth: '300px',
  priceAxisWidth: '64px',
  timeAxisHeight: '28px',
  buttonHeight: {
    sm: '28px',
    md: '34px',
    lg: '40px',
  },
  borderRadius: {
    sm: '4px',
    md: '6px',
    lg: '8px',
    pill: '9px',
  },
} as const;
```

### Recommended Component Patterns

1. **All toolbar buttons**: Transparent bg, no border, icon-only with aria-label tooltips
2. **Charts**: Use Plotly with custom dark template matching these colors
3. **Data tables**: Dense rows (30-36px), monospace prices, color-coded changes
4. **Side panels**: Draggable resize handle, collapsible widgets
5. **Legend overlays**: 50% opacity text on chart, clickable to toggle visibility

---

## 12. Key Takeaways for Our Design

1. **Simplicity wins**: TradingView uses almost no border-radius, minimal borders, and transparent backgrounds. The UI gets out of the way.

2. **Color = information**: Green/red is not decorative -- it's data. Use it consistently for buy/sell, profit/loss, up/down.

3. **Opacity hierarchy**: Instead of multiple gray shades, use opacity on white/black. 50% opacity white for secondary text is more elegant than a specific gray.

4. **Chart is king**: 80%+ of the viewport is the chart. Toolbars are thin, panels are optional/collapsible.

5. **Single font weight**: 400 (regular) for almost everything. No bold headings or decorative weights.

6. **No rounded corners on toolbars**: Sharp edges (0px radius) for toolbars. Slight rounding (4px) only for buttons, inputs, and popups.

7. **Blue-tinted darks**: Dark mode uses `#131722` (navy-tinted black), not pure `#000000` or `#1a1a1a`.

8. **19-shade color system**: Each semantic color has 19 shades from lightest to darkest. This enables consistent theming across all states (hover, active, pressed, disabled).

9. **Canvas-based chart**: The chart itself is rendered on HTML5 Canvas (not SVG/DOM). This is critical for performance with thousands of data points.

10. **Responsive sizing tiers**: Components scale across 5 size tiers (xs/sm/md/lg/xl) using CSS custom properties.

---

## 13. Official CSS Custom Properties Reference

From TradingView's Advanced Charts documentation -- these are the CSS variables that control the chart chrome:

```css
/* === Toolbar Colors === */
--tv-color-platform-background              /* Main page background */
--tv-color-pane-background                  /* Toolbar background */
--tv-color-toolbar-button-background-hover  /* Toolbar button hover */
--tv-color-toolbar-button-background-expanded    /* Expanded button hover */
--tv-color-toolbar-button-background-active      /* Active button bg */
--tv-color-toolbar-button-background-active-hover /* Active button hover */
--tv-color-toolbar-button-text             /* Button text/icon color */
--tv-color-toolbar-button-text-hover       /* Button text hover */
--tv-color-toolbar-button-text-active      /* Active button text */
--tv-color-toolbar-button-text-active-hover /* Active button text hover */
--tv-color-item-active-text                /* Toggle button text */
--tv-color-toolbar-toggle-button-background-active    /* Toggle bg */
--tv-color-toolbar-toggle-button-background-active-hover /* Toggle hover */
--tv-color-toolbar-divider-background      /* Divider lines */
--tv-color-toolbar-save-layout-loader      /* Save layout loader color */

/* === Popup / Context Menu Colors === */
--tv-color-popup-background                /* Popup background */
--tv-color-popup-element-text              /* Popup item text */
--tv-color-popup-element-text-hover        /* Popup item text hover */
--tv-color-popup-element-background-hover  /* Popup item bg hover */
--tv-color-popup-element-divider-background /* Popup divider */
--tv-color-popup-element-secondary-text    /* Secondary/hint text */
--tv-color-popup-element-hint-text         /* Hint text */
--tv-color-popup-element-text-active       /* Active item text */
--tv-color-popup-element-background-active /* Active item bg */
--tv-color-popup-element-toolbox-text      /* Toolbox icon color */
--tv-color-popup-element-toolbox-text-hover     /* Toolbox hover */
--tv-color-popup-element-toolbox-text-active-hover /* Toolbox active hover */
--tv-color-popup-element-toolbox-background-hover /* Toolbox bg hover */
--tv-color-popup-element-toolbox-background-active-hover /* Toolbox active bg hover */
```

---

*Document generated from live TradingView chart application extraction + official TradingView Advanced Charts Documentation.*
*Date: 2026-04-18*
