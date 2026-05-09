# Module 3: Visualization Specification

## 1. Objective

- **Purpose**: Convert processed financial features into clear, decision-ready visual outputs for analysis and reporting.
- **Input**: Processed DataFrames from Module 2 (price, return, risk, and indicator columns).
- **Output**: Interactive Plotly charts exported as HTML and PNG for local dashboard use and offline sharing.

## 2. Output Path and Naming Convention

- **Chart directory**: `data/processed/visualization/`
- **Price chart**: `<ticker>_price_volume_<timeframe>.html` and `.png`
- **Returns distribution**: `<ticker>_returns_distribution.html` and `.png`
- **Correlation heatmap**: `<ticker>_correlation_heatmap.html` and `.png`
- **Rolling stats**: `<ticker>_rolling_stats.html` and `.png`

Timeframe must support: `daily`, `weekly`, `monthly`, `yearly`, `all`.

## 3. Required Chart Set

### A. Price & Volume Master Chart

- **Price layer**: Candlestick as default, with optional OHLC/line fallback.
- **Trend overlays**: MA20, MA50, MA200.
- **Volume panel**: Separate subplot with volume bars and volume MA20.
- **Timeframes**: Must render correctly for daily/weekly/monthly/yearly.
- **Usability**: Hover tooltips, range selector, readable axis labels.

### B. Returns Distribution (Oscillators)

- **Main plot**: Histogram of clipped daily returns.
- **Density line**: KDE line overlaid on histogram.
- **Fallback rule**: If KDE is unstable or fails, render a smoothed density line from histogram bins.
- **Risk markers**: Vertical lines for `Return = 0`, `Mean`, `Median`, `VaR 95%`, `VaR 99%`.
- **Legend panel**: Dedicated right-side annotation box showing marker styles and values.

### C. Correlation Heatmap Across Selected Indicators

- **Scope**: Correlation matrix for selected indicator columns of the chosen ticker.
- **Default indicator set** should include representative return, momentum, trend, volatility, and risk fields.
- **Computation**: Pearson correlation with minimum valid observations per indicator.
- **Display**: Annotated cell values, bounded color scale from -1 to 1, readable labels.

### D. Rolling Statistics / Bands

- **Base**: Close price line.
- **Overlays**: MA curves and Bollinger bands.
- **Volume**: Optional lower subplot for volume where available.

## 4. Data Quality and Rendering Rules

- Skip charts for tickers with empty or invalid processed data.
- Replace `inf/-inf` with `NaN` before plotting.
- Ensure return-series charts only use non-null return values.
- Log chart-level exceptions without stopping batch rendering for other tickers.

## 5. Dashboard Integration Requirements

The local Streamlit dashboard must:

- Load charts from `data/processed/visualization/`.
- Regenerate missing charts automatically from processed data.
- Regenerate returns distribution and correlation heatmap for selected ticker when refreshing.
- Show both Returns Distribution and Correlation Heatmap in the Oscillators tab.

## 6. Visual Design Guidelines

- Use a dark financial-terminal style with high contrast axis text.
- Prevent title/legend overlap in constrained widths.
- Keep marker legend readable on desktop and laptop resolutions.
- Preserve consistent typography, spacing, and color semantics across all charts.

## 7. Copilot Execution Prompt

> "Implement visualization logic in `modules/visualizer.py` based on `#file:module3.md`.
> Use Plotly to render price-volume, returns distribution (histogram + KDE with fallback), correlation heatmap across selected indicators, and rolling stats.
> Save outputs to `data/processed/visualization/` as HTML (required) and PNG (best effort)."
