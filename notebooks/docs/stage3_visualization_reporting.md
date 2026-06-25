# Module 3: Visualization Specification


## 1. Objective
* **Purpose:** Convert processed financial features into clear, decision-ready visual outputs for analysis, reporting, and portfolio comparison.
* **Input:** Processed DataFrames from Module 2 (price, return, risk, and indicator columns for target stocks and S&P 500, VNIndex benchmark).
* **Output:** Interactive Plotly charts exported as HTML and PNG for local dashboard use and offline sharing.


## 2. Output Path and Naming Convention
* **Chart directory:** `data/processed/visualization/`
* **Individual Assets:**
    * Price chart: `<ticker>_price_volume_<timeframe>.html` and `.png`
    * Returns distribution: `<ticker>_returns_distribution.html` and `.png`
    * Rolling stats: `<ticker>_rolling_stats.html` and `.png`
* **Portfolio / Comparison (New):**
    * Performance Comparison: `portfolio_cumulative_performance.html` and `.png`
    * Efficient Frontier: `portfolio_efficient_frontier.html` and `.png`
    * Asset Correlation: `portfolio_asset_correlation.html` and `.png`
* **Timeframe must support:** daily, weekly, monthly, yearly, all.


## 3. Required Chart Set


### A. Price & Volume Master Chart (Individual)
* **Price layer:** Candlestick as default, with optional OHLC/line fallback.
* **Trend overlays:** MA20, MA50, MA200.
* **Volume panel:** Separate subplot with volume bars and volume MA20.
* **Timeframes:** Must render correctly for daily/weekly/monthly/yearly.
* **Usability:** Hover tooltips, range selector, readable axis labels.


### B. Returns Distribution (Individual)
* **Axes:** X-axis = Return (%), Y-axis = Frequency.
* **Main plot:** Histogram of clipped daily returns.
* **Density line:** KDE line overlaid on histogram.
* **Fallback rule:** If KDE is unstable/fails, render a smoothed density line from histogram bins.
* **Risk markers:** Vertical lines for Return = 0, Mean, Median, VaR 95%, VaR 99%.
* **Legend panel:** Dedicated right-side annotation box showing marker styles and values.


### C. Correlation Heatmap Across Selected Indicators


* **Scope:** Correlation matrix comparing key technical and risk indicators across the selected asset(s) and Benchmark (VN-Index), including:
  - **Trend & Return Signals:** Current Price, Recent Movement, % Return (1W / 1M / 3M / YTD)
  - **Moving Averages & Oscillators:** MA20, MA50, MA200, RSI, MACD
  - **Risk Metrics:** Historical Volatility (HV 30D / 60D), Beta vs VN-Index, VaR (95% / 99%), Max Drawdown, Sharpe Ratio


* **Computation:** Pearson correlation computed on time-aligned daily series; numeric anomaly flags (0/1 or magnitude scores) are used for categorical indicators; minimum valid observations enforced per pair.


* **Display:** Annotated cell values, bounded color scale from -1 to 1, readable axis labels per indicator group.




### D. Cumulative Performance Chart (Comparison)


* **Purpose:** 
  Compare the relative performance of Stock A, Stock B (optional), and the market benchmark over time.


* **Benchmark Selection Rule:**
  - Use `^VNINDEX` when `market = "VN"` for Vietnamese equities.
  - Use `^GSPC` (S&P 500) when `market = "GLOBAL"` for US/Global equities.


* **Scope:** 
  Compare:
  - Stock A
  - Stock B (if provided or auto-selected peer)
  - Benchmark Index (`^VNINDEX` or `^GSPC`)


* **Metric:** 
  Cumulative Return (%) calculated from the first shared available trading date across all compared assets.


  Formula:


  ```text
  Cumulative Return (%) =
  ((Current Price / Initial Price) - 1) × 100




### E. Efficient Frontier (Portfolio Simulation)
* **Axes:** X-axis = Volatility (Standard Deviation/Risk), Y-axis = Expected Return.
* **Computation:** Simulate N random portfolios using the individual time-series return data of Stock A and Stock B (and potentially other assets if provided). Calculate annualized return and volatility for each simulated portfolio.
* **Visual:** Scatter plot of all simulated portfolios. Highlight the "Optimal Portfolio" (Maximum Sharpe Ratio) with a distinct star marker, and the "Minimum Volatility Portfolio" with another marker.
* **Hover:** Tooltips must show the simulated weight allocations (e.g., A: 60%, B: 40%) and the Sharpe ratio for that specific point.


### F. Asset Correlation Heatmap (Comparison)
* **Scope:** Correlation matrix comparing daily returns of Stock A, Stock B, and the Benchmark (S&P 500).
* **Computation:** Pearson correlation using aligned dates with minimum valid observations.
* **Display:** Annotated cell values, bounded color scale from -1 to 1, readable labels.


## 4. Data Quality and Rendering Rules
* Skip charts for tickers with empty or invalid processed data.
* Replace inf/-inf with NaN before plotting.
* For comparison charts (Performance, Correlation), automatically inner-join/align dataframes on the Date index so all lines start and end correctly.
* Ensure return-series charts only use non-null return values.
* Log chart-level exceptions without stopping batch rendering for other tickers.


## 5. Dashboard Integration Requirements


* The local Streamlit dashboard must load charts from `data/processed/visualization/`.
* Regenerate missing charts automatically from processed data.


* **Tabs structure:**
    * *Individual Analysis:* Shows Price, Distribution, and Rolling Stats for the selected ticker.
    
    * *Portfolio & Comparison:* Shows:
        - Cumulative Performance (Stock A vs Stock B vs Benchmark)
        - Asset Correlation Heatmap Across Selected Indicators
        - Efficient Frontier


* **Benchmark Selection Rule:**
    - Use `^VNINDEX` when `market = "VN"` for Vietnamese equities.
    - Use `^GSPC` (S&P 500) when `market = "GLOBAL"` for US/Global equities.




## 6. Visual Design Guidelines
* Use a dark financial-terminal style with high contrast axis text.
* Prevent title/legend overlap in constrained widths.
* Keep marker legend readable on desktop and laptop resolutions.
* Preserve consistent typography, spacing, and color semantics across all charts (e.g., S&P 500 always uses a specific neutral color like white or gray, while A and B use distinct primary colors).


## 7. Copilot Execution Prompt


> "Implement visualization logic in `modules/visualizer.py` based on `#file:module3.md`.
> Use Plotly to render:
> 1. Price-volume with MA20/50/200.
> 2. Returns distribution (histogram + KDE, x=return, y=frequency).
> 3. Asset Correlation heatmap (Target Stocks vs S&P 500).
> 4. Cumulative Performance line chart (Target Stocks vs S&P 500 starting from day 1).
> 5. Efficient Frontier scatter plot (y=return, x=volatility) by simulating random portfolio weights using individual asset return series.
> Ensure data is date-aligned for comparison charts. Save outputs to `data/processed/visualization/` as HTML (required) and PNG (best effort)."