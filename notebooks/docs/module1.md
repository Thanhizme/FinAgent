# Module 1: Data Collection Specification


## 1. Objective


- **Purpose**: Establish a robust data collection pipeline to gather financial data from multiple sources.
- **Market Focus**: Primarily Vietnam (VNI) with support for US/Global markets.
- **Downstream Compatibility**: Ensure all collected data matches the exact schemas required for Feature Engineering (Module 2), Visualization (Module 3), and AI Analysis (Module 4) - specifically providing enough data for profit quality checks and macro-driven scenarios.


## 2. User Input Schema


The collection function must accept a configuration object/dictionary with the following fields:


- `tickers`: A list of strings (e.g., `["AAPL"]` or `["VCB", "TCB"]`).
- `start_date`: String in `YYYY-MM-DD` format. (Note: Must include a warm-up buffer of at least 12 months prior to the target analysis period to allow for MA200 calculations). 
- `end_date`: String in `YYYY-MM-DD` format (at the time the client runs the program).
- `market`: String (e.g., `"VN"` or `"GLOBAL"`).
- `stock_b`: Optional string — ticker of Stock B for comparison. When `stock_b` is provided by the client, use it directly as the comparison stock. If not provided, auto-select from the pre-defined sample list with the same Market Cap classification as Stock A.


## 3. Data Sources & Integration


- **Macro Data**: `FRED API` (via direct HTTP requests) for core US economic indicators, combined with `yfinance` for commodities and FX.
- **News Data**: `NewsAPI` with sentiment keyword filtering.
- **Fundamental Data**: `yfinance` (Income Statement, Balance Sheet, Cash Flow).
- **Price & Peers**: `yfinance` for US/Global, or local Vietnamese APIs for VNI.
- **VN Benchmark (Official)**: `vnstock` with `source="VCI"` to fetch `VNINDEX` EOD data (`open`, `high`, `low`, `close`, `volume`) for Vietnam benchmark analysis.
- **Benchmark Fallback Order (VN)**: `vnstock:VCI (VNINDEX official)` -> `yfinance VNINDEX candidates` -> `synthetic VNINDEX proxy`.


## 4. Detailed Data Schemas


The DataCollector must ensure the resulting CSV files/DataFrames strictly contain these exact column names and formats:


### A. Macro Data (`macro_df_wide`)


- **Fields (current implementation)**: `date`, `fed_funds_rate`, `us_10y_yield`, `us_cpi`, `dxy`, `gold_price`, `oil_price`, `gdp`, `unemployment_rate`, `domestic_interest_rate`, `fx_rate`, `fdi_inflow`.
- **Format Requirement**: Must be transformed into a **Wide Format** (columns as variables).
- **Frequency Handling**: Low-frequency macro series (monthly/quarterly/annual) are forward-filled (`ffill`) onto the daily timeline. In current code, `ffill` is applied across macro value columns in collector and then missing values are cleaned again in processor.
- **Sources (current implementation)**:
           `fed_funds_rate`, `us_10y_yield`, `us_cpi`, `gdp`, `unemployment_rate` -> FRED API
           `dxy`, `gold_price`, `oil_price`, `fx_rate` -> yfinance
           `fdi_inflow` -> World Bank API
           `domestic_interest_rate` -> derived fallback from `fed_funds_rate` when missing

           
### B. Industry Data (`industry_df`)


- **Fields**: date, industry_roe, industry_pe, industry_pb, industry_pe_1y, industry_pe_5y, industry_pb_1y, industry_pb_5y.
- **Purpose**: Calculated by averaging the fundamental metrics of the selected peer group for relative valuation.




### C. Corporate Events & News (`news_df`)


- **Fields**: `date`, `ticker`, `headline`, `summary`, `source`, `sentiment`, `event_type`.
- **event_type categories**: `dividend`, `earnings`, `m&a`, `management_change`, `expansion`, `legal`, `macro`, `general`.
- **sentiment labels**: `positive`, `negative`, `neutral` (based on keyword parsing).


### D. Fundamental Data (`fundamental_df`)


- **Fields**: date, ticker, revenue, gross_profit, operating_profit, net_income, eps, total_assets, total_liabilities, equity, total_debt, cash, operating_cash_flow, capital_expenditure, interest_expense, tax_rate, receivables, inventory, current_assets, current_liabilities, COGS, roe, roa, pe, pb, shares_outstanding, bvps, dividend, market_cap.
- **Key Requirement**: Must include `operating_cash_flow` to evaluate "Profit Quality" (CFO vs Net Income) as demanded by Module 4. Must include ‘operating_cash_flow’, ‘capital_expenditure’, ‘interest_expense’ and ‘tax_rate’ to calculate FCFE. Must include ‘receivables’, ‘inventory’, ‘current_assets’, ‘current_liabilities’, ‘COGS’ to calculate Activity và Liquidity ratios. Must include 'market_cap'. Note: Large/Mid/Small Cap classification uses a pre-defined mapping table instead of dynamic calculation.


### E. Price Data (`price_df`)


- **Fields**: `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`.
- **Note**: Ascending chronological order, dropping entirely empty rows.




### F. Benchmark & Peer Data (`benchmark_df`, `peer_df`)


- **Benchmark Fields**: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume` for VN official EOD benchmark; minimum required downstream fields are `date`, `ticker`, `close`, `volume`.
- **VN Benchmark Source Rule**: Prefer `VNINDEX` from `vnstock (VCI)` before any fallback source.