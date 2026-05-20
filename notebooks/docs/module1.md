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


## 4. Detailed Data Schemas


The DataCollector must ensure the resulting CSV files/DataFrames strictly contain these exact column names and formats:


### A. Macro Data (`macro_df_wide`)


- **Fields**: date, imf_global_growth, fed_funds_rate , oil_price, us_gdp_growth, us_interest_rate,  us_fx_rate, us_fdi_inflow, us_cpi, vn_unemployment.vn_gdp_growth, vn_interest_rate,  vn_fx_rate, vn_fdi_inflow, vn_cpi, vn_unemployment.
- **Format Requirement**: Must be transformed into a **Wide Format** (columns as variables).
- **Frequency Handling**: Monthly data (like `us_cpi`,`fed_funds_rate`, ‘vn_gdp_growth’ from FRED) must be Forward-Filled (`ffill`) to match daily stock price frequencies.
- Sources: fed_funds_rate → FRED API
           oil_price → yfinance
           imf_global_growth → IMF API 
           vn_gdp_growth, vn_interest_rate, vn_fx_rate, vn_fdi_inflow, vn_cpi, vn_unemployment → World Bank API or VN Stock


### B. Industry Data (`industry_df`)


- **Fields**: date, industry_pe, industry_pb, industry_pe_1y, industry_pe_5y, industry_pb_1y, industry_pb_5y.
- **Purpose**: Calculated by averaging the fundamental metrics of the selected peer group for relative valuation.




### C. Corporate Events & News (`news_df`)


- **Fields**: `date`, `ticker`, `headline`, `summary`, `source`, `sentiment`, `event_type`.
- **event_type categories**: `dividend`, `earnings`, `m&a`, `management_change`, `expansion`, `legal`, `macro`, `general`.
- **sentiment labels**: `positive`, `negative`, `neutral` (based on keyword parsing).


### D. Fundamental Data (`fundamental_df`)


- **Fields**: date, ticker, revenue, gross_profit, operating_profit, net_income, eps, total_assets, total_liabilities, equity, total_debt, cash, operating_cash_flow, capital_expenditure, interest_expense, tax_rate, receivables, inventory, payble, current_assets, current_liabilities, COGS, roe, roa, pe, pb, shares_outstanding, bvps, dividend, market_cap, risk free rate, market risk premium.
- **Key Requirement**: Must include `operating_cash_flow` to evaluate "Profit Quality" (CFO vs Net Income) as demanded by Module 4. Must include ‘operating_cash_flow’, ‘capital_expenditure’, ‘interest_expense’ and ‘tax_rate’ to calculate FCFE. Must include ‘receivables’, ‘inventory’, ‘payable’, ‘current_assets’, ‘current_liabilities’, ‘COGS’ to calculate Activity và Liquidity ratios. Must include 'market_cap'. Note: Large/Mid/Small Cap classification uses a pre-defined mapping table instead of dynamic calculation.


### E. Price Data (`price_df`)


- **Fields**: `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`.
- **Note**: Ascending chronological order, dropping entirely empty rows.




### F. Benchmark & Peer Data (`benchmark_df`, `peer_df`)


- **Benchmark Fields**: `date`, `ticker`, `close`, `volume` (Default: `^VNINDEX` for VN, `^GSPC` for Global).
- **Peer Fields**: `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`.
- **Peer Mapping Rule**: Peers must be dynamically selected based on matching Sector and Market Capitalization (Large/Mid/Small) to ensure accurate industry baseline comparisons. 




## 5. Processing Logic Requirements


- **API Rate Limiting & Fallbacks**: Implement retry logic (e.g., 3 attempts) for `yfinance` and direct API calls.
- **Intraday Support**: Optional short-term analysis data (`timestamp`, `ticker`, `price`, `volume`).
- **Data Validation (`_validate_df`)**:
 1. Log missing value counts.
 2. Drop rows where all value columns are NaN.
 3. Ensure deterministic sorting by the `date` column.


---


## 6. Copilot Execution Prompt (Internal Use)


> "Draft a Python class `DataCollector` that implements the logic in `#file:module1.md`.
> Integrate `yfinance`, `NewsAPI`, and `FRED API` (using requests to api.stlouisfed.org).
> Ensure the fundamental data extracts operating_cash_flow, capital_expenditure, interest_expense, tax_rate, receivables, inventory, COGS, current_assets, current_liabilities, market cap and macro data is structured in a Wide Format with Forward-Fill.
> Output DataFrames must strictly follow the schemas in Section 4."


## 7. Implementation Notes (Current Project)


- Raw data is persisted under `data/raw/`.
- Processed outputs are currently stored under `data/processed/processed_data/`.
- Visualization artifacts are exported under `data/processed/visualization/`.
- The downstream correlation chart requirement is implemented as a heatmap across selected indicators per ticker (not only cross-ticker return correlation).