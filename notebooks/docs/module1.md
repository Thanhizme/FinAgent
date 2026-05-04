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

## 3. Data Sources & Integration

- **Price & Peers**: `yfinance` for US/Global, or local Vietnamese APIs for VNI.
- **Fundamental Data**: `yfinance` (Income Statement, Balance Sheet, Cash Flow).
- **Macro Data**: `FRED API` (via direct HTTP requests) for core US economic indicators, combined with `yfinance` for commodities and FX.
- **News Data**: `NewsAPI` with sentiment keyword filtering.

## 4. Detailed Data Schemas

The DataCollector must ensure the resulting CSV files/DataFrames strictly contain these exact column names and formats:

### A. Price Data (`price_df`)
- **Fields**: `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`.
- **Note**: Ascending chronological order, dropping entirely empty rows.

### B. Benchmark & Peer Data (`benchmark_df`, `peer_df`)
- **Benchmark Fields**: `date`, `ticker`, `close`, `volume` (Default: `^VNINDEX` for VN, `^GSPC` for Global).
- **Peer Fields**: `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`.
- **Peer Mapping Rule**: Peers must be dynamically selected based on matching Sector and Market Capitalization (Large/Mid/Small) to ensure accurate industry baseline comparisons.

### C. Fundamental Data (`fundamental_df`)
- **Fields**: `date`, `ticker`, `revenue`, `gross_profit`, `operating_profit`, `net_income`, `eps`, `total_assets`, `total_liabilities`, `equity`, `total_debt`, `cash`, `operating_cash_flow`, `roe`, `roa`, `pe`, `pb`, `margin`, `debt_to_equity`, `shares_outstanding`, `bvps`, `dividend`.
- **Key Requirement**: Must include `operating_cash_flow` to evaluate "Profit Quality" (CFO vs Net Income) as demanded by Module 4.

### D. Corporate Events & News (`news_df`)
- **Fields**: `date`, `ticker`, `headline`, `summary`, `source`, `sentiment`, `event_type`.
- **event_type categories**: `dividend`, `earnings`, `m&a`, `management_change`, `expansion`, `legal`, `macro`, `general`.
- **sentiment labels**: `positive`, `negative`, `neutral` (based on keyword parsing).

### E. Macro Data (`macro_df_wide`)
- **Fields**: `date`, `fed_funds_rate`, `us_10y_yield`, `us_cpi`, `dxy`, `gold_price`, `oil_price`.
- **Format Requirement**: Must be transformed into a **Wide Format** (columns as variables). 
- **Frequency Handling**: Monthly data (like `us_cpi` or `fed_funds_rate` from FRED) must be Forward-Filled (`ffill`) to match daily stock price frequencies.

### F. Industry Data (`industry_df`)
- **Fields**: `date`, `industry_roe`, `industry_margin`, `industry_pe`, `industry_pb`.
- **Purpose**: Calculated by averaging the fundamental metrics of the selected peer group for relative valuation.

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
> Ensure the fundamental data extracts `operating_cash_flow` and macro data is structured in a Wide Format with Forward-Fill. 
> Output DataFrames must strictly follow the schemas in Section 4."