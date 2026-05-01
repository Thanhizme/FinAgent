# Module 1: Data Collection Specification

## 1. Objective

- **Purpose**: Establish a robust data collection pipeline to gather financial data from multiple sources.
- **Market Focus**: Primarily Vietnam (VNI) with support for Global markets.
- **Downstream Compatibility**: Ensure all collected data matches the schemas required for Feature Engineering (Module 2), Visualization (Module 3), and AI Analysis (Module 4).

## 2. User Input Schema

The collection function must accept a configuration object/dictionary with the following fields:

- `tickers`: A list of strings (e.g., `["VCB"]` or `["VCB", "TCB"]`).
- `start_date`: String in `YYYY-MM-DD` format ()
- `end_date`: String in `YYYY-MM-DD` format (at the time that client run the program)
- `market`: String (e.g., `"VN"`).

## 3. Data Sources & Integration

- **Price Data**: Utilize Yahoo Finance (`yfinance`) or local Vietnamese APIs (VNDirect, Fiintrade, CafeF).
- **Macro/Fundamental**: Integrate financial data providers suitable for the Vietnamese market.

## 4. Detailed Data Schemas

GitHub Copilot must ensure the resulting DataFrames contain these exact column names:

### A. Price Data (`price_df`)

- Fields: `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`.
- Includes: Historical stock prices for the primary ticker(s).

### B. Benchmark & Peer Data (`benchmark_df`, `peer_df`)

- Fields: `date`, `ticker`, `close`, `volume`.
- Purpose: Used for relative performance, beta calculation, and correlation heatmaps.
- Default Benchmark: `VNIndex` for the Vietnamese market.

### C. Fundamental Data (`fundamental_df`)

- **Income Statement**: `revenue`, `gross_profit`, `operating_profit`, `net_income`, `eps`.
- **Balance Sheet**: `total_assets`, `total_liabilities`, `equity`, `total_debt`, `cash`.
- **Ratios**: `roe`, `roa`, `pe`, `pb`, `margin`, `debt_to_equity`.
- **Share Data**: `shares_outstanding`, `bvps`, `dividend`.

### D. Corporate Events & News (`news_df`)

- Fields: `date`, `ticker`, `headline`, `summary`, `source`, `sentiment`, `event_type`.
- `event_type` categories: `dividend`, `earnings`, `m&a`, `management_change`, `expansion`, `legal`.

### E. Macro Data (`macro_df`)

- Fields: `date`, `variable`, `value`.
- Key Variables: `interest_rate`, `inflation_cpi`, `usd_vnd`, `gold_price`, `oil_price`, `vnindex`, `bond_yield`.

### F. Industry Data (`industry_df`)

- Fields: `date`, `industry_pe`, `industry_pb`, `industry_roe`, `industry_growth`.

## 5. Processing Logic Requirements

- **Intraday Support**: Optional short-term analysis data (timestamp, price, volume).
- **Multi-Ticker Handling**: If 2 tickers are provided, collect full price and fundamental data for both to enable comparative analysis.
- **Validation**: Check for missing values and ensure date-sorting for all time-series data.

---

## 6. Copilot Execution Prompt (Internal Use)

> "Draft a Python class `DataCollector` that implements the logic in `#file:module1.md`.
> Use libraries like `vnstock` or `yfinance`.
> Ensure the output DataFrames strictly follow the schemas in Section 4."
