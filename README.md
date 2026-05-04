# FinAgent

**AI-Powered Financial Data Agent**
IT Application in Banking and Finance - 2026

FinAgent is an end-to-end pipeline that autonomously collects live financial
data, cleans and engineers features, generates four categories of financial
charts, and delivers LLM-powered natural language analysis via Google Gemini,
Anthropic Claude, or OpenAI GPT.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Modules](#modules)
- [Processed Data Schema](#processed-data-schema)
- [Data Sources](#data-sources)
- [Visualisations](#visualisations)
- [AI Analysis](#ai-analysis)
- [Team](#team)

---

## Project Structure

```
FinAgent/
├── data/
│   ├── raw/               # Raw data fetched from APIs (git-ignored)
│   └── processed/         # Cleaned, feature-engineered CSVs (git-ignored)
├── modules/
│   ├── __init__.py        # Package exports
│   ├── collector.py       # Stage 1 - Data acquisition
│   ├── processor.py       # Stage 2 - Cleaning and feature engineering
│   ├── visualizer.py      # Stage 3 - Chart generation
│   └── ai_agent.py        # Stage 4 - LLM analysis
├── notebooks/
│   └── docs/
│       ├── module1.md     # Data collection specification
│       └── module2.md     # Processing & feature engineering specification
├── .env                   # Local secrets - never commit (git-ignored)
├── .env.example           # API key configuration template
├── .gitignore
├── requirements.txt
├── main.py                # Pipeline entry point
└── README.md
```

---

## Features

| Module         | Capability                                                                                                                                                                               |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Collector**  | Stock prices, financial statements, benchmark & peers (yfinance), news with relevance filtering (NewsAPI), macro indicators (FRED + yfinance), industry aggregates                       |
| **Processor**  | Missing value handling, duplicate removal, type normalisation, IQR outlier detection, full technical indicator suite, rolling beta/Sharpe, relative strength, news sentiment aggregation |
| **Visualizer** | Price trend + volume, correlation heatmap, returns distribution, rolling stats / Bollinger Bands                                                                                         |
| **AI Agent**   | Trend summary, anomaly report, risk commentary, multi-asset comparison via Gemini / Claude / GPT                                                                                         |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Thanhizme/FinAgent.git
cd FinAgent
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your API keys. See the [Configuration](#configuration) section for details.

### 5. Run the pipeline

```bash
python main.py
```

When `--tickers` is not provided, an interactive menu lets you choose the default ticker, pick from a suggested list, or enter any custom symbols.

---

## Configuration

All secrets and runtime settings are managed via the `.env` file.
See [`.env.example`](.env.example) for the complete list of supported variables.

| Variable            | Required      | Description                                  |
| ------------------- | ------------- | -------------------------------------------- |
| `GEMINI_API_KEY`    | Yes (default) | Google Gemini API key                        |
| `ANTHROPIC_API_KEY` | Optional      | Anthropic Claude API key                     |
| `OPENAI_API_KEY`    | Optional      | OpenAI GPT API key                           |
| `NEWS_API_KEY`      | Yes           | NewsAPI.org developer key                    |
| `FRED_API_KEY`      | Yes           | FRED (Federal Reserve Economic Data) API key |

Get your free Gemini API key at: https://aistudio.google.com/app/apikey  
Get your free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

---

## Usage

```bash
# Run with interactive ticker selection prompt
python main.py

# Specify tickers directly via CLI
python main.py --tickers AAPL MSFT

# Custom date range
python main.py --tickers TSLA --start 2024-01-01 --end 2024-12-31

# Use a different LLM provider
python main.py --provider anthropic
python main.py --provider openai

# Skip AI analysis (offline / cost-saving mode)
python main.py --skip-ai
```

### CLI Reference

| Argument     | Default       | Description                                                        |
| ------------ | ------------- | ------------------------------------------------------------------ |
| `--tickers`  | _(prompt)_    | One or more stock ticker symbols. Omit to get an interactive menu. |
| `--start`    | 18 months ago | Historical data start date (YYYY-MM-DD)                            |
| `--end`      | Yesterday     | Historical data end date (YYYY-MM-DD)                              |
| `--provider` | `gemini`      | LLM provider: `gemini`, `anthropic`, `openai`                      |
| `--skip-ai`  | `False`       | Skip the AI analysis stage                                         |

> **Note:** The default start date includes an 18-month warm-up buffer so that
> long-window indicators (MA200, rolling 252-day Sharpe) have sufficient history
> from day one of the target analysis period.

---

## Pipeline Overview

```
Stage 1             Stage 2             Stage 3             Stage 4
Data Collection --> Data Processing --> Visualisation  --> AI Analysis
collector.py        processor.py        visualizer.py       ai_agent.py
```

Each stage passes its output to the next. Failures are caught, logged, and
propagated cleanly to avoid silent data corruption.

---

## Modules

### collector.py — Data Collection

`DataCollector` fetches raw financial data from multiple sources and saves
everything to `data/raw/` as CSV files.

**Public methods:**

| Method                         | Output file                | Description                                               |
| ------------------------------ | -------------------------- | --------------------------------------------------------- |
| `fetch_stock_prices()`         | `<TICKER>_prices.csv`      | OHLCV daily history via yfinance                          |
| `fetch_benchmark()`            | `benchmark_<SYMBOL>.csv`   | Benchmark index (default: `^GSPC`) via yfinance           |
| `fetch_peers(peers)`           | `peer_<TICKER>.csv`        | Peer tickers matched by sector and market-cap             |
| `fetch_financial_statements()` | `<TICKER>_fundamental.csv` | Quarterly income statement, balance sheet, cash flow      |
| `fetch_macro_indicators()`     | `macro_indicators.csv`     | FRED rates/CPI + yfinance commodities/FX in wide format   |
| `fetch_industry_data(peers)`   | `industry_data.csv`        | Peer-averaged fundamental metrics                         |
| `fetch_news(query)`            | `news_<TICKER>.csv`        | Relevance-filtered news from NewsAPI, one row per article |
| `fetch_intraday()`             | `intraday_<TICKER>.csv`    | Optional short-term 5-minute OHLCV                        |

**News relevance filtering:**  
Short or ambiguous tickers (e.g. `V` for Visa) are automatically expanded to
their full company names using `_NEWS_ENTITY_ALIASES`. Each article is checked
against finance-context keywords and per-ticker exclusion keywords before being
kept. Articles that pass the filter carry one of eight `event_type` labels:
`dividend`, `earnings`, `m&a`, `management_change`, `expansion`, `legal`,
`macro`, `general`.

**Data validation (`_validate_df`)** runs after every fetch and:

1. Logs missing value counts.
2. Drops rows where all value columns are NaN.
3. Sorts by `date` ascending.

---

### processor.py — Data Cleaning & Feature Engineering

`DataProcessor` is initialised with a single raw DataFrame and a ticker
symbol. All methods return `self` for chaining. The full pipeline is
executed by `run_pipeline()` which saves the result to `data/processed/`.

**Cleaning steps:**

| Method                               | Description                                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `handle_missing_values(strategy)`    | `ffill` then `bfill` for price series                                                                         |
| `remove_duplicates()`                | Deduplicates on `(date, ticker)` with logging                                                                 |
| `normalise_types()`                  | Casts `date` to datetime, auto-detects and casts all numeric-like columns to float64, strips currency symbols |
| `detect_outliers(method, threshold)` | IQR method on `daily_return`; flags an `is_outlier` boolean column                                            |

**Feature engineering — return metrics:**

| Column                                                                | Description                         |
| --------------------------------------------------------------------- | ----------------------------------- |
| `daily_return`                                                        | $r_t = (P_t - P_{t-1}) / P_{t-1}$   |
| `log_return`                                                          | $\ln(P_t / P_{t-1})$                |
| `cum_return_7d`, `cum_return_30d`, `cum_return_90d`, `cum_return_ytd` | Cumulative returns over each window |

**Feature engineering — trend indicators:**

| Column                                 | Description                                   |
| -------------------------------------- | --------------------------------------------- |
| `ma7`, `ma20`, `ma30`, `ma50`, `ma200` | Simple moving averages                        |
| `ema12`, `ema26`                       | Exponential moving averages (inputs for MACD) |

**Feature engineering — volatility & risk:**

| Column                              | Description                                          |
| ----------------------------------- | ---------------------------------------------------- |
| `volatility_20`, `volatility_60`    | Rolling std dev of daily returns (annualised)        |
| `bb_upper`, `bb_middle`, `bb_lower` | Bollinger Bands (20-day, 2 σ)                        |
| `atr_14`                            | Average True Range (14-day)                          |
| `drawdown`                          | Peak-to-trough drawdown                              |
| `sharpe_ratio`                      | Rolling 252-day annualised Sharpe ratio              |
| `beta`                              | Rolling 60-day beta vs benchmark (first 60 rows NaN) |

**Feature engineering — momentum oscillators:**

| Column                                  | Description                                  |
| --------------------------------------- | -------------------------------------------- |
| `rsi_14`                                | Relative Strength Index (14-day)             |
| `macd_line`, `macd_signal`, `macd_hist` | MACD (EMA12 − EMA26, EMA9 signal, histogram) |

**Multi-asset features:**

| Method                                 | Description                                             |
| -------------------------------------- | ------------------------------------------------------- |
| `calc_correlation_matrix(other_dfs)`   | Pairwise return correlation across assets               |
| `calc_relative_strength(benchmark_df)` | Ratio of stock cumulative return to benchmark (RS Line) |

**News processing:**

`process_news()` is a standalone method (not part of the price pipeline).
It loads `news_<TICKER>.csv` files for all tickers, encodes sentiment
(`positive=1`, `neutral=0`, `negative=-1`), and aggregates to **one row per
date per ticker** with columns: `article_count`, `positive_count`,
`neutral_count`, `negative_count`. Result is saved to
`data/processed/news_processed.csv`.

**Pipeline execution:**

| Method                    | Description                                                                |
| ------------------------- | -------------------------------------------------------------------------- |
| `run_pipeline()`          | Runs all cleaning + feature engineering steps; returns processed DataFrame |
| `run_pipeline_and_save()` | Runs pipeline and saves result to `data/processed/<TICKER>_processed.csv`  |

---

### visualizer.py — Visualisation

Generates the four required chart types and saves them to `data/processed/`.

| Method                          | Description                                           |
| ------------------------------- | ----------------------------------------------------- |
| `price_trend_chart(ticker)`     | Closing price with MA overlays and volume bars        |
| `correlation_heatmap()`         | Pairwise return correlation heatmap across all assets |
| `returns_distribution(tickers)` | Histogram and KDE of daily returns                    |
| `rolling_stats_chart(ticker)`   | Moving averages with Bollinger Bands shading          |

---

### ai_agent.py — AI Analysis

Builds structured JSON context from processed DataFrames and submits
grounded prompts to the configured LLM provider.

| Method                               | Description                                       |
| ------------------------------------ | ------------------------------------------------- |
| `generate_trend_summary(data)`       | Per-asset trend and recent performance narrative  |
| `generate_anomaly_report(data)`      | Outlier events and notable dates                  |
| `generate_risk_commentary(data)`     | Volatility-based risk assessment                  |
| `generate_comparison(data, tickers)` | Side-by-side multi-asset comparison               |
| `run_full_analysis(data)`            | Executes all four tasks and returns a result dict |

---

## Processed Data Schema

All processed CSVs are saved to `data/processed/`. The pipeline produces
the following files for each run:

| File                                 | Description                                   | Key columns                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<TICKER>_processed.csv`             | Price + full indicator set per ticker         | `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `daily_return`, `log_return`, `ma7`…`ma200`, `ema12`, `ema26`, `rsi_14`, `macd_line`, `macd_signal`, `macd_hist`, `bb_upper`, `bb_middle`, `bb_lower`, `atr_14`, `volatility_20`, `volatility_60`, `drawdown`, `sharpe_ratio`, `beta`, `relative_strength`, `cum_return_ytd`, `is_outlier` |
| `benchmark_processed.csv`            | Same pipeline applied to the benchmark index  | Same schema minus `atr_14` (always NaN for index data, auto-dropped)                                                                                                                                                                                                                                                                                                |
| `<TICKER>_fundamental_processed.csv` | Cleaned quarterly fundamental data            | `date`, `ticker`, `revenue`, `gross_profit`, `operating_profit`, `net_income`, `eps`, `total_assets`, `total_liabilities`, `equity`, `total_debt`, `cash`, `operating_cash_flow`, `roe`, `roa`, `margin`, `debt_to_equity`, `shares_outstanding`, `bvps`, `dividend` (all-null columns auto-dropped)                                                                |
| `macro_processed.csv`                | Wide-format macro indicators, daily frequency | `date`, `fed_funds_rate`, `us_10y_yield`, `us_cpi`, `dxy`, `gold_price`, `oil_price`                                                                                                                                                                                                                                                                                |
| `industry_processed.csv`             | Peer-averaged fundamental metrics             | `date`, `industry_roe`, `industry_margin` (all-null columns auto-dropped)                                                                                                                                                                                                                                                                                           |
| `news_processed.csv`                 | Daily aggregated news sentiment per ticker    | `date`, `ticker`, `article_count`, `headline`, `summary`, `source`, `sentiment`, `sentiment_score`, `positive_count`, `neutral_count`, `negative_count`, `event_type`                                                                                                                                                                                               |

---

## Data Sources

| Source        | Library / API   | Data Type                                    | Free Tier           |
| ------------- | --------------- | -------------------------------------------- | ------------------- |
| Yahoo Finance | yfinance        | Stock prices, benchmark, peers, fundamentals | No API key required |
| NewsAPI       | REST API        | Financial news articles                      | 100 requests/day    |
| FRED          | REST API (HTTP) | Fed funds rate, 10Y yield, CPI               | Free with API key   |
| yfinance      | yfinance        | DXY, gold price (GC=F), oil price (CL=F)     | No API key required |

---

## Visualisations

| No. | Chart                         | Library        | Description                                        |
| --- | ----------------------------- | -------------- | -------------------------------------------------- |
| 1   | Price Trend + Volume          | Plotly         | Closing price with MA overlays and volume bars     |
| 2   | Correlation Heatmap           | Seaborn        | Pairwise return correlations across all assets     |
| 3   | Returns Distribution          | Plotly/Seaborn | Histogram and KDE of daily returns, normal overlay |
| 4   | Rolling Stats/Bollinger Bands | Plotly         | SMA with upper and lower band shading              |

---

## AI Analysis

The AI module serialises key statistics from processed DataFrames into a
structured JSON prompt, instructing the model to reference specific figures
in its output. This approach minimises hallucinations and ensures all
commentary is grounded in the actual dataset.

The default provider is Google Gemini. Anthropic Claude and OpenAI GPT are
available as alternatives via the `--provider` flag.

Output categories:

- **Trend Summary** - Current trend and recent performance per asset, citing
  closing prices, moving averages, and cumulative returns.
- **Anomaly Report** - Notable events or outlier dates identified in the
  dataset, with magnitude and potential causes.
- **Risk Commentary** - Volatility-based risk assessment citing rolling
  volatility, Sharpe ratio, and drawdown figures.
- **Comparison** - Side-by-side narrative comparing two or more assets
  across return, volatility, and trend dimensions.

Disclaimer: AI-generated outputs are for educational purposes only and do
not constitute financial advice.

---

## Team

| Member | Role                                    |
| ------ | --------------------------------------- |
| TBD    | Data Collection and API Integration     |
| TBD    | Data Processing and Feature Engineering |
| TBD    | Visualisation and Dashboard             |
| TBD    | AI Analysis and Report Writing          |
