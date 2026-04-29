# FinAgent

**AI-Powered Financial Data Agent**
IT Application in Banking and Finance -- 2026

FinAgent is an end-to-end pipeline that autonomously collects live financial
data, cleans and engineers features, generates four categories of financial
charts, and delivers LLM-powered natural language analysis via Anthropic
Claude or OpenAI GPT.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Modules](#modules)
- [Data Sources](#data-sources)
- [Visualisations](#visualisations)
- [AI Analysis](#ai-analysis)
- [Team](#team)

---

## Project Structure

```
FinAgent/
??? data/
?   ??? raw/               # Raw data fetched from APIs (git-ignored)
?   ??? processed/         # Cleaned data and chart exports (git-ignored)
??? modules/
?   ??? __init__.py        # Package exports
?   ??? collector.py       # Stage 1 - Data acquisition
?   ??? processor.py       # Stage 2 - Cleaning and feature engineering
?   ??? visualizer.py      # Stage 3 - Chart generation
?   ??? ai_agent.py        # Stage 4 - LLM analysis
??? notebooks/             # Jupyter notebooks for experimentation
??? .env                   # Local secrets -- never commit (git-ignored)
??? .env.example           # API key configuration template
??? .gitignore
??? requirements.txt
??? main.py                # Pipeline entry point
??? README.md
```

---

## Features

| Module         | Capability                                                                                            |
|----------------|-------------------------------------------------------------------------------------------------------|
| **Collector**  | Stock prices (yfinance), financial statements, news (NewsAPI), macro indicators (Alpha Vantage)       |
| **Processor**  | Missing value handling, duplicate removal, type normalisation, outlier detection, feature engineering |
| **Visualizer** | Price trend + volume, correlation heatmap, returns distribution, rolling stats / Bollinger Bands      |
| **AI Agent**   | Trend summary, anomaly report, risk commentary, multi-asset comparison (Claude / GPT)                 |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/FinAgent.git
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

---

## Configuration

All secrets and runtime settings are managed via the `.env` file.
See [`.env.example`](.env.example) for the complete list of supported variables.

| Variable                | Required            | Description               |
|-------------------------|---------------------|---------------------------|
| `ANTHROPIC_API_KEY`     | Yes (or OpenAI)     | Anthropic Claude API key  |
| `OPENAI_API_KEY`        | Yes (or Anthropic)  | OpenAI GPT API key        |
| `NEWS_API_KEY`          | Yes                 | NewsAPI.org developer key |
| `ALPHA_VANTAGE_API_KEY` | Yes                 | Alpha Vantage API key     |

**Important:** Never commit your `.env` file. It is listed in `.gitignore`
and must remain local to each developer's machine.

---

## Usage

```bash
# Run with default tickers (AAPL, MSFT, GOOGL, NVDA) over the last 365 days
python main.py

# Custom tickers and date range
python main.py --tickers TSLA AMZN META --start 2024-01-01 --end 2024-12-31

# Use OpenAI instead of Anthropic
python main.py --provider openai

# Skip AI analysis (offline / cost-saving mode)
python main.py --skip-ai
```

### CLI Reference

| Argument     | Default                | Description                               |
|--------------|------------------------|-------------------------------------------|
| `--tickers`  | `AAPL MSFT GOOGL NVDA` | One or more stock ticker symbols          |
| `--start`    | 1 year ago             | Historical data start date (YYYY-MM-DD)   |
| `--end`      | Today                  | Historical data end date (YYYY-MM-DD)     |
| `--provider` | `anthropic`            | LLM provider: `anthropic` or `openai`     |
| `--skip-ai`  | `False`                | Skip the AI analysis stage                |

---

## Pipeline Overview

```
Stage 1             Stage 2             Stage 3             Stage 4
Data Collection --> Data Processing --> Visualisation  --> AI Analysis
collector.py        processor.py        visualizer.py       ai_agent.py
```

Each stage passes its output directly to the next. Failures in any stage
are caught, logged, and propagated cleanly to avoid silent data corruption.

---

## Modules

### collector.py -- Data Collection

Fetches raw financial data from multiple sources and persists it to
`data/raw/` as CSV or JSON files.

- `fetch_stock_prices()` -- OHLCV history via yfinance
- `fetch_financial_statements()` -- Quarterly income statement, balance sheet, cash flow
- `fetch_news(query)` -- Latest articles via NewsAPI
- `fetch_macro_indicators(symbols)` -- Exchange rates and commodities via Alpha Vantage

### processor.py -- Data Processing

Cleans and enriches raw DataFrames, saving output to `data/processed/`.

- `handle_missing_values(strategy)` -- `ffill`, `interpolate`, or `drop`
- `remove_duplicates()` -- Detection and removal with audit logging
- `normalise_types()` -- DatetimeIndex, float64 casting, currency string stripping
- `detect_outliers(method, threshold)` -- IQR or Z-score; flags an `is_outlier` column
- `engineer_features()` -- `daily_return`, `rolling_avg_7`, `rolling_avg_30`, `volatility_30`, `cum_return`
- `run_pipeline()` -- Executes all steps in sequence

### visualizer.py -- Visualisation

Generates the four required chart types and saves them to `data/processed/`.

- `price_trend_chart(ticker)` -- Price line or candlestick with volume overlay
- `correlation_heatmap()` -- Pairwise asset return correlation matrix
- `returns_distribution(tickers)` -- Histogram and KDE of daily returns
- `rolling_stats_chart(ticker)` -- Moving averages with Bollinger Bands shading

### ai_agent.py -- AI Analysis

Builds structured JSON context from processed DataFrames and submits
grounded prompts to the configured LLM provider.

- `generate_trend_summary(data)` -- Per-asset trend and recent performance narrative
- `generate_anomaly_report(data)` -- Outlier events and notable dates
- `generate_risk_commentary(data)` -- Volatility-based risk assessment
- `generate_comparison(data, tickers)` -- Side-by-side multi-asset comparison
- `run_full_analysis(data)` -- Executes all four tasks and returns a result dict

---

## Data Sources

| Source        | Library / API | Data Type                | Free Tier           |
|---------------|---------------|--------------------------|---------------------|
| Yahoo Finance | yfinance      | Stock prices, financials | No API key required |
| NewsAPI       | REST API      | Financial news articles  | 100 requests/day    |
| Alpha Vantage | REST API      | FX rates, commodities    | 25 requests/day     |

---

## Visualisations

| No. | Chart                         | Library         | Description                                        |
|-----|-------------------------------|-----------------|----------------------------------------------------|
| 1   | Price Trend + Volume          | Plotly          | Closing price with 7/30-day MA and volume bars     |
| 2   | Correlation Heatmap           | Seaborn         | Pairwise return correlations across all assets     |
| 3   | Returns Distribution          | Plotly/Seaborn  | Histogram and KDE of daily returns, normal overlay |
| 4   | Rolling Stats/Bollinger Bands | Plotly          | SMA with upper and lower band shading              |

---

## AI Analysis

The AI module serialises key statistics from processed DataFrames into a
structured JSON prompt, instructing the model to reference specific figures
in its output. This approach minimises hallucinations and ensures all
commentary is grounded in the actual dataset.

Output categories:

- **Trend Summary** -- Current trend and recent performance per asset, citing
  closing prices, moving averages, and cumulative returns.
- **Anomaly Report** -- Notable events or outlier dates identified in the
  dataset, with magnitude and potential causes.
- **Risk Commentary** -- Volatility-based risk assessment citing 30-day
  annualised volatility and drawdown figures.
- **Comparison** -- Side-by-side narrative comparing two or more assets
  across return, volatility, and trend dimensions.

Disclaimer: AI-generated outputs are for educational purposes only and do
not constitute financial advice.

---

## Team

| Member | Role                                    |
|--------|-----------------------------------------|
| TBD    | Data Collection and API Integration     |
| TBD    | Data Processing and Feature Engineering |
| TBD    | Visualisation and Dashboard             |
| TBD    | AI Analysis and Report Writing          |

---

