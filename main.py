"""
main.py
-------
FinAgent - AI-Powered Financial Data Agent
Entry point and workflow orchestrator.

Pipeline stages
---------------
  1. Data Collection  : fetch stock prices, financials, news, macro indicators
  2. Data Processing  : clean, normalise, and engineer features
  3. Visualisation    : generate all four required chart types
  4. AI Analysis      : produce LLM-powered narrative reports

Usage
-----
  python main.py                          run full pipeline with defaults
  python main.py --tickers AAPL MSFT      specify tickers
  python main.py --start 2023-01-01       override start date
  python main.py --provider gemini        select LLM provider (default)

Environment
-----------
  Copy .env.example to .env and fill in your API keys before running.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta

from modules import DataCollector, DataProcessor, DataVisualizer, AIAgent

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finagent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_TODAY         = datetime.today()
_YESTERDAY     = _TODAY - timedelta(days=1)
_START_DEFAULT = _TODAY - relativedelta(months=30)      

DEFAULT_TICKERS  = ["V"]
DEFAULT_END      = _YESTERDAY.strftime("%Y-%m-%d")      
DEFAULT_START    = _START_DEFAULT.strftime("%Y-%m-%d")  
DEFAULT_PROVIDER = "gemini"

SUGGESTED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "V", "JPM", "META",
    "FPT", "VCB", "VHM", "HPG",
]


def _parse_ticker_input(raw_value: str) -> list[str]:
    """Parse tickers from a comma/space-separated terminal input string."""
    parts = raw_value.replace(",", " ").split()
    cleaned = []
    for item in parts:
        ticker = item.strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned


def prompt_tickers_from_terminal(default_tickers: list[str]) -> list[str]:
    """Prompt users for ticker selection when --tickers is not provided."""
    if not sys.stdin.isatty():
        logger.info("Non-interactive terminal detected, using default tickers: %s", default_tickers)
        return default_tickers

    print("\nTicker Selection")
    print("1) Use default ticker(s):", ", ".join(default_tickers))
    print("2) Pick from suggested list")
    print("3) Enter custom ticker(s)")

    choice = input("Choose option [1/2/3] (default=1): ").strip() or "1"

    if choice == "2":
        print("\nSuggested tickers:")
        for idx, ticker in enumerate(SUGGESTED_TICKERS, start=1):
            print(f"  {idx:>2}. {ticker}")
        raw_idx = input("Enter one or more indexes (e.g. 1 6 10): ").strip()
        selected = []
        for part in raw_idx.replace(",", " ").split():
            if part.isdigit():
                index = int(part)
                if 1 <= index <= len(SUGGESTED_TICKERS):
                    ticker = SUGGESTED_TICKERS[index - 1]
                    if ticker not in selected:
                        selected.append(ticker)
        if selected:
            return selected
        logger.warning("No valid index selected. Falling back to default tickers.")
        return default_tickers

    if choice == "3":
        raw_tickers = input("Enter ticker(s), separated by comma or space: ").strip()
        parsed = _parse_ticker_input(raw_tickers)
        if parsed:
            return parsed
        logger.warning("No valid ticker entered. Falling back to default tickers.")
        return default_tickers

    return default_tickers


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="FinAgent",
        description="AI-Powered Financial Data Agent - end-to-end pipeline runner.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="One or more stock ticker symbols. If omitted, terminal will prompt.",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        metavar="YYYY-MM-DD",
        help="Historical data start date (default: 18 months ago).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        metavar="YYYY-MM-DD",
        help="Historical data end date (default: yesterday).",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["gemini", "anthropic", "openai"],
        help="LLM provider for AI analysis (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip the AI analysis stage (useful for offline testing).",
    )
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=["daily", "weekly", "monthly", "yearly", "all"],
        help="Chart timeframe for auto-visualization export (default: %(default)s).",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_collection(tickers: list[str], start: str, end: str) -> dict:
    """Stage 1 - collect raw data from all configured sources."""
    logger.info("Stage 1: Data Collection")
    collector = DataCollector(tickers=tickers, start_date=start, end_date=end)

    raw_data = {
        "prices"      : collector.fetch_stock_prices(),
        "benchmark"   : collector.fetch_benchmark(),
        "peers"       : collector.fetch_peers(),
        "fundamental" : collector.fetch_financial_statements(),
        "macro"       : collector.fetch_macro_indicators(),
        "industry"    : collector.fetch_industry_data(),
        "news"        : collector.fetch_news(query=" ".join(tickers)),
    }
    logger.info("Data collection complete - %d ticker(s) collected.", len(tickers))
    return raw_data



def build_processors(raw_data: dict) -> dict:
    """
    Wrapper
    Returns dict:
        prices      : { ticker: cleaned_df }
        benchmark   : cleaned_df
        peers       : { ticker: cleaned_df }
        fundamental : { ticker: cleaned_df }
        macro       : cleaned_df
        industry    : cleaned_df
    """
    logger.info("Wrapper: build_processors() - bat dau clean toan bo data")
    processed = {}

    # 1. Price data
    processed["prices"] = {}
    for ticker, df in raw_data.get("prices", {}).items():
        if df is not None and not df.empty:
            logger.info("  Processing price: %s", ticker)
            processed["prices"][ticker] = DataProcessor(df=df, ticker=ticker).run_pipeline()

    # 2. Benchmark
    bm_df = raw_data.get("benchmark")
    if bm_df is not None and not bm_df.empty:
        logger.info("  Processing benchmark")
        processed["benchmark"] = DataProcessor(df=bm_df, ticker="benchmark").run_pipeline()

    # 3. Peer data
    processed["peers"] = {}
    for ticker, df in raw_data.get("peers", {}).items():
        if df is not None and not df.empty:
            logger.info("  Processing peer: %s", ticker)
            processed["peers"][ticker] = DataProcessor(df=df, ticker=ticker).run_pipeline()

    # 4. News - clean, encode sentiment, save
    news_df = raw_data.get("news")
    if news_df is not None and not news_df.empty:
        logger.info("  Processing news/sentiment data")
        p = DataProcessor(df=news_df, ticker="news")
        p.process_news()
        processed["news"] = p.df
        p._save_csv(filename="news_processed.csv")
    elif news_df is not None:
        processed["news"] = pd.DataFrame(
            columns=[
                "date", "ticker", "article_count", "headline", "summary", "source",
                "sentiment", "sentiment_score", "positive_count", "neutral_count",
                "negative_count", "event_type",
            ]
        )
        DataProcessor(df=processed["news"], ticker="news")._save_csv(filename="news_processed.csv")

    # 5. Fundamental data - clean, normalise, enrich, save
    processed["fundamental"] = {}
    for ticker, df in raw_data.get("fundamental", {}).items():
        if df is not None and not df.empty:
            logger.info("  Processing fundamental: %s", ticker)
            p = DataProcessor(df=df, ticker=ticker)
            p.normalise_types()
            p.remove_duplicates()
            p.handle_missing_values(strategy="ffill")
            p.engineer_fundamental_features(news_df=processed.get("news"))
            processed["fundamental"][ticker] = p.df
            p._save_csv(filename=f"{ticker}_fundamental_processed.csv")

    # 6. Macro - clean, normalise, save
    macro_df = raw_data.get("macro")
    if macro_df is not None and not macro_df.empty:
        logger.info("  Processing macro indicators")
        p = DataProcessor(df=macro_df, ticker="macro")
        p.normalise_types()
        p.remove_duplicates()
        p.handle_missing_values(strategy="ffill")
        processed["macro"] = p.df
        p._save_csv(filename="macro_processed.csv")

    # 7. Industry - clean, normalise, save
    industry_df = raw_data.get("industry")
    if industry_df is not None and not industry_df.empty:
        logger.info("  Processing industry data")
        p = DataProcessor(df=industry_df, ticker="industry")
        p.normalise_types()
        p.remove_duplicates()
        p.handle_missing_values(strategy="ffill")
        processed["industry"] = p.df
        p._save_csv(filename="industry_processed.csv")

    logger.info("Wrapper: build_processors() hoan tat.")
    return processed


def run_processing(raw_data: dict) -> dict:
    """Stage 2 - clean, normalise, va engineer features qua wrapper build_processors."""
    logger.info("Stage 2: Data Processing")

    processed_data = build_processors(raw_data)
    benchmark_df = processed_data.get("benchmark")
    peer_frames = processed_data.get("peers", {})
    price_frames = processed_data.get("prices", {})

    def enrich_and_save(collection_key: str, ticker: str, df, comparison_frames: list) -> None:
        processor = DataProcessor(df=df, ticker=ticker)
        processor.df = df.copy()

        if benchmark_df is not None:
            processor.calc_beta(benchmark_df)
            processor.calc_relative_strength(benchmark_df)

        if comparison_frames:
            processed_data.setdefault("correlation", {})[ticker] = processor.calc_correlation_matrix(comparison_frames)

        processor._save_csv()
        processed_data[collection_key][ticker] = processor.df

    for ticker, df in price_frames.items():
        comparison_frames = []
        if benchmark_df is not None:
            comparison_frames.append(benchmark_df)
        comparison_frames.extend(other_df for other_ticker, other_df in peer_frames.items() if other_ticker != ticker)
        enrich_and_save("prices", ticker, df, comparison_frames)

    for ticker, df in peer_frames.items():
        comparison_frames = []
        if benchmark_df is not None:
            comparison_frames.append(benchmark_df)
        comparison_frames.extend(other_df for other_ticker, other_df in peer_frames.items() if other_ticker != ticker)
        comparison_frames.extend(other_df for other_ticker, other_df in price_frames.items() if other_ticker != ticker)
        enrich_and_save("peers", ticker, df, comparison_frames)

    if benchmark_df is not None:
        benchmark_processor = DataProcessor(df=benchmark_df, ticker="benchmark")
        benchmark_processor.df = benchmark_df.copy()
        benchmark_processor.df["beta"] = 1.0
        benchmark_processor.df["relative_strength"] = 1.0
        all_null_columns = [
            column
            for column in benchmark_processor.df.columns
            if benchmark_processor.df[column].isna().all()
        ]
        if all_null_columns:
            logger.info("Dropping benchmark all-null columns before save: %s", all_null_columns)
            benchmark_processor.df = benchmark_processor.df.drop(columns=all_null_columns)
        benchmark_processor._save_csv()
        processed_data["benchmark"] = benchmark_processor.df

    logger.info("Processing complete.")
    return processed_data


def run_visualisation(processed_data: dict, timeframe: str = "daily") -> None:
    """Stage 3 - auto export charts only for user-requested tickers (prices)."""
    logger.info("Stage 3: Visualisation")

    chart_frames = {}

    for ticker, df in processed_data.get("prices", {}).items():
        if df is not None and not df.empty:
            chart_frames[ticker] = df

    if not chart_frames:
        logger.warning("No chart-ready price frames found. Skipping visualisation stage.")
        return

    visualizer = DataVisualizer(data=chart_frames)
    visualizer.render_all(timeframe=timeframe, chart_type="candlestick", include_rolling=True)
    logger.info("Visualisation complete.")


def generate_checklist_report(processed_data: dict) -> None:
    """Stage 5 - auto pass/fail checklist based on processed data columns and row counts."""

    # ---------------------------------------------------------------
    # Define expected columns per data category
    # ---------------------------------------------------------------
    _PRICE_COLS = [
        "open", "high", "low", "close", "volume",
        "rsi_14", "macd_line", "macd_signal", "atr_14",
        "sharpe_ratio", "beta", "relative_strength",
        "stoch_k", "stoch_d", "adx_14", "williams_r_14", "cci_14",
        "ultimate_oscillator", "roc_12",
        "pivot", "pivot_s1", "pivot_r1",
    ]
    _FUNDAMENTAL_COLS = [
        "current_ratio", "quick_ratio", "cash_ratio",
        "debt_to_assets", "debt_to_capital", "financial_leverage",
        "receivable_turnover", "inventory_turnover", "cash_conversion_cycle",
        "fcf", "fcff", "fcfe",
        "gross_profit_margin", "net_profit_margin", "roe", "roa",
        "pe", "pb",
    ]
    _MACRO_COLS = [
        "fed_funds_rate", "us_10y_yield", "us_cpi",
        "gdp", "unemployment_rate",
        "dxy", "gold_price", "oil_price",
        "fx_rate", "domestic_interest_rate", "fdi_inflow",
    ]
    _INDUSTRY_COLS = ["industry_roe", "industry_margin", "industry_pe", "industry_pb"]
    _NEWS_COLS     = ["sentiment", "sentiment_score", "event_type"]

    results = []   # list of (section, item, status, note)

    def check_df(label: str, df, required_cols: list[str], min_rows: int = 1) -> None:
        if df is None or (hasattr(df, "empty") and df.empty):
            results.append((label, "dataframe", "FAIL", "empty or None"))
            return
        row_count = len(df)
        results.append((label, "row_count", "PASS" if row_count >= min_rows else "FAIL",
                         f"{row_count} rows"))
        for col in required_cols:
            present   = col in df.columns
            non_null  = int(df[col].notna().sum()) if present else 0
            pct       = round(100 * non_null / row_count, 1) if row_count else 0
            status    = "PASS" if present and non_null > 0 else ("WARN" if present else "FAIL")
            note      = f"{pct}% non-null" if present else "missing column"
            results.append((label, col, status, note))

    # ---------------------------------------------------------------
    # Run checks
    # ---------------------------------------------------------------
    prices_dict = processed_data.get("prices", {})
    for ticker, df in prices_dict.items():
        check_df(f"price:{ticker}", df, _PRICE_COLS, min_rows=100)

    fundamental_dict = processed_data.get("fundamental", {})
    for ticker, df in fundamental_dict.items():
        check_df(f"fundamental:{ticker}", df, _FUNDAMENTAL_COLS, min_rows=1)

    check_df("macro",    processed_data.get("macro"),    _MACRO_COLS,    min_rows=10)
    check_df("industry", processed_data.get("industry"), _INDUSTRY_COLS, min_rows=1)
    check_df("news",     processed_data.get("news"),     _NEWS_COLS,     min_rows=1)

    # ---------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------
    pass_n = sum(1 for _, _, s, _ in results if s == "PASS")
    warn_n = sum(1 for _, _, s, _ in results if s == "WARN")
    fail_n = sum(1 for _, _, s, _ in results if s == "FAIL")

    separator = "-" * 72
    logger.info(separator)
    logger.info("CHECKLIST REPORT   PASS:%d  WARN:%d  FAIL:%d  (total %d checks)",
                pass_n, warn_n, fail_n, len(results))
    logger.info(separator)
    logger.info("%-28s %-26s %-6s %s", "SECTION", "ITEM", "STATUS", "NOTE")
    logger.info(separator)
    for section, item, status, note in results:
        if status != "PASS":   # only print non-pass rows to keep log concise
            logger.info("%-28s %-26s %-6s %s", section[:28], item[:26], status, note)
    logger.info(separator)
    if fail_n == 0 and warn_n == 0:
        logger.info("All checks PASSED.")
    else:
        logger.info("Review WARN/FAIL rows above. WARNs = column present but all NaN.")
    logger.info(separator)


def run_ai_analysis(processed_data: dict, provider: str) -> dict[str, str]:
    """Stage 4 - LLM-powered narrative analysis."""
    logger.info("Stage 4: AI Analysis (provider=%s)", provider)
    agent = AIAgent(provider=provider)
    # TODO: reports = agent.run_full_analysis(processed_data)
    reports = {}  # placeholder
    logger.info("AI analysis complete.")
    return reports


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.tickers = [ticker.upper() for ticker in args.tickers] if args.tickers else prompt_tickers_from_terminal(DEFAULT_TICKERS)

    logger.info("FinAgent pipeline starting.")
    logger.info("Tickers : %s", args.tickers)
    logger.info("Period  : %s to %s", args.start, args.end)
    logger.info("Provider: %s", args.provider)
    logger.info("Charts  : timeframe=%s", args.timeframe)

    try:
        raw_data       = run_collection(args.tickers, args.start, args.end)
        processed_data = run_processing(raw_data)
        run_visualisation(processed_data, timeframe=args.timeframe)

        if not args.skip_ai:
            reports = run_ai_analysis(processed_data, args.provider)
            for section, content in reports.items():
                logger.info("[AI] %s:\n%s", section.upper(), content)

        generate_checklist_report(processed_data)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed with an unexpected error: %s", exc)
        sys.exit(1)

    logger.info("FinAgent pipeline finished successfully.")


if __name__ == "__main__":
    main()