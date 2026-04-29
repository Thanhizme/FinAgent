"""
main.py
-------
FinAgent ??" AI-Powered Financial Data Agent
Entry point and workflow orchestrator.

Pipeline stages
---------------
  1. Data Collection   : fetch stock prices, financials, news, macro indicators
  2. Data Processing   : clean, normalise, and engineer features
  3. Visualisation     : generate all four required chart types
  4. AI Analysis       : produce LLM-powered narrative reports

Usage
-----
  python main.py                         # run full pipeline with defaults
  python main.py --tickers AAPL MSFT     # specify tickers
  python main.py --start 2023-01-01      # override start date
  python main.py --provider openai       # select LLM provider

Environment
-----------
  Copy .env.example ??' .env and fill in your API keys before running.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

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

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA"]
DEFAULT_START = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
DEFAULT_END = datetime.today().strftime("%Y-%m-%d")
DEFAULT_PROVIDER = "anthropic"


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="FinAgent",
        description="AI-Powered Financial Data Agent ??" end-to-end pipeline runner.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        metavar="TICKER",
        help="One or more stock ticker symbols (default: %(default)s).",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        metavar="YYYY-MM-DD",
        help="Historical data start date (default: 1 year ago).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        metavar="YYYY-MM-DD",
        help="Historical data end date (default: today).",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["anthropic", "openai"],
        help="LLM provider for AI analysis (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip the AI analysis stage (useful for offline testing).",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_collection(tickers: list[str], start: str, end: str) -> dict:
    """Stage 1 ??" collect raw data from all configured sources."""
    logger.info("=== Stage 1: Data Collection ===")
    collector = DataCollector(tickers=tickers, start_date=start, end_date=end)

    # TODO: call collector methods and store results
    # raw_prices = collector.fetch_stock_prices()
    # raw_news   = collector.fetch_news(query=" ".join(tickers))
    # raw_macro  = collector.fetch_macro_indicators(["FX_DAILY"])

    raw_data = {}  # placeholder
    logger.info("Data collection complete ??" %d ticker(s) collected.", len(tickers))
    return raw_data


def run_processing(raw_data: dict) -> dict[str, any]:
    """Stage 2 ??" clean, normalise, and engineer features."""
    logger.info("=== Stage 2: Data Processing ===")
    processed_data = {}

    for ticker, df in raw_data.items():
        logger.info("Processing ticker: %s", ticker)
        processor = DataProcessor(df=df, ticker=ticker)
        # TODO: processed_data[ticker] = processor.run_pipeline()

    logger.info("Processing complete ??" %d ticker(s) processed.", len(processed_data))
    return processed_data


def run_visualisation(processed_data: dict) -> None:
    """Stage 3 ??" generate all four chart types."""
    logger.info("=== Stage 3: Visualisation ===")
    visualizer = DataVisualizer(data=processed_data)
    # TODO: visualizer.render_all()
    logger.info("Visualisation complete.")


def run_ai_analysis(processed_data: dict, provider: str) -> dict[str, str]:
    """Stage 4 ??" LLM-powered narrative analysis."""
    logger.info("=== Stage 4: AI Analysis (provider=%s) ===", provider)
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

    logger.info("FinAgent pipeline starting.")
    logger.info("Tickers : %s", args.tickers)
    logger.info("Period  : %s ??' %s", args.start, args.end)
    logger.info("Provider: %s", args.provider)

    try:
        raw_data = run_collection(args.tickers, args.start, args.end)
        processed_data = run_processing(raw_data)
        run_visualisation(processed_data)

        if not args.skip_ai:
            reports = run_ai_analysis(processed_data, args.provider)
            for section, content in reports.items():
                logger.info("[AI] %s:\n%s", section.upper(), content)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed with an unexpected error: %s", exc)
        sys.exit(1)

    logger.info("FinAgent pipeline finished successfully.")


if __name__ == "__main__":
    main()

