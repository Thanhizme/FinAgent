"""
collector.py
------------
Responsible for acquiring raw financial data from external APIs and web source
Sources supported:
  - Stock prices         : yfinance (Yahoo Finance wrapper)
  - Financial statements : yfinance quarterly/annual financials
  - News & sentiment     : NewsAPI

All fetched data is persisted to data/raw/ as CSV or JSON files.
"""

import os
import json
import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataCollector:
    """
    Collects financial data from multiple sources and saves raw files to disk.

    Parameters
    ----------
    tickers : list[str]
        List of stock ticker symbols (e.g. ['AAPL', 'MSFT']).
    start_date : str
        Start date for historical data in 'YYYY-MM-DD' format.
    end_date : str
        End date for historical data in 'YYYY-MM-DD' format.
    """

    def __init__(self, tickers: list[str], start_date: str, end_date: str) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.news_api_key = os.getenv("NEWS_API_KEY")

    # ------------------------------------------------------------------
    # Stock Prices
    # ------------------------------------------------------------------

    def fetch_stock_prices(self) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV stock price history for all tickers via yfinance.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker symbol -> OHLCV DataFrame.
        """
        data_map = {}
        for ticker in self.tickers:                          
            logger.info("Fetching price data for %s ...", ticker)
            for attempt in range(1, 4):
                try:
                    df = yf.download(
                        ticker,
                        start=self.start_date,
                        end=self.end_date,
                        auto_adjust=True,
                        progress=False,
                    )
                    if not df.empty:
                        data_map[ticker] = df
                        self._save_csv(df, f"{ticker}_prices.csv")
                        logger.info("Fetched %d rows for %s.", len(df), ticker)
                    else:
                        logger.warning("No price data returned for %s.", ticker)
                    break
                except Exception as e:
                    logger.error(
                        "Attempt %d/3 failed for %s: %s", attempt, ticker, e
                    )
                    if attempt < 3:
                        time.sleep(3)
        return data_map

    # ------------------------------------------------------------------
    # Financial Statements
    # ------------------------------------------------------------------

    def fetch_financial_statements(self) -> dict[str, dict]:
        """
        Retrieve quarterly income statements, balance sheets, and cash flow
        statements for each ticker via yfinance.

        Returns
        -------
        dict[str, dict]
            Nested mapping: ticker -> {'income': df, 'balance': df, 'cashflow': df}
        """
        statements = {}
        for ticker in self.tickers:
            logger.info("Fetching financial statements for %s ...", ticker)
            t = yf.Ticker(ticker)
            statements[ticker] = {
                "income":   t.quarterly_financials,
                "balance":  t.quarterly_balance_sheet,
                "cashflow": t.quarterly_cashflow,
            }
            for key, df in statements[ticker].items():
                if df is not None and not df.empty:          # FIX 3: df.empty() -> df.empty
                    self._save_csv(df, f"{ticker}_{key}_statement.csv")
        return statements

    # ------------------------------------------------------------------
    # News & Sentiment
    # ------------------------------------------------------------------

    def fetch_news(self, query: str, page_size: int = 20) -> list[dict]:
        """
        Pull the latest financial news articles from NewsAPI.

        Parameters
        ----------
        query : str
            Search keywords (e.g. 'Apple stock earnings').
        page_size : int
            Number of articles to retrieve (max 100 on free tier).

        Returns
        -------
        list[dict]
            List of article metadata dicts.
        """
        if not self.news_api_key:
            logger.error("NEWS_API_KEY is not set in .env file.")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        query,
            "pageSize": page_size,
            "apiKey":   self.news_api_key,
            "language": "en",
            "sortBy":   "publishedAt",
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            # FIX 4: define filepath before using it
            safe_query = query[:30].replace(" ", "_")
            filepath = RAW_DATA_DIR / f"news_{safe_query}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)

            logger.info("Fetched %d articles for query '%s'.", len(articles), query)
            return articles

        except requests.exceptions.HTTPError as e:
            logger.error("NewsAPI HTTP error: %s", e)
            return []
        except Exception as e:
            logger.error("NewsAPI unexpected error: %s", e)
            return []

    # ------------------------------------------------------------------
    # Macro Indicators (reserved for future use)
    # ------------------------------------------------------------------

    # def fetch_macro_indicators(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
    #     """
    #     Fetch macroeconomic indicators (exchange rates, commodities) via
    #     Alpha Vantage REST API.
    #
    #     Parameters
    #     ----------
    #     symbols : list[str]
    #         Alpha Vantage function symbols, e.g. ['FX_DAILY', 'WTI'].
    #
    #     Returns
    #     -------
    #     dict[str, pd.DataFrame]
    #         Mapping of symbol -> time-series DataFrame.
    #     """
    #     # TODO: call Alpha Vantage endpoints, respect rate limits with time.sleep
    #     raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------

    def _save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Persist a DataFrame to data/raw/ as a CSV file.

        Parameters
        ----------
        df : pd.DataFrame
        filename : str
            Target filename (e.g. 'AAPL_prices.csv').

        Returns
        -------
        Path
            Absolute path to the saved file, or None on failure.
        """
        if df is None or df.empty:
            logger.warning("Skipping save - DataFrame is empty: %s", filename)
            return None
        filepath = RAW_DATA_DIR / filename
        try:
            df.to_csv(filepath, index=True)
            logger.info("Saved raw data -> %s", filepath)
            return filepath
        except Exception as e:
            logger.error("Error saving file %s: %s", filename, e)
            return Nonegger.info(f"Saved raw data in: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error while saving file {filename}: {e}")
            return None

