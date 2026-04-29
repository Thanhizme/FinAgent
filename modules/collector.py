"""
collector.py
------------
Responsible for acquiring raw financial data from external APIs and web sources.

Sources supported (at minimum two required):
  - Stock prices          : yfinance (Yahoo Finance wrapper)
  - Financial statements  : yfinance quarterly/annual financials
  - News & sentiment      : NewsAPI
  - Macro indicators      : Alpha Vantage (exchange rates, commodities)

All fetched data is persisted to data/raw/ as CSV or JSON files.
"""

import os
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
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    # ------------------------------------------------------------------
    # Stock Prices
    # ------------------------------------------------------------------

    def fetch_stock_prices(self) -> dict[str, pd.DataFrame]:
        """
        Download OHLCV stock price history for all tickers via yfinance.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker symbol ??' OHLCV DataFrame.
        """
        # TODO: implement yfinance download loop with retry logic
        raise NotImplementedError

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
            Nested mapping: ticker ??' {'income': df, 'balance': df, 'cashflow': df}
        """
        # TODO: iterate tickers, call yf.Ticker(t).quarterly_financials etc.
        raise NotImplementedError

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
        # TODO: GET https://newsapi.org/v2/everything with self.news_api_key
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Macro Indicators
    # ------------------------------------------------------------------

    def fetch_macro_indicators(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """
        Fetch macroeconomic indicators (exchange rates, commodities) via
        Alpha Vantage REST API.

        Parameters
        ----------
        symbols : list[str]
            Alpha Vantage function symbols, e.g. ['FX_DAILY', 'WTI'].

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of symbol ??' time-series DataFrame.
        """
        # TODO: call Alpha Vantage endpoints, respect rate limits with time.sleep
        raise NotImplementedError

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
            Absolute path to the saved file.
        """
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath)
        logger.info("Saved raw data ??' %s", filepath)
        return filepath

