"""
processor.py
------------
Handles all data cleaning, normalisation, and feature engineering steps
required before visualisation or AI analysis.

Responsibilities:
  - Missing value handling  : forward-fill, interpolation, or documented drop
  - Duplicate detection     : flag and remove with audit logging
  - Type normalisation      : dates ??' DatetimeIndex, currencies ??' float
  - Outlier detection       : IQR / Z-score flagging (stock splits, data errors)
  - Feature engineering     : daily returns, rolling averages (7d, 30d), volatility

Processed artefacts are persisted to data/processed/.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataProcessor:
    """
    Cleans, normalises, and enriches raw financial DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV (or similar) DataFrame produced by DataCollector.
    ticker : str
        Ticker symbol associated with the DataFrame (used in logging & filenames).
    """

    def __init__(self, df: pd.DataFrame, ticker: str) -> None:
        self.df = df.copy()
        self.ticker = ticker

    # ------------------------------------------------------------------
    # Missing Values
    # ------------------------------------------------------------------

    def handle_missing_values(self, strategy: str = "ffill") -> "DataProcessor":
        """
        Impute or remove missing values according to the chosen strategy.

        Parameters
        ----------
        strategy : {'ffill', 'interpolate', 'drop'}
            - 'ffill'       : propagate last valid observation forward.
            - 'interpolate' : linear interpolation between adjacent values.
            - 'drop'        : remove rows containing any NaN.

        Returns
        -------
        DataProcessor
            Self, for method chaining.
        """
        # TODO: implement strategy branching with before/after NaN count logging
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Duplicates
    # ------------------------------------------------------------------

    def remove_duplicates(self) -> "DataProcessor":
        """
        Detect and drop duplicate rows, logging the number of records removed.

        Returns
        -------
        DataProcessor
            Self, for method chaining.
        """
        # TODO: df.duplicated() ??' log count ??' df.drop_duplicates()
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Type Normalisation
    # ------------------------------------------------------------------

    def normalise_types(self) -> "DataProcessor":
        """
        Ensure correct dtypes across the DataFrame:
          - Index converted to pd.DatetimeIndex (UTC-aware).
          - Numeric columns cast to float64.
          - Currency strings (e.g. '$1,234.56') stripped and converted.

        Returns
        -------
        DataProcessor
            Self, for method chaining.
        """
        # TODO: pd.to_datetime on index, pd.to_numeric on price/volume cols
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Outlier Detection
    # ------------------------------------------------------------------

    def detect_outliers(self, method: str = "iqr", threshold: float = 3.0) -> "DataProcessor":
        """
        Flag anomalous values (e.g. caused by stock splits or data errors).

        Parameters
        ----------
        method : {'iqr', 'zscore'}
            Statistical method used for detection.
        threshold : float
            IQR multiplier or Z-score cutoff.

        Returns
        -------
        DataProcessor
            Self, for method chaining.

        Notes
        -----
        Flagged rows are marked in a boolean column ``is_outlier`` rather than
        being silently dropped, preserving data integrity for downstream review.
        """
        # TODO: implement IQR / Z-score logic; add 'is_outlier' boolean column
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def engineer_features(self) -> "DataProcessor":
        """
        Compute derived features required by the visualisation and AI modules:

          - ``daily_return``      : percentage change in closing price day-over-day.
          - ``rolling_avg_7``     : 7-day simple moving average of Close.
          - ``rolling_avg_30``    : 30-day simple moving average of Close.
          - ``volatility_30``     : 30-day rolling standard deviation of daily returns.
          - ``cum_return``        : cumulative return indexed from the start date.

        Returns
        -------
        DataProcessor
            Self, for method chaining.
        """
        # TODO: use df['Close'].pct_change(), rolling().mean(), rolling().std()
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the full cleaning and feature engineering pipeline in order.

        Order of operations:
          1. normalise_types
          2. remove_duplicates
          3. handle_missing_values
          4. detect_outliers
          5. engineer_features

        Returns
        -------
        pd.DataFrame
            Fully processed DataFrame, also saved to data/processed/.
        """
        # TODO: chain all steps, call _save_csv at the end
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------

    def _save_csv(self, filename: str | None = None) -> Path:
        """
        Save the current state of self.df to data/processed/.

        Parameters
        ----------
        filename : str, optional
            Target filename; defaults to '<ticker>_processed.csv'.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        filename = filename or f"{self.ticker}_processed.csv"
        filepath = PROCESSED_DATA_DIR / filename
        self.df.to_csv(filepath)
        logger.info("Saved processed data ??' %s", filepath)
        return filepath

