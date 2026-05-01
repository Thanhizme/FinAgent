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
        nan_before = self.df.isna().sum()
        logger.info(f"Number of Nan values before cleaning: \n{nan_before[nan_before > 0]}")
        if strategy == "ffill":
            self.df = self.df.ffill().bfill()
        elif strategy == "interpolate":
            self.df = self.df.interpolate(method='linear').bfill().ffill()
        elif strategy == "drop":
            self.df = self.df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        nan_after = self.df.isna().sum()
        logger.info(f"Number of Nan values after cleaning:\n{nan_after[nan_after > 0]}")
        filled = (nan_before - nan_after).sum()
        logger.info(f"Total cells filled: {filled}")
        return self        

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
        dup_count = self.df.duplicated().sum()
        logger.info(f"Number of duplicated rows: {dup_count}")
        self.df = self.df.drop_duplicates(subset=['date', 'ticker'])
        self.df = self.df.reset_index(drop=True)
        logger.info(f"Removed {dup_count} duplicate rows")
        return self

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
        def parse_currency(val):
            if not isinstance(val, str):
                return val
            val =val.strip()
            if val.startswith('$'): #usd
                return val.replace('$', '').replace(',', '')
            if ',' in val and '.' in val: #vnd
                val = val.replace(' VND', '').replace('.', '').replace(',', '.')
            return val
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        self.df = self.df.reset_index(drop=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for c in numeric_cols:
            if c not in self.df.columns:
                continue
            if self.df[c].dtype == object:
                self.df[c] = self.df[c].apply(parse_currency)
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce').astype('float64')
        
        logger.info('[%s] dtypes after normalise:\n%s', self.ticker, self.df.dtypes.to_string())
        return self

        

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
        if method == "iqr":
            q1 = self.df['close'].quantile(0.25)
            q3 = self.df['close'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold*iqr
            upper_bound = q3 + threshold*iqr
            self.df['is_outlier'] = (self.df['close'] < lower_bound) | (self.df['close'] > upper_bound)
            outlier_count = self.df['is_outlier'].sum()
            logger.info(f"[{self.ticker}] Found {outlier_count} outliers using {method}")
            return self
        elif method == "zscore":
            raise ValueError(f"Method will be implemented later")
        else:
            raise ValueError(f"Unknown method: {method}")

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
    # A. Return & Momentum
    # ------------------------------------------------------------------

    def calc_returns(self) -> "DataProcessor":
        # TODO: daily_return = close.pct_change()
        # TODO: log_return = np.log(close / close.shift(1))
        raise NotImplementedError

    def calc_cumulative_returns(self) -> "DataProcessor":
        # TODO: cum_return_1w, cum_return_1m, cum_return_3m, cum_return_ytd
        # Yeu cau: calc_returns() chay truoc
        raise NotImplementedError

    # ------------------------------------------------------------------
    # B. Moving Averages
    # ------------------------------------------------------------------

    def calc_moving_averages(self) -> "DataProcessor":
        # TODO: ma7, ma20, ma30, ma50, ma200 = close.rolling(N).mean()
        raise NotImplementedError

    # ------------------------------------------------------------------
    # C. Volatility & Bollinger Bands
    # ------------------------------------------------------------------

    def calc_volatility(self) -> "DataProcessor":
        # TODO: volatility_20, volatility_60 = daily_return.rolling(N).std()
        # Yeu cau: calc_returns() chay truoc
        raise NotImplementedError

    def calc_bollinger_bands(self) -> "DataProcessor":
        # TODO: bb_middle=ma20, bb_upper=ma20+2*std20, bb_lower=ma20-2*std20
        # Yeu cau: calc_moving_averages() chay truoc
        raise NotImplementedError

    # ------------------------------------------------------------------
    # D. Performance & Risk
    # ------------------------------------------------------------------

    def calc_max_drawdown(self) -> "DataProcessor":
        # TODO: rolling_max = close.cummax()
        # TODO: drawdown = (close - rolling_max) / rolling_max
        raise NotImplementedError

    def calc_sharpe_ratio(self, trading_days: int = 252) -> "DataProcessor":
        # TODO: sharpe = mean(daily_return) / std(daily_return) * sqrt(trading_days)
        # Yeu cau: calc_returns() chay truoc
        raise NotImplementedError

    def calc_beta(self, benchmark_df) -> "DataProcessor":
        # TODO: beta = Cov(r_stock, r_market) / Var(r_market)
        # benchmark_df phai co cot daily_return
        # Yeu cau: calc_returns() chay truoc
        raise NotImplementedError

    # ------------------------------------------------------------------
    # E. Multi-Asset
    # ------------------------------------------------------------------

    def calc_correlation_matrix(self, other_dfs: list):
        # TODO: merge tat ca daily_return theo date -> df.corr()
        # Tra ve DataFrame rieng (khong them vao self.df)
        raise NotImplementedError

    def calc_relative_strength(self, other_df) -> "DataProcessor":
        # TODO: relative_strength = cum_return_self / cum_return_other
        # other_df phai co cot close va ticker
        # Yeu cau: calc_cumulative_returns() chay truoc
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

