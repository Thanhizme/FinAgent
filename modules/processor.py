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
            dup_count = self.df.duplicated().sum()
            logger.info(f"Number of duplicated rows: {dup_count}")
            
            # Chỉ dùng subset các cột thực sự tồn tại trong DataFrame
            possible_subset = ['date', 'ticker']
            subset = [c for c in possible_subset if c in self.df.columns]
            self.df = self.df.drop_duplicates(subset=subset if subset else None)
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
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['rolling_avg_7'] = self.df['close'].rolling(window=7).mean()
        self.df['rolling_avg_30'] = self.df['close'].rolling(window=30).mean()
        self.df['volatility_30'] = self.df['daily_return'].rolling(window=30).std()
        self.df['cum_return'] = self.df['close']/self.df['close'].iloc[0] - 1
        logger.info("[%s] engineer_features done | cols added: daily_return, rolling_avg_7, rolling_avg_30, volatility_30, cum_return", self.ticker)
        return self
        
        


    # ------------------------------------------------------------------
    # A. Return & Momentum
    # ------------------------------------------------------------------

    def calc_returns(self) -> "DataProcessor":
        # TODO: daily_return = close.pct_change()
        # TODO: log_return = np.log(close / close.shift(1))
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        logger.info(f"[{self.ticker}] calc_returns done | daily_return mean= {self.df['daily_return'].mean()} | log_return = {self.df['log_return'].mean()}")
        return self

    def calc_cumulative_returns(self) -> "DataProcessor":
        # TODO: cum_return_1w, cum_return_1m, cum_return_3m, cum_return_ytd
        # Yeu cau: calc_returns() chay truoc
        if 'daily_return' not in self.df.columns:
            raise RuntimeError("calc_returns() must be called before calc_cumulative_returns()")
        self.df['cum_return_7'] = self.df['close']/self.df['close'].shift(7) - 1
        self.df['cum_return_30'] = self.df['close']/self.df['close'].shift(30) - 1
        self.df['cum_return_90'] = self.df['close']/self.df['close'].shift(90) - 1
        year = self.df['date'].dt.year
        first_close_of_year = self.df.groupby(year)['close'].transform('first')
        self.df['cum_return_ytd'] = self.df['close'] / first_close_of_year - 1
        logger.info(f"[{self.ticker}] cum_return_7 mean={self.df['cum_return_7'].mean():.4f} | "
            f"cum_return_30 mean={self.df['cum_return_30'].mean():.4f} | "
            f"cum_return_90 mean={self.df['cum_return_90'].mean():.4f}")
        return self
    # ------------------------------------------------------------------
    # B. Moving Averages
    # ------------------------------------------------------------------

    def calc_moving_averages(self) -> "DataProcessor":
        # TODO: ma7, ma20, ma30, ma50, ma200 = close.rolling(N).mean()
        self.df['ma7'] = self.df['close'].rolling(window=7).mean()
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma30'] = self.df['close'].rolling(window=30).mean()
        self.df['ma50'] = self.df['close'].rolling(window=50).mean()
        self.df['ma200'] = self.df['close'].rolling(window=200).mean()
        logger.info(f"[{self.ticker}] calc_moving_averages done | ma7, ma20, ma30, ma50, ma200 added")
        return self

    # ------------------------------------------------------------------
    # C. Volatility & Bollinger Bands
    # ------------------------------------------------------------------

    def calc_volatility(self) -> "DataProcessor":
        # TODO: volatility_20, volatility_60 = daily_return.rolling(N).std()
        # Yeu cau: calc_returns() chay truoc
        self.df['volatility_20'] = self.df['daily_return'].rolling(window=20).std()
        self.df['volatility_60'] = self.df['daily_return'].rolling(window=60).std()
        return self

    def calc_bollinger_bands(self) -> "DataProcessor":
        # TODO: bb_middle=ma20, bb_upper=ma20+2*std20, bb_lower=ma20-2*std20
        # Yeu cau: calc_moving_averages() chay truoc
        if 'ma20' not in self.df.columns:
            raise RuntimeError("calc_moving_averages() must be called before calc_bollinger_bands()")
        std20 = self.df['close'].rolling(20).std()
        self.df['bb_middle'] = self.df['ma20']
        self.df['bb_upper'] = self.df['ma20'] + 2*std20
        self.df['bb_lower'] = self.df['ma20'] - 2*std20
        logger.info(f"[{self.ticker}] Bollinger Bands done | bb_upper mean={self.df['bb_upper'].mean():.4f} | bb_lower mean={self.df['bb_lower'].mean():.4f}")
        return self
    # ------------------------------------------------------------------
    # D. Performance & Risk
    # ------------------------------------------------------------------

    def calc_max_drawdown(self) -> "DataProcessor":
        # TODO: rolling_max = close.cummax()
        # TODO: drawdown = (close - rolling_max) / rolling_max
        rolling_max = self.df['close'].cummax()
        self.df['drawdown'] = (self.df['close'] - rolling_max)/rolling_max
        max_drawdown = self.df['drawdown'].min()
        self.df['max_drawdown'] = max_drawdown
        logger.info(f"[{self.ticker}] max_drawdown = {max_drawdown:.4f}")
        return self
    #
    def calc_sharpe_ratio(self, trading_days: int = 252) -> "DataProcessor":
        # TODO: sharpe = mean(daily_return) / std(daily_return) * sqrt(trading_days)
        # Yeu cau: calc_returns() chay truoc
        if 'daily_return' not in self.df.columns:
            raise RuntimeError(f"calc_returns() must be called before calc_sharpe_ratio()")
        sharpe = self.df['daily_return'].mean() / self.df['daily_return'].std() * np.sqrt(trading_days) 
        self.df['sharpe_ratio'] = sharpe
        logger.info(f"[{self.ticker}] sharpe_ratio = {sharpe:.4f}")
        return self

    def calc_beta(self, benchmark_df) -> "DataProcessor":
        # TODO: beta = Cov(r_stock, r_market) / Var(r_market)
        # benchmark_df phai co cot daily_return
        # Yeu cau: calc_returns() chay truoc
        if 'daily_return' not in self.df.columns:
            raise RuntimeError(f"calc_returns() must be called before calc_beta()")
        if 'daily_return' not in benchmark_df.columns:
            raise RuntimeError(f"calc_returns() must be called before calc_beta()")
        merged = pd.merge(
        self.df[['date', 'daily_return']],
        benchmark_df[['date', 'daily_return']],
        on='date',
        suffixes=('_stock', '_market')
        )
        cov = float(merged['daily_return_stock'].cov(merged['daily_return_market']))
        var = merged['daily_return_market'].var()
        beta = cov/var
        self.df['beta'] = beta
        logger.info(f"[{self.ticker}] beta = {beta:.4f}")
        return self


    # ------------------------------------------------------------------
    # E. Multi-Asset
    # ------------------------------------------------------------------

    def calc_correlation_matrix(self, other_dfs: list):
        # TODO: merge tat ca daily_return theo date -> df.corr()
        # Tra ve DataFrame rieng (khong them vao self.df)
        if 'daily_return' not in self.df.columns:
            raise RuntimeError(f"calc_returns() must be called before calc_correlation_matrix()")
        data_dict = {
            self.ticker: self.df.set_index('date')['daily_return']
        }
        for other_df in other_dfs:
            t = other_df['ticker'].iloc[0]
            data_dict[t] = other_df.set_index('date')['daily_return']
        combined_df = pd.DataFrame(data_dict)
        corr_matrix = combined_df.corr()
        logger.info("[%s] calc_correlation_matrix done | Compared with %d other tickers", 
                self.ticker, len(other_dfs))
        return corr_matrix
        
    def calc_relative_strength(self, other_df) -> "DataProcessor":
        # TODO: relative_strength = cum_return_self / cum_return_other
        # other_df phai co cot close va ticker
        # Yeu cau: calc_cumulative_returns() chay truoc
        if 'cum_return_7' not in self.df.columns:
            raise RuntimeError("calc_cumulative_returns() must be called before calc_relative_strength()")
    
        if 'close' not in other_df.columns:
            raise ValueError("other_df must contain 'close' column to calculate relative strength")
        cum_other = (other_df['close'] / other_df['close'].iloc[0]) - 1
        other_tmp = pd.DataFrame({
            'date': other_df['date'],
            'cum_other': cum_other
        })
        merged = pd.merge(self.df, other_tmp, on='date', how='left')
        merged['relative_strength'] = (1 + merged['cum_return']) / (1 + merged['cum_other'])
        self.df['relative_strength'] = merged['relative_strength']
        ticker_other = other_df['ticker'].iloc[0] if 'ticker' in other_df.columns else "Benchmark"
        logger.info("[%s] calc_relative_strength done | Compared with %s", self.ticker, ticker_other)
        return self

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
        # Step 1: Cleaning
        self.normalise_types()
        self.remove_duplicates()
        self.df = self.df.ffill()
        self.df = self.df.dropna()          # xoá ~200 row đầu chưa đủ dữ liệu rolling
        self.df = self.df.reset_index(drop=True)
        self.detect_outliers(method="iqr")

        # Step 2: Feature engineering (rolling windows sẽ tạo NaN đầu chuỗi)
        self.calc_returns()
        self.calc_cumulative_returns()
        self.calc_moving_averages()
        self.calc_volatility()
        self.calc_bollinger_bands()
        self.calc_max_drawdown()
        self.calc_sharpe_ratio()

        # Step 3: Pass 2 - fill NaN sinh ra bởi rolling windows (ma200, bb, v.v.)
        self.df = self.df.ffill()
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        target_start = pd.Timestamp.today() - pd.DateOffset(months=18)
        self.df = self.df[self.df['date'] >= target_start].reset_index(drop=True)

        # Step 4: Save to disk
        filepath = self._save_csv()
        return self.df

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
        self.df.to_csv(filepath, index = False)
        logger.info("Saved processed data ??' %s", filepath)
        return filepath

