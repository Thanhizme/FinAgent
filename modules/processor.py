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

Processed artefacts are persisted to data/processed/processed_data/.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_BETA_WINDOW = 60
ROLLING_SHARPE_WINDOW = 252


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
        
        ohlcv_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        for c in ohlcv_cols:
            if c not in self.df.columns:
                continue
            if self.df[c].dtype == object:
                self.df[c] = self.df[c].apply(parse_currency)
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce').astype('float64')

        # Auto-cast remaining numeric-like columns (fundamental, macro, industry, etc.)
        _TEXT_COLS = {'ticker', 'headline', 'summary', 'source', 'sentiment', 'event_type'}
        ohlcv_set = set(ohlcv_cols)
        for c in self.df.columns:
            if c == 'date' or c in _TEXT_COLS or c in ohlcv_set:
                continue
            if self.df[c].dtype == object:
                converted = pd.to_numeric(self.df[c], errors='coerce')
                if converted.notna().mean() > 0.3:
                    self.df[c] = converted.astype('float64')

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
        if 'close' not in self.df.columns:
            self.df['is_outlier'] = False
            logger.info("[%s] No close column available for outlier detection", self.ticker)
            return self

        if method == "iqr":
            change_series = self.df['daily_return'] if 'daily_return' in self.df.columns else self.df['close'].pct_change()
            valid_changes = change_series.dropna()

            if valid_changes.empty:
                self.df['is_outlier'] = False
                logger.info("[%s] No valid returns available for outlier detection", self.ticker)
                return self

            q1 = valid_changes.quantile(0.25)
            q3 = valid_changes.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            self.df['is_outlier'] = ((change_series < lower_bound) | (change_series > upper_bound)).fillna(False)
            outlier_count = int(self.df['is_outlier'].sum())
            logger.info(
                "[%s] Found %d outliers using %s on daily returns",
                self.ticker,
                outlier_count,
                method,
            )
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
    # C2. Momentum Oscillators & ATR (NEW)
    # ------------------------------------------------------------------

    def calc_momentum_oscillators(self) -> "DataProcessor":
        """Calculate RSI and MACD using the 'ta' library."""
        # 1. RSI (14-day)
        self.df['rsi_14'] = RSIIndicator(close=self.df['close'], window=14).rsi()
        
        # 2. MACD
        macd = MACD(close=self.df['close'], window_slow=26, window_fast=12, window_sign=9)
        self.df['macd_line'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_hist'] = macd.macd_diff()
        
        logger.info(f"[{self.ticker}] Momentum Oscillators done | RSI, MACD added")
        return self

    def calc_extended_oscillators(self) -> "DataProcessor":
        """Calculate extended oscillators requested by valuation checklist."""
        if 'close' not in self.df.columns:
            for col in [
                'stoch_k', 'stoch_d', 'stochrsi_14', 'adx_14', 'williams_r_14',
                'cci_14', 'ultimate_oscillator', 'roc_12', 'bull_power_13',
                'bear_power_13', 'highs_lows_14',
            ]:
                self.df[col] = np.nan
            return self

        high = self.df['high'] if 'high' in self.df.columns else self.df['close']
        low = self.df['low'] if 'low' in self.df.columns else self.df['close']
        close = self.df['close']

        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()

        stochrsi = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        self.df['stochrsi_14'] = stochrsi.stochrsi_k()

        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        self.df['adx_14'] = adx.adx()

        williams = WilliamsRIndicator(high=high, low=low, close=close, lbp=14)
        self.df['williams_r_14'] = williams.williams_r()

        cci = CCIIndicator(high=high, low=low, close=close, window=14, constant=0.015)
        self.df['cci_14'] = cci.cci()

        # Ultimate oscillator implementation with standard weights (7,14,28)
        prev_close = close.shift(1)
        min_low_prev = pd.concat([low, prev_close], axis=1).min(axis=1)
        max_high_prev = pd.concat([high, prev_close], axis=1).max(axis=1)
        bp = close - min_low_prev
        tr = max_high_prev - min_low_prev
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)
        self.df['ultimate_oscillator'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

        roc = ROCIndicator(close=close, window=12)
        self.df['roc_12'] = roc.roc()

        ema13 = close.ewm(span=13, adjust=False).mean()
        self.df['bull_power_13'] = high - ema13
        self.df['bear_power_13'] = low - ema13

        self.df['highs_lows_14'] = close - high.rolling(14).max()

        logger.info(f"[{self.ticker}] Extended oscillators done | STOCH/STOCHRSI/ADX/Williams/CCI/UO/ROC/BullBear")
        return self

    def calc_pivot_points(self) -> "DataProcessor":
        """Calculate classic daily pivot points using previous period high/low/close."""
        needed = {'high', 'low', 'close'}
        if not needed.issubset(self.df.columns):
            for col in ['pivot', 'pivot_s1', 'pivot_s2', 'pivot_s3', 'pivot_r1', 'pivot_r2', 'pivot_r3']:
                self.df[col] = np.nan
            return self

        h_prev = self.df['high'].shift(1)
        l_prev = self.df['low'].shift(1)
        c_prev = self.df['close'].shift(1)

        pivot = (h_prev + l_prev + c_prev) / 3
        self.df['pivot'] = pivot
        self.df['pivot_r1'] = 2 * pivot - l_prev
        self.df['pivot_s1'] = 2 * pivot - h_prev
        self.df['pivot_r2'] = pivot + (h_prev - l_prev)
        self.df['pivot_s2'] = pivot - (h_prev - l_prev)
        self.df['pivot_r3'] = h_prev + 2 * (pivot - l_prev)
        self.df['pivot_s3'] = l_prev - 2 * (h_prev - pivot)
        return self

    def calc_atr(self) -> "DataProcessor":
        """Calculate Average True Range (14-day) for Stoploss sizing."""
        if not all(col in self.df.columns for col in ['high', 'low', 'close']):
            logger.warning(f"[{self.ticker}] Missing high/low/close cols for ATR. Setting atr_14 = NaN.")
            self.df["atr_14"] = float("nan")
            return self
            
        atr_indicator = AverageTrueRange(
            high=self.df['high'], 
            low=self.df['low'], 
            close=self.df['close'], 
            window=14
        )
        self.df['atr_14'] = atr_indicator.average_true_range()
        logger.info(f"[{self.ticker}] ATR (14) done")
        return self
    
    # ------------------------------------------------------------------
    # D. Performance & Risk
    # ------------------------------------------------------------------

    def calc_max_drawdown(self) -> "DataProcessor":
        # TODO: rolling_max = close.cummax()
        # TODO: drawdown = (close - rolling_max) / rolling_max
        rolling_max = self.df['close'].cummax()
        self.df['drawdown'] = (self.df['close'] - rolling_max) / rolling_max
        self.df['max_drawdown'] = self.df['drawdown'].cummin()
        max_drawdown = self.df['max_drawdown'].min()
        logger.info(f"[{self.ticker}] max_drawdown = {max_drawdown:.4f}")
        return self
    #
    def calc_sharpe_ratio(self, trading_days: int = 252, window: int = ROLLING_SHARPE_WINDOW) -> "DataProcessor":
        # TODO: sharpe = mean(daily_return) / std(daily_return) * sqrt(trading_days)
        # Yeu cau: calc_returns() chay truoc
        if 'daily_return' not in self.df.columns:
            raise RuntimeError(f"calc_returns() must be called before calc_sharpe_ratio()")

        rolling_mean = self.df['daily_return'].rolling(window=window, min_periods=window).mean()
        rolling_std = self.df['daily_return'].rolling(window=window, min_periods=window).std()
        annualized_return = rolling_mean * trading_days
        annualized_volatility = rolling_std * np.sqrt(trading_days)
        sharpe_series = annualized_return / annualized_volatility.replace(0, np.nan)
        self.df['sharpe_ratio'] = sharpe_series.replace([np.inf, -np.inf], np.nan)

        latest_sharpe = self.df['sharpe_ratio'].dropna()
        logger.info(
            "[%s] rolling sharpe_ratio computed | latest=%s",
            self.ticker,
            f"{latest_sharpe.iloc[-1]:.4f}" if not latest_sharpe.empty else "nan",
        )
        return self

    def calc_beta(self, benchmark_df, window: int = ROLLING_BETA_WINDOW) -> "DataProcessor":
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
            how='left',
            suffixes=('_stock', '_market')
        )

        rolling_cov = merged['daily_return_stock'].rolling(window=window, min_periods=window).cov(merged['daily_return_market'])
        rolling_var = merged['daily_return_market'].rolling(window=window, min_periods=window).var()
        beta_series = (rolling_cov / rolling_var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        self.df['beta'] = beta_series.values

        latest_beta = self.df['beta'].dropna()
        logger.info(
            "[%s] rolling beta computed | latest=%s",
            self.ticker,
            f"{latest_beta.iloc[-1]:.4f}" if not latest_beta.empty else "nan",
        )
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
        if "close" not in other_df.columns:
            raise ValueError("other_df must contain 'close' column to calculate relative strength")
        if "close" not in self.df.columns:
            raise ValueError("self.df must contain 'close' column to calculate relative strength")

        # Tinh cum_return inline - khong phu thuoc engineer_features()
        cum_self  = self.df["close"] / self.df["close"].iloc[0] - 1
        cum_other = other_df["close"] / other_df["close"].iloc[0] - 1

        other_tmp = pd.DataFrame({"date": other_df["date"].values, "cum_other": cum_other.values})
        tmp       = pd.DataFrame({"date": self.df["date"].values,  "cum_self":  cum_self.values})
        merged    = pd.merge(tmp, other_tmp, on="date", how="left")
        merged["cum_other"] = merged["cum_other"].ffill().bfill()

        self.df = self.df.copy()
        self.df["relative_strength"] = (1.0 + merged["cum_self"]) / (1.0 + merged["cum_other"])
        ticker_other = other_df["ticker"].iloc[0] if "ticker" in other_df.columns else "Benchmark"
        logger.info("[%s] calc_relative_strength done | Compared with %s", self.ticker, ticker_other)
        return self

    # ------------------------------------------------------------------
    # News / Sentiment processing
    # ------------------------------------------------------------------

    def process_news(self) -> "DataProcessor":
        """
        Clean, encode, and aggregate a news/sentiment DataFrame.

        Operations
        ----------
        - Normalise date column and sort chronologically.
        - Remove exact duplicate records on (date, ticker, headline).
        - Normalise sentiment labels to lowercase stripped strings.
        - Encode ``sentiment_score``: positive=1, neutral=0, negative=-1.
        - Normalise event_type to lowercase.
        - Aggregate to one row per (date, ticker) for safe downstream joins.

        Returns
        -------
        DataProcessor
            Self, for method chaining.
        """
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        dup_cols = [c for c in ['date', 'ticker', 'headline'] if c in self.df.columns]
        dup_count = self.df.duplicated(subset=dup_cols).sum()
        self.df = self.df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
        logger.info("[%s] News: removed %d duplicate records", self.ticker, dup_count)

        # headline is required — fill NaN with empty string to prevent downstream errors
        if 'headline' in self.df.columns:
            self.df['headline'] = self.df['headline'].fillna('').astype(str).str.strip()
        # summary and source are optional fields; leave legitimate NaN values intact
        for col in ['summary', 'source']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().replace('nan', pd.NA)

        if 'sentiment' in self.df.columns:
            self.df['sentiment'] = (
                self.df['sentiment']
                .fillna('neutral')
                .astype(str)
                .str.lower()
                .str.strip()
            )
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            self.df['sentiment_score'] = (
                self.df['sentiment'].map(sentiment_map).fillna(0).astype(int)
            )
            logger.info(
                "[%s] News sentiment distribution:\n%s",
                self.ticker,
                self.df['sentiment'].value_counts().to_string(),
            )

        if 'event_type' in self.df.columns:
            self.df['event_type'] = (
                self.df['event_type'].fillna('general').astype(str).str.lower().str.strip()
            )

        def dominant_label(avg_score: float) -> str:
            if avg_score > 0:
                return "positive"
            if avg_score < 0:
                return "negative"
            return "neutral"

        # Keep one row per article to preserve news volume after processing.
        # Add day-level volume context for downstream joins and analytics.
        day_counts = (
            self.df.groupby(['date', 'ticker'])
            .size()
            .rename('article_count')
            .reset_index()
        )
        self.df = self.df.merge(day_counts, on=['date', 'ticker'], how='left')
        self.df['article_count'] = self.df['article_count'].fillna(1).astype(int)

        self.df['positive_count'] = (self.df['sentiment_score'] > 0).astype(int)
        self.df['neutral_count'] = (self.df['sentiment_score'] == 0).astype(int)
        self.df['negative_count'] = (self.df['sentiment_score'] < 0).astype(int)
        self.df['sentiment_score'] = self.df['sentiment_score'].astype(float)
        self.df['sentiment'] = self.df['sentiment_score'].apply(dominant_label)
        self.df = self.df[
            [
                'date', 'ticker', 'article_count', 'headline', 'summary', 'source',
                'sentiment', 'sentiment_score', 'positive_count', 'neutral_count',
                'negative_count', 'event_type'
            ]
        ].sort_values(['date', 'ticker', 'headline']).reset_index(drop=True)

        logger.info(
            "[%s] process_news() done | %d article-level records from %d articles",
            self.ticker,
            len(self.df),
            len(self.df),
        )
        return self

    # ------------------------------------------------------------------
    # Fundamental feature engineering
    # ------------------------------------------------------------------

    def engineer_fundamental_features(self, news_df: pd.DataFrame | None = None) -> "DataProcessor":
        """
        Add financial-health features for fundamental analysis.

        Added columns:
          - net_debt
          - net_debt_to_ebitda
          - interest_coverage
          - asset_turnover
          - altman_z_score
          - latest_event_type, latest_sentiment, news_article_count
        """
        numeric_candidates = [
            'revenue', 'operating_profit', 'total_assets', 'total_liabilities',
            'total_debt', 'cash', 'interest_expense', 'ebitda', 'current_assets',
            'current_liabilities', 'retained_earnings', 'market_cap', 'equity',
            'net_income', 'gross_profit', 'operating_cash_flow', 'capex',
            'accounts_receivable', 'inventory', 'accounts_payable', 'cogs',
            'short_term_investments', 'income_tax_expense', 'pre_tax_income',
        ]
        for column in numeric_candidates:
            if column in self.df.columns:
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')

        if 'total_debt' in self.df.columns and 'cash' in self.df.columns:
            self.df['net_debt'] = self.df['total_debt'] - self.df['cash']
        else:
            self.df['net_debt'] = np.nan

        # If EBITDA is unavailable, use operating_profit as a conservative proxy for EBIT-based leverage.
        if 'ebitda' not in self.df.columns:
            self.df['ebitda'] = self.df['operating_profit'] if 'operating_profit' in self.df.columns else np.nan

        self.df['net_debt_to_ebitda'] = np.where(
            self.df['ebitda'].notna() & (self.df['ebitda'] != 0),
            self.df['net_debt'] / self.df['ebitda'],
            np.nan,
        )

        if 'interest_expense' in self.df.columns and 'operating_profit' in self.df.columns:
            interest_base = self.df['interest_expense'].abs()
            self.df['interest_coverage'] = np.where(
                interest_base.notna() & (interest_base != 0),
                self.df['operating_profit'] / interest_base,
                np.nan,
            )
        else:
            self.df['interest_coverage'] = np.nan

        if 'revenue' in self.df.columns and 'total_assets' in self.df.columns:
            self.df['asset_turnover'] = np.where(
                self.df['total_assets'].notna() & (self.df['total_assets'] != 0),
                self.df['revenue'] / self.df['total_assets'],
                np.nan,
            )
        else:
            self.df['asset_turnover'] = np.nan

        if 'operating_profit' in self.df.columns:
            self.df['ebit'] = self.df['operating_profit']
        else:
            self.df['ebit'] = np.nan

        # Profitability block
        self.df['gross_profit_margin'] = np.where(
            self.df.get('revenue', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('revenue', pd.Series(np.nan, index=self.df.index)) != 0),
            self.df.get('gross_profit', pd.Series(np.nan, index=self.df.index))
            / self.df.get('revenue', pd.Series(np.nan, index=self.df.index)),
            np.nan,
        )
        self.df['net_profit_margin'] = np.where(
            self.df.get('revenue', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('revenue', pd.Series(np.nan, index=self.df.index)) != 0),
            self.df.get('net_income', pd.Series(np.nan, index=self.df.index))
            / self.df.get('revenue', pd.Series(np.nan, index=self.df.index)),
            np.nan,
        )

        # Liquidity block
        self.df['current_ratio'] = np.where(
            self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)) != 0),
            self.df.get('current_assets', pd.Series(np.nan, index=self.df.index))
            / self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)),
            np.nan,
        )
        self.df['quick_ratio'] = np.where(
            self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)) != 0),
            (
                self.df.get('cash', pd.Series(np.nan, index=self.df.index))
                + self.df.get('short_term_investments', pd.Series(0.0, index=self.df.index)).fillna(0)
                + self.df.get('accounts_receivable', pd.Series(0.0, index=self.df.index)).fillna(0)
            ) / self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)),
            np.nan,
        )
        self.df['cash_ratio'] = np.where(
            self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)) != 0),
            (
                self.df.get('cash', pd.Series(np.nan, index=self.df.index))
                + self.df.get('short_term_investments', pd.Series(0.0, index=self.df.index)).fillna(0)
            ) / self.df.get('current_liabilities', pd.Series(np.nan, index=self.df.index)),
            np.nan,
        )

        # Solvency block
        self.df['debt_to_assets'] = np.where(
            self.df.get('total_assets', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('total_assets', pd.Series(np.nan, index=self.df.index)) != 0),
            self.df.get('total_debt', pd.Series(np.nan, index=self.df.index))
            / self.df.get('total_assets', pd.Series(np.nan, index=self.df.index)),
            np.nan,
        )
        self.df['debt_to_capital'] = np.where(
            (
                self.df.get('total_debt', pd.Series(np.nan, index=self.df.index))
                + self.df.get('equity', pd.Series(np.nan, index=self.df.index))
            ).notna()
            & (
                self.df.get('total_debt', pd.Series(np.nan, index=self.df.index))
                + self.df.get('equity', pd.Series(np.nan, index=self.df.index))
            != 0),
            self.df.get('total_debt', pd.Series(np.nan, index=self.df.index))
            / (
                self.df.get('total_debt', pd.Series(np.nan, index=self.df.index))
                + self.df.get('equity', pd.Series(np.nan, index=self.df.index))
            ),
            np.nan,
        )
        self.df['financial_leverage'] = np.where(
            self.df.get('equity', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('equity', pd.Series(np.nan, index=self.df.index)) != 0),
            self.df.get('total_assets', pd.Series(np.nan, index=self.df.index))
            / self.df.get('equity', pd.Series(np.nan, index=self.df.index)),
            np.nan,
        )

        # Activity block (quarterly approximation)
        avg_receivable = (
            self.df.get('accounts_receivable', pd.Series(np.nan, index=self.df.index))
            + self.df.get('accounts_receivable', pd.Series(np.nan, index=self.df.index)).shift(1)
        ) / 2
        avg_inventory = (
            self.df.get('inventory', pd.Series(np.nan, index=self.df.index))
            + self.df.get('inventory', pd.Series(np.nan, index=self.df.index)).shift(1)
        ) / 2
        avg_payable = (
            self.df.get('accounts_payable', pd.Series(np.nan, index=self.df.index))
            + self.df.get('accounts_payable', pd.Series(np.nan, index=self.df.index)).shift(1)
        ) / 2

        self.df['receivable_turnover'] = np.where(
            avg_receivable.notna() & (avg_receivable != 0),
            self.df.get('revenue', pd.Series(np.nan, index=self.df.index)) / avg_receivable,
            np.nan,
        )
        self.df['days_sales_outstanding'] = np.where(
            self.df['receivable_turnover'].notna() & (self.df['receivable_turnover'] != 0),
            365 / self.df['receivable_turnover'],
            np.nan,
        )

        self.df['inventory_turnover'] = np.where(
            avg_inventory.notna() & (avg_inventory != 0),
            self.df.get('cogs', pd.Series(np.nan, index=self.df.index)) / avg_inventory,
            np.nan,
        )
        self.df['days_inventory_outstanding'] = np.where(
            self.df['inventory_turnover'].notna() & (self.df['inventory_turnover'] != 0),
            365 / self.df['inventory_turnover'],
            np.nan,
        )

        purchase = (
            self.df.get('inventory', pd.Series(np.nan, index=self.df.index))
            - self.df.get('inventory', pd.Series(np.nan, index=self.df.index)).shift(1)
            + self.df.get('cogs', pd.Series(np.nan, index=self.df.index))
        )
        self.df['payable_turnover'] = np.where(
            avg_payable.notna() & (avg_payable != 0),
            purchase / avg_payable,
            np.nan,
        )
        self.df['days_payable_outstanding'] = np.where(
            self.df['payable_turnover'].notna() & (self.df['payable_turnover'] != 0),
            365 / self.df['payable_turnover'],
            np.nan,
        )
        self.df['cash_conversion_cycle'] = (
            self.df['days_sales_outstanding']
            + self.df['days_inventory_outstanding']
            - self.df['days_payable_outstanding']
        )

        # Cash flow block
        capex = self.df.get('capex', pd.Series(np.nan, index=self.df.index)).abs()
        self.df['fcf'] = self.df.get('operating_cash_flow', pd.Series(np.nan, index=self.df.index)) - capex
        tax_rate = np.where(
            self.df.get('pre_tax_income', pd.Series(np.nan, index=self.df.index)).notna()
            & (self.df.get('pre_tax_income', pd.Series(np.nan, index=self.df.index)) != 0),
            self.df.get('income_tax_expense', pd.Series(np.nan, index=self.df.index))
            / self.df.get('pre_tax_income', pd.Series(np.nan, index=self.df.index)),
            0.21,
        )
        net_borrowing = self.df.get('total_debt', pd.Series(np.nan, index=self.df.index)).diff()
        self.df['fcff'] = (
            self.df.get('operating_cash_flow', pd.Series(np.nan, index=self.df.index))
            + self.df.get('interest_expense', pd.Series(np.nan, index=self.df.index)).abs() * (1 - tax_rate)
            - capex
        )
        self.df['fcfe'] = self.df['fcf'] + net_borrowing

        # Altman Z-score (public manufacturing form) with graceful NaN fallback.
        wc = self.df['current_assets'] - self.df['current_liabilities'] if {'current_assets', 'current_liabilities'}.issubset(self.df.columns) else np.nan
        retained_earnings = self.df['retained_earnings'] if 'retained_earnings' in self.df.columns else np.nan
        market_value_equity = self.df['market_cap'] if 'market_cap' in self.df.columns else self.df.get('equity', np.nan)
        total_assets = self.df['total_assets'] if 'total_assets' in self.df.columns else np.nan
        total_liabilities = self.df['total_liabilities'] if 'total_liabilities' in self.df.columns else np.nan
        sales = self.df['revenue'] if 'revenue' in self.df.columns else np.nan

        x1 = np.where(pd.notna(total_assets) & (total_assets != 0), wc / total_assets, np.nan)
        x2 = np.where(pd.notna(total_assets) & (total_assets != 0), retained_earnings / total_assets, np.nan)
        x3 = np.where(pd.notna(total_assets) & (total_assets != 0), self.df['ebit'] / total_assets, np.nan)
        x4 = np.where(pd.notna(total_liabilities) & (total_liabilities != 0), market_value_equity / total_liabilities, np.nan)
        x5 = np.where(pd.notna(total_assets) & (total_assets != 0), sales / total_assets, np.nan)
        self.df['altman_z_score'] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        self.df['latest_event_type'] = pd.NA
        self.df['latest_sentiment'] = pd.NA
        self.df['news_article_count'] = 0
        if news_df is not None and not news_df.empty and {'date', 'ticker'}.issubset(news_df.columns):
            tmp = news_df.copy()
            tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
            if 'article_count' not in tmp.columns:
                tmp['article_count'] = 1
            grouped = (
                tmp.sort_values('date')
                .groupby(['date', 'ticker'], as_index=False)
                .agg(
                    latest_event_type=('event_type', 'last'),
                    latest_sentiment=('sentiment', 'last'),
                    news_article_count=('article_count', 'sum'),
                )
            )
            merged = pd.merge(
                self.df,
                grouped,
                on=['date', 'ticker'],
                how='left',
                suffixes=('', '_news'),
            )
            for col in ['latest_event_type', 'latest_sentiment']:
                news_col = f"{col}_news"
                if news_col in merged.columns:
                    merged[col] = merged[news_col].combine_first(merged[col])
                    merged = merged.drop(columns=[news_col])
            if 'news_article_count_news' in merged.columns:
                merged['news_article_count'] = (
                    merged['news_article_count_news'].fillna(merged['news_article_count']).fillna(0).astype(int)
                )
                merged = merged.drop(columns=['news_article_count_news'])
            self.df = merged

        logger.info(
            "[%s] engineer_fundamental_features done | added leverage/coverage/zscore/turnover/news columns",
            self.ticker,
        )
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
            Fully processed DataFrame.
        """
        # Step 1: Cleaning
        self.normalise_types()
        self.remove_duplicates()
        self.handle_missing_values(strategy="ffill")

        # Step 2: Feature engineering (rolling windows sẽ tạo NaN đầu chuỗi)
        self.calc_returns()
        self.detect_outliers(method="iqr")
        self.calc_cumulative_returns()
        self.calc_moving_averages()
        self.calc_volatility()
        self.calc_bollinger_bands()
        self.calc_max_drawdown()
        self.calc_momentum_oscillators()
        self.calc_extended_oscillators()
        self.calc_pivot_points()
        self.calc_atr()
        self.calc_sharpe_ratio()

        # NOTE: beta va relative_strength duoc tinh sau trong run_processing()
        # vi can benchmark_df tu ben ngoai.
        return self.df

    def run_pipeline_and_save(self) -> "pd.DataFrame":
        """Chay pipeline va luu CSV ngay (dung khi khong can beta/RS)."""
        self.run_pipeline()
        self._save_csv()
        return self.df

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------

    def _save_csv(self, filename: str | None = None) -> Path:
        """
        Save the current state of self.df to data/processed/processed_data/.

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
        logger.info("Saved processed data -> %s", filepath)
        return filepath

