
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

    def __init__(self, df: pd.DataFrame, ticker: str) -> None:
        self.df = df.copy()
        self.ticker = ticker

    def handle_missing_values(self, strategy: str = "ffill") -> "DataProcessor":
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

    def remove_duplicates(self) -> "DataProcessor":
            dup_count = self.df.duplicated().sum()
            logger.info(f"Number of duplicated rows: {dup_count}")
            
            possible_subset = ['date', 'ticker']
            subset = [c for c in possible_subset if c in self.df.columns]
            self.df = self.df.drop_duplicates(subset=subset if subset else None)
            self.df = self.df.reset_index(drop=True)
            logger.info(f"Removed {dup_count} duplicate rows")
            return self

    def normalise_types(self) -> "DataProcessor":
        def parse_currency(val):
            if not isinstance(val, str):
                return val
            val =val.strip()
            if val.startswith('$'):
                return val.replace('$', '').replace(',', '')
            if ',' in val and '.' in val:
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

    def detect_outliers(self, method: str = "iqr", threshold: float = 3.0) -> "DataProcessor":
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

    def engineer_features(self) -> "DataProcessor":
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['rolling_avg_7'] = self.df['close'].rolling(window=7).mean()
        self.df['rolling_avg_30'] = self.df['close'].rolling(window=30).mean()
        self.df['volatility_30'] = self.df['daily_return'].rolling(window=30).std()
        self.df['cum_return'] = self.df['close']/self.df['close'].iloc[0] - 1
        logger.info("[%s] engineer_features done | cols added: daily_return, rolling_avg_7, rolling_avg_30, volatility_30, cum_return", self.ticker)
        return self
        
    def calc_returns(self) -> "DataProcessor":
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))
        logger.info(f"[{self.ticker}] calc_returns done | daily_return mean= {self.df['daily_return'].mean()} | log_return = {self.df['log_return'].mean()}")
        return self

    def calc_cumulative_returns(self) -> "DataProcessor":
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

    def calc_moving_averages(self) -> "DataProcessor":
        self.df['ma7'] = self.df['close'].rolling(window=7).mean()
        self.df['ma20'] = self.df['close'].rolling(window=20).mean()
        self.df['ma30'] = self.df['close'].rolling(window=30).mean()
        self.df['ma50'] = self.df['close'].rolling(window=50).mean()
        self.df['ma200'] = self.df['close'].rolling(window=200).mean()
        self.df['ma20_signal'] = np.where(self.df['close'] > self.df['ma20'], 'BUY', 'SELL')
        self.df['ma50_signal'] = np.where(self.df['close'] > self.df['ma50'], 'BUY', 'SELL')
        self.df['ma200_signal'] = np.where(self.df['close'] > self.df['ma200'], 'BUY', 'SELL')
        logger.info(f"[{self.ticker}] calc_moving_averages done | ma7, ma20, ma30, ma50, ma200 and BUY/SELL signals added")
        return self

    def calc_volatility(self) -> "DataProcessor":
        self.df['volatility_30'] = self.df['daily_return'].rolling(window=30).std()
        self.df['volatility_60'] = self.df['daily_return'].rolling(window=60).std()
        logger.info("[%s] calc_volatility done | volatility_30, volatility_60 added", self.ticker)
        return self

    def calc_bollinger_bands(self) -> "DataProcessor":
        if 'ma20' not in self.df.columns:
            raise RuntimeError("calc_moving_averages() must be called before calc_bollinger_bands()")
        std20 = self.df['close'].rolling(20).std()
        self.df['bb_middle'] = self.df['ma20']
        self.df['bb_upper'] = self.df['ma20'] + 2*std20
        self.df['bb_lower'] = self.df['ma20'] - 2*std20
        logger.info(f"[{self.ticker}] Bollinger Bands done | bb_upper mean={self.df['bb_upper'].mean():.4f} | bb_lower mean={self.df['bb_lower'].mean():.4f}")
        return self
    
    def calc_momentum_oscillators(self) -> "DataProcessor":
        self.df['rsi_14'] = RSIIndicator(close=self.df['close'], window=14).rsi()
        
        macd = MACD(close=self.df['close'], window_slow=26, window_fast=12, window_sign=9)
        self.df['macd_line'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_hist'] = macd.macd_diff()
        
        logger.info(f"[{self.ticker}] Momentum Oscillators done | RSI, MACD added")
        return self

    def calc_extended_oscillators(self) -> "DataProcessor":
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
    
    def calc_max_drawdown(self) -> "DataProcessor":
        rolling_max = self.df['close'].cummax()
        self.df['drawdown'] = (self.df['close'] - rolling_max) / rolling_max
        self.df['max_drawdown'] = self.df['drawdown'].cummin()
        max_drawdown = self.df['max_drawdown'].min()
        logger.info(f"[{self.ticker}] max_drawdown = {max_drawdown:.4f}")
        return self
    def calc_sharpe_ratio(self, trading_days: int = 252, window: int = ROLLING_SHARPE_WINDOW) -> "DataProcessor":
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

    def calc_price_volume_anomalies(self) -> "DataProcessor":
        if 'volume' in self.df.columns:
            rolling_vol_avg = self.df['volume'].rolling(window=20).mean()
            self.df['volume_spike'] = (self.df['volume'] > 2 * rolling_vol_avg).fillna(False)
        else:
            self.df['volume_spike'] = False

        if all(c in self.df.columns for c in ['open', 'high', 'low']):
            prev_high = self.df['high'].shift(1)
            prev_low = self.df['low'].shift(1)
            self.df['gap_up'] = (self.df['open'] > prev_high).fillna(False)
            self.df['gap_down'] = (self.df['open'] < prev_low).fillna(False)
        else:
            self.df['gap_up'] = False
            self.df['gap_down'] = False

        if 'daily_return' in self.df.columns:
            rolling_std = self.df['daily_return'].rolling(window=20).std()
            self.df['sudden_price_movement'] = (
                self.df['daily_return'].abs() > 3 * rolling_std
            ).fillna(False)
        else:
            self.df['sudden_price_movement'] = False

        logger.info(
            "[%s] calc_price_volume_anomalies done | volume_spike, gap_up, gap_down, sudden_price_movement",
            self.ticker,
        )
        return self

    def calc_var(self, window: int = 252) -> "DataProcessor":
        if 'daily_return' not in self.df.columns:
            self.df['var_95'] = np.nan
            self.df['var_99'] = np.nan
            return self
        self.df['var_95'] = (
            self.df['daily_return']
            .rolling(window=window, min_periods=window)
            .quantile(0.05)
        )
        self.df['var_99'] = (
            self.df['daily_return']
            .rolling(window=window, min_periods=window)
            .quantile(0.01)
        )
        logger.info("[%s] calc_var done | var_95, var_99 added", self.ticker)
        return self

    def calc_beta(self, benchmark_df, window: int = ROLLING_BETA_WINDOW) -> "DataProcessor":
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

    def calc_correlation_matrix(self, other_dfs: list):
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
        if "close" not in other_df.columns:
            raise ValueError("other_df must contain 'close' column to calculate relative strength")
        if "close" not in self.df.columns:
            raise ValueError("self.df must contain 'close' column to calculate relative strength")

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

    def process_news(self) -> "DataProcessor":
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)

        dup_cols = [c for c in ['date', 'ticker', 'headline'] if c in self.df.columns]
        dup_count = self.df.duplicated(subset=dup_cols).sum()
        self.df = self.df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
        logger.info("[%s] News: removed %d duplicate records", self.ticker, dup_count)

        if 'headline' in self.df.columns:
            self.df['headline'] = self.df['headline'].fillna('').astype(str).str.strip()
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

    def engineer_fundamental_features(
        self,
        news_df: pd.DataFrame | None = None,
        industry_df: pd.DataFrame | None = None,
    ) -> "DataProcessor":
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
            frames = []
            for ticker, base in self.df.groupby('ticker', sort=False):
                news_slice = grouped[grouped['ticker'] == ticker].sort_values('date')
                base_sorted = base.copy()
                base_sorted['date'] = pd.to_datetime(base_sorted['date'], errors='coerce')
                base_sorted = base_sorted.sort_values('date')
                if news_slice.empty:
                    frames.append(base_sorted)
                    continue

                merged = pd.merge_asof(
                    base_sorted,
                    news_slice.drop(columns=['ticker']),
                    on='date',
                    direction='nearest',
                    tolerance=pd.Timedelta(days=90),
                )
                if 'news_article_count' in merged.columns:
                    merged['news_article_count'] = merged['news_article_count'].fillna(0).astype(int)
                merged['ticker'] = ticker
                frames.append(merged)
            self.df = pd.concat(frames, ignore_index=True).sort_values(['ticker', 'date']).reset_index(drop=True)
        if 'latest_event_type' not in self.df.columns:
            self.df['latest_event_type'] = pd.NA
        if 'latest_sentiment' not in self.df.columns:
            self.df['latest_sentiment'] = pd.NA
        if 'news_article_count' not in self.df.columns:
            self.df['news_article_count'] = 0

        # --- Revenue growth ---
        if 'revenue' in self.df.columns:
            self.df['revenue_growth'] = self.df['revenue'].pct_change()
        else:
            self.df['revenue_growth'] = np.nan

        # --- D/E ratio alias (spec output name: de_ratio) ---
        if 'debt_to_equity' in self.df.columns:
            self.df['de_ratio'] = self.df['debt_to_equity']
        elif 'total_debt' in self.df.columns and 'equity' in self.df.columns:
            self.df['de_ratio'] = np.where(
                self.df['equity'].notna() & (self.df['equity'] != 0),
                self.df['total_debt'] / self.df['equity'],
                np.nan,
            )
        else:
            self.df['de_ratio'] = np.nan

        # --- Rolling P/E and P/B averages ---
        for ratio_col, avg_1y, avg_5y in [
            ('pe', 'pe_1y_avg', 'pe_5y_avg'),
            ('pb', 'pb_1y_avg', 'pb_5y_avg'),
        ]:
            if ratio_col in self.df.columns:
                self.df[avg_1y] = self.df[ratio_col].rolling(window=4, min_periods=1).mean()
                self.df[avg_5y] = self.df[ratio_col].rolling(window=20, min_periods=1).mean()
            else:
                self.df[avg_1y] = np.nan
                self.df[avg_5y] = np.nan

        # --- Industry P/E and P/B passthrough ---
        self.df['pe_industry'] = np.nan
        self.df['pb_industry'] = np.nan
        if industry_df is not None and not industry_df.empty:
            ind = industry_df.copy()
            ind['date'] = pd.to_datetime(ind['date'], errors='coerce')
            ind = ind.sort_values('date')
            if 'ticker' in ind.columns and self.ticker in ind['ticker'].values:
                ind = ind[ind['ticker'] == self.ticker]
            if 'industry_pe' in ind.columns:
                tmp_pe = ind[['date', 'industry_pe']].dropna(subset=['industry_pe'])
                if not tmp_pe.empty:
                    merged_pe = pd.merge_asof(
                        self.df[['date']].sort_values('date'),
                        tmp_pe.rename(columns={'industry_pe': 'pe_industry'}),
                        on='date',
                        direction='backward',
                    )
                    self.df['pe_industry'] = merged_pe['pe_industry'].values
            if 'industry_pb' in ind.columns:
                tmp_pb = ind[['date', 'industry_pb']].dropna(subset=['industry_pb'])
                if not tmp_pb.empty:
                    merged_pb = pd.merge_asof(
                        self.df[['date']].sort_values('date'),
                        tmp_pb.rename(columns={'industry_pb': 'pb_industry'}),
                        on='date',
                        direction='backward',
                    )
                    self.df['pb_industry'] = merged_pb['pb_industry'].values

        logger.info(
            "[%s] engineer_fundamental_features done | added leverage/coverage/zscore/turnover/news/revenue_growth/de_ratio/pe_pb_avgs columns",
            self.ticker,
        )
        return self

    def calc_dcf_valuation(
        self,
        beta: float | None = None,
        risk_free_rate: float = 0.045,
        market_risk_premium: float = 0.055,
        current_price: float | None = None,
        projection_years: int = 5,
    ) -> "DataProcessor":
        """FCFE-based DCF intrinsic price using CAPM cost of equity.

        Ke  = risk_free_rate + beta * market_risk_premium
        g   = trailing 3-year annualised FCFE CAGR (quarterly data → 12-period lag),
              floored at 0 and capped at Ke − 0.01 to avoid terminal value blow-up.

        Equity value per row = annual_fcfe * (1 + g) / (Ke − g)
        (mathematically equivalent to n-year projected DCF + terminal value at g)
        Intrinsic price = equity_value / shares_outstanding
        Upside          = (intrinsic_price − current_price) / current_price
        """
        _beta = beta if beta is not None else 1.0
        ke = risk_free_rate + _beta * market_risk_premium

        if 'fcfe' not in self.df.columns:
            self.df['dcf_intrinsic_price'] = np.nan
            self.df['dcf_upside'] = np.nan
            logger.info("[%s] calc_dcf_valuation: fcfe column missing, setting NaN", self.ticker)
            return self

        # Annual FCFE = rolling sum of 4 quarters
        annual_fcfe = self.df['fcfe'].rolling(window=4, min_periods=4).sum()

        # Growth rate: 3-year CAGR (12-quarter lag)
        fcfe_3y_ago = annual_fcfe.shift(12)
        with np.errstate(divide='ignore', invalid='ignore'):
            g_raw = np.where(
                fcfe_3y_ago.notna() & (fcfe_3y_ago > 0)
                & annual_fcfe.notna() & (annual_fcfe > 0),
                (annual_fcfe / fcfe_3y_ago) ** (1.0 / 3.0) - 1.0,
                0.05,
            )
        g = np.clip(g_raw, 0.0, max(ke - 0.01, 0.001))

        shares = self.df.get('shares_outstanding')
        if not isinstance(shares, pd.Series):
            shares = pd.Series(np.nan, index=self.df.index)
        shares = pd.to_numeric(shares, errors='coerce').reindex(self.df.index)
        if shares.isna().all() and current_price is not None and current_price > 0:
            market_cap = self.df.get('market_cap')
            if isinstance(market_cap, pd.Series):
                market_cap = pd.to_numeric(market_cap, errors='coerce').reindex(self.df.index)
                shares = market_cap / current_price
                logger.info(
                    "[%s] calc_dcf_valuation: shares_outstanding missing, derived from market_cap/current_price",
                    self.ticker,
                )

        kd_safe = np.where(np.abs(ke - g) < 1e-6, 1e-6, ke - g)
        annual_fcfe_arr = np.asarray(annual_fcfe, dtype=float)
        has_fcfe_history = annual_fcfe.notna()
        has_positive_fcfe = annual_fcfe > 0
        has_shares = shares.notna() & (shares > 0)
        equity_value = np.where(
            has_fcfe_history,
            annual_fcfe_arr * (1.0 + g) / kd_safe,
            np.nan,
        )

        intrinsic_price = np.where(
            pd.notna(equity_value) & has_shares,
            equity_value / shares.to_numpy(dtype=float),
            np.nan,
        )
        self.df['dcf_intrinsic_price'] = np.where(
            np.isfinite(intrinsic_price),
            intrinsic_price,
            np.nan,
        )

        ref_price = current_price if (current_price is not None and current_price > 0) else None
        if ref_price is not None:
            self.df['dcf_upside'] = np.where(
                self.df['dcf_intrinsic_price'].notna(),
                (self.df['dcf_intrinsic_price'] - ref_price) / ref_price,
                np.nan,
            )
        else:
            self.df['dcf_upside'] = np.nan

        # Quality flags: keep computed values visible while surfacing why a row may be unreliable.
        invalid_reasons = np.select(
            [
                ~has_fcfe_history,
                ~has_shares,
                ~np.isfinite(self.df['dcf_intrinsic_price']),
                ~has_positive_fcfe,
                self.df['dcf_intrinsic_price'] <= 0,
            ],
            [
                "insufficient_fcfe_history",
                "missing_or_nonpositive_shares_outstanding",
                "intrinsic_price_not_finite",
                "non_positive_annual_fcfe",
                "non_positive_intrinsic_price",
            ],
            default="",
        )
        self.df['dcf_invalid_reason'] = pd.Series(invalid_reasons, index=self.df.index).replace("", pd.NA)
        self.df['dcf_is_valid'] = self.df['dcf_invalid_reason'].isna()

        valid_count = int(self.df['dcf_intrinsic_price'].notna().sum())
        valid_quality_count = int(self.df['dcf_is_valid'].sum())
        logger.info(
            "[%s] calc_dcf_valuation done | Ke=%.4f | %d rows with intrinsic price | %d rows marked valid",
            self.ticker, ke, valid_count, valid_quality_count,
        )
        return self

    def run_pipeline(self) -> pd.DataFrame:
        self.normalise_types()
        self.remove_duplicates()
        self.handle_missing_values(strategy="ffill")

        self.calc_returns()
        self.detect_outliers(method="iqr")
        self.calc_cumulative_returns()
        self.calc_moving_averages()
        self.calc_volatility()
        self.calc_bollinger_bands()
        self.calc_max_drawdown()
        self.calc_momentum_oscillators()
        self.calc_extended_oscillators()
        self.calc_price_volume_anomalies()
        self.calc_pivot_points()
        self.calc_atr()
        self.calc_sharpe_ratio()
        self.calc_var()

        return self.df

    def run_pipeline_and_save(self) -> "pd.DataFrame":
        self.run_pipeline()
        self._save_csv()
        return self.df

    def _save_csv(self, filename: str | None = None) -> Path:
        filename = filename or f"{self.ticker}_processed.csv"
        filepath = PROCESSED_DATA_DIR / filename
        self.df.to_csv(filepath, index = False)
        logger.info("Saved processed data -> %s", filepath)
        return filepath

