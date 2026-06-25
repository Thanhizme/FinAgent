"""Stage 5 - Signal generation for QuantStrategy.

This module reads Stage 2 processed price data, computes three factor scores
(trend, momentum, risk), builds Buy/Sell/Hold signals, and exports a signal
file per ticker under data/quant_outputs/signals.
"""

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_data"
SIGNALS_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "signals"

DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 0.40,
    "momentum": 0.35,
    "risk": 0.25,
}

DEFAULT_BUY_THRESHOLD = 0.30
DEFAULT_SELL_THRESHOLD = -0.20
DEFAULT_VOL_LOOKBACK = 252

REQUIRED_PRICE_COLS = [
    "date",
    "close",
    "ma20",
    "ma50",
    "ma200",
    "rsi_14",
    "macd_hist",
    "volatility_30",
    "drawdown",
]

SIGNAL_OUTPUT_COLS = [
    "date",
    "ticker",
    "close",
    "trend_score",
    "momentum_score",
    "risk_score",
    "quant_score",
    "confidence",
    "raw_signal",
    "exec_signal",
    "target_position",
    "position_change",
]


class QuantStrategy:
    """Multi-factor scoring strategy that outputs Buy/Sell/Hold signals."""

    def __init__(
        self,
        ticker: str,
        processed_dir: Path | str = PROCESSED_DATA_DIR,
        output_dir: Path | str = SIGNALS_OUTPUT_DIR,
        weights: Mapping[str, float] | None = None,
        buy_threshold: float = DEFAULT_BUY_THRESHOLD,
        sell_threshold: float = DEFAULT_SELL_THRESHOLD,
        vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    ) -> None:
        self.ticker = ticker.upper().strip()
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.weights = dict(weights) if weights is not None else DEFAULT_WEIGHTS.copy()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.vol_lookback = vol_lookback

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_processed_price(self) -> pd.DataFrame:
        """Load and validate processed price data for the ticker."""
        file_path = self.processed_dir / f"{self.ticker}_processed.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed file not found: {file_path}")

        df = pd.read_csv(file_path)
        if "date" not in df.columns:
            raise ValueError(f"'date' column missing in {file_path}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        missing = [col for col in REQUIRED_PRICE_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"[{self.ticker}] Missing required columns: {missing}")

        numeric_cols = [
            "close",
            "ma20",
            "ma50",
            "ma200",
            "rsi_14",
            "macd_hist",
            "stoch_k",
            "roc_12",
            "volatility_30",
            "drawdown",
            "var_95",
            "atr_14",
            "relative_strength",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("[%s] Loaded processed rows=%d", self.ticker, len(df))
        return df

    def compute_trend_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute trend score in range [-1, 1]."""
        conditions = [
            (df["close"] > df["ma50"]) & (df["ma50"] > df["ma200"]),
            (df["close"] > df["ma50"]) & (df["ma50"] <= df["ma200"]),
            (df["close"] <= df["ma50"]) & (df["close"] > df["ma20"]),
            (df["close"] < df["ma50"]) & (df["ma50"] < df["ma200"]),
        ]
        base_scores = [1.0, 0.5, 0.0, -1.0]
        trend = np.select(conditions, base_scores, default=-0.5).astype(float)

        if "relative_strength" in df.columns:
            rs = pd.to_numeric(df["relative_strength"], errors="coerce")
            trend = trend + np.where(rs > 1.05, 0.2, 0.0) + np.where(rs < 0.95, -0.2, 0.0)

        return pd.Series(trend, index=df.index, name="trend_score").clip(-1.0, 1.0)

    def compute_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute momentum score in range [-1, 1]."""
        rsi = pd.to_numeric(df.get("rsi_14", pd.Series(np.nan, index=df.index)), errors="coerce")
        macd = pd.to_numeric(df.get("macd_hist", pd.Series(np.nan, index=df.index)), errors="coerce")
        roc = pd.to_numeric(df.get("roc_12", pd.Series(np.nan, index=df.index)), errors="coerce")
        stoch = pd.to_numeric(df.get("stoch_k", pd.Series(np.nan, index=df.index)), errors="coerce")

        rsi_sub = np.select(
            [
                rsi < 30,
                (rsi >= 30) & (rsi < 45),
                (rsi >= 45) & (rsi <= 55),
                (rsi > 55) & (rsi <= 70),
                rsi > 70,
            ],
            [1.0, 0.5, 0.0, -0.5, -1.0],
            default=0.0,
        )

        macd_prev = macd.shift(1)
        macd_sub = np.select(
            [
                (macd > 0) & (macd > macd_prev),
                (macd > 0) & (macd <= macd_prev),
                (macd < 0) & (macd < macd_prev),
                (macd < 0) & (macd >= macd_prev),
                macd == 0,
            ],
            [1.0, 0.5, -1.0, -0.5, 0.0],
            default=0.0,
        )

        roc_sub = np.select(
            [roc > 5, (roc >= -5) & (roc <= 5), roc < -5],
            [0.5, 0.0, -0.5],
            default=0.0,
        )

        stoch_sub = np.select(
            [stoch < 20, stoch > 80],
            [0.5, -0.5],
            default=0.0,
        )

        momentum = (rsi_sub + macd_sub + roc_sub + stoch_sub) / 4.0
        return pd.Series(momentum, index=df.index, name="momentum_score").clip(-1.0, 1.0)

    def compute_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Compute risk penalty score in range [-1, 0]."""
        vol = pd.to_numeric(df.get("volatility_30", pd.Series(np.nan, index=df.index)), errors="coerce")

        def last_rank(window_values: pd.Series) -> float:
            series = pd.Series(window_values)
            return float(series.rank(pct=True).iloc[-1])

        vol_rank = vol.rolling(self.vol_lookback, min_periods=30).apply(last_rank, raw=False)

        vol_penalty = np.select(
            [vol_rank > 0.80, vol_rank > 0.60],
            [-0.50, -0.25],
            default=0.0,
        )

        dd = pd.to_numeric(df.get("drawdown", pd.Series(np.nan, index=df.index)), errors="coerce")
        dd_penalty = np.select(
            [dd < -0.15, dd < -0.05],
            [-0.50, -0.25],
            default=0.0,
        )

        risk = vol_penalty + dd_penalty
        return pd.Series(risk, index=df.index, name="risk_score").clip(lower=-1.0, upper=0.0)

    def compute_quant_score(
        self,
        df: pd.DataFrame,
        trend_score: pd.Series,
        momentum_score: pd.Series,
        risk_score: pd.Series,
    ) -> pd.Series:
        """Compute weighted total score from component scores."""
        _ = df
        w_trend = float(self.weights.get("trend", 0.4))
        w_momentum = float(self.weights.get("momentum", 0.35))
        w_risk = float(self.weights.get("risk", 0.25))

        total_weight = w_trend + w_momentum + w_risk
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight:.6f}")

        quant_score = (w_trend * trend_score) + (w_momentum * momentum_score) + (w_risk * risk_score)
        return pd.Series(quant_score, index=df.index, name="quant_score")

    def build_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build final signal DataFrame for one ticker."""
        out = df.copy()

        out["trend_score"] = self.compute_trend_score(out)
        out["momentum_score"] = self.compute_momentum_score(out)
        out["risk_score"] = self.compute_risk_score(out)
        out["quant_score"] = self.compute_quant_score(
            out,
            out["trend_score"],
            out["momentum_score"],
            out["risk_score"],
        )
        out["confidence"] = out["quant_score"].abs().clip(0.0, 1.0)

        out["raw_signal"] = np.select(
            [
                out["quant_score"] >= self.buy_threshold,
                out["quant_score"] <= self.sell_threshold,
            ],
            ["Buy", "Sell"],
            default="Hold",
        )

        # Anti look-ahead: execute tomorrow based on today's closed-bar signal.
        out["exec_signal"] = out["raw_signal"].shift(1).fillna("Hold")

        # State-driven position: Buy -> 1, Sell -> 0, Hold -> keep previous state.
        state = np.where(out["exec_signal"] == "Buy", 1.0, np.nan)
        state = np.where(out["exec_signal"] == "Sell", 0.0, state)
        out["target_position"] = pd.Series(state, index=out.index).ffill().fillna(0.0).astype(int)
        out["position_change"] = out["target_position"].diff().fillna(out["target_position"]).astype(int)

        out["ticker"] = self.ticker
        out = out.dropna(subset=["close"])

        for col in SIGNAL_OUTPUT_COLS:
            if col not in out.columns:
                out[col] = np.nan

        return out[SIGNAL_OUTPUT_COLS].reset_index(drop=True)

    def save_signals(self, signal_df: pd.DataFrame) -> Path:
        """Save signal DataFrame to data/quant_outputs/signals."""
        out_path = self.output_dir / f"{self.ticker}_signals.csv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        signal_df.to_csv(out_path, index=False)
        logger.info("[%s] Signals saved: %s | rows=%d", self.ticker, out_path, len(signal_df))
        return out_path

    def run_for_ticker(self) -> pd.DataFrame:
        """Run full signal generation flow for one ticker."""
        df = self.load_processed_price()
        signal_df = self.build_signals(df)
        self.save_signals(signal_df)
        return signal_df

    @classmethod
    def run_for_universe(
        cls,
        tickers: list[str],
        processed_dir: Path | str = PROCESSED_DATA_DIR,
        output_dir: Path | str = SIGNALS_OUTPUT_DIR,
        weights: Mapping[str, float] | None = None,
        buy_threshold: float = DEFAULT_BUY_THRESHOLD,
        sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    ) -> dict[str, pd.DataFrame]:
        """Run strategy for multiple tickers. Errors are isolated per ticker."""
        results: dict[str, pd.DataFrame] = {}
        failed: list[str] = []

        for ticker in tickers:
            try:
                engine = cls(
                    ticker=ticker,
                    processed_dir=processed_dir,
                    output_dir=output_dir,
                    weights=weights,
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                )
                results[ticker] = engine.run_for_ticker()
            except Exception as exc:
                failed.append(ticker)
                logger.warning("[run_for_universe] %s failed: %s", ticker, exc)

        logger.info(
            "run_for_universe done | success=%d | failed=%d | failed_tickers=%s",
            len(results),
            len(failed),
            failed,
        )
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    sample_ticker = "AAPL"
    strategy = QuantStrategy(sample_ticker)
    signal_df = strategy.run_for_ticker()

    print("\n=== Tail 5 rows ===")
    print(signal_df.tail(5))
    print("\n=== Raw signal counts ===")
    print(signal_df["raw_signal"].value_counts(dropna=False))
    print("\n=== Position counts ===")
    print(signal_df["target_position"].value_counts(dropna=False))
