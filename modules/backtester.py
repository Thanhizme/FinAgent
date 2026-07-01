import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIGNALS_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "signals"
BACKTEST_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "backtests"


class Backtester:
	"""Vectorized long/cash backtester that consumes Stage 5 signal files."""

	def __init__(
		self,
		ticker: str,
		signals_dir: Path | str = SIGNALS_DIR,
		output_dir: Path | str = BACKTEST_DIR,
		initial_capital: float = 100_000.0,
		transaction_cost_bps: float = 10.0,
		slippage_bps: float = 5.0,
		trading_days: int = 252,
	) -> None:
		self.ticker = ticker.upper().strip()
		self.signals_dir = Path(signals_dir)
		self.output_dir = Path(output_dir)
		self.initial_capital = float(initial_capital)
		self.transaction_cost = float(transaction_cost_bps) / 10_000.0
		self.slippage = float(slippage_bps) / 10_000.0
		self.trading_days = int(trading_days)

		self.output_dir.mkdir(parents=True, exist_ok=True)

	def load_signals(self) -> pd.DataFrame:
		"""Load and validate <TICKER>_signals.csv produced by Stage 5."""
		signal_path = self.signals_dir / f"{self.ticker}_signals.csv"
		if not signal_path.exists():
			raise FileNotFoundError(f"Signal file not found: {signal_path}")

		df = pd.read_csv(signal_path)
		required_cols = ["date", "close", "target_position", "position_change", "exec_signal", "raw_signal"]
		missing = [c for c in required_cols if c not in df.columns]
		if missing:
			raise ValueError(f"[{self.ticker}] Missing signal columns: {missing}")

		df["date"] = pd.to_datetime(df["date"], errors="coerce")
		df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

		df["close"] = pd.to_numeric(df["close"], errors="coerce")
		df["target_position"] = pd.to_numeric(df["target_position"], errors="coerce").fillna(0).clip(0, 1)
		df["position_change"] = pd.to_numeric(df["position_change"], errors="coerce").fillna(0)

		if "ticker" not in df.columns:
			df["ticker"] = self.ticker
		else:
			df["ticker"] = df["ticker"].astype(str).str.upper().fillna(self.ticker)

		logger.info("[%s] loaded signal rows=%d", self.ticker, len(df))
		return df

	def run_backtest(self, signal_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		"""Run vectorized backtest and return (equity_curve, trades, metrics)."""
		bt = signal_df.copy()

		bt["daily_return"] = bt["close"].pct_change().fillna(0.0)
		bt["turnover"] = bt["target_position"].diff().abs().fillna(bt["target_position"].abs())

		gross_return = bt["target_position"] * bt["daily_return"]
		cost_per_turn = self.transaction_cost + self.slippage
		cost_return = bt["turnover"] * cost_per_turn
		net_return = gross_return - cost_return

		bt["gross_return"] = gross_return
		bt["cost_return"] = cost_return
		bt["strategy_return"] = net_return

		bt["equity"] = self.initial_capital * (1.0 + bt["strategy_return"]).cumprod()
		bt["buyhold_equity"] = self.initial_capital * (1.0 + bt["daily_return"]).cumprod()

		rolling_peak = bt["equity"].cummax()
		bt["drawdown"] = bt["equity"] / rolling_peak - 1.0

		equity_curve = bt[
			[
				"date",
				"ticker",
				"close",
				"target_position",
				"turnover",
				"daily_return",
				"strategy_return",
				"equity",
				"buyhold_equity",
				"drawdown",
				"raw_signal",
				"exec_signal",
			]
		].copy()

		trades = bt.loc[bt["turnover"] > 0, ["date", "ticker", "close", "target_position", "position_change", "turnover", "raw_signal", "exec_signal"]].copy()
		trades["action"] = np.where(trades["position_change"] > 0, "BUY", "SELL")
		trades = trades[["date", "ticker", "action", "close", "target_position", "position_change", "turnover", "raw_signal", "exec_signal"]]

		metrics = self._build_metrics(bt)
		return equity_curve, trades, metrics

	def _build_metrics(self, bt: pd.DataFrame) -> pd.DataFrame:
		"""Compute one-row performance metrics DataFrame."""
		if bt.empty:
			return pd.DataFrame([{"ticker": self.ticker}])

		first_date = bt["date"].iloc[0]
		last_date = bt["date"].iloc[-1]
		days = max((last_date - first_date).days, 1)
		years = max(days / 365.25, 1 / self.trading_days)

		strat_ret = bt["strategy_return"]
		bh_ret = bt["daily_return"]
		final_equity = float(bt["equity"].iloc[-1])
		final_bh_equity = float(bt["buyhold_equity"].iloc[-1])

		total_return = final_equity / self.initial_capital - 1.0
		buyhold_return = final_bh_equity / self.initial_capital - 1.0
		cagr = (final_equity / self.initial_capital) ** (1.0 / years) - 1.0
		buyhold_cagr = (final_bh_equity / self.initial_capital) ** (1.0 / years) - 1.0

		vol = float(strat_ret.std(ddof=0))
		bh_vol = float(bh_ret.std(ddof=0))
		sharpe = (np.sqrt(self.trading_days) * float(strat_ret.mean()) / vol) if vol > 0 else np.nan
		bh_sharpe = (np.sqrt(self.trading_days) * float(bh_ret.mean()) / bh_vol) if bh_vol > 0 else np.nan

		downside = strat_ret[strat_ret < 0]
		downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
		sortino = (np.sqrt(self.trading_days) * float(strat_ret.mean()) / downside_std) if downside_std > 0 else np.nan

		max_drawdown = float(bt["drawdown"].min())
		avg_daily_return = float(strat_ret.mean())
		win_rate = float((strat_ret > 0).mean())
		exposure = float(bt["target_position"].mean())

		turnover_events = int((bt["turnover"] > 0).sum())
		cost_paid_pct = float(bt["cost_return"].sum())

		return pd.DataFrame(
			[
				{
					"ticker": self.ticker,
					"start_date": first_date.date().isoformat(),
					"end_date": last_date.date().isoformat(),
					"rows": int(len(bt)),
					"initial_capital": self.initial_capital,
					"final_equity": final_equity,
					"total_return": total_return,
					"cagr": cagr,
					"max_drawdown": max_drawdown,
					"sharpe": sharpe,
					"sortino": sortino,
					"avg_daily_return": avg_daily_return,
					"win_rate": win_rate,
					"exposure": exposure,
					"turnover_events": turnover_events,
					"cost_paid_pct": cost_paid_pct,
					"buyhold_final_equity": final_bh_equity,
					"buyhold_total_return": buyhold_return,
					"buyhold_cagr": buyhold_cagr,
					"buyhold_sharpe": bh_sharpe,
				}
			]
		)

	def save_outputs(
		self,
		equity_curve: pd.DataFrame,
		trades: pd.DataFrame,
		metrics: pd.DataFrame,
	) -> dict[str, Path]:
		"""Persist backtest artifacts and return output paths."""
		eq_path = self.output_dir / f"{self.ticker}_equity_curve.csv"
		trade_path = self.output_dir / f"{self.ticker}_trades.csv"
		metrics_path = self.output_dir / f"{self.ticker}_metrics.csv"

		equity_curve.to_csv(eq_path, index=False)
		trades.to_csv(trade_path, index=False)
		metrics.to_csv(metrics_path, index=False)

		logger.info("[%s] backtest outputs saved | equity=%s | trades=%s | metrics=%s", self.ticker, eq_path, trade_path, metrics_path)
		return {"equity": eq_path, "trades": trade_path, "metrics": metrics_path}

	def run_for_ticker(self) -> dict[str, pd.DataFrame | dict[str, Path]]:
		"""Load signals, run backtest, save artifacts, and return all results."""
		signal_df = self.load_signals()
		equity_curve, trades, metrics = self.run_backtest(signal_df)
		output_paths = self.save_outputs(equity_curve, trades, metrics)
		return {
			"signal": signal_df,
			"equity_curve": equity_curve,
			"trades": trades,
			"metrics": metrics,
			"paths": output_paths,
		}

	@classmethod
	def run_for_universe(
		cls,
		tickers: list[str],
		signals_dir: Path | str = SIGNALS_DIR,
		output_dir: Path | str = BACKTEST_DIR,
		initial_capital: float = 100_000.0,
		transaction_cost_bps: float = 10.0,
		slippage_bps: float = 5.0,
	) -> dict[str, dict[str, pd.DataFrame | dict[str, Path]]]:
		"""Run backtests for all tickers; isolate failures per ticker."""
		results: dict[str, dict[str, pd.DataFrame | dict[str, Path]]] = {}
		failed: list[str] = []

		for ticker in tickers:
			try:
				engine = cls(
					ticker=ticker,
					signals_dir=signals_dir,
					output_dir=output_dir,
					initial_capital=initial_capital,
					transaction_cost_bps=transaction_cost_bps,
					slippage_bps=slippage_bps,
				)
				results[ticker] = engine.run_for_ticker()
			except Exception as exc:
				failed.append(ticker)
				logger.warning("[backtester.run_for_universe] %s failed: %s", ticker, exc)

		logger.info(
			"backtester.run_for_universe done | success=%d | failed=%d | failed_tickers=%s",
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

	sample = Backtester("AMZN")
	result = sample.run_for_ticker()

	print("\n=== Metrics ===")
	print(result["metrics"].to_string(index=False))
	print("\n=== Last 5 Equity Rows ===")
	print(result["equity_curve"].tail(5).to_string(index=False))
