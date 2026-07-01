"""
Stage 7: Portfolio Optimization

Multi-asset portfolio construction using signals from Stage 5 and returns from Stage 6.
Implements equal-weight, max Sharpe, and risk parity allocation strategies.
"""

import logging
from pathlib import Path
from scipy.optimize import minimize

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SIGNALS_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "signals"
BACKTEST_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "backtests"
PORTFOLIO_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "portfolio"


class PortfolioOptimizer:
    """Multi-asset portfolio construction and backtesting engine."""

    def __init__(
        self,
        tickers: list[str],
        signals_dir: Path | str = SIGNALS_DIR,
        backtest_dir: Path | str = BACKTEST_DIR,
        output_dir: Path | str = PORTFOLIO_DIR,
        initial_capital: float = 100_000.0,
        rebalance_freq: str = "monthly",
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        transaction_cost_bps: float = 10.0,
    ) -> None:
        """Initialize portfolio optimizer with universe of tickers and constraints."""
        self.tickers = [t.upper().strip() for t in tickers]
        self.signals_dir = Path(signals_dir)
        self.backtest_dir = Path(backtest_dir)
        self.output_dir = Path(output_dir)

        self.initial_capital = float(initial_capital)
        self.rebalance_freq = str(rebalance_freq).lower()
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.transaction_cost = float(transaction_cost_bps)/10000.0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.asset_cols: list[str] = []

    def load_asset_returns(self) -> dict[str, pd.DataFrame]:
        """Load strategy_return series from Stage 6 equity curves for all tickers."""
        returns_dict: dict[str, pd.DataFrame] = {}

        for ticker in self.tickers:
            fp = self.backtest_dir / f"{ticker}_equity_curve.csv"
            if not fp.exists():
                logger.warning("[%s] missing equity curve: %s", ticker, fp)
                continue
            df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

            if "strategy_return" not in df.columns:
                logger.warning("[%s] missing strategy_return in %s", ticker, fp)
                continue

            sub = df[["date", "strategy_return"]].copy()
            sub.columns = ["date", ticker]
            returns_dict[ticker] = sub

        if not returns_dict:
            logger.warning("No valid asset returns loaded")
        else:
            logger.info("Loaded return series for %d tickers", len(returns_dict))

        return returns_dict



    def build_return_matrix(self, returns_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all asset returns into aligned matrix (date on rows, tickers on columns)."""
        if not returns_dict:
            raise ValueError("returns_dict is empty")
        
        keys = list(returns_dict.keys())
        matrix = returns_dict[keys[0]].copy()
        
        for k in keys[1:]:
            matrix = matrix.merge(returns_dict[k], on="date", how="inner")

        matrix = matrix.sort_values("date").drop_duplicates(subset=["date"])

        self.asset_cols = [c for c in matrix.columns if c != "date"]
        if not self.asset_cols:
            raise ValueError("No asset columns in return matrix")
        logger.info("Return matrix built: rows=%d, asset=%d", len(matrix), len(self.asset_cols))
        return matrix

    def compute_equal_weights(self) -> dict[str, float]:
        """Baseline strategy: assign equal weight to each ticker (1/n)."""
        cols = self.asset_cols if self.asset_cols else self.tickers
        n = len(cols)
        if n == 0:
            raise ValueError("No asset available")
        w = 1.0/n
        return {c: w for c in cols}

    def compute_max_sharpe_weights(
        self,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Optimize for max Sharpe ratio using scipy.optimize.minimize (SLSQP method)."""
        cols = [c for c in returns.columns if c != "date"]
        R = returns[cols].to_numpy(dtype=float)

        mu = R.mean(axis=0)
        cov = np.cov(R.T)
        n = len(cols)

        if n==1:
            return {cols[0]: 1.0}
        
        def objective(w: np.ndarray) -> float:
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(w @ cov @ w))
            if port_vol <= 1e-12:
                return 1e6
            return -(port_ret / port_vol)
        
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) -1.0}]
        bnds = [(self.min_weight, self.max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        res = minimize(objective, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)

        if not res.success:
            logger.warning("Max Sharpe optimization failed: %s. Fallback to equal weight", res.message)
            return {k: float(v) for k,v in zip(cols, np.ones(n) / n)}
        w = np.clip(res.x, self.min_weight, self.max_weight)
        w = w / w.sum()
        return {k: float(v) for k, v in zip(cols, w)}

    def compute_risk_parity_weights(
        self,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """Risk parity: inverse volatility weighting normalized to sum to 1.0."""
        cols = [c for c in returns.columns if c != "date"]
        R = returns[cols].to_numpy(dtype=float)

        vol = R.std(axis=0, ddof=0)
        vol = np.where(vol <= 1e-12, 1e-12, vol)

        inv_vol = 1.0 / vol
        w = inv_vol / inv_vol.sum()

        w = np.clip(w, self.min_weight, self.max_weight)
        w = w / w.sum()
        return {k: float(v) for k, v in zip(cols, w)}

    def apply_rebalance_schedule(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
        freq: str = "monthly",
    ) -> pd.DataFrame:
        """Extend single weight snapshot into time series, optionally with rebalance dates."""
        out = returns.copy()
        cols = [c for c in out.columns if c != "date"]

        for c in cols:
            out[f"weight_{c}"] = float(weights.get(c, 0.0))

        out["rebalance_flag"] = False
        if len(out) > 0:
            out.loc[out.index[0], "rebalance_flag"] = True
        return out

    def backtest_portfolio(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float],
    ) -> pd.DataFrame:
        """Run portfolio backtest: compute weighted portfolio returns and equity curve."""
        bt = returns.copy()
        cols = [c for c in bt.columns if c != "date"]
        w = np.array([float(weights.get(c, 0.0)) for c in cols], dtype=float)
        if w.sum() <=0:
            raise ValueError("Weight sum must be >0")
        
        w = w / w.sum()
        R = bt[cols].to_numpy(dtype=float)
        bt["portfolio_return_gross"] = R @ w

        bt["turnover"] = 0.0
        if len(bt) > 0:
            bt.loc[bt.index[0], "turnover"] = float(np.abs(w).sum())
        
        bt["portfolio_cost"] = bt["turnover"] * self.transaction_cost
        bt["portfolio_return"] = bt["portfolio_return_gross"] - bt["portfolio_cost"]

        bt["portfolio_equity"] = self.initial_capital * (1.0 + bt["portfolio_return"]).cumprod()

        bt["benchmark_return"] = bt[cols].mean(axis=1)
        bt["benchmark_equity"] = self.initial_capital * (1.0 + bt["benchmark_return"]).cumprod()

        peak = bt["portfolio_equity"].cummax()
        bt["portfolio_drawdown"] = bt["portfolio_equity"] / peak - 1.0

        return bt
        


    def compute_portfolio_metrics(
        self,
        bt_result: pd.DataFrame,
        static_weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Compute performance metrics: total_return, CAGR, Sharpe, Sortino, MDD, win_rate."""
        if bt_result.empty:
            return pd.DataFrame([{"strategy": "portfolio", "message": "empty result"}])
        r = bt_result["portfolio_return"]
        br = bt_result["benchmark_return"]

        first_date = bt_result["date"].iloc[0]
        last_date = bt_result["date"].iloc[-1]
        days = max((last_date - first_date).days, 1)
        years = max(days / 365.25, 1.0 / 365.25)

        final_eq = float(bt_result["portfolio_equity"].iloc[-1])
        final_bm = float(bt_result["benchmark_equity"].iloc[-1])

        total_return = final_eq / self.initial_capital - 1.0
        cagr = (final_eq / self.initial_capital) ** (1.0 / years) - 1.0

        vol = float(r.std(ddof=0))
        sharpe = (np.sqrt(252) * float(r.mean()) / vol) if vol > 1e-12 else np.nan

        downside = r[r < 0]
        downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
        sortino = (np.sqrt(252) * float(r.mean()) / downside_std) if downside_std > 1e-12 else np.nan

        max_dd = float(bt_result["portfolio_drawdown"].min())
        win_rate = float((r > 0).mean())
        avg_daily_return = float(r.mean())

        bench_total_return = final_bm / self.initial_capital -1.0
        bench_vol = float(br.std(ddof=0))
        bench_sharpe = (np.sqrt(252) * float(br.mean())/ bench_vol) if bench_vol > 1e-12 else np.nan

        turnover_events = int((bt_result["turnover"] > 0).sum())
        cost_paid_pct = float(bt_result["portfolio_cost"].sum())
        out = pd.DataFrame([{
            "strategy": "portfolio",
            "start_date": pd.to_datetime(first_date).date().isoformat(),
            "end_date": pd.to_datetime(last_date).date().isoformat(),
            "rows": int(len(bt_result)),
            "initial_capital": self.initial_capital,
            "final_equity": final_eq,
            "total_return": total_return,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "avg_daily_return": avg_daily_return,
            "win_rate": win_rate,
            "turnover_events": turnover_events,
            "cost_paid_pct": cost_paid_pct,
            "benchmark_final_equity": final_bm,
            "benchmark_total_return": bench_total_return,
            "benchmark_sharpe": bench_sharpe,
            "weights": str(static_weights) if static_weights is not None else "",
        }])

        return out


    def save_portfolio_outputs(
        self,
        bt_result: pd.DataFrame,
        metrics: pd.DataFrame,
        weights: dict[str, float],
        strategy_name: str = "portfolio",
    ) -> dict[str, Path]:
        """Save equity curve, metrics, and weights to CSV files."""
        eq_path = self.output_dir / f"{strategy_name}_equity.csv"
        metrics_path = self.output_dir / f"{strategy_name}_metrics.csv"
        weights_path = self.output_dir / f"{strategy_name}_weights.csv"

        bt_result.to_csv(eq_path, index=False)
        metrics.to_csv(metrics_path, index=False)
        pd.DataFrame([weights]).to_csv(weights_path, index=False)

        logger.info(
            "[%s] saved outputs | equity=%s | metrics=%s | weights=%s",
            strategy_name, eq_path, metrics_path, weights_path
        )
        return {"equity": eq_path, "metrics": metrics_path, "weights": weights_path}

    def run_for_universe(self) -> dict[str, dict]:
        """Orchestrate full portfolio optimization: load, optimize, backtest, save for all strategies."""
        returns_dict = self.load_asset_returns()
        matrix = self.build_return_matrix(returns_dict)

        results: dict[str, dict] = {}

        # Equal Weight
        ew = self.compute_equal_weights()
        ew_bt = self.backtest_portfolio(matrix, ew)
        ew_m = self.compute_portfolio_metrics(ew_bt, ew)
        ew_p = self.save_portfolio_outputs(ew_bt, ew_m, ew, strategy_name="equal_weight")
        results["equal_weight"] = {"weights": ew, "backtest": ew_bt, "metrics": ew_m, "paths": ew_p}

        # Max Sharpe
        ms = self.compute_max_sharpe_weights(matrix)
        ms_bt = self.backtest_portfolio(matrix, ms)
        ms_m = self.compute_portfolio_metrics(ms_bt, ms)
        ms_p = self.save_portfolio_outputs(ms_bt, ms_m, ms, strategy_name="max_sharpe")
        results["max_sharpe"] = {"weights": ms, "backtest": ms_bt, "metrics": ms_m, "paths": ms_p}

        # Risk Parity
        rp = self.compute_risk_parity_weights(matrix)
        rp_bt = self.backtest_portfolio(matrix, rp)
        rp_m = self.compute_portfolio_metrics(rp_bt, rp)
        rp_p = self.save_portfolio_outputs(rp_bt, rp_m, rp, strategy_name="risk_parity")
        results["risk_parity"] = {"weights": rp, "backtest": rp_bt, "metrics": rp_m, "paths": rp_p}

        logger.info("Portfolio optimization complete. strategies=%s", list(results.keys()))
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    optimizer = PortfolioOptimizer(["AAPL", "MSFT", "TSLA", "AMZN"])
    out = optimizer.run_for_universe()

    print("\n=== Equal Weight Metrics ===")
    print(out["equal_weight"]["metrics"].to_string(index=False))

    print("\n=== Max Sharpe Metrics ===")
    print(out["max_sharpe"]["metrics"].to_string(index=False))

    print("\n=== Risk Parity Metrics ===")
    print(out["risk_parity"]["metrics"].to_string(index=False))
