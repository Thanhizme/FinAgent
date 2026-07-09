"""Stage 8: Risk overlay and position management (skeleton)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PORTFOLIO_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "portfolio"
RISK_DIR = Path(__file__).resolve().parents[1] / "data" / "quant_outputs" / "risk"


class RiskManager:
    """Risk overlay engine for Stage 7 portfolio outputs."""

    def __init__(
        self,
        portfolio_dir: Path | str = PORTFOLIO_DIR,
        output_dir: Path | str = RISK_DIR,
        strategy_names: list[str] | None = None,
        initial_capital: float = 100_000.0,
        target_vol: float = 0.15,
        vol_lookback: int = 20,
        min_exposure: float = 0.0,
        max_leverage: float = 1.0,
        drawdown_soft: float = -0.10,
        drawdown_hard: float = -0.15,
        soft_exposure: float = 0.50,
        asset_cap: float = 0.35,
        transaction_cost_bps: float = 5.0,
        trading_days: int = 252,
    ) -> None:
        self.portfolio_dir = Path(portfolio_dir)
        self.output_dir = Path(output_dir)
        self.strategy_names = strategy_names or ["equal_weight", "max_sharpe", "risk_parity"]
        self.initial_capital = float(initial_capital)
        self.target_vol = float(target_vol)
        self.vol_lookback = int(vol_lookback)
        self.min_exposure = float(min_exposure)
        self.max_leverage = float(max_leverage)
        self.drawdown_soft = float(drawdown_soft)
        self.drawdown_hard = float(drawdown_hard)
        self.soft_exposure = float(soft_exposure)
        self.asset_cap = float(asset_cap)
        self.transaction_cost_bps = float(transaction_cost_bps) / 10000
        self.trading_days = int(trading_days)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_portfolio_equity(self, strategy_name: str) -> pd.DataFrame:
        fp = self.portfolio_dir / f"{strategy_name}_equity.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Portfolio equity file not found: {fp}")

        df = pd.read_csv(fp)
        if "date" not in df.columns:
            raise ValueError(f"[{strategy_name}] Missing 'date' column in {fp}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    def load_portfolio_weights(self, strategy_name: str) -> dict[str, float]:
        fp = self.portfolio_dir / f"{strategy_name}_weights.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Portfolio weights file not found: {fp}")

        df = pd.read_csv(fp)
        if df.empty:
            raise ValueError(f"[{strategy_name}] Empty weights file: {fp}")

        row = df.iloc[0].to_dict()
        weights: dict[str, float] = {}
        for k, v in row.items():
            value = pd.to_numeric(v, errors="coerce")
            if pd.notna(value):
                weights[str(k)] = float(value)

        if not weights:
            raise ValueError(f"[{strategy_name}] No valid numeric weights in {fp}")
        return weights

    def _build_base_weights(self, columns: list[str], raw_weights: dict[str, float]) -> np.ndarray:
        vals = np.array([max(float(raw_weights.get(c, 0.0)), 0.0) for c in columns], dtype=float)
        total = float(vals.sum())
        if total <= 0:
            raise ValueError("Weight sum must be > 0")
        return vals / total

    def _drawdown_scale(self, drawdown: pd.Series) -> np.ndarray:
        dd = pd.to_numeric(drawdown, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.where(
            dd <= self.drawdown_hard,
            0.0,
            np.where(dd <= self.drawdown_soft, self.soft_exposure, 1.0),
        )

    def _vol_scale(self, returns: pd.Series) -> np.ndarray:
        realized_vol = (
            pd.to_numeric(returns, errors="coerce")
            .fillna(0.0)
            .rolling(self.vol_lookback, min_periods=max(5, self.vol_lookback // 2))
            .std(ddof=0)
            * np.sqrt(self.trading_days)
        )
        realized_vol = realized_vol.replace(0.0, np.nan)
        scale = self.target_vol / realized_vol
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        return scale.clip(self.min_exposure, self.max_leverage).to_numpy(dtype=float)

    def apply_risk_overlays(self, equity_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
        out = equity_df.copy()

        if "portfolio_return" in out.columns:
            base_ret = pd.to_numeric(out["portfolio_return"], errors="coerce").fillna(0.0)
        elif "portfolio_return_gross" in out.columns:
            base_ret = pd.to_numeric(out["portfolio_return_gross"], errors="coerce").fillna(0.0)
        else:
            raise ValueError("Missing portfolio return column in Stage 7 equity output")

        if "portfolio_drawdown" in out.columns:
            base_dd = pd.to_numeric(out["portfolio_drawdown"], errors="coerce").fillna(0.0)
        else:
            base_eq = self.initial_capital * (1.0 + base_ret).cumprod()
            base_dd = base_eq / base_eq.cummax() - 1.0

        vol_scale = self._vol_scale(base_ret)
        dd_scale = self._drawdown_scale(base_dd)
        gross_scale = np.clip(vol_scale * dd_scale, self.min_exposure, self.max_leverage)

        asset_cols = [c for c in out.columns if c in weights]
        if asset_cols:
            R = out[asset_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            base_w = self._build_base_weights(asset_cols, weights)
            n = len(out)
            W = np.zeros((n, len(asset_cols)), dtype=float)

            for i in range(n):
                w_scaled = base_w * float(gross_scale[i])
                w_capped = np.minimum(w_scaled, self.asset_cap)
                W[i, :] = np.maximum(w_capped, 0.0)

            gross_ret = np.sum(R * W, axis=1)
            risky_weight = W.sum(axis=1)
            cash_weight = 1.0 - risky_weight

            w_cols = [f"risk_weight_{c}" for c in asset_cols]
            w_df = pd.DataFrame(W, columns=w_cols, index=out.index)
            out = pd.concat([out, w_df], axis=1)
        else:
            gross_ret = base_ret.to_numpy(dtype=float) * gross_scale
            risky_weight = gross_scale
            cash_weight = 1.0 - risky_weight

        out["vol_scale"] = vol_scale
        out["drawdown_scale"] = dd_scale
        out["gross_scale"] = gross_scale
        out["risky_weight"] = risky_weight
        out["cash_weight"] = cash_weight

        turnover = np.abs(np.diff(risky_weight, prepend=0.0))
        out["risk_turnover"] = turnover
        out["risk_cost"] = out["risk_turnover"] * self.transaction_cost_bps

        out["risk_return_gross"] = gross_ret
        out["risk_return"] = out["risk_return_gross"] - out["risk_cost"]
        out["risk_equity"] = self.initial_capital * (1.0 + out["risk_return"]).cumprod()

        peak = out["risk_equity"].cummax()
        out["risk_drawdown"] = out["risk_equity"] / peak - 1.0
        return out

    def compute_risk_metrics(self, risk_df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        if risk_df.empty:
            return pd.DataFrame([{"strategy": strategy_name, "message": "empty result"}])

        r = pd.to_numeric(risk_df["risk_return"], errors="coerce").fillna(0.0)
        first_date = risk_df["date"].iloc[0]
        last_date = risk_df["date"].iloc[-1]

        days = max((last_date - first_date).days, 1)
        years = max(days / 365.25, 1.0 / 365.25)

        final_eq = float(risk_df["risk_equity"].iloc[-1])
        total_return = final_eq / self.initial_capital - 1.0
        cagr = (final_eq / self.initial_capital) ** (1.0 / years) - 1.0

        vol = float(r.std(ddof=0))
        sharpe = (np.sqrt(self.trading_days) * float(r.mean()) / vol) if vol > 1e-12 else np.nan

        downside = r[r < 0]
        downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
        sortino = (np.sqrt(self.trading_days) * float(r.mean()) / downside_std) if downside_std > 1e-12 else np.nan

        return pd.DataFrame(
            [
                {
                    "strategy": strategy_name,
                    "start_date": pd.to_datetime(first_date).date().isoformat(),
                    "end_date": pd.to_datetime(last_date).date().isoformat(),
                    "rows": int(len(risk_df)),
                    "initial_capital": self.initial_capital,
                    "final_equity": final_eq,
                    "total_return": total_return,
                    "cagr": cagr,
                    "volatility": vol,
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "max_drawdown": float(risk_df["risk_drawdown"].min()),
                    "avg_daily_return": float(r.mean()),
                    "win_rate": float((r > 0).mean()),
                    "avg_risky_weight": float(pd.to_numeric(risk_df["risky_weight"], errors="coerce").mean()),
                    "avg_cash_weight": float(pd.to_numeric(risk_df["cash_weight"], errors="coerce").mean()),
                    "turnover_events": int((pd.to_numeric(risk_df["risk_turnover"], errors="coerce") > 0).sum()),
                    "cost_paid_pct": float(pd.to_numeric(risk_df["risk_cost"], errors="coerce").sum()),
                }
            ]
        )

    def save_outputs(
        self,
        strategy_name: str,
        risk_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> dict[str, Path]:
        overlay_path = self.output_dir / f"{strategy_name}_risk_overlay.csv"
        metrics_path = self.output_dir / f"{strategy_name}_risk_metrics.csv"

        risk_df.to_csv(overlay_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        logger.info(
            "[%s] risk outputs saved | overlay=%s | metrics=%s",
            strategy_name,
            overlay_path,
            metrics_path,
        )
        return {"overlay": overlay_path, "metrics": metrics_path}

    def run_for_strategy(self, strategy_name: str) -> dict[str, pd.DataFrame | dict[str, Path]]:
        equity_df = self.load_portfolio_equity(strategy_name)
        weights = self.load_portfolio_weights(strategy_name)
        risk_df = self.apply_risk_overlays(equity_df, weights)
        metrics_df = self.compute_risk_metrics(risk_df, strategy_name)
        paths = self.save_outputs(strategy_name, risk_df, metrics_df)

        return {
            "risk_overlay": risk_df,
            "metrics": metrics_df,
            "paths": paths,
        }

    def run_for_universe(self) -> dict[str, dict[str, pd.DataFrame | dict[str, Path]]]:
        results: dict[str, dict[str, pd.DataFrame | dict[str, Path]]] = {}
        failed: list[str] = []

        for strategy_name in self.strategy_names:
            try:
                results[strategy_name] = self.run_for_strategy(strategy_name)
            except Exception as exc:
                failed.append(strategy_name)
                logger.warning("[risk_manager.run_for_universe] %s failed: %s", strategy_name, exc)

        logger.info(
            "risk_manager.run_for_universe done | success=%d | failed=%d | failed_strategies=%s",
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

    manager = RiskManager()
    output = manager.run_for_universe()

    for name, payload in output.items():
        print(f"\n=== {name} Risk Metrics ===")
        print(payload["metrics"].to_string(index=False))
