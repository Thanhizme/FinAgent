"""
visualizer.py
-------------
Generates the four required chart types for the FinAgent pipeline.

Chart catalogue:
  1. price_trend_chart     : Price trend line with volume overlay (candlestick optional)
  2. correlation_heatmap   : Correlation matrix across selected assets / indicators
  3. returns_distribution  : Histogram / KDE of daily returns per asset
  4. rolling_stats_chart   : Moving averages + Bollinger Bands overlay

All figures can be rendered interactively (Plotly) or saved to disk as PNG/HTML.
Recommended libraries: plotly, matplotlib, seaborn.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import CCIIndicator, MACD

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DataVisualizer:
    """
    Produces publication-quality financial charts from processed DataFrames.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping of ticker symbol ??' processed DataFrame (output of DataProcessor).
    output_dir : Path, optional
        Directory where chart files are saved. Defaults to data/processed/visualization/.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ) -> None:
        self.data = data
        self.output_dir = output_dir or OUTPUT_DIR

    def _get_ticker_frame(self, ticker: str) -> pd.DataFrame:
        if ticker not in self.data:
            raise KeyError(f"Ticker '{ticker}' not found in visualizer data")

        df = self.data[ticker].copy()
        if "date" not in df.columns:
            raise ValueError(f"Ticker '{ticker}' data has no 'date' column")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        timeframe = (timeframe or "daily").lower()
        if timeframe in {"d", "day", "daily"}:
            return df.copy()

        freq_map = {
            "weekly": "W-FRI",
            "w": "W-FRI",
            "month": "ME",
            "monthly": "ME",
            "m": "ME",
            "quarter": "Q",
            "quarterly": "Q",
            "year": "YE",
            "yearly": "YE",
            "y": "YE",
        }
        freq = freq_map.get(timeframe)
        if freq is None:
            raise ValueError("timeframe must be one of: daily, weekly, monthly, yearly")

        if "date" not in df.columns:
            raise ValueError("DataFrame must contain a 'date' column for resampling")

        frame = df.set_index("date")
        if not any(c in frame.columns for c in ["open", "high", "low", "close", "volume"]):
            raise ValueError("No OHLCV columns available for resampling")

        grouped = frame.resample(freq)
        resampled = pd.DataFrame(index=grouped.size().index)

        if "open" in frame.columns:
            resampled["open"] = grouped["open"].first()
        if "high" in frame.columns:
            resampled["high"] = grouped["high"].max()
        if "low" in frame.columns:
            resampled["low"] = grouped["low"].min()
        if "close" in frame.columns:
            resampled["close"] = grouped["close"].last()
        if "volume" in frame.columns:
            resampled["volume"] = grouped["volume"].sum()

        if "close" in resampled.columns:
            resampled = resampled.dropna(subset=["close"])
        resampled = resampled.reset_index()
        return resampled

    def _ensure_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            return df
        out = df.copy()
        out["ma20"] = out["close"].rolling(20).mean()
        out["ma50"] = out["close"].rolling(50).mean()
        out["ma200"] = out["close"].rolling(200).mean()
        rolling_std20 = out["close"].rolling(20).std()
        out["bb_upper"] = out["ma20"] + 2.0 * rolling_std20
        out["bb_lower"] = out["ma20"] - 2.0 * rolling_std20
        return out

    def _normalise_timeframes(self, timeframe: str) -> list[str]:
        key = (timeframe or "daily").lower()
        if key in {"all", "*"}:
            return ["daily", "weekly", "monthly", "yearly"]
        return [key]

    def _save_figure(self, fig: go.Figure, filename_stub: str, save: bool) -> None:
        if not save:
            return

        html_path = self.output_dir / f"{filename_stub}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    def price_trend_chart(
        self,
        ticker: str,
        chart_type: str = "candlestick",
        timeframe: str = "daily",
        save: bool = True,
    ) -> None:
        """
        Render a price trend chart with a volume bar overlay on a secondary axis.

        Parameters
        ----------
        ticker : str
            Ticker symbol to plot (must exist in self.data).
        chart_type : {'line', 'candlestick', 'ohlc'}
            Visual encoding for price.
        timeframe : {'daily', 'weekly', 'monthly', 'yearly'}
            Aggregation window used to build the chart.
        save : bool
            Whether to export the figure to output_dir.

        Notes
        -----
        Expected columns in the DataFrame: open, high, low, close, volume.
        """
        df = self._get_ticker_frame(ticker)
        df = self._resample_ohlcv(df, timeframe)
        df = self._ensure_moving_averages(df)

        missing = [c for c in ["close", "volume"] if c not in df.columns]
        if missing:
            raise ValueError(f"Ticker '{ticker}' data missing required columns: {', '.join(missing)}")

        title_timeframe = timeframe.capitalize()
        title = f"{ticker} Price & Volume Master Chart ({title_timeframe})"

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        chart_type = (chart_type or "candlestick").lower()
        if chart_type == "candlestick" and all(col in df.columns for col in ["open", "high", "low", "close"]):
            fig.add_trace(
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="Price",
                    increasing_line_color="#22c55e",
                    decreasing_line_color="#ef4444",
                    increasing_fillcolor="#22c55e",
                    decreasing_fillcolor="#ef4444",
                    whiskerwidth=0.4,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        elif chart_type == "ohlc" and all(col in df.columns for col in ["open", "high", "low", "close"]):
            fig.add_trace(
                go.Ohlc(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="Price",
                    increasing_line_color="#22c55e",
                    decreasing_line_color="#ef4444",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["close"],
                    mode="lines",
                    line=dict(color="#60a5fa", width=2),
                    name="Close",
                    hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        ma_styles = {
            "ma50": {"color": "#a855f7", "width": 2.1},
            "ma200": {"color": "#22c55e", "width": 2.4},
        }
        for col, style in ma_styles.items():
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[col],
                        mode="lines",
                        line=dict(color=style["color"], width=style["width"]),
                        name=col.upper(),
                        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{col.upper()}: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["bb_upper"],
                    mode="lines",
                    line=dict(color="rgba(56, 189, 248, 0.9)", width=1.2, dash="dot"),
                    name="BB Upper",
                    hovertemplate="%{x|%Y-%m-%d}<br>BB Upper: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["bb_lower"],
                    mode="lines",
                    line=dict(color="rgba(56, 189, 248, 0.9)", width=1.2, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(59, 130, 246, 0.10)",
                    name="BB Lower",
                    hovertemplate="%{x|%Y-%m-%d}<br>BB Lower: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if "volume" in df.columns:
            if "open" in df.columns:
                volume_colors = [
                    "#22c55e" if c >= o else "#ef4444"
                    for c, o in zip(df["close"], df["open"])
                ]
            else:
                volume_colors = ["#94a3b8"] * len(df)

            bar_width = None
            if len(df) > 1:
                step = df["date"].diff().dropna().median()
                if pd.notna(step):
                    step_delta = pd.to_timedelta(step, errors="coerce")
                    if pd.notna(step_delta):
                        step_ms = float(step_delta / pd.Timedelta(milliseconds=1))
                        bar_width = max(step_ms * 0.72, 12 * 60 * 60 * 1000)

            fig.add_trace(
                go.Bar(
                    x=df["date"],
                    y=df["volume"],
                    width=bar_width,
                    marker_color=volume_colors,
                    marker_line_width=0,
                    opacity=0.94,
                    name="Volume",
                    hovertemplate="%{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            # Removed VOL MA20 trace

        fig.update_layout(
            title=dict(text=title, x=0.02, xanchor="left", y=0.97, yanchor="top"),
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=13),
            margin=dict(l=60, r=30, t=100, b=50),
            height=900,
            bargap=0.05,
            bargroupgap=0.0,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.65)",
                bordercolor="rgba(148, 163, 184, 0.2)",
                borderwidth=1,
            ),
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                title=None,
                rangeselector=dict(
                    x=0.01,
                    y=1.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="rgba(15, 23, 42, 0.95)",
                    activecolor="rgba(56, 189, 248, 0.35)",
                    bordercolor="rgba(148, 163, 184, 0.35)",
                    borderwidth=1,
                    font=dict(color="#e2e8f0", size=11),
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ],
                ),
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.12)",
                zeroline=False,
                title="Price",
            ),
            xaxis2=dict(showgrid=False, title=None),
            yaxis2=dict(
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.10)",
                title="Volume",
            ),
        )

        fig.update_xaxes(
            rangeslider_visible=False,
            showline=True,
            linecolor="rgba(148, 163, 184, 0.25)",
            mirror=False,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            showline=True,
            linecolor="rgba(148, 163, 184, 0.18)",
            row=2,
            col=1,
        )
        fig.update_yaxes(row=1, col=1, fixedrange=False)
        fig.update_yaxes(row=2, col=1, fixedrange=False)

        fig.add_annotation(
            text=f"{ticker} • {len(df)} bars • {timeframe.upper()}",
            xref="paper",
            yref="paper",
            x=0.995,
            y=1.12,
            showarrow=False,
            font=dict(size=12, color="#94a3b8"),
            align="right",
        )

        filename_stub = f"{ticker.lower()}_price_volume_{timeframe.lower()}"
        self._save_figure(fig, filename_stub, save)
        logger.info("Saved price trend chart -> %s", self.output_dir / f"{filename_stub}.html")

    def _returns_series(self, frame: pd.DataFrame) -> pd.Series:
        if "daily_return" in frame.columns:
            returns = pd.to_numeric(frame["daily_return"], errors="coerce")
        elif "close" in frame.columns:
            close = pd.to_numeric(frame["close"], errors="coerce")
            returns = close.pct_change()
        else:
            return pd.Series(dtype=float)
        return returns.replace([np.inf, -np.inf], np.nan)

    def indicator_correlation_heatmap(
        self,
        ticker_a: str,
        ticker_b: Optional[str] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
        benchmark_label: str = "Benchmark",
        min_periods: int = 8,
        save: bool = True,
    ) -> Optional[go.Figure]:
        """Correlation heatmap across selected indicators for A/B/Benchmark."""
        if ticker_a not in self.data:
            return None

        # User-configured view: use primary ticker only; concise Trend/Return signals.
        indicator_specs = [
            ("trend", "daily_ret", "daily_return", "DRET"),
            ("trend", "volume", "volume", "VOL"),
            ("osc", "ma20", "ma20", "MA20"),
            ("osc", "ma50", "ma50", "MA50"),
            ("osc", "ma200", "ma200", "MA200"),
            ("osc", "bb_upper", "bb_upper", "BBU"),
            ("osc", "bb_lower", "bb_lower", "BBL"),
            ("osc", "rsi", "rsi_14", "RSI"),
            ("osc", "macd", "macd_line", "MACD"),
            ("risk", "hv30", "volatility_30", "HV30"),
            ("risk", "hv60", "volatility_60", "HV60"),
            ("risk", "beta", "beta", "BETA"),
            ("risk", "var95", "var_95", "VaR95"),
            ("risk", "var99", "var_99", "VaR99"),
            ("risk", "max_dd", "max_drawdown", "MDD"),
            ("risk", "sharpe", "sharpe_ratio", "SHRP"),
        ]
        indicator_groups = {
            "trend": "TR",
            "osc": "OSC",
            "risk": "RK",
        }

        def _ensure_indicator_columns(work: pd.DataFrame) -> pd.DataFrame:
            out = work.copy()
            if "daily_return" not in out.columns and "close" in out.columns:
                close = pd.to_numeric(out["close"], errors="coerce")
                out["daily_return"] = close.pct_change()
            if "log_return" not in out.columns and "close" in out.columns:
                close = pd.to_numeric(out["close"], errors="coerce")
                out["log_return"] = np.log(close / close.shift(1))

            if "close" in out.columns:
                close = pd.to_numeric(out["close"], errors="coerce")
                if "cum_return_7" not in out.columns:
                    out["cum_return_7"] = close / close.shift(7) - 1
                if "cum_return_30" not in out.columns:
                    out["cum_return_30"] = close / close.shift(30) - 1
                if "cum_return_90" not in out.columns:
                    out["cum_return_90"] = close / close.shift(90) - 1
                if "cum_return_ytd" not in out.columns:
                    year = pd.to_datetime(out["date"], errors="coerce").dt.year
                    first_close_of_year = close.groupby(year).transform("first")
                    out["cum_return_ytd"] = close / first_close_of_year - 1
                if "ma20" not in out.columns:
                    out["ma20"] = close.rolling(20).mean()
                if "ma50" not in out.columns:
                    out["ma50"] = close.rolling(50).mean()
                if "ma200" not in out.columns:
                    out["ma200"] = close.rolling(200).mean()
                rolling_std20 = close.rolling(20).std()
                if "bb_upper" not in out.columns:
                    out["bb_upper"] = out["ma20"] + 2.0 * rolling_std20
                if "bb_lower" not in out.columns:
                    out["bb_lower"] = out["ma20"] - 2.0 * rolling_std20
                if "max_drawdown" not in out.columns:
                    rolling_max = close.cummax()
                    drawdown = (close - rolling_max) / rolling_max
                    out["max_drawdown"] = drawdown.cummin()

            if "daily_return" in out.columns:
                ret = pd.to_numeric(out["daily_return"], errors="coerce")
                if "volatility_30" not in out.columns:
                    out["volatility_30"] = ret.rolling(30).std()
                if "volatility_60" not in out.columns:
                    out["volatility_60"] = ret.rolling(60).std()
                if "var_95" not in out.columns:
                    out["var_95"] = ret.rolling(252, min_periods=252).quantile(0.05)
                if "var_99" not in out.columns:
                    out["var_99"] = ret.rolling(252, min_periods=252).quantile(0.01)

                if "sharpe_ratio" not in out.columns:
                    rolling_mean = ret.rolling(252, min_periods=252).mean()
                    rolling_std = ret.rolling(252, min_periods=252).std()
                    annualized_return = rolling_mean * 252
                    annualized_volatility = rolling_std * np.sqrt(252)
                    out["sharpe_ratio"] = (annualized_return / annualized_volatility.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

            if "beta" not in out.columns and "ticker" in out.columns:
                ticker_values = out["ticker"].dropna().astype(str).str.upper().unique()
                if len(ticker_values) == 1 and ticker_values[0] in {"^VNINDEX", "VNINDEX", "^GSPC"}:
                    out["beta"] = 1.0

            return out

        def _frame_to_indicator_columns(frame: pd.DataFrame, label: str) -> pd.DataFrame:
            work = frame.copy()
            work["date"] = pd.to_datetime(work["date"], errors="coerce")
            work = work.dropna(subset=["date"]).sort_values("date")
            work = _ensure_indicator_columns(work)
            out = pd.DataFrame({"date": work["date"]})
            for group, alias, col, short in indicator_specs:
                if col in work.columns:
                    series = work[col]
                    if series.dtype == bool:
                        values = series.astype(int)
                    else:
                        values = pd.to_numeric(series, errors="coerce")
                    out[f"{label}|{indicator_groups[group]}:{short}"] = values
            return out

        frame_a = self._get_ticker_frame(ticker_a)
        merged = _frame_to_indicator_columns(frame_a, ticker_a)
        labels: list[str] = [ticker_a]
        if merged.empty:
            return None

        numeric = merged.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
        valid_cols = [c for c in numeric.columns if numeric[c].notna().sum() >= min_periods]
        if len(valid_cols) < 2:
            return None

        corr = numeric[valid_cols].corr(method="pearson", min_periods=min_periods).round(2)

        axis_labels = [
            col.split("|", 1)[-1].replace(":", " ")
            for col in corr.columns
        ]
        n_cols = len(corr.columns)
        text_size = 8 if n_cols <= 36 else 6

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=corr.values,
                    x=axis_labels,
                    y=axis_labels,
                    zmin=-1,
                    zmax=1,
                    colorscale=[[0.0, "#1e3a8a"], [0.5, "#0f172a"], [1.0, "#b91c1c"]],
                    text=corr.values,
                    texttemplate="%{text:.2f}",
                    textfont=dict(size=text_size, color="#e2e8f0"),
                    hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
                    colorbar=dict(title="Corr"),
                )
            ]
        )
        fig.update_layout(
            title=dict(
                text=f"Asset Indicator Correlation: {' vs '.join(labels)}",
                x=0.02,
                xanchor="left",
            ),
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=11),
            margin=dict(l=160, r=70, t=80, b=150),
            height=max(820, min(1500, 24 * n_cols)),
        )
        fig.update_xaxes(tickangle=-45, side="bottom")
        fig.update_yaxes(autorange="reversed")

        filename_stub = f"indicator_corr_{ticker_a}"
        self._save_figure(fig, filename_stub, save)
        logger.info("Saved indicator correlation heatmap -> %s", self.output_dir / f"{filename_stub}.html")
        return fig

    def asset_return_correlation_heatmap(
        self,
        ticker_a: str,
        ticker_b: str,
        benchmark_df: Optional[pd.DataFrame] = None,
        benchmark_label: str = "Benchmark",
        min_periods: int = 8,
        save: bool = True,
    ) -> Optional[go.Figure]:
        """Correlation matrix of aligned daily returns for A/B/Benchmark."""
        if ticker_a not in self.data or ticker_b not in self.data:
            return None

        frame_a = self._get_ticker_frame(ticker_a)
        frame_b = self._get_ticker_frame(ticker_b)

        ret_a = pd.DataFrame({"date": frame_a["date"], ticker_a: self._returns_series(frame_a)})
        ret_b = pd.DataFrame({"date": frame_b["date"], ticker_b: self._returns_series(frame_b)})
        merged = ret_a.merge(ret_b, on="date", how="inner")

        if benchmark_df is not None and not benchmark_df.empty and "date" in benchmark_df.columns:
            bm = benchmark_df.copy()
            bm["date"] = pd.to_datetime(bm["date"], errors="coerce")
            bm = bm.dropna(subset=["date"]).sort_values("date")
            ret_bm = pd.DataFrame({"date": bm["date"], benchmark_label: self._returns_series(bm)})
            merged = merged.merge(ret_bm, on="date", how="inner")

        merged = merged.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if merged.empty:
            return None

        numeric = merged.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
        valid_cols = [c for c in numeric.columns if numeric[c].notna().sum() >= min_periods]
        if len(valid_cols) < 2:
            return None

        corr = numeric[valid_cols].corr(method="pearson", min_periods=min_periods).round(2)
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    zmin=-1,
                    zmax=1,
                    colorscale=[[0.0, "#1e3a8a"], [0.5, "#0f172a"], [1.0, "#b91c1c"]],
                    text=corr.values,
                    texttemplate="%{text:.2f}",
                    hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=dict(text=f"Asset Return Correlation: {ticker_a} vs {ticker_b}", x=0.02, xanchor="left"),
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=12),
            margin=dict(l=110, r=60, t=70, b=80),
            height=520,
        )
        fig.update_yaxes(autorange="reversed")
        filename_stub = "portfolio_asset_correlation"
        self._save_figure(fig, filename_stub, save)
        logger.info("Saved asset return correlation heatmap -> %s", self.output_dir / f"{filename_stub}.html")
        return fig

    def correlation_heatmap(
        self,
        ticker: Optional[str] = None,
        columns: Optional[list[str]] = None,
        save: bool = True,
    ) -> None:
        """Backward-compatible wrapper used by current app flows."""
        _ = columns
        selected_ticker = ticker or next(iter(self.data.keys()))
        self.indicator_correlation_heatmap(ticker_a=selected_ticker, save=save)

    def returns_distribution(
        self,
        tickers: Optional[list[str]] = None,
        plot_type: str = "both",
        save: bool = True,
    ) -> None:
        """
        Visualise the distribution of daily returns for one or more assets.

        Parameters
        ----------
        tickers : list[str], optional
            Subset of tickers to plot. Defaults to all tickers in self.data.
        plot_type : {'histogram', 'kde', 'both'}
            Type of distribution visualisation.
        save : bool
            Whether to export the figure to output_dir.

        Notes
        -----
        Overlay a normal distribution curve for reference.
        Annotate with mean and standard deviation statistics.
        """
        selected = tickers or list(self.data.keys())
        selected = [t for t in selected if t in self.data]
        if not selected:
            raise ValueError("No valid tickers supplied for returns_distribution")

        plot_type = (plot_type or "both").lower()
        if plot_type not in {"histogram", "kde", "both"}:
            raise ValueError("plot_type must be one of: histogram, kde, both")

        for ticker in selected:
            df = self._get_ticker_frame(ticker)
            if "daily_return" not in df.columns:
                if "close" not in df.columns:
                    raise ValueError(f"Ticker '{ticker}' data requires daily_return or close column")
                returns = df["close"].pct_change().dropna()
            else:
                returns = df["daily_return"].dropna()

            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            if returns.empty:
                raise ValueError(f"Ticker '{ticker}' has no valid return values")

            q01 = returns.quantile(0.01)
            q99 = returns.quantile(0.99)
            clipped = returns.clip(lower=q01, upper=q99)

            iqr = float(clipped.quantile(0.75) - clipped.quantile(0.25))
            n = len(clipped)
            span = float(clipped.max() - clipped.min())
            if iqr > 0 and n > 1:
                bin_width = max(2 * iqr / (n ** (1 / 3)), 1e-6)
            else:
                bin_width = max(span / 40 if span > 0 else 1e-4, 1e-6)
            bins = int(np.clip(np.ceil(span / bin_width) if span > 0 else 30, 25, 90))

            fig = go.Figure()

            if plot_type in {"histogram", "both"}:
                fig.add_trace(
                    go.Histogram(
                        x=clipped,
                        nbinsx=bins,
                        name="Histogram",
                        showlegend=True,
                        marker=dict(color="rgba(96, 165, 250, 0.55)", line=dict(width=0)),
                        hovertemplate="Return: %{x:.3%}<br>Frequency: %{y}<extra></extra>",
                    )
                )

            if plot_type in {"kde", "both"}:
                kde_added = False
                x_values = clipped.to_numpy(dtype=float)

                if len(x_values) > 5 and np.nanstd(x_values) > 1e-12:
                    try:
                        kde = gaussian_kde(x_values)
                        x_grid = np.linspace(float(clipped.min()), float(clipped.max()), 300)
                        density = kde(x_grid)

                        scale = len(clipped) * (span / bins if bins > 0 and span > 0 else 1)
                        y_kde = density * scale
                        if np.isfinite(y_kde).any() and float(np.nanmax(y_kde)) > 0:
                            fig.add_trace(
                                go.Scatter(
                                    x=x_grid,
                                    y=y_kde,
                                    mode="lines",
                                    line=dict(color="#22d3ee", width=2.6),
                                    name="KDE",
                                    showlegend=True,
                                    hovertemplate="Return: %{x:.3%}<br>Density (scaled): %{y:.2f}<extra></extra>",
                                )
                            )
                            kde_added = True
                    except Exception:
                        kde_added = False

                if not kde_added and len(x_values) > 1:
                    counts, edges = np.histogram(x_values, bins=bins)
                    centers = (edges[:-1] + edges[1:]) / 2
                    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
                    kernel = kernel / kernel.sum()
                    smooth_counts = np.convolve(counts.astype(float), kernel, mode="same")
                    fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=smooth_counts,
                            mode="lines",
                            line=dict(color="#22d3ee", width=2.6),
                            name="KDE",
                            showlegend=True,
                            hovertemplate="Return: %{x:.3%}<br>Density (smoothed): %{y:.2f}<extra></extra>",
                        )
                    )

            mean_v = float(clipped.mean())
            median_v = float(clipped.median())
            var95 = float(clipped.quantile(0.05))
            var99 = float(clipped.quantile(0.01))

            fig.add_vline(x=0.0, line_width=1.4, line_dash="dot", line_color="#cbd5e1")
            fig.add_vline(x=mean_v, line_width=1.7, line_dash="dash", line_color="#f59e0b")
            fig.add_vline(x=median_v, line_width=1.7, line_dash="dash", line_color="#a855f7")
            fig.add_vline(x=var95, line_width=1.9, line_dash="dash", line_color="#ef4444")
            fig.add_vline(x=var99, line_width=2.0, line_dash="dot", line_color="#fb7185")

            marker_lines = [
                ("Return = 0", "#cbd5e1", "dot", 0.0),
                ("Mean", "#f59e0b", "dash", mean_v),
                ("Median", "#a855f7", "dash", median_v),
                ("VaR 95%", "#ef4444", "dash", var95),
                ("VaR 99%", "#fb7185", "dot", var99),
            ]

            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0.78,
                x1=0.995,
                y0=0.16,
                y1=0.90,
                line=dict(color="rgba(148, 163, 184, 0.35)", width=1),
                fillcolor="rgba(15, 23, 42, 0.82)",
                layer="above",
            )

            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.79,
                y=0.88,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#e2e8f0"),
                text="<b>Vertical Markers</b>",
            )

            y_top = 0.82
            y_step = 0.12
            for idx, (label, color, dash_style, value) in enumerate(marker_lines):
                y_pos = y_top - idx * y_step
                fig.add_shape(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=0.80,
                    x1=0.86,
                    y0=y_pos,
                    y1=y_pos,
                    line=dict(color=color, width=2, dash=dash_style),
                    layer="above",
                )
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.87,
                    y=y_pos,
                    xanchor="left",
                    yanchor="middle",
                    showarrow=False,
                    align="left",
                    font=dict(size=11, color="#e2e8f0"),
                    text=f"{label}: {value:.2%}",
                )

            fig.update_layout(
                title=dict(
                    text=f"{ticker} Daily Returns Distribution",
                    x=0.01,
                    xanchor="left",
                    y=0.99,
                    yanchor="top",
                ),
                template="plotly_dark",
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0f172a",
                font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=13),
                margin=dict(l=60, r=230, t=105, b=55),
                height=600,
                hovermode="x",
                bargap=0.03,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=0.955,
                    xanchor="left",
                    x=0.0,
                    bgcolor="rgba(15, 23, 42, 0.65)",
                    bordercolor="rgba(148, 163, 184, 0.2)",
                    borderwidth=1,
                    font=dict(size=11),
                ),
                xaxis=dict(
                    title="Daily Return",
                    domain=[0.0, 0.74],
                    tickformat=".1%",
                    showgrid=True,
                    gridcolor="rgba(148, 163, 184, 0.10)",
                ),
                yaxis=dict(
                    title="Frequency",
                    showgrid=True,
                    gridcolor="rgba(148, 163, 184, 0.10)",
                ),
            )

            filename_stub = f"{ticker.lower()}_returns_distribution"
            self._save_figure(fig, filename_stub, save)
            logger.info("Saved returns distribution chart -> %s", self.output_dir / f"{filename_stub}.html")

    def rolling_stats_chart(
        self,
        ticker: str,
        window: int = 20,
        num_std: float = 2.0,
        save: bool = True,
    ) -> None:
        """
        Plot rolling moving averages and Bollinger Bands for a given ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol to visualise.
        window : int
            Look-back window in trading days for the Bollinger Band calculation.
        num_std : float
            Number of standard deviations for the upper/lower bands.
        save : bool
            Whether to export the figure to output_dir.

        Notes
        -----
        Bands: upper = SMA(window) + num_std ?-- ??(window)
               lower = SMA(window) ??' num_std ?-- ??(window)
        Shade the band region for readability.
        """
        df = self._get_ticker_frame(ticker)
        if "close" not in df.columns:
            raise ValueError(f"Ticker '{ticker}' data missing required column: close")

        close = df["close"]
        ma20 = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = ma20 + num_std * std
        lower = ma20 - num_std * std

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.74, 0.26],
            specs=[[{}], [{}]],
        )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=close,
                mode="lines",
                line=dict(color="#60a5fa", width=2.2),
                name="Close",
                hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Removed MA20 trace
        if "ma50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["ma50"],
                    mode="lines",
                    line=dict(color="#a855f7", width=2.0),
                    name="MA50",
                    hovertemplate="%{x|%Y-%m-%d}<br>MA50: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if "ma200" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["ma200"],
                    mode="lines",
                    line=dict(color="#22c55e", width=2.4),
                    name="MA200",
                    hovertemplate="%{x|%Y-%m-%d}<br>MA200: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=upper,
                mode="lines",
                line=dict(color="rgba(56, 189, 248, 0.6)", width=1.1, dash="dot"),
                name=f"BB Upper ({window}, {num_std}σ)",
                hovertemplate="%{x|%Y-%m-%d}<br>Upper: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=lower,
                mode="lines",
                line=dict(color="rgba(56, 189, 248, 0.6)", width=1.1, dash="dot"),
                name=f"BB Lower ({window}, {num_std}σ)",
                hovertemplate="%{x|%Y-%m-%d}<br>Lower: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=ma20,
                mode="lines",
                line=dict(color="rgba(56, 189, 248, 0.18)", width=0.5),
                fill="tonexty",
                fillcolor="rgba(59, 130, 246, 0.12)",
                name="Bollinger Band Fill",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        if "volume" in df.columns:
            if "open" in df.columns:
                volume_colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["close"], df["open"])]
            else:
                volume_colors = ["#94a3b8"] * len(df)
            fig.add_trace(
                go.Bar(
                    x=df["date"],
                    y=df["volume"],
                    marker_color=volume_colors,
                    opacity=0.75,
                    name="Volume",
                    hovertemplate="%{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title=dict(text=f"{ticker} Rolling Stats & Bollinger Bands", x=0.02, xanchor="left"),
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=13),
            margin=dict(l=60, r=30, t=70, b=50),
            height=860,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.65)",
                bordercolor="rgba(148, 163, 184, 0.2)",
                borderwidth=1,
            ),
            xaxis_rangeslider_visible=False,
            yaxis=dict(
                title="Price",
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.12)",
                zeroline=False,
            ),
            yaxis2=dict(
                title="Volume",
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.10)",
                zeroline=False,
            ),
        )

        fig.update_xaxes(showline=True, linecolor="rgba(148, 163, 184, 0.18)", row=1, col=1)
        fig.update_xaxes(showline=True, linecolor="rgba(148, 163, 184, 0.18)", row=2, col=1)

        filename_stub = f"{ticker.lower()}_rolling_stats"
        self._save_figure(fig, filename_stub, save)
        logger.info("Saved rolling stats chart -> %s", self.output_dir / f"{filename_stub}.html")

    def render_all(
        self,
        timeframe: str = "daily",
        chart_type: str = "candlestick",
        include_rolling: bool = True,
    ) -> None:
        """
        Generate chart set for every ticker in self.data.

        Focuses on the price/volume master chart and optional rolling stats.
        Errors on individual tickers are logged without halting batch export.
        """
        timeframes = self._normalise_timeframes(timeframe)
        for ticker, df in self.data.items():
            if df is None or df.empty:
                continue
            try:
                if {"close", "volume"}.issubset(df.columns):
                    for tf in timeframes:
                        self.price_trend_chart(ticker=ticker, chart_type=chart_type, timeframe=tf, save=True)
                    self.returns_distribution(tickers=[ticker], plot_type="both", save=True)
                    if include_rolling:
                        self.rolling_stats_chart(ticker=ticker, save=True)
            except Exception as exc:
                logger.exception("Chart rendering failed for %s: %s", ticker, exc)

    def comparison_metrics_chart(
        self,
        ticker_a: str,
        ticker_b: str,
        fund_a: Optional[pd.DataFrame] = None,
        fund_b: Optional[pd.DataFrame] = None,
        clip_threshold: Optional[float] = 50.0,
        use_log_scale: bool = False,
        save: bool = True,
    ) -> go.Figure:
        """
        Side-by-side grouped bar chart comparing Stock A vs Stock B across three
        categories: Company Financial Health, Fundamental Valuation, Technical Valuation.

        Parameters
        ----------
        ticker_a, ticker_b : str
            The two tickers to compare.
        fund_a, fund_b : pd.DataFrame, optional
            Latest-row fundamental data for each ticker (already loaded by caller).
        clip_threshold : float, optional
            Absolute cap for display values. Any value with |x| > clip_threshold will
            be clipped on chart height and labeled as "{threshold}+" or "-{threshold}+".
            Set to None or <= 0 to disable clipping.
        use_log_scale : bool
            If True, use log scale for Y-axis on subplots where all displayed values are positive.
        save : bool
            Whether to save the figure to output_dir.
        """

        def _latest(df: Optional[pd.DataFrame], col: str) -> Optional[float]:
            if df is None or df.empty or col not in df.columns:
                return None
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(vals.iloc[-1]) if not vals.empty else None

        def _tech(ticker: str, col: str) -> Optional[float]:
            try:
                df = self._get_ticker_frame(ticker)
            except KeyError:
                return None
            if col not in df.columns:
                return None
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(vals.iloc[-1]) if not vals.empty else None

        def _clip_for_display(value: float) -> float:
            if clip_threshold is None or clip_threshold <= 0:
                return value
            if abs(value) > float(clip_threshold):
                return float(clip_threshold) if value > 0 else -float(clip_threshold)
            return value

        health_metrics = {
            "ROE (%)": (
                (_latest(fund_a, "roe") or 0) * 100,
                (_latest(fund_b, "roe") or 0) * 100,
            ),
            "ROA (%)": (
                (_latest(fund_a, "roa") or 0) * 100,
                (_latest(fund_b, "roa") or 0) * 100,
            ),
            "Debt/Equity": (
                _latest(fund_a, "debt_to_equity") or 0,
                _latest(fund_b, "debt_to_equity") or 0,
            ),
            "Current Ratio": (
                _latest(fund_a, "current_ratio") or 0,
                _latest(fund_b, "current_ratio") or 0,
            ),
            "Interest Coverage": (
                _latest(fund_a, "interest_coverage") or 0,
                _latest(fund_b, "interest_coverage") or 0,
            ),
            "Altman Z": (
                _latest(fund_a, "altman_z_score") or 0,
                _latest(fund_b, "altman_z_score") or 0,
            ),
        }

        fund_metrics = {
            "P/E Ratio": (
                _latest(fund_a, "pe") or 0,
                _latest(fund_b, "pe") or 0,
            ),
            "P/B Ratio": (
                _latest(fund_a, "pb") or 0,
                _latest(fund_b, "pb") or 0,
            ),
            "EPS": (
                _latest(fund_a, "eps") or 0,
                _latest(fund_b, "eps") or 0,
            ),
            "Net Margin (%)": (
                (_latest(fund_a, "net_profit_margin") or 0) * 100,
                (_latest(fund_b, "net_profit_margin") or 0) * 100,
            ),
            "Gross Margin (%)": (
                (_latest(fund_a, "gross_profit_margin") or 0) * 100,
                (_latest(fund_b, "gross_profit_margin") or 0) * 100,
            ),
            "Dividend ($)": (
                _latest(fund_a, "dividend") or 0,
                _latest(fund_b, "dividend") or 0,
            ),
        }

        tech_metrics = {
            "RSI 14": (
                _tech(ticker_a, "rsi_14") or 0,
                _tech(ticker_b, "rsi_14") or 0,
            ),
            "Sharpe Ratio": (
                _tech(ticker_a, "sharpe_ratio") or 0,
                _tech(ticker_b, "sharpe_ratio") or 0,
            ),
            "Volatility 30d (%)": (
                (_tech(ticker_a, "volatility_30") or 0) * 100,
                (_tech(ticker_b, "volatility_30") or 0) * 100,
            ),
            "Beta": (
                _tech(ticker_a, "beta") or 0,
                _tech(ticker_b, "beta") or 0,
            ),
            "Max Drawdown (%)": (
                (_tech(ticker_a, "max_drawdown") or 0) * 100,
                (_tech(ticker_b, "max_drawdown") or 0) * 100,
            ),
            "Rel. Strength": (
                _tech(ticker_a, "relative_strength") or 0,
                _tech(ticker_b, "relative_strength") or 0,
            ),
        }

        categories = [
            ("Company Financial Health", health_metrics),
            ("Fundamental Valuation", fund_metrics),
            ("Technical Valuation", tech_metrics),
        ]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[c[0] for c in categories],
            horizontal_spacing=0.08,
        )

        color_a = "#38bdf8"
        color_b = "#f97316"

        for col_idx, (_, metrics) in enumerate(categories, start=1):
            labels = list(metrics.keys())
            vals_a_raw = [float(metrics[k][0]) for k in labels]
            vals_b_raw = [float(metrics[k][1]) for k in labels]

            vals_a = [_clip_for_display(v) for v in vals_a_raw]
            vals_b = [_clip_for_display(v) for v in vals_b_raw]
            text_a = [f"{v:.2f}" for v in vals_a_raw]
            text_b = [f"{v:.2f}" for v in vals_b_raw]

            show_legend = col_idx == 1
            fig.add_trace(
                go.Bar(
                    name=ticker_a,
                    x=labels,
                    y=vals_a,
                    marker_color=color_a,
                    text=text_a,
                    textposition="outside",
                    textfont=dict(size=10),
                    customdata=[[v] for v in vals_a_raw],
                    hovertemplate="<b>%{x}</b><br>Displayed: %{y:.2f}<br>Original: %{customdata[0]:.2f}<extra></extra>",
                    showlegend=show_legend,
                    legendgroup="a",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Bar(
                    name=ticker_b,
                    x=labels,
                    y=vals_b,
                    marker_color=color_b,
                    text=text_b,
                    textposition="outside",
                    textfont=dict(size=10),
                    customdata=[[v] for v in vals_b_raw],
                    hovertemplate="<b>%{x}</b><br>Displayed: %{y:.2f}<br>Original: %{customdata[0]:.2f}<extra></extra>",
                    showlegend=show_legend,
                    legendgroup="b",
                ),
                row=1,
                col=col_idx,
            )

            panel_vals = vals_a + vals_b
            if use_log_scale and panel_vals and min(panel_vals) > 0:
                fig.update_yaxes(type="log", row=1, col=col_idx)
            else:
                fig.update_yaxes(type="linear", row=1, col=col_idx)

        fig.update_layout(
            title=dict(
                text=f"Comparison: {ticker_a} vs {ticker_b}",
                x=0.02,
                xanchor="left",
            ),
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=12),
            margin=dict(l=50, r=30, t=90, b=120),
            height=560,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.04,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.65)",
                bordercolor="rgba(148, 163, 184, 0.2)",
                borderwidth=1,
            ),
        )
        for i in range(1, 4):
            fig.update_xaxes(tickangle=-30, row=1, col=i)
            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.12)",
                zeroline=True,
                zerolinecolor="rgba(148, 163, 184, 0.3)",
                row=1,
                col=i,
            )

        stub = f"comparison_{ticker_a.lower()}_vs_{ticker_b.lower()}"
        self._save_figure(fig, stub, save)
        logger.info("Saved comparison chart -> %s", self.output_dir / f"{stub}.html")
        return fig

    def performance_comparison_chart(
        self,
        ticker_a: str,
        ticker_b: str,
        benchmark_df: Optional[pd.DataFrame] = None,
        benchmark_label: str = "VNI",
        save: bool = True,
    ) -> Optional[go.Figure]:
        def _return_series_from_ticker(ticker: str) -> Optional[pd.DataFrame]:
            if ticker not in self.data:
                return None
            try:
                frame = self._get_ticker_frame(ticker)
            except Exception:
                return None
            return _return_series_from_frame(frame, ticker)

        def _return_series_from_frame(frame: Optional[pd.DataFrame], label: str) -> Optional[pd.DataFrame]:
            if frame is None or frame.empty or "date" not in frame.columns:
                return None

            work = frame.copy()
            work["date"] = pd.to_datetime(work["date"], errors="coerce")
            work = work.dropna(subset=["date"]).sort_values("date")
            if work.empty:
                return None

            if "daily_return" in work.columns:
                returns = pd.to_numeric(work["daily_return"], errors="coerce")
            elif "close" in work.columns:
                close = pd.to_numeric(work["close"], errors="coerce")
                returns = close.pct_change()
            else:
                return None

            out = pd.DataFrame({"date": work["date"], label: returns})
            out[label] = pd.to_numeric(out[label], errors="coerce").fillna(0.0)
            return out

        s_a = _return_series_from_ticker(ticker_a)
        s_b = _return_series_from_ticker(ticker_b)
        s_benchmark = _return_series_from_frame(benchmark_df, benchmark_label)
        has_benchmark = s_benchmark is not None

        series = [s for s in [s_a, s_b, s_benchmark] if s is not None]
        if len(series) < 2:
            return None

        merged = series[0]
        for s in series[1:]:
            merged = merged.merge(s, on="date", how="inner")
        if merged.empty:
            return None

        ret_cols = [c for c in merged.columns if c != "date"]
        for col in ret_cols:
            merged[col] = (1.0 + merged[col]).cumprod() - 1.0

        fig = go.Figure()
        palette = {
            ticker_a: "#1d4ed8",
            ticker_b: "#ef4444",
            benchmark_label: "#14b8a6",
        }

        for col in [c for c in merged.columns if c != "date"]:
            fig.add_trace(
                go.Scatter(
                    x=merged["date"],
                    y=merged[col] * 100,
                    mode="lines",
                    name=col,
                    line=dict(width=2.8, color=palette.get(col, "#94a3b8")),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2f}%<extra></extra>",
                )
            )

        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="rgba(148,163,184,0.5)")
        fig.update_layout(
            title=dict(
                text=f"Performance Chart: {ticker_a} vs {ticker_b}" + (f" vs {benchmark_label}" if has_benchmark else ""),
                x=0.02,
                xanchor="left",
            ),
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0", size=12),
            height=520,
            margin=dict(l=55, r=20, t=70, b=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            xaxis=dict(title=None, showgrid=False),
            yaxis=dict(
                title="Return (%)",
                ticksuffix="%",
                showgrid=True,
                gridcolor="rgba(148,163,184,0.12)",
            ),
        )

        stub = "portfolio_cumulative_performance"
        self._save_figure(fig, stub, save)
        logger.info("Saved performance chart -> %s", self.output_dir / f"{stub}.html")
        return fig

    def efficient_frontier_chart(
        self,
        ticker_a: str,
        ticker_b: str,
        all_price_dfs: Optional[dict[str, pd.DataFrame]] = None,
        n_portfolios: int = 5000,
        risk_free_rate: float = 0.03,
        save: bool = True,
    ) -> go.Figure:
        """Random-portfolio efficient frontier for Stock A and Stock B."""
        price_dfs = all_price_dfs or self.data
        if ticker_a not in price_dfs or ticker_b not in price_dfs:
            raise ValueError("Both ticker_a and ticker_b must exist in provided price data")

        df_a = self._get_ticker_frame(ticker_a)
        df_b = self._get_ticker_frame(ticker_b)
        ra = self._returns_series(df_a)
        rb = self._returns_series(df_b)
        aligned = pd.DataFrame({"date": df_a["date"], ticker_a: ra}).merge(
            pd.DataFrame({"date": df_b["date"], ticker_b: rb}),
            on="date",
            how="inner",
        )
        aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(subset=[ticker_a, ticker_b])
        if len(aligned) < 10:
            raise ValueError("Insufficient aligned return observations for efficient frontier")

        ret_matrix = aligned[[ticker_a, ticker_b]].to_numpy(dtype=float)
        mean_returns = ret_matrix.mean(axis=0) * 252
        cov_matrix = np.cov(ret_matrix.T) * 252

        rng = np.random.default_rng(42)
        weights_a = rng.random(n_portfolios)
        weights_b = 1.0 - weights_a
        weights = np.column_stack([weights_a, weights_b])

        port_returns = weights @ mean_returns
        port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_matrix, weights))
        sharpe = (port_returns - risk_free_rate) / np.where(port_vols == 0, np.nan, port_vols)

        # Endpoints for 100% single-asset allocations.
        vol_a = float(np.sqrt(cov_matrix[0, 0]))
        vol_b = float(np.sqrt(cov_matrix[1, 1]))
        ret_a = float(mean_returns[0])
        ret_b = float(mean_returns[1])

        idx_max_sharpe = int(np.nanargmax(sharpe))
        idx_min_vol = int(np.nanargmin(port_vols))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=port_vols,
                y=port_returns,
                mode="markers",
                name="Random portfolios",
                marker=dict(
                    size=6,
                    color=sharpe,
                    colorscale="Viridis",
                    colorbar=dict(title="Sharpe"),
                    opacity=0.78,
                ),
                customdata=np.column_stack([weights_a, weights_b, sharpe]),
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                    f"<br>{ticker_a} Weight: %{{customdata[0]:.1%}}"
                    f"<br>{ticker_b} Weight: %{{customdata[1]:.1%}}"
                    "<br>Sharpe: %{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[port_vols[idx_max_sharpe]],
                y=[port_returns[idx_max_sharpe]],
                mode="markers",
                name="Optimal Portfolio (Max Sharpe)",
                marker=dict(size=17, symbol="star", color="#f59e0b", line=dict(width=1.4, color="#fde68a")),
                customdata=[[weights_a[idx_max_sharpe], weights_b[idx_max_sharpe], sharpe[idx_max_sharpe]]],
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                    f"<br>{ticker_a} Weight: %{{customdata[0]:.1%}}"
                    f"<br>{ticker_b} Weight: %{{customdata[1]:.1%}}"
                    "<br>Sharpe: %{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )

        fig.add_annotation(
            x=float(port_vols[idx_max_sharpe]),
            y=float(port_returns[idx_max_sharpe]),
            xanchor="center",
            yanchor="bottom",
            xshift=0,
            yshift=22,
            align="left",
            showarrow=False,
            bgcolor="rgba(15, 23, 42, 0.90)",
            bordercolor="rgba(245, 158, 11, 0.65)",
            borderwidth=1,
            font=dict(size=11, color="#fef08a"),
            text=(
                "<b>Optimal Portfolio (Max Sharpe)</b>"
                f"<br>{ticker_a}: {weights_a[idx_max_sharpe]:.1%}"
                f"<br>{ticker_b}: {weights_b[idx_max_sharpe]:.1%}"
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[vol_a],
                y=[ret_a],
                mode="markers",
                name=f"{ticker_a} Endpoint",
                marker=dict(size=13, symbol="circle", color="#22d3ee", line=dict(width=1.2, color="#cffafe")),
                showlegend=False,
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                    f"<br>{ticker_a} Weight: 100.0%"
                    f"<br>{ticker_b} Weight: 0.0%<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[vol_b],
                y=[ret_b],
                mode="markers",
                name=f"{ticker_b} Endpoint",
                marker=dict(size=13, symbol="circle", color="#f97316", line=dict(width=1.2, color="#ffedd5")),
                showlegend=False,
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                    f"<br>{ticker_a} Weight: 0.0%"
                    f"<br>{ticker_b} Weight: 100.0%<extra></extra>"
                ),
            )
        )

        # Capital Market Line from risk-free point to tangency portfolio (max Sharpe).
        tangent_vol = float(port_vols[idx_max_sharpe])
        tangent_ret = float(port_returns[idx_max_sharpe])
        max_vol = float(np.nanmax(np.concatenate([port_vols, np.array([vol_a, vol_b])])) )
        slope = 0.0
        if tangent_vol > 0:
            slope = (tangent_ret - risk_free_rate) / tangent_vol
        cml_x = np.array([0.0, max_vol * 1.05])
        cml_y = risk_free_rate + slope * cml_x
        fig.add_trace(
            go.Scatter(
                x=cml_x,
                y=cml_y,
                mode="lines",
                name="CML",
                line=dict(color="#f43f5e", width=2.0, dash="dash"),
                hovertemplate="Volatility: %{x:.2%}<br>CML Return: %{y:.2%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[risk_free_rate],
                mode="markers+text",
                name="Risk-free",
                text=[f"Rf: {risk_free_rate:.2%}"],
                textposition="top right",
                marker=dict(size=10, symbol="x", color="#fb7185"),
                showlegend=False,
                hovertemplate="Risk-free Rate: %{y:.2%}<extra></extra>",
            )
        )

        # Use anchored annotations for endpoint weights to avoid text overlap near markers.
        fig.add_annotation(
            x=vol_a,
            y=ret_a,
            xanchor="left",
            yanchor="top",
            xshift=16,
            yshift=-18,
            align="left",
            showarrow=False,
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor="rgba(125, 211, 252, 0.55)",
            borderwidth=1,
            font=dict(size=11, color="#7dd3fc"),
            text=f"<b>{ticker_a}</b><br>{ticker_a}: 100%<br>{ticker_b}: 0%",
        )
        fig.add_annotation(
            x=vol_b,
            y=ret_b,
            xanchor="right",
            yanchor="top",
            xshift=-16,
            yshift=-18,
            align="left",
            showarrow=False,
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor="rgba(251, 146, 60, 0.55)",
            borderwidth=1,
            font=dict(size=11, color="#fdba74"),
            text=f"<b>{ticker_b}</b><br>{ticker_a}: 0%<br>{ticker_b}: 100%",
        )

        fig.add_trace(
            go.Scatter(
                x=[port_vols[idx_min_vol]],
                y=[port_returns[idx_min_vol]],
                mode="markers",
                name="Minimum Volatility Portfolio",
                marker=dict(size=14, symbol="diamond", color="#38bdf8", line=dict(width=1.2, color="#bae6fd")),
                customdata=[[weights_a[idx_min_vol], weights_b[idx_min_vol], sharpe[idx_min_vol]]],
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                    f"<br>{ticker_a} Weight: %{{customdata[0]:.1%}}"
                    f"<br>{ticker_b} Weight: %{{customdata[1]:.1%}}"
                    "<br>Sharpe: %{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )

        fig.add_annotation(
            x=float(port_vols[idx_min_vol]),
            y=float(port_returns[idx_min_vol]),
            xanchor="left",
            yanchor="bottom",
            xshift=14,
            yshift=14,
            align="left",
            showarrow=False,
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor="rgba(56, 189, 248, 0.55)",
            borderwidth=1,
            font=dict(size=11, color="#7dd3fc"),
            text=(
                "<b>Minimum Volatility</b>"
                f"<br>{ticker_a}: {weights_a[idx_min_vol]:.1%}"
                f"<br>{ticker_b}: {weights_b[idx_min_vol]:.1%}"
            ),
        )

        fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.3)", line_width=1)

        fig.update_layout(
            title=dict(
                text=f"Efficient Frontier — {ticker_a} vs {ticker_b}",
                x=0.02,
                xanchor="left",
            ),
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=13),
            margin=dict(l=70, r=30, t=80, b=70),
            height=580,
            xaxis=dict(
                title="Annualised Volatility (Risk)",
                tickformat=".0%",
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.12)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Annualised Return",
                tickformat=".0%",
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.12)",
                zeroline=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.65)",
                bordercolor="rgba(148, 163, 184, 0.2)",
                borderwidth=1,
            ),
        )

        stub = "portfolio_efficient_frontier"
        self._save_figure(fig, stub, save)
        logger.info("Saved efficient frontier -> %s", self.output_dir / f"{stub}.html")
        return fig

