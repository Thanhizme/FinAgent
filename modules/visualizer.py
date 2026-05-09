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

        try:
            png_path = self.output_dir / f"{filename_stub}.png"
            fig.write_image(str(png_path), scale=2)
        except Exception as exc:
            logger.info("PNG export skipped for %s (%s)", filename_stub, exc)

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
            "ma20": {"color": "#f59e0b", "width": 1.8},
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
            vol_ma20 = df["volume"].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=vol_ma20,
                    mode="lines",
                    line=dict(color="#cbd5e1", width=1.5),
                    name="VOL MA20",
                    hovertemplate="%{x|%Y-%m-%d}<br>VOL MA20: %{y:,.0f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

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

    def correlation_heatmap(
        self,
        ticker: Optional[str] = None,
        columns: Optional[list[str]] = None,
        save: bool = True,
    ) -> None:
        """
        Plot a correlation matrix heatmap across selected indicators.

        Parameters
        ----------
        ticker : str, optional
            Ticker to visualise. Defaults to the first ticker in self.data.
        columns : list[str], optional
            Specific indicator columns to include.
            Defaults to a curated set of technical and return indicators.
        save : bool
            Whether to export the figure to output_dir.

        Notes
        -----
        Uses Plotly heatmap with correlation values annotated in each cell.
        """
        if not self.data:
            raise ValueError("No data available for correlation heatmap")

        selected_ticker = ticker or next(iter(self.data.keys()))
        df = self._get_ticker_frame(selected_ticker)

        default_columns = [
            "daily_return",
            "log_return",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "stoch_k",
            "stoch_d",
            "adx_14",
            "williams_r_14",
            "cci_14",
            "ultimate_oscillator",
            "roc_12",
            "atr_14",
            "beta",
            "sharpe_ratio",
            "relative_strength",
            "volume",
        ]

        selected_columns = columns or default_columns
        available_columns = [c for c in selected_columns if c in df.columns]
        if len(available_columns) < 2:
            raise ValueError(
                f"Ticker '{selected_ticker}' does not have enough indicator columns for correlation heatmap"
            )

        numeric = df[available_columns].apply(pd.to_numeric, errors="coerce")
        valid_columns = [c for c in numeric.columns if numeric[c].notna().sum() >= 8]
        if len(valid_columns) < 2:
            raise ValueError(
                f"Ticker '{selected_ticker}' has insufficient non-null indicator data for heatmap"
            )

        corr = numeric[valid_columns].corr(method="pearson", min_periods=8).round(2)

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    zmin=-1,
                    zmax=1,
                    colorscale=[
                        [0.0, "#1e3a8a"],
                        [0.5, "#0f172a"],
                        [1.0, "#b91c1c"],
                    ],
                    colorbar=dict(
                        title="Corr",
                        ticks="outside",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=["-1.0", "-0.5", "0", "0.5", "1.0"],
                    ),
                    text=corr.values,
                    texttemplate="%{text:.2f}",
                    textfont=dict(size=10, color="#e2e8f0"),
                    hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(text=f"{selected_ticker} Correlation Heatmap (Selected Indicators)", x=0.02, xanchor="left"),
            template="plotly_dark",
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0f172a",
            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#e2e8f0", size=12),
            margin=dict(l=120, r=70, t=70, b=120),
            height=760,
        )
        fig.update_xaxes(tickangle=-35, side="bottom")
        fig.update_yaxes(autorange="reversed")

        filename_stub = f"{selected_ticker.lower()}_correlation_heatmap"
        self._save_figure(fig, filename_stub, save)
        logger.info("Saved correlation heatmap -> %s", self.output_dir / f"{filename_stub}.html")

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

        if "ma20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["ma20"],
                    mode="lines",
                    line=dict(color="#f59e0b", width=2.0),
                    name="MA20",
                    hovertemplate="%{x|%Y-%m-%d}<br>MA20: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
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
            "Volatility 20d (%)": (
                (_tech(ticker_a, "volatility_20") or 0) * 100,
                (_tech(ticker_b, "volatility_20") or 0) * 100,
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
            vals_a = [metrics[k][0] for k in labels]
            vals_b = [metrics[k][1] for k in labels]

            show_legend = col_idx == 1
            fig.add_trace(
                go.Bar(
                    name=ticker_a,
                    x=labels,
                    y=vals_a,
                    marker_color=color_a,
                    text=[f"{v:.2f}" for v in vals_a],
                    textposition="outside",
                    textfont=dict(size=10),
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
                    text=[f"{v:.2f}" for v in vals_b],
                    textposition="outside",
                    textfont=dict(size=10),
                    showlegend=show_legend,
                    legendgroup="b",
                ),
                row=1,
                col=col_idx,
            )

        fig.update_layout(
            title=dict(
                text=f"Comparison: {ticker_a} (Stock A) vs {ticker_b} (Stock B)",
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

    def efficient_frontier_chart(
        self,
        ticker_a: str,
        ticker_b: str,
        all_price_dfs: Optional[dict[str, pd.DataFrame]] = None,
        save: bool = True,
    ) -> go.Figure:
        """
        Plot the 2-stock efficient frontier (Stock A + Stock B) plus individual
        stock risk-return scatter for all available tickers.

        Parameters
        ----------
        ticker_a, ticker_b : str
            The two tickers to highlight as Stock A / Stock B.
        all_price_dfs : dict, optional
            Mapping of ticker -> price DataFrame for ALL tickers to plot as background dots.
            Defaults to self.data.
        save : bool
            Whether to save the figure to output_dir.
        """
        price_dfs = all_price_dfs or self.data

        points: dict[str, dict] = {}
        for t, df in price_dfs.items():
            if df is None or df.empty:
                continue
            try:
                frame = df.copy()
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
                frame = frame.dropna(subset=["date"]).sort_values("date")
                if "daily_return" in frame.columns:
                    rets = pd.to_numeric(frame["daily_return"], errors="coerce").dropna()
                elif "close" in frame.columns:
                    rets = frame["close"].pct_change().dropna()
                else:
                    continue
                rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
                if len(rets) < 10:
                    continue
                ann_ret = float(rets.mean() * 252)
                ann_vol = float(rets.std() * np.sqrt(252))
                points[t] = {"return": ann_ret, "vol": ann_vol, "n": len(rets)}
            except Exception:
                continue

        fig = go.Figure()

        bg_tickers = [t for t in points if t not in (ticker_a, ticker_b)]
        if bg_tickers:
            fig.add_trace(
                go.Scatter(
                    x=[points[t]["vol"] for t in bg_tickers],
                    y=[points[t]["return"] for t in bg_tickers],
                    mode="markers+text",
                    text=bg_tickers,
                    textposition="top center",
                    textfont=dict(size=10, color="#94a3b8"),
                    marker=dict(size=9, color="#475569", line=dict(color="#94a3b8", width=1)),
                    name="Other stocks",
                    hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
                )
            )

        if ticker_a in points and ticker_b in points:
            try:
                df_a = price_dfs[ticker_a].copy()
                df_b = price_dfs[ticker_b].copy()
                df_a["date"] = pd.to_datetime(df_a["date"], errors="coerce")
                df_b["date"] = pd.to_datetime(df_b["date"], errors="coerce")
                col_r = "daily_return" if "daily_return" in df_a.columns else "close"
                ra = df_a.set_index("date")[col_r].dropna()
                rb = df_b.set_index("date")[col_r].dropna()
                if col_r == "close":
                    ra = ra.pct_change().dropna()
                    rb = rb.pct_change().dropna()
                common = ra.index.intersection(rb.index)
                if len(common) >= 10:
                    rho = float(ra.loc[common].corr(rb.loc[common]))
                else:
                    rho = 0.5
            except Exception:
                rho = 0.5

            r_a = points[ticker_a]["return"]
            r_b = points[ticker_b]["return"]
            v_a = points[ticker_a]["vol"]
            v_b = points[ticker_b]["vol"]

            weights = np.linspace(0, 1, 200)
            port_ret = weights * r_a + (1 - weights) * r_b
            port_vol = np.sqrt(
                (weights * v_a) ** 2
                + ((1 - weights) * v_b) ** 2
                + 2 * weights * (1 - weights) * v_a * v_b * rho
            )

            fig.add_trace(
                go.Scatter(
                    x=port_vol,
                    y=port_ret,
                    mode="lines",
                    line=dict(color="rgba(148, 163, 184, 0.45)", width=2, dash="dot"),
                    name=f"Frontier ({ticker_a}↔{ticker_b})",
                    hovertemplate="Volatility: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>",
                )
            )

        if ticker_a in points:
            fig.add_trace(
                go.Scatter(
                    x=[points[ticker_a]["vol"]],
                    y=[points[ticker_a]["return"]],
                    mode="markers+text",
                    text=[f"▲ {ticker_a} (A)"],
                    textposition="top right",
                    textfont=dict(size=13, color="#38bdf8"),
                    marker=dict(size=16, color="#38bdf8", symbol="star", line=dict(color="#0ea5e9", width=1.5)),
                    name=f"{ticker_a} (Stock A)",
                    hovertemplate=f"<b>{ticker_a} (A)</b><br>Volatility: %{{x:.1%}}<br>Return: %{{y:.1%}}<extra></extra>",
                )
            )

        if ticker_b in points:
            fig.add_trace(
                go.Scatter(
                    x=[points[ticker_b]["vol"]],
                    y=[points[ticker_b]["return"]],
                    mode="markers+text",
                    text=[f"● {ticker_b} (B)"],
                    textposition="top right",
                    textfont=dict(size=13, color="#f97316"),
                    marker=dict(size=16, color="#f97316", symbol="circle", line=dict(color="#ea580c", width=1.5)),
                    name=f"{ticker_b} (Stock B)",
                    hovertemplate=f"<b>{ticker_b} (B)</b><br>Volatility: %{{x:.1%}}<br>Return: %{{y:.1%}}<extra></extra>",
                )
            )

        fig.add_hline(y=0, line_dash="dash", line_color="rgba(148, 163, 184, 0.3)", line_width=1)

        fig.update_layout(
            title=dict(
                text=f"Efficient Frontier — {ticker_a} (A) vs {ticker_b} (B)",
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

        stub = f"frontier_{ticker_a.lower()}_vs_{ticker_b.lower()}"
        self._save_figure(fig, stub, save)
        logger.info("Saved efficient frontier -> %s", self.output_dir / f"{stub}.html")
        return fig

