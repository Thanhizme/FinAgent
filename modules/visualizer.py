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

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    # ------------------------------------------------------------------
    # Chart 1 ??" Price Trend + Volume Overlay
    # ------------------------------------------------------------------

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
                    bar_width = max(step.total_seconds() * 1000 * 0.72, 12 * 60 * 60 * 1000)

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

    # ------------------------------------------------------------------
    # Chart 2 ??" Correlation Heatmap
    # ------------------------------------------------------------------

    def correlation_heatmap(
        self,
        columns: Optional[list[str]] = None,
        save: bool = True,
    ) -> None:
        """
        Plot a correlation matrix heatmap across all tracked assets.

        Parameters
        ----------
        columns : list[str], optional
            Specific columns (e.g. ['Close', 'daily_return']) to include.
            Defaults to closing prices of all tickers.
        save : bool
            Whether to export the figure to output_dir.

        Notes
        -----
        Uses seaborn.heatmap with annotated correlation coefficients.
        Combines daily_return columns from all tickers into a single DataFrame
        before computing df.corr().
        """
        # TODO: pivot combined_df = {ticker: df['daily_return']} ??' pd.DataFrame
        #       compute corr_matrix = combined_df.corr()
        #       render seaborn heatmap with annot=True, fmt='.2f'
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Chart 3 ??" Returns Distribution
    # ------------------------------------------------------------------

    def returns_distribution(
        self,
        tickers: Optional[list[str]] = None,
        plot_type: str = "histogram",
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
        # TODO: for each ticker plot daily_return distribution
        #       use plotly.express.histogram or seaborn.histplot(kde=True)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Chart 4 ??" Rolling Statistics (MA + Bollinger Bands)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Convenience ??" render all charts for all tickers
    # ------------------------------------------------------------------

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

