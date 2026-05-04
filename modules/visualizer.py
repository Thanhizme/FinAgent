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

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DataVisualizer:
    """
    Produces publication-quality financial charts from processed DataFrames.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping of ticker symbol ??' processed DataFrame (output of DataProcessor).
    output_dir : Path, optional
        Directory where chart files are saved. Defaults to data/processed/.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ) -> None:
        self.data = data
        self.output_dir = output_dir or OUTPUT_DIR

    # ------------------------------------------------------------------
    # Chart 1 ??" Price Trend + Volume Overlay
    # ------------------------------------------------------------------

    def price_trend_chart(
        self,
        ticker: str,
        chart_type: str = "line",
        save: bool = True,
    ) -> None:
        """
        Render a price trend chart with a volume bar overlay on a secondary axis.

        Parameters
        ----------
        ticker : str
            Ticker symbol to plot (must exist in self.data).
        chart_type : {'line', 'candlestick', 'ohlc'}
            Visual encoding for price.  'candlestick' and 'ohlc' are bonus options.
        save : bool
            Whether to export the figure to output_dir.

        Notes
        -----
        Expected columns in the DataFrame: open, high, low, close, volume.
        """
        # TODO: build Plotly (or matplotlib) figure with two subplots:
        #       - top: price trace (line / candlestick / OHLC)
        #       - bottom: volume bar chart
        #       Add rolling_avg_7 and rolling_avg_30 as overlays if present
        raise NotImplementedError

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
        # TODO: compute SMA, upper_band, lower_band on df['Close']
        #       plot with shaded fill_between region
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience ??" render all charts for all tickers
    # ------------------------------------------------------------------

    def render_all(self) -> None:
        """
        Generate all four chart types for every ticker in self.data.

        Calls each chart method in sequence; errors on individual charts are
        caught and logged rather than halting the entire pipeline.
        """
        # TODO: loop tickers, call each chart method, catch + log exceptions
        raise NotImplementedError

