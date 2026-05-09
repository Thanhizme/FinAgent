"""
Simple exchange-style web UI for FinAgent.

Run:
    streamlit run web_app.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules.visualizer import DataVisualizer

APP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = APP_DIR / "data" / "processed" / "processed_data"
VIS_DIR = APP_DIR / "data" / "processed" / "visualization"


def parse_tickers(raw: str) -> list[str]:
    parts = raw.replace(",", " ").split()
    tickers: list[str] = []
    for item in parts:
        t = item.strip().upper()
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def run_pipeline(tickers: list[str], start: date, end: date, timeframe: str) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "main.py",
        "--tickers",
        *tickers,
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--skip-ai",
        "--timeframe",
        timeframe,
    ]
    proc = subprocess.run(
        cmd,
        cwd=APP_DIR,
        text=True,
        capture_output=True,
    )
    logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, logs.strip()


def load_price_df(ticker: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{ticker}_processed.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_fundamental_df(ticker: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{ticker}_fundamental_processed.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def available_tickers() -> list[str]:
    tickers = []
    if PROCESSED_DIR.exists():
        for path in PROCESSED_DIR.glob("*_processed.csv"):
            stem = path.stem
            if stem.endswith("_fundamental_processed"):
                continue
            if stem in {"benchmark_processed", "industry_processed", "macro_processed", "news_processed"}:
                continue
            ticker = stem.replace("_processed", "")
            if ticker and ticker not in tickers:
                tickers.append(ticker.upper())
    return sorted(tickers)


def chart_files_for_ticker(ticker: str, timeframe: str) -> list[Path]:
    t = ticker.lower()
    if timeframe == "all":
        return [
            VIS_DIR / f"{t}_price_volume_daily.html",
            VIS_DIR / f"{t}_price_volume_weekly.html",
            VIS_DIR / f"{t}_price_volume_monthly.html",
            VIS_DIR / f"{t}_price_volume_yearly.html",
        ]
    return [VIS_DIR / f"{t}_price_volume_{timeframe}.html"]


def returns_distribution_path(ticker: str) -> Path:
    return VIS_DIR / f"{ticker.lower()}_returns_distribution.html"


def correlation_heatmap_path(ticker: str) -> Path:
    return VIS_DIR / f"{ticker.lower()}_correlation_heatmap.html"


def ensure_chart_file(ticker: str, timeframe: str, price_df: pd.DataFrame) -> Path | None:
    files = chart_files_for_ticker(ticker, timeframe)
    existing = next((p for p in files if p.exists()), None)
    if existing is not None:
        return existing

    if price_df is None or price_df.empty:
        return None

    try:
        visualizer = DataVisualizer({ticker: price_df})
        visualizer.price_trend_chart(ticker=ticker, chart_type="candlestick", timeframe=timeframe, save=True)
        if timeframe == "all":
            # generate rolling stats once for the selected ticker when all timeframes are requested
            visualizer.rolling_stats_chart(ticker=ticker, save=True)
        return next((p for p in files if p.exists()), None)
    except Exception as exc:
        st.error(f"Unable to generate chart for {ticker}: {exc}")
        return None


def ensure_returns_distribution_chart(ticker: str, price_df: pd.DataFrame) -> Path | None:
    path = returns_distribution_path(ticker)
    if price_df is None or price_df.empty:
        return path if path.exists() else None

    try:
        visualizer = DataVisualizer({ticker: price_df})
        visualizer.returns_distribution(tickers=[ticker], plot_type="both", save=True)
        return path if path.exists() else None
    except Exception as exc:
        st.error(f"Unable to generate returns distribution for {ticker}: {exc}")
        return None


def ensure_correlation_heatmap_chart(ticker: str, price_df: pd.DataFrame) -> Path | None:
    path = correlation_heatmap_path(ticker)
    if price_df is None or price_df.empty:
        return path if path.exists() else None

    try:
        visualizer = DataVisualizer({ticker: price_df})
        visualizer.correlation_heatmap(ticker=ticker, save=True)
        return path if path.exists() else None
    except Exception as exc:
        st.error(f"Unable to generate correlation heatmap for {ticker}: {exc}")
        return None


def render_chart(path: Path) -> None:
    if not path.exists():
        st.warning(f"Chart not found: {path.name}")
        return
    html = path.read_text(encoding="utf-8")
    components.html(html, height=980, scrolling=True)


def metric_value(df: pd.DataFrame, col: str, fmt: str = "{:.2f}") -> str:
    if col not in df.columns or df.empty:
        return "N/A"
    value = df[col].dropna()
    if value.empty:
        return "N/A"
    try:
        return fmt.format(float(value.iloc[-1]))
    except Exception:
        return str(value.iloc[-1])


st.set_page_config(
    page_title="FinAgent Exchange UI",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0b1220 0%, #111827 100%); color: #e5e7eb; }
    .block-container { padding-top: 1.1rem; padding-bottom: 1rem; }
    .card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("FinAgent Market Terminal")
st.caption("Exchange-style dashboard with ticker picker, refresh, and instant results.")

with st.sidebar:
    st.header("Run Settings")

    existing_tickers = available_tickers()
    picker_mode = st.radio("Ticker source", ["Dropdown", "Manual input"], horizontal=False)

    if picker_mode == "Dropdown" and existing_tickers:
        selected_ticker = st.selectbox("Select ticker", existing_tickers, index=0)
        raw_tickers = selected_ticker
    else:
        raw_tickers = st.text_input("Ticker(s) comma or space separated", value=st.session_state.get("last_manual_tickers", "TSLA"))
        selected_ticker = parse_tickers(raw_tickers)[0] if parse_tickers(raw_tickers) else "TSLA"

    default_end = date.today() - timedelta(days=1)
    default_start = date.today() - timedelta(days=30 * 18)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

    timeframe = st.selectbox("Chart timeframe", ["daily", "weekly", "monthly", "yearly", "all"], index=0)

    col_a, col_b = st.columns(2)
    run_clicked = col_a.button("Run", use_container_width=True)
    refresh_clicked = col_b.button("Refresh Data", use_container_width=True)

if "last_run_ok" not in st.session_state:
    st.session_state["last_run_ok"] = False
if "last_logs" not in st.session_state:
    st.session_state["last_logs"] = ""
if "last_tickers" not in st.session_state:
    st.session_state["last_tickers"] = []
if "last_timeframe" not in st.session_state:
    st.session_state["last_timeframe"] = "daily"
if "last_manual_tickers" not in st.session_state:
    st.session_state["last_manual_tickers"] = "TSLA"
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = selected_ticker if "selected_ticker" in locals() else "TSLA"

if refresh_clicked:
    run_clicked = True

if run_clicked:
    tickers = parse_tickers(raw_tickers)
    if not tickers:
        st.error("Please enter at least one ticker.")
    elif start_date >= end_date:
        st.error("Start date must be earlier than end date.")
    else:
        with st.spinner("Running pipeline and rendering charts..."):
            ok, logs = run_pipeline(tickers, start_date, end_date, timeframe)
            st.session_state["last_run_ok"] = ok
            st.session_state["last_logs"] = logs
            st.session_state["last_tickers"] = tickers
            st.session_state["last_timeframe"] = timeframe
            st.session_state["selected_ticker"] = tickers[0]
            st.session_state["last_manual_tickers"] = raw_tickers

        if ok:
            st.success("Pipeline completed successfully.")
        else:
            st.error("Pipeline failed. Check logs below.")

if st.session_state["last_logs"]:
    with st.expander("Pipeline logs", expanded=not st.session_state["last_run_ok"]):
        st.text(st.session_state["last_logs"])

dashboard_ticker = st.session_state.get("selected_ticker") or (st.session_state["last_tickers"][0] if st.session_state["last_tickers"] else None)

if dashboard_ticker:
    st.markdown(f"## {dashboard_ticker}")

    price_df = load_price_df(dashboard_ticker)
    if price_df is None:
        st.warning(f"No processed price file found for {dashboard_ticker}.")
    else:
        latest = price_df.tail(1).iloc[0]
        prev = price_df.tail(2).iloc[0] if len(price_df) > 1 else latest
        change = float(latest["close"] - prev["close"])
        change_pct = float(change / prev["close"]) if prev["close"] else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Close", f"{latest['close']:.2f}", f"{change:+.2f} ({change_pct:+.2%})")
        m2.metric("RSI 14", metric_value(price_df, "rsi_14", "{:.2f}"))
        m3.metric("Beta", metric_value(price_df, "beta", "{:.3f}"))
        m4.metric("Sharpe", metric_value(price_df, "sharpe_ratio", "{:.3f}"))
        m5.metric("Volume", f"{int(latest['volume']):,}")

        if st.session_state["last_tickers"] and dashboard_ticker not in st.session_state["last_tickers"]:
            st.info("This ticker exists in processed files. Use Refresh Data to rerun analysis for it.")

        main_tab, osc_tab = st.tabs(["Price & Volume", "Oscillators - Returns Distribution"])

        with main_tab:
            chart_path = ensure_chart_file(dashboard_ticker, st.session_state["last_timeframe"], price_df)
            if chart_path is not None:
                if st.session_state["last_timeframe"] == "all":
                    nested_tabs = []
                    chart_paths = [p for p in chart_files_for_ticker(dashboard_ticker, "all") if p.exists()]
                    if chart_paths:
                        labels = [p.stem.split("_")[-1].upper() for p in chart_paths]
                        nested_tabs = st.tabs(labels)
                        for tab, path in zip(nested_tabs, chart_paths):
                            with tab:
                                render_chart(path)
                else:
                    render_chart(chart_path)
            else:
                st.warning(f"Chart not available for {dashboard_ticker}.")

        with osc_tab:
            dist_path = ensure_returns_distribution_chart(dashboard_ticker, price_df)
            if dist_path is not None:
                render_chart(dist_path)
            else:
                st.warning(f"Returns distribution chart not available for {dashboard_ticker}.")

            heatmap_path = ensure_correlation_heatmap_chart(dashboard_ticker, price_df)
            if heatmap_path is not None:
                render_chart(heatmap_path)
            else:
                st.warning(f"Correlation heatmap not available for {dashboard_ticker}.")

            if "daily_return" in price_df.columns:
                returns = price_df["daily_return"].dropna()
                if not returns.empty:
                    q95 = returns.quantile(0.05)
                    q99 = returns.quantile(0.01)
                    s1, s2, s3, s4, s5, s6 = st.columns(6)
                    s1.metric("Mean Return", f"{returns.mean():.3%}")
                    s2.metric("Std Dev", f"{returns.std():.3%}")
                    s3.metric("Skewness", f"{returns.skew():.3f}")
                    s4.metric("Kurtosis", f"{returns.kurtosis():.3f}")
                    s5.metric("VaR 95%", f"{q95:.3%}")
                    s6.metric("VaR 99%", f"{q99:.3%}")

        left, right = st.columns([1.35, 1])
        with left:
            st.markdown("### Recent rows")
            st.dataframe(price_df.tail(25), use_container_width=True, height=420)

        with right:
            st.markdown("### Fundamentals")
            fund_df = load_fundamental_df(dashboard_ticker)
            if fund_df is not None and not fund_df.empty:
                st.dataframe(fund_df, use_container_width=True, height=420)
            else:
                st.info("No fundamental file found for this ticker.")

        st.markdown("### Data snapshot")
        summary_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "daily_return", "rsi_14", "macd_line", "beta", "sharpe_ratio"] if c in price_df.columns]
        st.dataframe(price_df[summary_cols].tail(10), use_container_width=True)
else:
    st.info("Choose a ticker from the sidebar and press Run to load the dashboard.")
