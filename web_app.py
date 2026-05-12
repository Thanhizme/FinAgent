
from __future__ import annotations

import base64
import html
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.ai_agent import AnalysisAgent
from modules.visualizer import DataVisualizer

APP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = APP_DIR / "data" / "processed" / "processed_data"
VIS_DIR = APP_DIR / "data" / "processed" / "visualization"

PRESET_TICKERS = [
    "AAPL", "MSFT", "MANH", "TER", "IDCC", "KLIC",
    "JPM", "V", "SF", "JEF", "DFIN", "VBTX",
    "AMGN", "ELV", "HALO", "EHC", "HIMS", "NSTG",
    "AMZN", "TSLA", "DECK", "CROX", "CMG", "BOOT", "SONO",
    "MDLZ", "KMB", "CASY", "CELH", "CALM", "JJSF",
    "LMT", "GE", "DE", "UPS", "BYRN", "MLKN",
    "XOM", "CVX", "OVV", "APA", "REPX", "PARR",
    "D", "NEE", "VST", "NRG", "AWR", "AVA",
    "PLD", "EQIX", "REXR", "OHI", "LGIH", "UTL",
    "LIN", "SHW", "RS", "STLD", "MLI", "IOSP",
    "GOOGL", "META", "PINS", "TTWO", "CNK", "YELP", "FOX",
]

def parse_tickers(raw: str) -> list[str]:
    parts = raw.replace(",", " ").split()
    tickers: list[str] = []
    for item in parts:
        t = item.strip().upper()
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def render_plain_text_block(text: str) -> None:
    safe_text = html.escape(str(text) if text is not None else "N/A")
    st.markdown(
        f"<div style='white-space: pre-wrap; line-height: 1.65;'>{safe_text}</div>",
        unsafe_allow_html=True,
    )

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

def load_benchmark_df() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "benchmark_processed.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

def available_tickers() -> list[str]:
    tickers = []
    seen = set()
    if PROCESSED_DIR.exists():
        for path in PROCESSED_DIR.glob("*_processed.csv"):
            stem = path.stem
            if stem.endswith("_fundamental_processed"):
                continue
            if stem in {"benchmark_processed", "industry_processed", "macro_processed", "news_processed"}:
                continue
            ticker = stem.replace("_processed", "")
            upper = ticker.upper()
            if ticker and upper not in seen:
                tickers.append(upper)
                seen.add(upper)

    for ticker in PRESET_TICKERS:
        upper = ticker.upper()
        if upper not in seen:
            tickers.append(upper)
            seen.add(upper)

    return sorted(tickers)

def available_processed_tickers() -> list[str]:
    tickers = []
    seen = set()
    if PROCESSED_DIR.exists():
        for path in PROCESSED_DIR.glob("*_processed.csv"):
            stem = path.stem
            if stem.endswith("_fundamental_processed"):
                continue
            if stem in {"benchmark_processed", "industry_processed", "macro_processed", "news_processed"}:
                continue
            ticker = stem.replace("_processed", "").upper()
            if ticker and ticker not in seen:
                tickers.append(ticker)
                seen.add(ticker)
    return sorted(tickers)

def available_fundamental_tickers() -> list[str]:
    tickers = []
    seen = set()
    if PROCESSED_DIR.exists():
        for path in PROCESSED_DIR.glob("*_fundamental_processed.csv"):
            stem = path.stem.replace("_fundamental_processed", "").upper()
            if stem and stem not in seen:
                tickers.append(stem)
                seen.add(stem)
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

def render_chart(path: Path, height: int = 980) -> None:
    if not path.exists():
        st.warning(f"Chart not found: {path.name}")
        return
    html = path.read_text(encoding="utf-8")
    encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.iframe(src=f"data:text/html;base64,{encoded_html}", height=height, width="stretch")

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

def has_configured_api_key(env_var: str) -> bool:
    load_dotenv(APP_DIR / ".env", override=True)
    value = (os.getenv(env_var) or "").strip()
    if not value:
        return False
    return not value.lower().startswith("your_")

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
        raw_tickers = st.text_input("Ticker(s) comma or space separated", value=st.session_state.get("last_manual_tickers", "TSLA")) or ""
        parsed_manual = parse_tickers(raw_tickers)
        selected_ticker = parsed_manual[0] if parsed_manual else "TSLA"

    default_end = date.today() - timedelta(days=1)
    default_start = date.today() - timedelta(days=30 * 18)
    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

    timeframe = st.selectbox("Chart timeframe", ["daily", "weekly", "monthly", "yearly", "all"], index=0)

    st.markdown("---")
    st.markdown("**AI Settings**")
    st.selectbox(
        "Gemini model",
        [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
        ],
        index=0,
        key="gemini_model_selector",
    )

    st.markdown("---")
    st.markdown("**Comparison**")
    _all_tickers = available_tickers()
    _other_tickers = [t for t in _all_tickers if t != (selected_ticker if "selected_ticker" in locals() else "")]
    stock_b_sidebar = st.selectbox(
        "Compare vs (Stock B)",
        _other_tickers if _other_tickers else ["—"],
        key="stock_b_selector",
    )
    st.caption("Stock B will be collected and processed together with Stock A when you press Run.")

    col_a, col_b = st.columns(2)
    run_clicked = col_a.button("Run", width="stretch")
    refresh_clicked = col_b.button("Refresh Data", width="stretch")

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
if "ai_analysis_cache" not in st.session_state:
    st.session_state["ai_analysis_cache"] = {}
if "active_dashboard_panel" not in st.session_state:
    st.session_state["active_dashboard_panel"] = "Price & Volume"

if refresh_clicked:
    run_clicked = True

if run_clicked:
    tickers = parse_tickers(raw_tickers or "")
    stock_b_for_run = (st.session_state.get("stock_b_selector") or "").strip().upper()
    if stock_b_for_run and stock_b_for_run != "—" and stock_b_for_run not in tickers:
        tickers.append(stock_b_for_run)
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
        st.markdown(f"### Comparison — {dashboard_ticker} vs {st.session_state.get('stock_b_selector', 'Stock B')}")
        st.info("Comparison is unavailable because Stock A has no processed data yet. Pick a ticker with processed data or press Run.")
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

        active_panel = st.radio(
            "View",
            ["Price & Volume", "Oscillators - Returns Distribution", "Comparison"],
            key="active_dashboard_panel",
            horizontal=True,
            label_visibility="collapsed",
        )

        if active_panel == "Price & Volume":
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

        elif active_panel == "Oscillators - Returns Distribution":
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

        else:
            st.markdown(f"### Comparison — {dashboard_ticker} vs {st.session_state.get('stock_b_selector', 'Stock B')}")

            stock_b = st.session_state.get("stock_b_selector", "")
            if not stock_b or stock_b == "—":
                st.warning("Please choose Stock B from the sidebar.")
            elif stock_b == dashboard_ticker:
                st.warning("Choose a different Stock B to compare against Stock A.")
            else:
                price_b = load_price_df(stock_b)
                fund_a = load_fundamental_df(dashboard_ticker)
                fund_b = load_fundamental_df(stock_b)
                benchmark_df = load_benchmark_df()

                missing_for_pair = []
                if price_b is None or price_b.empty:
                    missing_for_pair.append(stock_b)
                if fund_a is None or fund_a.empty:
                    missing_for_pair.append(dashboard_ticker)
                if fund_b is None or fund_b.empty:
                    missing_for_pair.append(stock_b)

                if missing_for_pair:
                    st.info(
                        "Missing comparison data for: "
                        + ", ".join(sorted(set(missing_for_pair)))
                        + ". Press Run to collect and process Stock A + Stock B together."
                    )

                st.markdown("#### Performance Chart")
                perf_data = {dashboard_ticker: price_df}
                if price_b is not None:
                    perf_data[stock_b] = price_b
                perf_visualizer = DataVisualizer(perf_data)
                perf_fig = perf_visualizer.performance_comparison_chart(
                    ticker_a=dashboard_ticker,
                    ticker_b=stock_b,
                    benchmark_df=benchmark_df,
                    benchmark_label="VNI",
                    save=True,
                )
                if perf_fig is not None:
                    st.plotly_chart(perf_fig, width="stretch")
                else:
                    st.warning("Not enough data to build performance chart for Stock A, Stock B, and VNI.")

                def _last_val(df, col, pct=False, dollar=False):
                    if df is None or df.empty or col not in df.columns:
                        return "N/A"
                    v = pd.to_numeric(df[col], errors="coerce").dropna()
                    if v.empty:
                        return "N/A"
                    val = float(v.iloc[-1])
                    if pct:
                        return f"{val * 100:.2f}%"
                    if dollar:
                        return f"${val:,.2f}"
                    return f"{val:.4f}"

                def _last_tech(df, col, pct=False):
                    if df is None or df.empty or col not in df.columns:
                        return "N/A"
                    v = pd.to_numeric(df[col], errors="coerce").dropna()
                    if v.empty:
                        return "N/A"
                    val = float(v.iloc[-1])
                    return f"{val * 100:.2f}%" if pct else f"{val:.4f}"

                health_rows = [
                    ("ROE", _last_val(fund_a, "roe", pct=True), _last_val(fund_b, "roe", pct=True)),
                    ("ROA", _last_val(fund_a, "roa", pct=True), _last_val(fund_b, "roa", pct=True)),
                    ("Debt/Equity", _last_val(fund_a, "debt_to_equity"), _last_val(fund_b, "debt_to_equity")),
                    ("Current Ratio", _last_val(fund_a, "current_ratio"), _last_val(fund_b, "current_ratio")),
                    ("Interest Coverage", _last_val(fund_a, "interest_coverage"), _last_val(fund_b, "interest_coverage")),
                    ("Altman Z-Score", _last_val(fund_a, "altman_z_score"), _last_val(fund_b, "altman_z_score")),
                    ("Net Profit Margin", _last_val(fund_a, "net_profit_margin", pct=True), _last_val(fund_b, "net_profit_margin", pct=True)),
                    ("Gross Profit Margin", _last_val(fund_a, "gross_profit_margin", pct=True), _last_val(fund_b, "gross_profit_margin", pct=True)),
                ]

                fund_rows = [
                    ("P/E Ratio", _last_val(fund_a, "pe"), _last_val(fund_b, "pe")),
                    ("P/B Ratio", _last_val(fund_a, "pb"), _last_val(fund_b, "pb")),
                    ("EPS", _last_val(fund_a, "eps", dollar=True), _last_val(fund_b, "eps", dollar=True)),
                    ("BVPS", _last_val(fund_a, "bvps", dollar=True), _last_val(fund_b, "bvps", dollar=True)),
                    ("Dividend", _last_val(fund_a, "dividend", dollar=True), _last_val(fund_b, "dividend", dollar=True)),
                    ("Market Cap", _last_val(fund_a, "market_cap"), _last_val(fund_b, "market_cap")),
                    ("Net Debt/EBITDA", _last_val(fund_a, "net_debt_to_ebitda"), _last_val(fund_b, "net_debt_to_ebitda")),
                ]

                tech_rows = [
                    ("RSI 14", _last_tech(price_df, "rsi_14"), _last_tech(price_b, "rsi_14")),
                    ("MACD Line", _last_tech(price_df, "macd_line"), _last_tech(price_b, "macd_line")),
                    ("Sharpe Ratio", _last_tech(price_df, "sharpe_ratio"), _last_tech(price_b, "sharpe_ratio")),
                    ("Volatility 20d", _last_tech(price_df, "volatility_20", pct=True), _last_tech(price_b, "volatility_20", pct=True)),
                    ("Beta", _last_tech(price_df, "beta"), _last_tech(price_b, "beta")),
                    ("Max Drawdown", _last_tech(price_df, "max_drawdown", pct=True), _last_tech(price_b, "max_drawdown", pct=True)),
                    ("Rel. Strength", _last_tech(price_df, "relative_strength"), _last_tech(price_b, "relative_strength")),
                    ("ADX 14", _last_tech(price_df, "adx_14"), _last_tech(price_b, "adx_14")),
                ]

                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### Company Financial Health")
                    h_df = pd.DataFrame(health_rows, columns=["Metric", dashboard_ticker, stock_b])
                    st.dataframe(h_df.set_index("Metric"), width="stretch")

                    st.markdown("#### Fundamental Valuation")
                    f_df = pd.DataFrame(fund_rows, columns=["Metric", dashboard_ticker, stock_b])
                    st.dataframe(f_df.set_index("Metric"), width="stretch")

                with col_right:
                    st.markdown("#### Technical Valuation")
                    t_df = pd.DataFrame(tech_rows, columns=["Metric", dashboard_ticker, stock_b])
                    st.dataframe(t_df.set_index("Metric"), width="stretch")

                st.markdown("#### Visual Comparison")
                vc_col1, vc_col2 = st.columns(2)
                use_log_scale_cmp = vc_col1.checkbox(
                    "Log scale Y-axis (positive-only panels)",
                    value=False,
                    key=f"cmp_log_scale_{dashboard_ticker}_{stock_b}",
                )
                clip_threshold_cmp = vc_col2.number_input(
                    "Clip threshold (0 = off)",
                    min_value=0.0,
                    value=50.0,
                    step=5.0,
                    key=f"cmp_clip_threshold_{dashboard_ticker}_{stock_b}",
                )
                st.caption("Bars above threshold are clipped and labeled like 50+ to keep smaller metrics readable.")

                all_price_data = {}
                for t in available_tickers():
                    df_t = load_price_df(t)
                    if df_t is not None:
                        all_price_data[t] = df_t

                try:
                    vis_data = {dashboard_ticker: price_df}
                    if price_b is not None:
                        vis_data[stock_b] = price_b
                    visualizer = DataVisualizer(all_price_data)
                    cmp_fig = visualizer.comparison_metrics_chart(
                        ticker_a=dashboard_ticker,
                        ticker_b=stock_b,
                        fund_a=fund_a,
                        fund_b=fund_b,
                        clip_threshold=clip_threshold_cmp if clip_threshold_cmp > 0 else None,
                        use_log_scale=use_log_scale_cmp,
                        save=True,
                    )
                    st.plotly_chart(cmp_fig, width="stretch")
                except Exception as exc:
                    st.warning(f"Could not render comparison bar chart: {exc}")

                st.markdown("#### Efficient Frontier")
                try:
                    ef_visualizer = DataVisualizer(all_price_data)
                    ef_fig = ef_visualizer.efficient_frontier_chart(
                        ticker_a=dashboard_ticker,
                        ticker_b=stock_b,
                        save=True,
                    )
                    st.plotly_chart(ef_fig, width="stretch")
                except Exception as exc:
                    st.warning(f"Could not render efficient frontier: {exc}")

                st.markdown("#### AI Analysis")
                st.caption("Generate grounded narrative analysis using Gemini from the current processed metrics.")

                can_run_ai = price_df is not None and not price_df.empty
                gemini_ready = has_configured_api_key("GEMINI_API_KEY")
                ai_col_run, ai_col_clear = st.columns([1, 1])

                selected_ai_model = st.session_state.get("gemini_model_selector", "gemini-2.5-flash")
                pair_key = f"{dashboard_ticker}|{stock_b}|{selected_ai_model}"

                if not gemini_ready:
                    st.warning("Gemini is not configured. Add a valid GEMINI_API_KEY to .env, then restart or refresh the app.")

                if ai_col_run.button(
                    "Generate AI Analysis (Gemini)",
                    key=f"ai_run_{dashboard_ticker}_{stock_b}",
                    width="stretch",
                    disabled=(not can_run_ai) or (not gemini_ready),
                ):
                    try:
                        with st.spinner("Generating AI analysis with Gemini..."):
                            ai_data = {dashboard_ticker: price_df}
                            if price_b is not None and not price_b.empty:
                                ai_data[stock_b] = price_b

                            agent = AnalysisAgent(provider="gemini", model=selected_ai_model)
                            analysis = agent.generate_full_analysis(
                                ai_data,
                                comparison_tickers=[dashboard_ticker, stock_b] if stock_b in ai_data else [dashboard_ticker],
                                primary_ticker=dashboard_ticker,
                            )
                            st.session_state["ai_analysis_cache"][pair_key] = analysis
                        st.success("AI analysis generated.")
                    except Exception as exc:
                        st.error(f"AI analysis failed: {exc}")

                if ai_col_clear.button(
                    "Clear AI Result",
                    key=f"ai_clear_{dashboard_ticker}_{stock_b}",
                    width="stretch",
                ):
                    st.session_state["ai_analysis_cache"].pop(pair_key, None)

                analysis = st.session_state["ai_analysis_cache"].get(pair_key)
                if analysis:
                    mode = analysis.get("analysis_mode")
                    model_used = analysis.get("model_used")
                    fallback_reason = analysis.get("fallback_reason")
                    if mode == "llm_single_call":
                        if model_used:
                            st.success(f"AI mode: LLM single-call (grounded narrative output) - model: {model_used}")
                        else:
                            st.success("AI mode: LLM single-call (grounded narrative output).")
                    elif mode == "llm_recovered_full_sections":
                        msg = "AI mode: full 4-section LLM output recovered after single-call failure."
                        if model_used:
                            msg += f" Model: {model_used}"
                        st.success(msg)
                    elif mode == "llm_sectional_fallback":
                        msg = "AI mode: sectional LLM fallback (single-call JSON failed, but narrative still generated by LLM)."
                        if model_used:
                            msg += f" Model: {model_used}"
                        st.info(msg)
                    elif mode == "deterministic_fallback":
                        if model_used:
                            st.warning(f"AI mode: deterministic fallback (LLM unavailable or invalid output) - last model: {model_used}")
                        else:
                            st.warning("AI mode: deterministic fallback (LLM unavailable or invalid output).")

                    if fallback_reason:
                        st.caption(f"Fallback reason: {fallback_reason}")

                    with st.expander("Trend Summary", expanded=True):
                        render_plain_text_block(analysis.get("trend_summary", "N/A"))

                    with st.expander("Anomaly Report", expanded=True):
                        render_plain_text_block(analysis.get("anomaly_report", "N/A"))

                    with st.expander("Risk Commentary", expanded=True):
                        render_plain_text_block(analysis.get("risk_commentary", "N/A"))

                    with st.expander("Comparison", expanded=True):
                        render_plain_text_block(analysis.get("comparison", "N/A"))
                else:
                    st.info("No AI analysis yet. Click 'Generate AI Analysis (Gemini)'.")

        left, right = st.columns([1.35, 1])
        with left:
            st.markdown("### Recent rows")
            st.dataframe(price_df.tail(25), width="stretch", height=420)

        with right:
            st.markdown("### Fundamentals")
            fund_df = load_fundamental_df(dashboard_ticker)
            if fund_df is not None and not fund_df.empty:
                st.dataframe(fund_df, width="stretch", height=420)
            else:
                st.info("No fundamental file found for this ticker.")

        st.markdown("### Data snapshot")
        summary_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "daily_return", "rsi_14", "macd_line", "beta", "sharpe_ratio"] if c in price_df.columns]
        st.dataframe(price_df[summary_cols].tail(10), width="stretch")
else:
    st.info("Choose a ticker from the sidebar and press Run to load the dashboard.")

