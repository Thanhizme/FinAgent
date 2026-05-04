from pathlib import Path

import pandas as pd

from main import run_processing
from modules.processor import DataProcessor


RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_price_frame(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / filename)


def build_processing_input() -> dict:
    return {
        "prices": {
            "AAPL": load_price_frame("AAPL_prices.csv"),
        },
        "benchmark": load_price_frame("benchmark_GSPC.csv"),
        "peers": {
            ticker: load_price_frame(f"peer_{ticker}.csv")
            for ticker in ["IDCC", "KLIC", "MANH", "MSFT", "TER"]
        },
        "fundamental": {},
        "macro": pd.DataFrame(),
        "industry": pd.DataFrame(),
        "news": pd.read_csv(RAW_DATA_DIR / "news_AAPL.csv"),
    }


def test_run_pipeline_preserves_raw_date_coverage():
    raw_df = load_price_frame("AAPL_prices.csv")
    raw_dates = pd.to_datetime(raw_df["date"])

    processed_df = DataProcessor(df=raw_df, ticker="AAPL").run_pipeline()

    assert len(processed_df) == len(raw_df)
    assert processed_df["date"].iloc[0] == raw_dates.iloc[0]
    assert processed_df["date"].iloc[-1] == raw_dates.iloc[-1]


def test_calc_beta_produces_rolling_series_after_60_day_window():
    stock_df = DataProcessor(df=load_price_frame("AAPL_prices.csv"), ticker="AAPL").run_pipeline()
    benchmark_df = DataProcessor(df=load_price_frame("benchmark_GSPC.csv"), ticker="benchmark").run_pipeline()

    processor = DataProcessor(df=stock_df, ticker="AAPL")
    processor.df = stock_df.copy()
    processor.calc_beta(benchmark_df)

    beta_series = processor.df["beta"]
    assert beta_series.notna().sum() == len(beta_series) - 60
    assert beta_series.iloc[:60].isna().all()
    assert beta_series.iloc[60:].notna().all()
    assert beta_series.dropna().nunique() > 1


def test_run_processing_adds_relative_strength_and_saves_processed_csvs():
    processed_data = run_processing(build_processing_input())

    aapl_df = processed_data["prices"]["AAPL"]
    benchmark_df = processed_data["benchmark"]
    saved_aapl_df = pd.read_csv(PROCESSED_DATA_DIR / "AAPL_processed.csv")
    saved_benchmark_df = pd.read_csv(PROCESSED_DATA_DIR / "benchmark_processed.csv")

    assert "relative_strength" in aapl_df.columns
    assert aapl_df["relative_strength"].notna().all()
    assert aapl_df["relative_strength"].nunique() > 1
    assert "beta" in aapl_df.columns
    assert aapl_df["beta"].iloc[60:].notna().all()

    assert "relative_strength" in benchmark_df.columns
    assert benchmark_df["relative_strength"].eq(1.0).all()
    assert benchmark_df["beta"].eq(1.0).all()
    assert "atr_14" not in benchmark_df.columns

    assert list(saved_aapl_df.columns) == list(aapl_df.columns)
    assert list(saved_benchmark_df.columns) == list(benchmark_df.columns)
    assert len(saved_aapl_df) == len(aapl_df)
    assert len(saved_benchmark_df) == len(benchmark_df)
    assert "atr_14" not in saved_benchmark_df.columns


def test_run_processing_aggregates_news_to_daily_ticker_rows():
    processed_data = run_processing(build_processing_input())

    news_df = processed_data["news"]
    saved_news_df = pd.read_csv(PROCESSED_DATA_DIR / "news_processed.csv")
    raw_news_df = pd.read_csv(RAW_DATA_DIR / "news_AAPL.csv")

    assert {"date", "ticker", "article_count", "sentiment_score", "positive_count", "neutral_count", "negative_count"}.issubset(news_df.columns)
    assert news_df.duplicated(subset=["date", "ticker"]).sum() == 0
    assert int(news_df["article_count"].sum()) == len(raw_news_df)
    assert (news_df["article_count"] >= 1).all()
    assert (news_df["positive_count"] + news_df["neutral_count"] + news_df["negative_count"] == news_df["article_count"]).all()

    assert list(saved_news_df.columns) == list(news_df.columns)
    assert len(saved_news_df) == len(news_df)