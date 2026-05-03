"""
test_collector.py
-----------------
Quick smoke test for all DataCollector methods.
Run with: python test_collector.py
"""

import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

from modules.collector import DataCollector

# Dynamic dates - mirrors the logic in main.py
_TODAY      = datetime.today()
_END_DATE   = (_TODAY - timedelta(days=1)).strftime("%Y-%m-%d")         # yesterday
_START_DATE = (_TODAY - relativedelta(months=18)).strftime("%Y-%m-%d")  # 18 months ago

print(f"Date range: {_START_DATE} to {_END_DATE}")
print()

collector = DataCollector(
    tickers=["AAPL"],   # list cong ty
    start_date=_START_DATE,
    end_date=_END_DATE,
    market="GLOBAL",            # "GLOBAL" or "VN"
)

# ------------------------------------------------------------------
# TEST 1 - Stock Prices
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 1: fetch_stock_prices()")
print("=" * 60)
prices = collector.fetch_stock_prices()
for ticker, df in prices.items():
    print(f"  {ticker}: {len(df)} rows | columns: {list(df.columns)}")
    print(f"  Sample:\n{df.tail(3)}\n")

# ------------------------------------------------------------------
# TEST 2 - News
# ------------------------------------------------------------------
print("=" * 60)
print("TEST 2: fetch_news() -> news_df")
print("=" * 60)
news_df = collector.fetch_news(query="AAPL MSFT stock", page_size=5)
if not news_df.empty:
    print(f"  Rows: {len(news_df)} | Columns: {list(news_df.columns)}")
    print()
    for _, row in news_df.iterrows():
        d, t, s, e, h = row["date"], row["ticker"], row["sentiment"], row["event_type"], row["headline"]
        print(f"  [{d}] [{t}] [{s}] [{e}]")
        print(f"    {h}")
    print()
else:
    print("  news_df: empty")

# ------------------------------------------------------------------
# TEST 3 - Fundamental Data
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 3: fetch_financial_statements() -> fundamental_df")
print("=" * 60)
fundamentals = collector.fetch_financial_statements()
for ticker, df in fundamentals.items():
    if df is not None and not df.empty:
        print(f"  {ticker}: {len(df)} quarters | columns: {list(df.columns)}")
        print(f"  Sample:\n{df.tail(3).to_string()}\n")
    else:
        print(f"  {ticker}: empty")

# ------------------------------------------------------------------
# TEST 4 - Benchmark
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 4: fetch_benchmark() -> benchmark_df")
print("=" * 60)
benchmark = collector.fetch_benchmark()
if not benchmark.empty:
    print(f"  Benchmark: {len(benchmark)} rows | columns: {list(benchmark.columns)}")
    print(f"  Sample:\n{benchmark.tail(3).to_string()}\n")
else:
    print("  benchmark_df: empty")

# ------------------------------------------------------------------
# TEST 5 - Peers
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 5: fetch_peers() -> peer_df")
print("=" * 60)
peers = collector.fetch_peers()
for ticker, df in peers.items():
    if df is not None and not df.empty:
        print(f"  {ticker}: {len(df)} rows | columns: {list(df.columns)}")
        print(f"  Sample:\n{df.tail(2).to_string()}\n")
    else:
        print(f"  {ticker}: empty")

print()


# ------------------------------------------------------------------
# TEST 6 - Macro Indicators
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 6: fetch_macro_indicators() -> macro_df")
print("=" * 60)
macro_df = collector.fetch_macro_indicators()
if not macro_df.empty:
    print(f"  Rows: {len(macro_df)} | Variables: {macro_df['variable'].unique().tolist()}")
    print(f"  Sample:\n{macro_df.groupby('variable').tail(1).to_string()}\n")
else:
    print("  macro_df: empty")

# ------------------------------------------------------------------
# TEST 7 - Industry Data
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 7: fetch_industry_data() -> industry_df")
print("=" * 60)
industry_df = collector.fetch_industry_data()
if not industry_df.empty:
    print(f"  Rows: {len(industry_df)} | Columns: {list(industry_df.columns)}")
    print(f"  Sample:\n{industry_df.tail(3).to_string()}\n")
else:
    print("  industry_df: empty")

# ------------------------------------------------------------------
# TEST 8 - Intraday Data
# ------------------------------------------------------------------
print()
print("=" * 60)
print("TEST 8: fetch_intraday(interval=5m, period=5d) -> intraday_df")
print("=" * 60)
intraday = collector.fetch_intraday(interval="5m", period="5d")
for ticker, df in intraday.items():
    print(f"  {ticker}: {len(df)} bars | columns: {list(df.columns)}")
    print(f"  Sample:\n{df.tail(3).to_string()}\n")

print("All tests completed. Check data/raw/ for saved files.")