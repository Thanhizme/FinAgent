"""
test_processor.py
-----------------
Test cac ham clean data cua DataProcessor.
Run: python test_processor.py
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

from modules.processor import DataProcessor

# ------------------------------------------------------------------ Load data
df = pd.read_csv("data/raw/AAPL_prices.csv")
print(f"Raw data: {len(df)} rows | columns: {list(df.columns)}")
print(f"dtypes:\n{df.dtypes}\n")

# ------------------------------------------------------------------ Inject test cases
# Them NaN de test handle_missing_values
df_test = df.copy()
df_test.loc[5, "close"]  = np.nan
df_test.loc[10, "open"]  = np.nan
df_test.loc[15, "volume"]= np.nan

# Them duplicate de test remove_duplicates
df_test = pd.concat([df_test, df_test.iloc[[0, 1]]], ignore_index=True)

# Them outlier de test detect_outliers
df_test.loc[20, "close"] = 99999.0   # gia bat thuong

print(f"After inject: {len(df_test)} rows | NaN count: {df_test.isna().sum().sum()} | duplicates: {df_test.duplicated().sum()}\n")

# ------------------------------------------------------------------ TEST 1: normalise_types
print("=" * 60)
print("TEST 1: normalise_types()")
print("=" * 60)
p = DataProcessor(df=df_test, ticker="AAPL")
p.normalise_types()
print(f"date dtype : {p.df['date'].dtype}")
print(f"close dtype: {p.df['close'].dtype}")
print(f"Sorted ascending: {p.df['date'].is_monotonic_increasing}")
print()

# ------------------------------------------------------------------ TEST 2: remove_duplicates
print("=" * 60)
print("TEST 2: remove_duplicates()")
print("=" * 60)
p.remove_duplicates()
print(f"Rows after dedup: {len(p.df)}")
print()

# ------------------------------------------------------------------ TEST 3: handle_missing_values
print("=" * 60)
print("TEST 3: handle_missing_values(strategy='ffill')")
print("=" * 60)
p.handle_missing_values(strategy="ffill")
print(f"NaN remaining: {p.df.isna().sum().sum()}")
print()

# ------------------------------------------------------------------ TEST 4: detect_outliers
print("=" * 60)
print("TEST 4: detect_outliers(method='iqr')")
print("=" * 60)
p.detect_outliers(method="iqr", threshold=3.0)
print(f"is_outlier column exists: {'is_outlier' in p.df.columns}")
print(f"Outliers flagged:\n{p.df[p.df['is_outlier']][['date','close','is_outlier']]}")
print()

# ------------------------------------------------------------------ TEST 5: invalid inputs
print("=" * 60)
print("TEST 5: invalid inputs -> should raise ValueError")
print("=" * 60)
try:
    p.handle_missing_values(strategy="invalid")
except ValueError as e:
    print(f"  handle_missing_values: caught -> {e}")

try:
    p.detect_outliers(method="invalid")
except ValueError as e:
    print(f"  detect_outliers      : caught -> {e}")

print()
print("All tests completed.")
