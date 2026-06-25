# Stage 5 - Signal Generation (`quant_strategy.py`)

## Objective

Convert Stage 2 processed price data into explicit trading signals (`Buy / Sell / Hold`) using a multi-factor scoring framework.

Stage 5 output is saved to:

- `data/quant_outputs/signals/<TICKER>_signals.csv`

---

## Data Flow

```text
data/processed/processed_data/<TICKER>_processed.csv
        |
        v
[QuantStrategy]
        |
        +- compute_trend_score()
        +- compute_momentum_score()
        +- compute_risk_score()
        +- compute_quant_score()
        |
        v
build_signals() -> raw_signal + exec_signal (shifted by +1 day)
        |
        v
save_signals()
        |
        v
data/quant_outputs/signals/<TICKER>_signals.csv
```

---

## Output Schema

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Trading date |
| `ticker` | str | Symbol |
| `close` | float | Closing price |
| `trend_score` | float | Trend score in [-1, +1] |
| `momentum_score` | float | Momentum score in [-1, +1] |
| `risk_score` | float | Risk penalty score in [-1, 0] |
| `quant_score` | float | Weighted total score |
| `confidence` | float | Confidence proxy, usually `abs(quant_score)` clipped to [0, 1] |
| `raw_signal` | str | Signal computed on day `t` |
| `exec_signal` | str | Signal executed on day `t+1` (`raw_signal.shift(1)`) |
| `target_position` | int | Strategy state: 1 = Long, 0 = Cash |
| `position_change` | int | Daily change in position state |

---

## Score Components

### 1. Trend Score (`weight = 0.40`)

Inputs: `close`, `ma20`, `ma50`, `ma200`, optional `relative_strength`

- Bullish alignment (`close > ma50` and `ma50 > ma200`) -> +1.0
- Mild bullish (`close > ma50` and `ma50 <= ma200`) -> +0.5
- Neutral (`close <= ma50` and `close > ma20`) -> 0.0
- Bearish alignment (`close < ma50` and `ma50 < ma200`) -> -1.0
- Optional relative strength adjustment:
  - `relative_strength > 1.05` -> +0.2
  - `relative_strength < 0.95` -> -0.2

Final value is clipped to `[-1, +1]`.

### 2. Momentum Score (`weight = 0.35`)

Inputs: `rsi_14`, `macd_hist`, `roc_12`, `stoch_k`

- RSI sub-score from oversold to overbought buckets
- MACD histogram sub-score using sign and slope (`hist` vs `hist.shift(1)`)
- ROC sub-score for positive/neutral/negative momentum regimes
- Stochastic sub-score for overbought/oversold states

Final momentum score is the average of sub-scores, clipped to `[-1, +1]`.

### 3. Risk Score (`weight = 0.25`, penalty only)

Inputs: `volatility_30`, `drawdown` (optional support from `atr_14`, `var_95` if available)

- Volatility penalty via rolling percentile rank
- Drawdown penalty via threshold buckets

Risk score is always non-positive and clipped to `[-1, 0]`.

---

## Total Score Formula

```python
quant_score = (
    0.40 * trend_score
    + 0.35 * momentum_score
    + 0.25 * risk_score
)
```

---

## Signal Mapping

| Condition | `raw_signal` |
|---|---|
| `quant_score >= 0.30` | `Buy` |
| `quant_score <= -0.20` | `Sell` |
| Otherwise | `Hold` |

Execution signal is delayed by one bar:

```python
exec_signal = raw_signal.shift(1)
```

This avoids look-ahead bias.

---

## Anti Look-Ahead Rule

- `raw_signal[t]`: computed using fully closed data at day `t`
- `exec_signal[t]`: equals `raw_signal[t-1]`
- Backtesting must use `exec_signal` or derived `target_position`, not `raw_signal`

---

## Class Structure (`QuantStrategy`)

```text
QuantStrategy
+- __init__(...)
+- load_processed_price()
+- compute_trend_score(df)
+- compute_momentum_score(df)
+- compute_risk_score(df)
+- compute_quant_score(df, trend_score, momentum_score, risk_score)
+- build_signals(df)
+- save_signals(signal_df)
+- run_for_ticker()
+- run_for_universe(tickers, ...)
```

---

## Tunable Parameters

| Parameter | Default | Description |
|---|---|---|
| `trend_weight` | 0.40 | Weight for trend component |
| `momentum_weight` | 0.35 | Weight for momentum component |
| `risk_weight` | 0.25 | Weight for risk component |
| `buy_threshold` | 0.30 | Buy trigger threshold |
| `sell_threshold` | -0.20 | Sell trigger threshold |
| `vol_lookback` | 252 | Rolling window for volatility percentile |

---

## Sequential Implementation Steps

1. Load `data/processed/processed_data/<TICKER>_processed.csv`
2. Validate required columns and data types
3. Compute `trend_score`
4. Compute `momentum_score`
5. Compute `risk_score`
6. Compute `quant_score` as weighted sum
7. Compute `confidence`
8. Map `quant_score` to `raw_signal`
9. Build `exec_signal = raw_signal.shift(1)`
10. Build state columns (`target_position`, `position_change`)
11. Keep only output schema columns
12. Save to `data/quant_outputs/signals/<TICKER>_signals.csv`
