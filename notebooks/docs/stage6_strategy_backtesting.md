# Stage 6 - Strategy Backtesting (`backtester.py`)

## Objective

Evaluate Stage 5 signals in a realistic baseline setup by:

- Applying transaction costs and slippage
- Comparing strategy performance against a Buy-and-Hold baseline
- Exporting reproducible artifacts for analysis and reporting

Outputs are saved to:

- `data/quant_outputs/backtests/<TICKER>_equity_curve.csv`
- `data/quant_outputs/backtests/<TICKER>_trades.csv`
- `data/quant_outputs/backtests/<TICKER>_metrics.csv`

---

## Data Flow

```text
data/quant_outputs/signals/<TICKER>_signals.csv
	|
	v
[Backtester]
	|
	+- load_signals()
	+- run_backtest()
	|    +- daily_return
	|    +- turnover
	|    +- gross_return
	|    +- cost_return
	|    +- strategy_return
	|    +- equity / buyhold_equity / drawdown
	|    +- trades
	|    +- _build_metrics()
	|
	+- save_outputs()
	|
	v
data/quant_outputs/backtests/*.csv
```

---

## Input Schema (from Stage 5)

The backtester requires these columns:

- `date`
- `close`
- `target_position`
- `position_change`
- `exec_signal`
- `raw_signal`

If any required column is missing, `load_signals()` raises a `ValueError` immediately.

---

## Core Calculation Logic

### 1. Base Return

```python
daily_return = close.pct_change().fillna(0.0)
```

Purpose: close-to-close daily market return.

### 2. Turnover (position state change)

```python
turnover = abs(target_position - target_position.shift(1))
```

Current implementation:

```python
bt["turnover"] = bt["target_position"].diff().abs().fillna(bt["target_position"].abs())
```

- `turnover = 1` when state flips (`0 -> 1` or `1 -> 0`)
- `turnover = 0` when state is unchanged

### 3. Gross Return (before costs)

```python
gross_return = target_position * daily_return
```

- In cash (`target_position = 0`): return is 0
- In long (`target_position = 1`): receive full daily return

### 4. Cost Return (fees + slippage)

```python
cost_per_turn = transaction_cost + slippage
cost_return = turnover * cost_per_turn
```

Costs are applied only on turnover events.

### 5. Net Strategy Return

```python
strategy_return = gross_return - cost_return
```

This is the net return series used for portfolio growth.

### 6. Equity Curves

```python
equity = initial_capital * (1 + strategy_return).cumprod()
buyhold_equity = initial_capital * (1 + daily_return).cumprod()
```

- `equity`: strategy portfolio value
- `buyhold_equity`: always-long benchmark portfolio value

### 7. Drawdown

```python
rolling_peak = equity.cummax()
drawdown = equity / rolling_peak - 1
```

Measures percentage decline from the running equity peak.

---

## Trade Log Construction

Trades are extracted from rows with `turnover > 0`:

```python
trades = bt.loc[bt["turnover"] > 0, [...]].copy()
trades["action"] = np.where(trades["position_change"] > 0, "BUY", "SELL")
```

Interpretation:

- `position_change > 0` -> `BUY`
- `position_change < 0` -> `SELL`

The trade log is used for auditability and execution diagnostics.

---

## Metrics (`_build_metrics`)

Main reported statistics:

- `total_return`: total strategy return
- `cagr`: annualized growth rate
- `max_drawdown`: worst peak-to-trough decline
- `sharpe`: return adjusted by total volatility
- `sortino`: return adjusted by downside volatility
- `avg_daily_return`: average daily strategy return
- `win_rate`: percentage of positive strategy-return days
- `exposure`: fraction of time in long position
- `turnover_events`: number of state changes
- `cost_paid_pct`: cumulative cost drag
- `buyhold_*`: corresponding Buy-and-Hold comparison metrics

---

## Class Structure (`Backtester`)

```text
Backtester
+- __init__(...)
+- load_signals()
+- run_backtest(signal_df)
+- _build_metrics(bt)
+- save_outputs(equity_curve, trades, metrics)
+- run_for_ticker()
+- run_for_universe(tickers, ...)
```

---

## Quick Run Instructions

### Single ticker

```bash
python modules/backtester.py
```

or

```bash
python -c "from modules.backtester import Backtester; r=Backtester('AAPL').run_for_ticker(); print(r['metrics'].to_string(index=False))"
```

### Multiple tickers

```python
from modules.backtester import Backtester
Backtester.run_for_universe(["AAPL", "MSFT", "TSLA"])
```

---

## Validation Checklist

After execution, verify:

1. All 3 output files exist in `data/quant_outputs/backtests/`
2. No critical `NaN` in `equity_curve` core columns (`strategy_return`, `equity`, `drawdown`)
3. `turnover_events` matches expected trading frequency
4. `cost_paid_pct` is in a reasonable range
5. `equity` vs `buyhold_equity` comparison is coherent with strategy behavior

---

## Practical Notes

- This is a vectorized baseline backtest, not an intraday execution simulator.
- It does not include walk-forward train/test splitting.
- Cost modeling is linear in turnover and does not include advanced market impact.
- For serious evaluation, add out-of-sample tests and stress scenarios.

