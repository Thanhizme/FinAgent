# Stage 7 - Portfolio Optimization (`portfolio.py`)

## Objective

Convert single-asset trading strategies (from Stage 5-6) into a multi-asset portfolio using three allocation approaches:

1. **Equal Weight**: baseline (1/n across all assets)
2. **Max Sharpe**: risk-return optimized for maximum risk-adjusted return
3. **Risk Parity**: volatility-balanced allocation

Outputs are saved to:

- `data/quant_outputs/portfolio/<STRATEGY>_equity.csv`
- `data/quant_outputs/portfolio/<STRATEGY>_metrics.csv`
- `data/quant_outputs/portfolio/<STRATEGY>_weights.csv`

---

## Data Flow

```text
data/quant_outputs/backtests/<TICKER>_equity_curve.csv (multiple tickers)
        |
        v
[PortfolioOptimizer]
        |
        +- load_asset_returns()        -> dict of returns per ticker
        +- build_return_matrix()       -> aligned returns matrix
        |
        +- compute_equal_weights()     -> baseline (1/n)
        +- compute_max_sharpe_weights()-> optimization
        +- compute_risk_parity_weights()-> volatility-based
        |
        +- backtest_portfolio()        -> portfolio equity curve
        +- compute_portfolio_metrics() -> performance stats
        +- save_portfolio_outputs()    -> CSV artifacts
        |
        v
data/quant_outputs/portfolio/<STRATEGY>_*.csv
```

---

## Input Schema

Portfolio optimizer reads from Stage 6 backtest output:

- **Source**: `data/quant_outputs/backtests/<TICKER>_equity_curve.csv`
- **Required columns**: `date`, `strategy_return`
- **Frequency**: daily returns

All tickers are merged using **inner join** on date, ensuring synchronized timeline.

---

## Portfolio Construction Strategies

### 1. Equal Weight (Baseline)

**Formula**:
$$w_i = \frac{1}{n} \quad \forall i$$

**Characteristics**:
- No optimization; purely mechanical rebalancing.
- Useful as control benchmark.
- Often beats sophisticated models in crisis periods (simplicity advantage).

---

### 2. Maximum Sharpe Ratio

**Optimization Problem**:

Maximize:
$$\text{Sharpe} = \frac{\mu_p}{\sigma_p}$$

Subject to:
- $\sum_{i=1}^{n} w_i = 1$ (fully invested constraint)
- $w_{\min} \leq w_i \leq w_{\max}$ (position limits, default: [0, 1] for long-only)

**Method**: scipy.optimize.minimize with SLSQP (Sequential Least Squares Programming)

**Output**:
- Weights that maximize risk-adjusted return
- Usually concentrated (not 1/n)
- Sensitive to historical mean/covariance estimates

---

### 3. Risk Parity

**Formula**:
$$w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$$

Where $\sigma_i$ is historical volatility of asset $i$.

**Characteristics**:
- Each asset contributes equally to portfolio risk (in variance terms)
- Favors lower-volatility assets with higher weights
- Robust to estimation error vs. mean-variance optimization
- Commonly used in institutional portfolios (commodities, bonds, equities mix)

---

## Portfolio Return Calculation

**Portfolio daily return**:
$$r_p(t) = \sum_{i=1}^{n} w_i \cdot r_i(t)$$

Where:
- $r_i(t)$ = asset $i$ return on day $t$ (from Stage 6)
- $w_i$ = allocation weight (static or rebalance-dependent)

**Portfolio equity curve**:
$$V(t) = V_0 \cdot \prod_{s=1}^{t} (1 + r_p(s))$$

---

## Performance Metrics

PortfolioOptimizer computes identical metrics as Stage 6 backtester, but at portfolio level:

| Metric | Formula | Interpretation |
|---|---|---|
| `total_return` | $\frac{V_T}{V_0} - 1$ | Cumulative return |
| `cagr` | $\left(\frac{V_T}{V_0}\right)^{1/y} - 1$ | Annualized growth |
| `volatility` | $\text{std}(r_p)$ | Daily return volatility |
| `sharpe` | $\frac{\sqrt{252} \cdot \text{mean}(r_p)}{\text{std}(r_p)}$ | Return per risk unit |
| `sortino` | $\frac{\sqrt{252} \cdot \text{mean}(r_p)}{\text{std}(r_p \text{ if } r_p < 0)}$ | Return per downside risk |
| `max_drawdown` | $\min\left(\frac{V(t)}{V_{\text{peak}}(t)} - 1\right)$ | Worst peak-to-trough |
| `win_rate` | $\frac{\# \text{days where } r_p > 0}{|\text{all days}|}$ | Positive return %  |

---

## Comparison: Strategy vs. Strategy

After backtesting all three strategies on the same return matrix, compare:

- **Sharpe ratio**: Does max Sharpe outperform baseline on risk-adjusted basis?
- **Drawdown**: Does risk parity reduce volatility vs. equal weight?
- **Turnover**: How often does rebalancing occur (if dynamic)?
- **Interpretation**: When does each strategy excel?

---

## Rebalancing

**Current Version (V1)**:
- Static weights applied across entire backtest period
- No dynamic rebalance schedule yet
- Future: monthly/quarterly rebalance logic with transaction cost impact

**Implementation Plan**:
1. Define rebalance dates (e.g., first trading day of each month)
2. Recompute weights on rebalance dates
3. Apply transaction cost: $\text{cost} = \sum_i |\Delta w_i| \times \text{tc\_bps}$
4. Deduct from portfolio return on rebalance day

---

## Class Structure

```text
PortfolioOptimizer
+- __init__(tickers, signals_dir, backtest_dir, output_dir, initial_capital, ...)
+- load_asset_returns()
+- build_return_matrix(returns_dict)
+- compute_equal_weights()
+- compute_max_sharpe_weights(returns)
+- compute_risk_parity_weights(returns)
+- apply_rebalance_schedule(returns, weights, freq)
+- backtest_portfolio(returns, weights)
+- compute_portfolio_metrics(bt_result, static_weights)
+- save_portfolio_outputs(bt_result, metrics, weights, strategy_name)
+- run_for_universe()
```

---

## Running Portfolio Optimization

### Single run (3 strategies at once)

```python
from modules.portfolio import PortfolioOptimizer

optimizer = PortfolioOptimizer(
    tickers=["AAPL", "MSFT", "TSLA"],
    initial_capital=100_000.0,
)
results = optimizer.run_for_universe()

# results keys: 'equal_weight', 'max_sharpe', 'risk_parity'
print(results['max_sharpe']['metrics'])
```

### CLI test

```bash
python modules/portfolio.py
```

---

## Output Files

For each strategy (e.g., `max_sharpe`):

1. **{strategy}_equity.csv**
   - Columns: date, ticker_1, ..., ticker_n, portfolio_return, portfolio_equity, portfolio_drawdown
   - Daily returns for all assets + portfolio aggregates
   - Used for time-series analysis and charting

2. **{strategy}_metrics.csv**
   - Single row: strategy, start_date, end_date, total_return, cagr, sharpe, sortino, max_drawdown, win_rate, weights
   - Summary statistics for comparison

3. **{strategy}_weights.csv**
   - Single row: weight_AAPL, weight_MSFT, weight_TSLA, ...
   - Final allocation used in backtest

---

## Validation Checklist

After running `run_for_universe()`, verify:

1. All three strategy folders created (`equal_weight`, `max_sharpe`, `risk_parity`)
2. Each strategy has 3 CSV files (equity, metrics, weights)
3. Portfolio weights sum to 1.0 (or within rounding error)
4. Portfolio equity curve is smooth (no NaN jumps)
5. Max Sharpe weights are different from equal (1/n), indicating optimization worked
6. Risk Parity weights inversely correlate with individual volatilities

---

## Advanced Topics (Future)

- **Rebalancing frequency**: monthly/quarterly dynamic updates
- **Factor exposure**: tilt towards value, momentum, or quality
- **Constraints**: sector limits, correlation limits, ESG screens
- **Out-of-sample**: rolling window optimization to validate robustness
- **Transaction cost modeling**: impact, bid-ask, market depth
- **Stress testing**: portfolio behavior in historical crisis scenarios

---

## Key Assumptions

- Returns are independently and identically distributed (IID) → May not hold
- No short selling allowed (long-only weights in [0, 1])
- Static weights over backtest period → Ignores rebalance cost
- Historical volatility is good proxy for future risk → Can be wrong in regime shifts
- Correlations are stable → Often breaks down in crashes
