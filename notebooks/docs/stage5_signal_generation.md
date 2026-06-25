# Stage 5 – Signal Generation (`quant_strategy.py`)

## Mục tiêu

Chuyển đổi dữ liệu đã được xử lý ở Stage 2 (processed prices) thành tín hiệu giao dịch rõ ràng
(`Buy / Sell / Hold`) thông qua một hệ thống tính điểm đa thành phần (multi-factor scoring).
Output là file CSV tại `data/quant_outputs/signals/<TICKER>_signals.csv`.

---

## Luồng dữ liệu

```
data/processed/processed_data/<TICKER>_processed.csv
        │
        ▼
[QuantStrategy]
        │
        ├── compute_trend_score()
        ├── compute_momentum_score()
        ├── compute_risk_score()
        └── compute_quant_score()
                │
                ▼
        build_signals()  →  raw_signal + exec_signal (shift +1, chống look-ahead)
                │
                ▼
        save_signals()
                │
                ▼
data/quant_outputs/signals/<TICKER>_signals.csv
```

---

## Output Schema (file tín hiệu)

| Cột | Kiểu | Mô tả |
|---|---|---|
| `date` | datetime | Ngày |
| `ticker` | str | Mã cổ phiếu |
| `close` | float | Giá đóng cửa |
| `trend_score` | float | Điểm xu hướng [-1, +1] |
| `momentum_score` | float | Điểm momentum [-1, +1] |
| `risk_score` | float | Điểm rủi ro [-1, 0] (luôn âm hoặc bằng 0) |
| `quant_score` | float | Tổng điểm có trọng số |
| `raw_signal` | str | Tín hiệu ngày t (Buy/Sell/Hold) |
| `exec_signal` | str | Tín hiệu thực thi ngày t+1 (shift(1)) |
| `confidence` | float | Độ tin cậy [0, 1] = abs(quant_score) / max_possible_score |

---

## Các thành phần tính điểm

### 1. Trend Score (`weight = 0.4`)

Dùng các cột: `close`, `ma20`, `ma50`, `ma200`, `relative_strength`

| Điều kiện | Điểm |
|---|---|
| close > ma50 **và** ma50 > ma200 (Golden Cross) | +1.0 |
| close > ma50 **nhưng** ma50 <= ma200 | +0.5 |
| close <= ma50 **và** close > ma20 | 0.0 |
| close < ma50 **và** ma50 < ma200 (Death Cross) | -1.0 |
| relative_strength > 1.05 (outperform benchmark ≥ 5%) | +0.2 thêm |
| relative_strength < 0.95 (underperform ≥ 5%) | -0.2 thêm |

*Trend score được clip về [-1, +1] sau khi cộng tất cả thành phần.*

---

### 2. Momentum Score (`weight = 0.35`)

Dùng các cột: `rsi_14`, `macd_hist`, `roc_12`, `stoch_k`

**RSI sub-score:**
| Ngưỡng RSI | Sub-score |
|---|---|
| < 30 (oversold) | +1.0 |
| 30 – 45 | +0.5 |
| 45 – 55 (neutral) | 0.0 |
| 55 – 70 | -0.5 |
| > 70 (overbought) | -1.0 |

**MACD Histogram sub-score:**
- macd_hist > 0 và đang tăng (hist > hist.shift(1)): +1.0
- macd_hist > 0 nhưng đang giảm: +0.5
- macd_hist < 0 và đang giảm: -1.0
- macd_hist < 0 nhưng đang tăng: -0.5

**ROC sub-score:**
- roc_12 > 5%: +0.5
- roc_12 ∈ [-5%, 5%]: 0.0
- roc_12 < -5%: -0.5

**Stochastic sub-score:**
- stoch_k < 20: +0.5
- stoch_k > 80: -0.5
- else: 0.0

*Momentum score = trung bình 4 sub-score, clip về [-1, +1].*

---

### 3. Risk Score (`weight = 0.25`, chỉ trừ điểm)

Dùng các cột: `volatility_30`, `atr_14`, `drawdown`, `var_95`

Risk score luôn ∈ [-1, 0]. Mục đích là giảm nhẹ tổng điểm khi rủi ro cao.

**Volatility penalty:**
- Tính `vol_rank = rolling percentile 252d của volatility_30`
- vol_rank > 0.8: penalty = -0.5
- vol_rank > 0.6: penalty = -0.25
- else: 0.0

**Drawdown penalty:**
- drawdown < -15%: penalty = -0.5
- drawdown ∈ [-15%, -5%]: penalty = -0.25
- drawdown > -5%: 0.0

*Risk score = clip(sum của penalties, -1, 0)*

---

### 4. Quant Score tổng hợp

```
quant_score = (0.4 * trend_score) + (0.35 * momentum_score) + (0.25 * risk_score)
```

---

### 5. Map điểm sang tín hiệu

| Ngưỡng | raw_signal |
|---|---|
| quant_score >= 0.3 | `Buy` |
| quant_score <= -0.2 | `Sell` |
| còn lại | `Hold` |

`exec_signal = raw_signal.shift(1)` — tín hiệu được chấp nhận để thực thi vào ngày **hôm sau**, tránh look-ahead bias.

---

## Cấu trúc class `QuantStrategy`

```
QuantStrategy
├── __init__(ticker, processed_dir, output_dir, weights, buy_threshold, sell_threshold)
├── load_processed_price()
├── compute_trend_score(df)
├── compute_momentum_score(df)
├── compute_risk_score(df)
├── compute_quant_score(df)
├── build_signals(df)
├── save_signals(signal_df)
├── run_for_ticker()                  ← orchestrator đơn lẻ
└── run_for_universe(tickers, ...)    ← classmethod, chạy nhiều ticker
```

---

## Tham số có thể tuning

| Tham số | Default | Mô tả |
|---|---|---|
| `trend_weight` | 0.40 | Trọng số trend trong quant_score |
| `momentum_weight` | 0.35 | Trọng số momentum |
| `risk_weight` | 0.25 | Trọng số risk |
| `buy_threshold` | 0.30 | Ngưỡng tối thiểu để phát tín hiệu Buy |
| `sell_threshold` | -0.20 | Ngưỡng tối đa để phát tín hiệu Sell |
| `vol_lookback` | 252 | Cửa sổ rolling để tính vol percentile |

---

## Chống look-ahead bias

- `raw_signal[t]` = tín hiệu tính từ dữ liệu ngày `t`
- `exec_signal[t]` = `raw_signal[t-1]` → chỉ thực thi vào đầu ngày `t`
- Trong backtester, **luôn dùng `exec_signal`**, không bao giờ dùng `raw_signal`

---

## Tích hợp vào `main.py`

```python
# sau Stage 2 (run_processing)
if args.run_quant:
    from modules.quant_strategy import QuantStrategy
    QuantStrategy.run_for_universe(tickers=tickers, processed_dir=..., output_dir=...)
```

Flag `--run-quant` được thêm vào CLI để bật/tắt độc lập với pipeline AI.

---

## Các bước triển khai tuần tự

1. **Load dữ liệu** từ `data/processed/processed_data/<TICKER>_processed.csv`
   - Validate tồn tại đủ cột cần thiết, raise lỗi rõ ràng nếu thiếu
2. **Tính `trend_score`** từng hàng theo rule bảng trên
3. **Tính `momentum_score`** — cộng 4 sub-score rồi lấy trung bình
4. **Tính `risk_score`** — rolling percentile volatility + drawdown penalties
5. **Tính `quant_score`** = weighted sum
6. **Tính `confidence`** = abs(quant_score) / max_possible_score (= 1.0)
7. **Map sang `raw_signal`** theo ngưỡng buy/sell threshold
8. **Tạo `exec_signal`** = raw_signal.shift(1)
9. **Giữ lại subset cột** theo Output Schema (loại bỏ cột intermediate)
10. **Lưu CSV** → `data/quant_outputs/signals/<TICKER>_signals.csv`
