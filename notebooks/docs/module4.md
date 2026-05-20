# Module 4: AI Analysis Specification

## 1. Objective
* **Purpose:** Generate a professional-grade financial analysis report by feeding processed data into a Large Language Model (LLM).
* **Input:** A structured JSON object containing:
    * `ticker`: Target stock (Stock A) and comparison stock (Stock B).
    * `price_summary`: Returns, price levels, and volatility.
    * `technical_metrics`: MA, RSI, MACD, and anomalies.
    * `fundamental_metrics`: Financial ratios (Profitability, Liquidity, Solvency).
    * `macro_summary`: Global & Domestic economic indicators.
    * `news_summary`: Recent corporate events and sentiment.
    * `chart_descriptions`: Qualitative insights from Module 3 charts.

---

## 2. Report Structure & Prompt Logic

### Output Format Rule (Apply to ALL sections)
For every indicator or metric, the LLM **must** output in the following format:
{Indicator Name}: {Actual Value or Range} → {2–3 sentence analytical interpretation explaining what this number means, why it matters for this specific company, and what signal it gives for investment decision-making.} 
**The LLM must never output numbers or values alone without interpretation.**
Every data point must be followed by a substantive analytical paragraph.
Default output language: **English**.

---

### A. Executive Summary

**Basic Information**
Output the following fields as labelled lines:
- Ticker & Exchange
- Company Name
- Industry & Sub-sector
- Market Cap Classification (Large / Mid / Small Cap) with approximate USD value

# Change a little bit from the previous version
**Summary Statement**
- `Short-term View (1–3 months):`
    - 1 sentence summarizing Fundamental signal
    - 1 sentence summarizing Technical Valuation signal
- `Long-term View (1–3 years):`
    -1 sentence summarizing Fundamental signal
    - 1 sentence summarizing Technical Valuation signal

**Comparison Snippet** *(if Stock B provided)*
→ 2–3 sentences comparing the two stocks' investment profiles (e.g., Defensive vs. Growth, income vs. capital appreciation).

**Call to Action**
- Strategic recommendation: `Accumulate / Hold / Wait`
  → 2–3 sentences specifying which investor profile this stock suits and under what market conditions accumulation is most attractive.
- *Standard AI Disclaimer:* "This analysis is AI-generated for informational purposes only and is not investment advice."

---

### B. Macro Analysis

* **Purpose:** 
  Analyze macroeconomic conditions affecting Stock A’s business performance, valuation, capital flows, and market sentiment.

---

## Case 1 — Vietnam Market (`market = "VN"`)

### Global Macro Indicators *(source: `macro_df_wide`)*

For each indicator below, output:
- Latest value
- Trend direction
- 2–3 sentence interpretation of impact on Vietnam market conditions and Stock A.

1. `imf_global_growth` — IMF global GDP growth forecast
2. `fed_funds_rate` — US Federal Reserve policy rate
3. `oil_price` — Global crude oil price level

### Domestic (Vietnam) Macro Indicators *(source: `macro_df_wide`)*

For each indicator below, output:
- Latest value
- Trend direction
- 2–3 sentence interpretation of direct impact on Stock A’s business and valuation.

1. `vn_gdp_growth` — Vietnam GDP growth rate
2. `vn_cpi` — Vietnam inflation rate

---

## Case 2 — Global / US Market (`market = "GLOBAL"`)

### Global & US Macro Indicators *(source: `macro_df_wide`)*

For each indicator below, output:
- Latest value
- Trend direction
- 2–3 sentence interpretation of direct impact on Stock A’s business, valuation, and market sentiment.

1. `imf_global_growth` — IMF global GDP growth forecast
2. `fed_funds_rate` — US Federal Reserve policy rate
3. `oil_price` — Global crude oil price level
4. `us_gdp_growth` — US GDP growth rate
5. `us_cpi` — US inflation level

**Industry Valuation & Profitability** *(source: `industry_df`)*
# Change a little bit
*(Each indicator as a bullet point, followed by a 1-sentence comparative conclusion)*


- `P/E Industry 1Y`: {value}
- `P/E Industry 5Y`: {value}
  → 1 sentence comparing P/E 1Y vs. P/E 5Y and concluding which stage the industry is currently in.


- `P/B Industry 1Y`: {value}
- `P/B Industry 5Y`: {value}
  → 1 sentence comparing P/B 1Y vs. P/B 5Y and concluding which stage the industry is currently in.


**Corporate Events & News** *(source: `news_df`)*

For each detected `event_type`, output the event + 2–3 sentence interpretation of its likely impact on stock price, sentiment, or fundamentals:

- `dividend` / `earnings` / `m&a` / `management_change`
- `expansion` / `legal` / `macro` / `general`

### C. Company's Financial Health

**Profitability** *(source: `fundamental_metrics`)*

1. `Revenue Growth (YoY)`: {value}
   → Interpret acceleration/deceleration, compare to industry, and flag key drivers.
2. `ROE`: {value}
   → Interpret capital efficiency, sustainability, and comparison to industry benchmark.

**Activity Ratios**
3. `Cash Conversion Cycle (DSO + DIO − DPO)`: {value} days
   → Interpret working capital efficiency and liquidity cycle vs. industry norms.

**Liquidity & Solvency**
4. `Current Ratio`: {value}
    → Interpret short-term liquidity adequacy and buffer against operational stress.
5. `Debt-to-Equity (D/E)`: {value}
    → Interpret leverage risk, financial flexibility, and impact on cost of capital.

**Cash Flow Analysis**
6. `FCFF`: {value}
    → Interpret firm-level free cash flow capacity and reinvestment headroom.
7. `FCFE`: {value}
    → Interpret equity cash flow available for dividends, buybacks, or growth investment.

→ **Overall Financial Health Conclusion:** 2–3 sentences summarizing the company's balance sheet strength, earnings quality, and cash flow profile relative to sector peers.

---

### D. Valuation Analysis

**Fundamental Valuation**
# Change a little bit
1. **P/E**
- `P/E 1Y`: {value}
- `P/E 5Y`: {value}
- `P/E Industry`: {value}
  → 1 sentence comparing P/E 1Y vs. P/E 5Y; 1 sentence comparing P/E 1Y vs. Industry P/E → conclude: `Undervalued / Fairly Valued / Overvalued`
2. **P/B**
- `P/B 1Y`: {value}
- `P/B 5Y`: {value}
- `P/B Industry`: {value}
  → 1 sentence comparing P/B 1Y vs. P/B 5Y; 1 sentence comparing P/B 1Y vs. Industry P/B → conclude: `Undervalued / Fairly Valued / Overvalued`
3. `DCF Valuation (FCFE-Based)`:
   - Intrinsic Price: {value}
   - Current Market Price: {value}
   - Upside / Downside: {value}%
   - Valuation Status: `Undervalued / Fairly Valued / Overvalued`
   → Interpret the reliability of the DCF result given the growth assumptions used, and the margin of safety implied by the current market price.

**Technical Analysis**

*Trend Summary*

4. `Current Price`: {value}
   → Interpret where price sits relative to 12-month range and what it implies for market structure (trending vs. sideways).
5. `1W Return`: {value}%
   → Interpret short-term momentum direction and any catalysts behind recent movement.
6. `1M Return`: {value}%
   → Interpret 1-month price action and sentiment shift vs. prior period.
7. `3M Return`: {value}%
   → Interpret medium-term trend recovery or deterioration.
8. `YTD Return`: {value}% vs. VN-Index YTD
   → Interpret relative performance and whether the stock is leading or lagging the market.

*Moving Averages & Oscillators*

9. `MA20`: Price vs. MA20 → {above / below / crossing}
   → Interpret short-term momentum and whether selling pressure has eased.
10. `MA50`: Price vs. MA50 → {above / below / crossing}
    → Interpret medium-term trend direction and consolidation signals.
11. `MA200`: Price vs. MA200 → {above / below / crossing}
    → Interpret long-term trend status and whether a structural reversal is forming.
12. `RSI(14)`: {value}
    → Interpret momentum condition (overbought / neutral / oversold) and what it signals for near-term price action.
13. `MACD`: {signal}
    → Interpret bullish/bearish crossover and momentum confirmation or divergence.
14. `Bollinger Bands`: {price_relative_to_bands} → Evaluate volatility and overbought/oversold levels; identify potential mean reversion or trend breakouts when price touches or pierges the upper/lower bands. 

*Price & Volume Anomalies*

14. `Volume Spike`: {detected / not detected} — {context}
    → Explain the likely cause (institutional activity, news event, etc.) and what it implies for short-term supply/demand dynamics.
15. `Gap Up / Gap Down`: {detected / not detected} — {direction & context}
    → Explain the trigger and whether the gap has been filled or remains open as a technical reference level.
16. `Sudden Price Movement`: {detected / not detected} — {direction & magnitude}
    → Explain the probable cause (earnings surprise, macro shock, foreign flow) and its implication for trend continuation or reversal.

**Risk Metrics**

17. `Historical Volatility 30D`: {value}%
    → Interpret risk level relative to sector peers and what it implies for position sizing.
18. `Historical Volatility 60D`: {value}%
    → Interpret medium-term volatility stability and speculative activity level.
19. `Beta vs. VN-Index`: {value}
    → Interpret market sensitivity and expected behavior during broad market up/down cycles.
20. `VaR 95%`: {value}% daily
    → Interpret expected maximum daily loss under normal market conditions 95% of the time.
21. `VaR 99%`: {value}% daily
    → Interpret tail-risk exposure under extreme market stress scenarios.
22. `Max Drawdown`: {value}%
    → Interpret historical worst-case loss scenario and what it implies for long-term holders.
23. `Sharpe Ratio`: {value}
    → Interpret risk-adjusted return efficiency and compare to benchmark or peer.

→ **Technical Conclusion:** 2–3 sentences summarizing overall trend structure, momentum quality, and recommended approach (accumulate gradually / wait for breakout / avoid near-term).

---

### E. Peer Comparison (Stock A vs. Stock B)

* **Auto-selection logic:** If no Stock B is provided, system auto-selects a peer with similar Market Cap from the sample universe.

**Financial Health Comparison**

### Comparison Analysis (Stock A vs. Stock B)

* **Peer Selection Rule:**
  - If `stock_b` is provided by the client, use it directly.
  - If not provided, automatically select a peer from the predefined sample universe with similar:
    - Sector
    - Market capitalization group (Large/Mid/Small Cap)

---

## Financial Health Comparison

Use  indicators available from `fundamental_df` to compare Stock A and Stock B.

*Profitability*
1. `Revenue Growth (YoY)`
2. `ROE`

*Liquidity & Solvency*
3. `Current Ratio`
4. `Debt-to-Equity (D/E)`

*Cash Flow Analysis*
5. `FCFE`

Output Requirements:
- Highlight relative strengths and weaknesses between the two companies.
- Explain which company demonstrates:
  - better financial health

---

## Fundamental Valuation Comparison

Use valuation indicators and historical valuation context to compare Stock A and Stock B.

1. ‘Current P/E: {value} vs. Industry avg {value}
2. ‘Current P/B’: {value} vs. Industry avg {value}
3. `DCF Valuation (FCFE-Based)`:
- Intrinsic Price: {value}
- Upside / Downside: {value}%

Output Requirements:
- Explain whether either stock appears:
  - undervalued
  - fairly valued
  - relatively expensive
- Discuss valuation attractiveness under the current market environment.

---

## Technical & Risk Profile Comparison

Use technical and risk indicators derived from `price_df` and risk analytics modules.

*Moving Averages & Oscillators*
1. `MACD`
2. `RSI(14)`

*Risk Metrics*
3. `Historical Volatility 30D`
4. `Max Drawdown`
5. `Sharpe Ratio`


Output Requirements:
- Explain differences in:
  - price momentum
  - volatility/risk profile
  - defensive vs. cyclical characteristics
  - short-term trading sentiment

---

## Comparison Summary

Generate a concise comparative conclusion that:
- Positions Stock A versus Stock B under current market conditions.
- Identifies which stock may be more suitable for:
  - growth-oriented investors
  - value-focused investors
- Summarizes the key differentiating factors driving the recommendation.


## 3. Visual Design & Formatting (LLM Output)
* **Tone:** Professional, objective, and data-driven (Financial Analyst style).
* **Formatting:** Markdown tables for comparison matrices; numbered bullet format (`1. Indicator: Value → Analysis`) for all metric sections.
* **Language:** English by default. Vietnamese supported if specified by user.
* **Strict Rule:** The LLM must never return raw numbers without analytical interpretation. Every metric must include a 2–3 sentence explanatory paragraph.

---

## 4. Copilot Execution Prompt

> "Implement the AI Analysis logic in `modules/analyzer.py` based on `#file:module4.md`.
> Construct a comprehensive prompt template that pipes Module 2 data and Module 3 insights into the LLM.
> The output must strictly follow the Executive Summary → Macro → Financial Health → Valuation → Comparison structure.
> For every single metric and indicator across all sections, the LLM must output:
> `{Indicator}: {Value} → {2–3 sentence analytical interpretation}`
> The LLM must never return standalone numbers without interpretation.
> Ensure the DCF calculation and Peer Comparison logic are dynamically handled based on the input tickers.
> Target Output: A structured Markdown report for the Streamlit 'AI Insight' tab."

