"""
collector.py
------------
Sources:
  - Stock prices         : yfinance
  - Financial statements : yfinance
  - News & sentiment     : NewsAPI
  - Macro indicators     : yfinance (Schema E)
  - Industry data        : yfinance averaged from peers (Schema F)
  - Intraday data        : yfinance (Section 5)
"""

import os
import re
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class DataCollector:
    """
    Collects financial data from multiple sources and saves raw files to disk.

    Parameters
    ----------
    tickers : list[str]
    start_date : str  (YYYY-MM-DD)
    end_date   : str  (YYYY-MM-DD)
    market     : str  "GLOBAL" | "VN"
    """

    # ------------------------------------------------------------------ constants

    _BENCHMARK_TICKER = {
        "GLOBAL": "^GSPC",
        "VN":     "^VNINDEX",
    }

    _PEER_MAP = {
        # ========================================================
        # US MARKET SECTORS
        # ========================================================
        # Information Technology
        "AAPL":  ["MSFT", "MANH", "TER", "IDCC", "KLIC"],
        "MSFT":  ["AAPL", "MANH", "TER", "IDCC", "KLIC"],
        
        # Financials
        "JPM":   ["V", "SF", "JEF", "DFIN", "VBTX"],
        "V":     ["JPM", "SF", "JEF", "DFIN", "VBTX"],
        
        # Health Care
        "AMGN":  ["ELV", "HALO", "EHC", "HIMS", "NSTG"],
        
        # Consumer Discretionary
        "AMZN":  ["TSLA", "DECK", "CROX", "BOOT", "SONO"],
        "TSLA":  ["AMZN", "DECK", "CROX", "BOOT", "SONO"],
        
        # Consumer Staples
        "MDLZ":  ["KMB", "CASY", "CELH", "CALM", "JJSF"],
        
        # Industrials
        "LMT":   ["GE", "DE", "UPS", "BYRN", "MLKN"],
        "GE":    ["LMT", "DE", "UPS", "BYRN", "MLKN"],
        
        # Energy
        "XOM":   ["CVX", "OVV", "APA", "REPX", "PARR"],
        "CVX":   ["XOM", "OVV", "APA", "REPX", "PARR"],
        
        # Utilities
        "D":     ["NEE", "VST", "NRG", "AWR", "AVA"],
        
        # Real Estate
        "PLD":   ["EQIX", "REXR", "OHI", "LGIH", "UTL"],
        
        # Materials
        "LIN":   ["SHW", "RS", "STLD", "MLI", "IOSP"],
        
        # Communication Services
        "GOOGL": ["META", "PINS", "TTWO", "CNK", "YELP"],
        "META":  ["GOOGL", "PINS", "TTWO", "CNK", "YELP"],

        # ========================================================
        # VIETNAM MARKET SECTORS
        # ========================================================
        # 1. Ngân hàng
        "VCB":   ["BID", "LPB", "MSB", "BVB", "ABB"],
        "BID":   ["VCB", "LPB", "MSB", "BVB", "ABB"],
        
        # 2. Bất động sản
        "VHM":   ["VIC", "KDH", "NLG", "DRH", "HQC"],
        "VIC":   ["VHM", "KDH", "NLG", "DRH", "HQC"],
        
        # 3. Thực phẩm & Đồ uống
        "VNM":   ["MSN", "PAN", "VHC", "ANV", "IDI"],
        "MSN":   ["VNM", "PAN", "VHC", "ANV", "IDI"],
        
        # 4. Dịch vụ tài chính (Chứng khoán)
        "SSI":   ["VND", "VCI", "HCM", "BSI", "FTS"],
        "VND":   ["SSI", "VCI", "HCM", "BSI", "FTS"],
        
        # 5. Tài nguyên cơ bản (Thép)
        "HPG":   ["GVR", "HSG", "NKG", "TVN", "VGS"],
        
        # 6. Dầu khí
        "GAS":   ["PLX", "PVS", "PVD", "PVC", "PVB"],
        
        # 7. Bán lẻ
        "MWG":   ["PNJ", "FRT", "DGW", "PET", "ASG"],
        "PNJ":   ["MWG", "FRT", "DGW", "PET", "ASG"],
        
        # 11. Công nghệ thông tin
        "FPT":   ["VGI", "CMG", "FOX", "ELC", "ITD"],
        "VGI":   ["FPT", "CMG", "FOX", "ELC", "ITD"],
    }

    _MACRO_TICKERS = {
        "gold_price":    "GC=F",
        "oil_price":     "CL=F",
        "usd_vnd":       "USDVND=X",
        #"vnindex":       "^VNINDEX",
        "bond_yield":    "^TNX",
        "interest_rate": "^IRX",
    }

    _SENTIMENT_POSITIVE = [
        "surge", "beat", "profit", "gain", "growth", "record", "rally",
        "up", "rise", "strong", "boost", "exceed", "outperform", "high",
        "upgrade", "bullish", "optimistic", "recovery", "rebound", "soar",
        "skyrocket", "buyback"
    ]
    _SENTIMENT_NEGATIVE = [
        "crash", "loss", "miss", "fall", "drop", "down", "cut", "risk",
        "warn", "decline", "weak", "below", "disappoint", "layoff", "fine",
        "downgrade", "bearish", "pessimistic", "plummet", "tumble", "slump",
        "bankrupt", "uncertainty"
    ]
    _EVENT_KEYWORDS = {
        "earnings":          ["earnings", "eps", "revenue", "profit", "quarterly", "results", "guidance", "beating", "missing"],
        "dividend":          ["dividend", "bonus", "ex-dividend", "payout", "yield", "distribution"],
        "m&a":               ["acquire", "merger", "buyout", "takeover", "deal", "acquisition"],
        "management_change": ["ceo", "resign", "appoint", "executive", "board", "chief"],
        "expansion":         ["expand", "launch", "open", "new market", "partnership", "contract"],
        "macro":             ["fed", "inflation", "cpi", "gdp", "interest rate", "central bank"],
        "legal":             ["lawsuit", "regulatory", "sanction", "compliance", "fine", "penalty", "sec", "investigation", "fraud"],
    }
    _NEWS_ENTITY_ALIASES = {
        "AAPL": ["Apple", "Apple Inc"],
        "MSFT": ["Microsoft", "Microsoft Corp"],
        "JPM": ["JPMorgan", "JPMorgan Chase", "JPMorgan Chase & Co"],
        "V": ["Visa", "Visa Inc"],
        "SF": ["Stifel", "Stifel Financial"],
        "JEF": ["Jefferies", "Jefferies Financial"],
        "DFIN": ["Donnelley Financial", "Donnelley Financial Solutions"],
        "VBTX": ["Veritex", "Veritex Holdings"],
    }
    _AMBIGUOUS_NEWS_TICKERS = {"V", "D", "GE", "F", "T"}
    _NEWS_FINANCE_CONTEXT_KEYWORDS = [
        "stock", "shares", "earnings", "revenue", "profit", "guidance",
        "company", "corp", "inc", "bank", "financial", "finance",
        "payment", "payments", "card", "cards", "merchant", "transaction",
        "analyst", "wall street", "quarter", "results", "investor", "market",
    ]
    _NEWS_EXCLUSION_KEYWORDS = {
        "V": [
            "h-1b", "immigration", "passport", "tourist visa", "visa assistance",
            "airport", "travel package", "luxury tour", "hotel", "trip",
        ],
    }

    # ------------------------------------------------------------------ init

    def __init__(self, tickers, start_date, end_date, market="GLOBAL"):
        self.tickers      = tickers
        self.start_date   = start_date
        self.end_date     = end_date
        self.market       = market.upper()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        logger.info("DataCollector initialised | tickers=%s | market=%s | %s to %s",
                    self.tickers, self.market, self.start_date, self.end_date)

    def _get_news_search_terms(self, ticker: str) -> list[str]:
        ticker = ticker.upper()
        terms = [ticker]
        aliases = self._NEWS_ENTITY_ALIASES.get(ticker, [])
        for alias in aliases:
            if alias not in terms:
                terms.append(alias)
        return terms

    def _empty_news_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["date", "ticker", "headline", "summary", "source", "sentiment", "event_type"]
        )

    def _build_ticker_news_query(self, ticker: str, user_query: str | None = None) -> str:
        terms = self._get_news_search_terms(ticker)
        if len(ticker) <= 2 and len(terms) > 1:
            # Single-character and other short tickers need company-name anchors.
            return " OR ".join(f'"{term}"' if " " in term else term for term in terms[1:])

        query_terms = [f'"{term}"' if " " in term else term for term in terms]
        if user_query and ticker.upper() not in user_query.upper():
            query_terms.append(f'"{user_query}"' if " " in user_query else user_query)
        return " OR ".join(query_terms)

    def _text_contains_term(self, text: str, term: str) -> bool:
        text = text.lower()
        term = term.lower()
        if term.isalpha() and len(term) <= 5 and " " not in term:
            return re.search(rf"\b{re.escape(term)}\b", text) is not None
        return term in text

    def _is_relevant_news_article(self, article: dict, ticker: str) -> bool:
        text = " ".join(
            [
                article.get("title") or "",
                article.get("description") or "",
                article.get("content") or "",
                ((article.get("source") or {}).get("name") or ""),
            ]
        ).lower()

        terms = self._get_news_search_terms(ticker)
        strong_terms = [term for term in terms if term.upper() != ticker.upper()]
        alias_match = any(self._text_contains_term(text, term) for term in strong_terms)
        ticker_match = self._text_contains_term(text, ticker)

        exclusion_terms = self._NEWS_EXCLUSION_KEYWORDS.get(ticker.upper(), [])
        if any(exclusion in text for exclusion in exclusion_terms):
            return False

        if ticker.upper() in self._AMBIGUOUS_NEWS_TICKERS:
            finance_context = any(
                self._text_contains_term(text, keyword)
                for keyword in self._NEWS_FINANCE_CONTEXT_KEYWORDS
            )
            return alias_match and finance_context

        if strong_terms:
            return alias_match or ticker_match
        return ticker_match

    # ------------------------------------------------------------------ helpers

    def _flatten(self, raw):
        """Flatten yfinance MultiIndex columns and reset index to plain date column."""
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        return raw.reset_index().rename(columns={"Date": "date", "Price": "date", "Datetime": "date"})

    def _save_csv(self, df, filename):
        """Persist DataFrame to data/raw/ as CSV."""
        if df is None or df.empty:
            logger.warning("Skipping save - empty: %s", filename)
            return None
        fp = RAW_DATA_DIR / filename
        try:
            df.to_csv(fp, index=False)
            logger.info("Saved raw data -> %s", fp)
            return fp
        except Exception as e:
            logger.error("Error saving %s: %s", filename, e)
            return None

    def _validate_df(self, df, name="DataFrame", date_col="date"):
        """
        Validate a time-series DataFrame (module1.md Section 5):
          1. Log missing value counts per column.
          2. Drop rows where ALL non-date columns are NaN.
          3. Ensure ascending date sort.
        """
        if df is None or df.empty:
            logger.warning("[validate] %s is empty.", name)
            return df
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            logger.warning("[validate] %s missing values:\n%s", name, missing.to_string())
        else:
            logger.info("[validate] %s - no missing values.", name)
        value_cols = [c for c in df.columns if c != date_col]
        before = len(df)
        df = df.dropna(subset=value_cols, how="all")
        if len(df) < before:
            logger.warning("[validate] %s dropped %d fully-empty rows.", name, before - len(df))
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ A. Price Data

    def fetch_stock_prices(self):
        """
        Schema (price_df): date, ticker, open, high, low, close, adj_close, volume
        """
        data_map = {}
        for ticker in self.tickers:
            logger.info("Fetching price data for %s ...", ticker)
            for attempt in range(1, 4):
                try:
                    raw = yf.download(ticker, start=self.start_date, end=self.end_date,
                                      auto_adjust=True, progress=False)
                    if not raw.empty:
                        df = self._flatten(raw)
                        df.insert(1, "ticker", ticker)
                        df["adj_close"] = df["close"]
                        cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
                        df = df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)
                        df = self._validate_df(df, f"{ticker}_price")
                        data_map[ticker] = df
                        self._save_csv(df, f"{ticker}_prices.csv")
                        logger.info("Fetched %d rows for %s.", len(df), ticker)
                    else:
                        logger.warning("No price data for %s.", ticker)
                    break
                except Exception as e:
                    logger.error("Attempt %d/3 failed for %s: %s", attempt, ticker, e)
                    if attempt < 3:
                        time.sleep(3)
        return data_map

    # ------------------------------------------------------------------ B. Benchmark & Peers

    def fetch_benchmark(self):
        """
        Schema (benchmark_df): date, ticker, close, volume
        GLOBAL -> ^GSPC | VN -> ^VNINDEX
        """
        symbol = self._BENCHMARK_TICKER.get(self.market, "^GSPC")
        logger.info("Fetching benchmark %s for market=%s ...", symbol, self.market)
        try:
            raw = yf.download(symbol, start=self.start_date, end=self.end_date,
                              auto_adjust=True, progress=False)
            if raw.empty:
                return pd.DataFrame()
            df = self._flatten(raw)
            df.insert(1, "ticker", symbol)
            df = df[[c for c in ["date", "ticker", "close", "volume"] if c in df.columns]]
            df = df.sort_values("date").reset_index(drop=True)
            df = self._validate_df(df, "benchmark_df")
            self._save_csv(df, f"benchmark_{symbol.replace('^', '')}.csv")
            logger.info("Fetched %d rows for benchmark %s.", len(df), symbol)
            return df
        except Exception as e:
            logger.error("Failed benchmark %s: %s", symbol, e)
            return pd.DataFrame()

    def fetch_peers(self, peers=None):
        """
        Schema per ticker (peer_df): date, ticker, open, high, low, close, adj_close, volume
        """
        if peers is None:
            primary = self.tickers[0] if self.tickers else ""
            peers = self._PEER_MAP.get(primary, [t for t in self.tickers if t != primary])
        if not peers:
            logger.warning("No peers resolved.")
            return {}
        logger.info("Fetching peer data for: %s ...", peers)
        peer_map = {}
        for ticker in peers:
            try:
                raw = yf.download(ticker, start=self.start_date, end=self.end_date,
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    continue
                df = self._flatten(raw)
                df.insert(1, "ticker", ticker)
                df["adj_close"] = df["close"]
                cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
                df = df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)
                df = self._validate_df(df, f"{ticker}_peer")
                peer_map[ticker] = df
                self._save_csv(df, f"peer_{ticker}.csv")
                logger.info("Fetched %d rows for peer %s.", len(df), ticker)
            except Exception as e:
                logger.error("Failed peer %s: %s", ticker, e)
        return peer_map

    # ------------------------------------------------------------------ C. Fundamental Data

    def fetch_financial_statements(self):
        result = {}
        for ticker in self.tickers:
            logger.info("Fetching financial statements for %s ...", ticker)
            t = yf.Ticker(ticker)
            income  = t.quarterly_financials
            balance = t.quarterly_balance_sheet
            cash_flow = t.quarterly_cashflow  # BỔ SUNG DÒNG NÀY
            info    = t.info
            # TRUYỀN THÊM cash_flow VÀO HÀM DƯỚI
            df = self._build_fundamental(income, balance, cash_flow, info, ticker)
            df = self._validate_df(df, f"{ticker}_fundamental")
            result[ticker] = df
            self._save_csv(df, f"{ticker}_fundamental.csv")
        return result

    def _build_fundamental(self, income, balance, cash_flow, info, ticker):
        def _row(df, *keys):
            if df is None or df.empty:
                return pd.Series(dtype=float)
            for k in keys:
                if k in df.index:
                    return df.loc[k]
            return pd.Series(dtype=float)

        dates = set()
        
        if income is not None and not income.empty:
            dates.update(income.columns)
        if balance is not None and not balance.empty:
            dates.update(balance.columns)
        if cash_flow is not None and not cash_flow.empty:
            dates.update(cash_flow.columns)
        dates = sorted(list(dates), reverse=True)

        rows = []
        for date in dates:
            def _v(s, _d=date):
                if _d in s.index:
                    v = s[_d]
                    return float(v) if pd.notna(v) else None
                return None
            
            rev = _v(_row(income, "Total Revenue"))
            net_inc = _v(_row(income, "Net Income"))
            assets = _v(_row(balance, "Total Assets"))
            equity = _v(_row(balance, "Stockholders Equity", "Common Stock Equity"))
            debt = _v(_row(balance, "Total Debt"))
            

            if rev is None and net_inc is None and assets is None and equity is None:
                continue


            shares = _v(_row(balance, "Ordinary Shares Number", "Share Issued"))
            if shares is None:
                shares = _v(_row(income, "Basic Average Shares", "Diluted Average Shares"))

            roe = (net_inc / equity) if (net_inc and equity and equity != 0) else None
            roa = (net_inc / assets) if (net_inc and assets and assets != 0) else None
            margin = (net_inc / rev) if (net_inc and rev and rev != 0) else None
            debt_to_equity = (debt / equity) if (debt and equity and equity != 0) else None

            bvps = (equity / shares) if (equity and shares and shares != 0) else None

            eps = _v(_row(income, "Basic EPS", "Diluted EPS"))
            if eps is None and net_inc is not None and shares is not None and shares != 0:
                eps = net_inc / shares # Lợi nhuận ròng / Số cổ phiếu
            cfo = _v(_row(cash_flow, "Operating Cash Flow", "Total Cash From Operating Activities"))
            rows.append({
                "date":               pd.Timestamp(date).strftime("%Y-%m-%d"),
                "ticker":             ticker,
                "revenue":            rev,
                "gross_profit":       _v(_row(income, "Gross Profit")),
                "operating_profit":   _v(_row(income, "Operating Income", "Operating Revenue")),
                "net_income":         net_inc,
                "eps":                eps,  
                "total_assets":       assets,
                "total_liabilities":  _v(_row(balance, "Total Liabilities Net Minority Interest")),
                "equity":             equity,
                "total_debt":         debt,
                "cash":               _v(_row(balance, "Cash And Cash Equivalents")),
                "operating_cash_flow": cfo,
                "roe":                roe,
                "roa":                roa,
                "margin":             margin,
                "debt_to_equity":     debt_to_equity,
                "shares_outstanding": shares, 
                "bvps":               bvps    
            })

        if not rows:
            logger.warning("No fundamental rows for %s.", ticker)
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        def _i(k):
            v = info.get(k)
            return float(v) if v is not None else None

        import numpy as np
        df["pe"]                 = np.nan 
        df["pb"]                 = np.nan 
        df["dividend"]           = _i("dividendRate")

        cols = [
            "date", "ticker",
            "revenue", "gross_profit", "operating_profit", "net_income", "eps",
            "total_assets", "total_liabilities", "equity", "total_debt", "cash", "operating_cash_flow",
            "roe", "roa", "pe", "pb", "margin", "debt_to_equity",
            "shares_outstanding", "bvps", "dividend",
        ]
        return df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------ D. News & Sentiment

    def fetch_news(self, query, page_size=20):
        """
        Schema (news_df): date, ticker, headline, summary, source, sentiment, event_type
        sentiment  : positive | negative | neutral
        event_type : earnings | dividend | m&a | management_change | expansion | legal | general
        """
        if not self.news_api_key:
            logger.error("NEWS_API_KEY not set in .env")
            return pd.DataFrame()
        try:
            frames = []
            for ticker in self.tickers:
                ticker_query = self._build_ticker_news_query(ticker, query)
                r = requests.get("https://newsapi.org/v2/everything", timeout=10, params={
                    "q": ticker_query, "pageSize": page_size,
                    "apiKey": self.news_api_key, "language": "en", "sortBy": "publishedAt",
                })
                r.raise_for_status()
                articles = r.json().get("articles", [])
                relevant_articles = [
                    article for article in articles
                    if self._is_relevant_news_article(article, ticker)
                ]
                logger.info(
                    "News query for %s kept %d/%d relevant articles",
                    ticker,
                    len(relevant_articles),
                    len(articles),
                )
                ticker_df = self._build_news(relevant_articles, ticker)
                if ticker_df is not None and not ticker_df.empty:
                    frames.append(ticker_df)

            if not frames:
                logger.warning("No relevant news articles found for tickers=%s", self.tickers)
                empty_df = self._empty_news_frame()
                filename_stub = "_".join(self.tickers[:3])
                empty_df.to_csv(RAW_DATA_DIR / f"news_{filename_stub}.csv", index=False)
                logger.info("Saved empty raw data -> %s", RAW_DATA_DIR / f"news_{filename_stub}.csv")
                return empty_df

            df = pd.concat(frames, ignore_index=True)
            df = self._validate_df(df, "news_df")
            filename_stub = "_".join(self.tickers[:3])
            self._save_csv(df, f"news_{filename_stub}.csv")
            logger.info("Fetched %d relevant articles for tickers=%s.", len(df), self.tickers)
            return df
        except Exception as e:
            logger.error("NewsAPI error: %s", e)
            return pd.DataFrame()

    def _build_news(self, articles, ticker):
        rows = []
        for a in articles:
            headline = a.get("title") or ""
            hl = headline.lower()
            sentiment = "neutral"
            if any(k in hl for k in self._SENTIMENT_POSITIVE):
                sentiment = "positive"
            elif any(k in hl for k in self._SENTIMENT_NEGATIVE):
                sentiment = "negative"
            event_type = "general"
            for etype, kws in self._EVENT_KEYWORDS.items():
                if any(k in hl for k in kws):
                    event_type = etype
                    break
            rows.append({
                "date":       (a.get("publishedAt") or "")[:10],
                "ticker":     ticker,
                "headline":   headline,
                "summary":    a.get("description") or "",
                "source":     (a.get("source") or {}).get("name") or "",
                "sentiment":  sentiment,
                "event_type": event_type,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------ E. Macro Indicators

    def fetch_macro_indicators(self):
        """
        Schema (macro_df_wide): date, fed_funds_rate, us_10y_yield, us_cpi, dxy, gold_price, oil_price
        """
        import requests
        data_frames = []

        # =================================================================
        # 1. LẤY DỮ LIỆU TỪ FRED API BẰNG REQUESTS (Không dùng pandas_datareader nữa)
        # =================================================================
        fred_indicators = {
            "fed_funds_rate": "FEDFUNDS",
            "us_10y_yield": "DGS10",  
            "us_cpi": "CPIAUCSL"
        }
        
        for name, series_id in fred_indicators.items():
            logger.info("Fetching macro from FRED API: %s (%s) ...", name, series_id)
            try:
                # Đường dẫn (Endpoint) chuẩn xác của FRED để lấy chuỗi dữ liệu
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": getattr(self, 'fred_api_key', None),
                    "file_type": "json",
                    "observation_start": self.start_date, # Bắt buộc phải có để không tải từ năm 1947
                    "observation_end": self.end_date
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                observations = data.get("observations", [])
                if observations:
                    # Lọc bỏ giá trị "." (những ngày FRED không có dữ liệu)
                    valid_obs = [obs for obs in observations if obs.get("value") != "."]
                    
                    df = pd.DataFrame(valid_obs)[["date", "value"]]
                    df["value"] = df["value"].astype(float)
                    df = df.rename(columns={"value": name})
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    data_frames.append(df)
                else:
                    logger.warning("No data found for FRED series %s", series_id)
            except Exception as e:
                logger.error("Failed FRED macro %s: %s", name, e)

        # =================================================================
        # 2. LẤY DỮ LIỆU TỪ YAHOO FINANCE (DXY, Vàng, Dầu)
        # =================================================================
        yf_indicators = {
            "dxy": "DX-Y.NYB",       
            "gold_price": "GC=F",    
            "oil_price": "CL=F"      
        }

        for name, symbol in yf_indicators.items():
            logger.info("Fetching macro from yfinance: %s (%s) ...", name, symbol)
            try:
                raw = yf.download(symbol, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
                if not raw.empty:
                    df = self._flatten(raw)
                    df = df[["date", "close"]].rename(columns={"close": name})
                    df["date"] = pd.to_datetime(df["date"]) 
                    df.set_index("date", inplace=True)
                    data_frames.append(df)
            except Exception as e:
                logger.error("Failed yfinance macro %s: %s", name, e)

        # =================================================================
        # 3. GỘP BẢNG VÀ LÀM SẠCH DỮ LIỆU
        # =================================================================
        if not data_frames:
            logger.warning("No macro data fetched.")
            return pd.DataFrame(columns=["date"])

        # Gộp tất cả các bảng
        macro_df = pd.concat(data_frames, axis=1).reset_index()
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        macro_df = macro_df.sort_values("date").reset_index(drop=True)

        # Trải phẳng dữ liệu theo ngày (Forward Fill)
        value_cols = [c for c in macro_df.columns if c != "date"]
        macro_df[value_cols] = macro_df[value_cols].ffill()

        # Loại bỏ rác đầu mút
        macro_df = macro_df.dropna(how='all', subset=value_cols).reset_index(drop=True)

        # Định dạng lại chuỗi ngày tháng cho chuẩn Form
        macro_df["date"] = macro_df["date"].dt.strftime("%Y-%m-%d")

        df = self._validate_df(macro_df, "macro_df_wide")
        self._save_csv(df, "macro_indicators.csv")
        logger.info("Macro fetched: %d rows (FRED Direct API + YFinance).", len(df))
        
        return df
    
    def fetch_industry_data(self, peers=None):
        """
        Schema (industry_df): date, industry_roe, industry_margin, industry_pe, industry_pb
        """
        if peers is None:
            primary = self.tickers[0] if self.tickers else ""
            peers = self._PEER_MAP.get(primary, [t for t in self.tickers if t != primary])

        if not peers:
            logger.warning("No peers for industry data - returning empty.")
            return pd.DataFrame(columns=["date", "industry_roe", "industry_margin"])

        logger.info("Computing historical industry data from peers: %s ...", peers)
        all_peer_data = []

        for ticker in peers:
            try:
                t = yf.Ticker(ticker)
                inc = t.quarterly_financials
                bal = t.quarterly_balance_sheet

                dates = set()
                if inc is not None and not inc.empty: dates.update(inc.columns)
                if bal is not None and not bal.empty: dates.update(bal.columns)
                
                for d in dates:
                    def _v(df, key1, key2=None):
                        if df is not None and d in df.columns:
                            if key1 in df.index: return float(df.loc[key1, d]) if pd.notna(df.loc[key1, d]) else None
                            if key2 and key2 in df.index: return float(df.loc[key2, d]) if pd.notna(df.loc[key2, d]) else None
                        return None
                    
                    net_inc = _v(inc, "Net Income")
                    rev = _v(inc, "Total Revenue")
                    eq = _v(bal, "Stockholders Equity", "Common Stock Equity")
                        
                    roe = (net_inc / eq) if (net_inc and eq and eq != 0) else None
                    margin = (net_inc / rev) if (net_inc and rev and rev != 0) else None
                    
                    if roe is not None or margin is not None:
                        all_peer_data.append({
                            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "roe": roe,
                            "margin": margin
                        })
            except Exception as e:
                logger.error("Failed historical data for peer %s: %s", ticker, e)

        if not all_peer_data:
            logger.warning("No historical peer data found.")
            return pd.DataFrame()

        df_peers = pd.DataFrame(all_peer_data)
        industry_df = df_peers.groupby('date').mean().reset_index()
        
        industry_df = industry_df.rename(columns={
            "roe": "industry_roe",
            "margin": "industry_margin"
        })

        import numpy as np
        industry_df["industry_pe"] = np.nan
        industry_df["industry_pb"] = np.nan
        
        industry_df = industry_df.sort_values("date").reset_index(drop=True)
        df = self._validate_df(industry_df, "industry_df")
        self._save_csv(df, "industry_data.csv")
        logger.info("Historical Industry data compiled: %d quarters.", len(df))
        
        return df

    # ------------------------------------------------------------------ Intraday (Section 5)

    def fetch_intraday(self, interval="5m", period="5d"):
        """
        Schema (intraday_df): timestamp, ticker, price, volume
        interval : "1m" | "5m" | "15m" | "30m" | "60m" | "90m"
        period   : "1d" | "5d" | "1mo"
        """
        data_map = {}
        for ticker in self.tickers:
            logger.info("Fetching intraday %s for %s (period=%s)...", interval, ticker, period)
            try:
                raw = yf.download(ticker, period=period, interval=interval,
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    logger.warning("No intraday data for %s.", ticker)
                    continue
                df = self._flatten(raw)
                df = df.rename(columns={"date": "timestamp"})
                df = pd.DataFrame({
                    "timestamp": df["timestamp"],
                    "ticker":    ticker,
                    "price":     df["close"],
                    "volume":    df["volume"],
                }).sort_values("timestamp").reset_index(drop=True)
                df = self._validate_df(df, f"{ticker}_intraday", date_col="timestamp")
                data_map[ticker] = df
                self._save_csv(df, f"{ticker}_intraday_{interval}.csv")
                logger.info("Fetched %d intraday bars for %s.", len(df), ticker)
            except Exception as e:
                logger.error("Failed intraday %s: %s", ticker, e)
        return data_map
