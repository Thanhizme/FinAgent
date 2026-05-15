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

try:
    from vnstock.api.quote import Quote as VnQuote
except Exception:  # pragma: no cover - optional dependency
    VnQuote = None

try:
    from vnstock import Finance as VnFinance
except Exception:  # pragma: no cover - optional dependency
    VnFinance = None

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

    _BENCHMARK_TICKER = {
        "GLOBAL": "^GSPC",
        "VN":     "^VNINDEX",
    }

    _VN_BENCHMARK_PROXY_COMPONENTS = [
        "VCB", "BID", "VHM", "VIC", "VNM", "MSN", "HPG", "FPT", "GAS", "MWG"
    ]

    _PEER_MAP = {
        # US peer pairs from the provided sector/cap table.
        # Each ticker maps to exactly one same-sector, same-cap peer.

        # Information Technology
        # Large cap
        "AAPL":  ["MSFT"],
        "MSFT":  ["AAPL"],
        # Mid cap
        "MANH":  ["TER"],
        "TER":   ["MANH"],
        # Small cap
        "IDCC":  ["KLIC"],
        "KLIC":  ["IDCC"],

        # Financials
        # Large cap
        "JPM":   ["V"],
        "V":     ["JPM"],
        # Mid cap
        "SF":    ["JEF"],
        "JEF":   ["SF"],
        # Small cap 
        "DFIN":  ["VBTX"],
        "VBTX":  ["DFIN"],

        # Health Care
        # Large cap
        "AMGN":  ["ELV"],
        "ELV":   ["AMGN"],
        # Mid cap
        "HALO":  ["EHC"],
        "EHC":   ["HALO"],
        # Small cap
        "HIMS":  ["NSTG"],
        "NSTG":  ["HIMS"],

        # Consumer Discretionary
        # Large cap
        "AMZN":  ["TSLA"],
        "TSLA":  ["AMZN"],
        # Mid cap
        "DECK":  ["CROX"],
        "CROX":  ["DECK"],
        # Small cap
        "BOOT":  ["SONO"],
        "SONO":  ["BOOT"],

        # Consumer Staples
        # Large cap
        "MDLZ":  ["KMB"],
        "KMB":   ["MDLZ"],
        # Mid cap
        "CASY":  ["CELH"],
        "CELH":  ["CASY"],
        # Small cap
        "CALM":  ["JJSF"],
        "JJSF":  ["CALM"],

        # Industrials
        # Large cap 
        "LMT":   ["GE"],
        "GE":    ["LMT"],
        # Mid cap
        "DE":    ["UPS"],
        "UPS":   ["DE"],
        # Small cap
        "BYRN":  ["MLKN"],
        "MLKN":  ["BYRN"],

        # Energy
        # Large cap
        "XOM":   ["CVX"],
        "CVX":   ["XOM"],
        # Mid cap
        "OVV":   ["APA"],
        "APA":   ["OVV"],
        # Small cap
        "REPX":  ["PARR"],
        "PARR":  ["REPX"],

        # Utilities
        # Large cap
        "D":     ["NEE"],
        "NEE":   ["D"],
        # Mid cap
        "VST":   ["NRG"],
        "NRG":   ["VST"],
        # Small cap
        "AWR":   ["AVA"],
        "AVA":   ["AWR"],

        # Real Estate
        # Large cap
        "PLD":   ["EQIX"],
        "EQIX":  ["PLD"],
        # Mid cap
        "REXR":  ["OHI"],
        "OHI":   ["REXR"],
        # Small cap
        "LGIH":  ["UTL"],
        "UTL":   ["LGIH"],

        # Materials
        # Large cap
        "LIN":   ["SHW"],
        "SHW":   ["LIN"],
        # Mid cap
        "RS":    ["STLD"],
        "STLD":  ["RS"],
        # Small cap
        "MLI":   ["IOSP"],
        "IOSP":  ["MLI"],

        # Communication Services
        # Large cap
        "GOOGL": ["META"],
        "META":  ["GOOGL"],
        # Mid cap
        "PINS":  ["TTWO"],
        "TTWO":  ["PINS"],
        # Small cap
        "CNK":   ["YELP"],
        "YELP":  ["CNK"],
    # VN peer pairs from the provided sector/cap table.
    # Each ticker maps to one same-sector, same-cap peer.

    # 1) Ngan hang
    "VCB": ["BID"],
    "BID": ["VCB"],
    "LPB": ["MSB"],
    "MSB": ["LPB"],
    "BVB": ["ABB"],
    "ABB": ["BVB"],

    # 2) Bat dong san
    "VHM": ["VIC"],
    "VIC": ["VHM"],
    "KDH": ["NLG"],
    "NLG": ["KDH"],
    "DRH": ["HQC"],
    "HQC": ["DRH"],

    # 3) Thuc pham & Do uong
    "VNM": ["MSN"],
    "MSN": ["VNM"],
    "PAN": ["VHC"],
    "VHC": ["PAN"],
    "ANV": ["IDI"],
    "IDI": ["ANV"],

    # 4) Dich vu tai chinh
    "SSI": ["VND"],
    "VND": ["SSI"],
    "VCI": ["HCM"],
    "HCM": ["VCI"],
    "BSI": ["FTS"],
    "FTS": ["BSI"],

    # 5) Tai nguyen co ban (Thep)
    "HPG": ["GVR"],
    "GVR": ["HPG"],
    "HSG": ["NKG"],
    "NKG": ["HSG"],
    "TVN": ["VGS"],
    "VGS": ["TVN"],

    # 6) Dau khi
    "GAS": ["PLX"],
    "PLX": ["GAS"],
    "PVS": ["PVD"],
    "PVD": ["PVS"],
    "PVC": ["PVB"],
    "PVB": ["PVC"],

    # 7) Ban le
    "MWG": ["PNJ"],
    "PNJ": ["MWG"],
    "FRT": ["DGW"],
    "DGW": ["FRT"],
    "PET": ["ASG"],
    "ASG": ["PET"],

    # 8) Hoa chat
    "DGC": ["DPM"],
    "DPM": ["DGC"],
    "DCM": ["CSV"],
    "CSV": ["DCM"],
    "BFC": ["LAS"],
    "LAS": ["BFC"],

    # 9) Xay dung & Vat lieu
    "VCG": ["REE"],
    "REE": ["VCG"],
    "CTD": ["HHV"],
    "HHV": ["CTD"],
    "LCG": ["C4G"],
    "C4G": ["LCG"],

    # 10) Tien ich (Dien, Nuoc)
    "POW": ["PGV"],
    "PGV": ["POW"],
    "GEG": ["HDG"],
    "HDG": ["GEG"],
    "TDM": ["BWE"],
    "BWE": ["TDM"],

    # 11) Cong nghe thong tin
    "FPT": ["VGI"],
    "VGI": ["FPT"],
    "CMG": ["FOX"],
    "FOX": ["CMG"],
    "ELC": ["ITD"],
    "ITD": ["ELC"],

    # 12) Van tai & Kho bai
    "VJC": ["GMD"],
    "GMD": ["VJC"],
    "HAH": ["PVT"],
    "PVT": ["HAH"],
    "VIP": ["VTO"],
    "VTO": ["VIP"],

    # 13) Y te & Duoc pham
    "DHG": ["DMC"],
    "DMC": ["DHG"],
    "TRA": ["IMP"],
    "IMP": ["TRA"],
    "DBD": ["OPC"],
    "OPC": ["DBD"],

    # 14) Bao hiem
    "BVH": ["PVI"],
    "PVI": ["BVH"],
    "MIG": ["BMI"],
    "BMI": ["MIG"],
    "BIC": ["PGI"],
    "PGI": ["BIC"],

    # 15) Hang ca nhan & Gia dung
    "MSH": ["PNJ"],
    "TNG": ["GIL"],
    "GIL": ["TNG"],
    "TCM": ["ADS"],
    "ADS": ["TCM"],

    # 16) Du lich & Giai tri
    "VNG": ["HVN"],
    "HVN": ["VNG"],
    "OCH": ["NVT"],
    "NVT": ["OCH"],
    "DSN": ["SKG"],
    "SKG": ["DSN"],

    # 17) Truyen thong
    "YEG": ["ABC"],
    "ABC": ["YEG"],
    "VNB": ["TTN"],
    "TTN": ["VNB"],

    # 18) O to & Phu tung
    "HAX": ["DRC"],
    "DRC": ["HAX"],
    "CSM": ["TMT"],
    "TMT": ["CSM"],
    "SVC": ["HTL"],
    "HTL": ["SVC"],

    # 19) Dich vu luu tru & An uong
    "DAH": ["RIC"],
    "RIC": ["DAH"],
    "CTC": ["DXL"],
    "DXL": ["CTC"],
    }

    _VN_TICKERS = {
        "VCB", "BID", "LPB", "MSB", "BVB", "ABB",
        "VHM", "VIC", "KDH", "NLG", "DRH", "HQC",
        "VNM", "MSN", "PAN", "VHC", "ANV", "IDI",
        "SSI", "VND", "VCI", "HCM", "BSI", "FTS",
        "HPG", "GVR", "HSG", "NKG", "TVN", "VGS",
        "GAS", "PLX", "PVS", "PVD", "PVC", "PVB",
        "MWG", "PNJ", "FRT", "DGW", "PET", "ASG",
        "DGC", "DPM", "DCM", "CSV", "BFC", "LAS",
        "VCG", "REE", "CTD", "HHV", "LCG", "C4G",
        "POW", "PGV", "GEG", "HDG", "TDM", "BWE",
        "FPT", "VGI", "CMG", "FOX", "ELC", "ITD",
        "VJC", "GMD", "HAH", "PVT", "VIP", "VTO",
        "DHG", "DMC", "TRA", "IMP", "DBD", "OPC",
        "BVH", "PVI", "MIG", "BMI", "BIC", "PGI",
        "MSH", "TNG", "GIL", "TCM", "ADS",
        "VNG", "HVN", "OCH", "NVT", "DSN", "SKG",
        "YEG", "ABC", "VNB", "TTN",
        "HAX", "DRC", "CSM", "TMT", "SVC", "HTL",
        "DAH", "RIC", "CTC", "DXL",
    }

    _MACRO_TICKERS = {
        "gold_price":    "GC=F",
        "oil_price":     "CL=F",
        "usd_vnd":       "USDVND=X",
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

    def __init__(self, tickers, start_date, end_date, market="GLOBAL"):
        self.tickers      = tickers
        self.start_date   = start_date
        self.end_date     = end_date
        self.market       = market.upper()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        logger.info("DataCollector initialised | tickers=%s | market=%s | %s to %s",
                    self.tickers, self.market, self.start_date, self.end_date)

    def _yf_symbol(self, ticker: str) -> str:
        """Normalize local ticker to provider symbol for yfinance."""
        t = (ticker or "").strip().upper()
        if not t:
            return t
        if t in self._VN_TICKERS and not t.endswith(".VN"):
            return f"{t}.VN"
        return t

    def _benchmark_candidates(self) -> list[str]:
        """Return benchmark symbol candidates by market.

        VNINDEX is not consistently available on yfinance, so keep fallbacks.
        """
        if self.market == "VN":
            return ["^VNINDEX", "VNINDEX", "VNINDEX.VN", "^VNI"]
        return ["^GSPC"]

    def _build_vnindex_proxy(self) -> pd.DataFrame:
        """Build a synthetic VNINDEX series from liquid VN large-cap constituents."""
        close_series = []
        used_components = []

        for ticker in self._VN_BENCHMARK_PROXY_COMPONENTS:
            symbol = self._yf_symbol(ticker)
            try:
                raw = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    progress=False,
                )
                if raw is None or raw.empty:
                    continue

                flat = self._flatten(raw)
                if "date" not in flat.columns or "close" not in flat.columns:
                    continue

                s = (
                    flat[["date", "close"]]
                    .dropna(subset=["close"])
                    .drop_duplicates(subset=["date"])
                    .set_index("date")["close"]
                    .sort_index()
                    .rename(ticker)
                )
                if s.empty:
                    continue

                close_series.append(s)
                used_components.append(ticker)
            except Exception as e:
                logger.warning("Proxy component fetch failed for %s (%s): %s", ticker, symbol, e)

        if not close_series:
            return pd.DataFrame()

        closes = pd.concat(close_series, axis=1).sort_index().ffill()
        returns = closes.pct_change().replace([np.inf, -np.inf], np.nan)
        proxy_returns = returns.mean(axis=1, skipna=True)
        proxy_returns = proxy_returns.fillna(0.0)

        proxy_level = (1.0 + proxy_returns).cumprod() * 1000.0
        benchmark_df = proxy_level.rename("close").to_frame().reset_index()
        benchmark_df = benchmark_df.rename(columns={benchmark_df.columns[0]: "date"})
        benchmark_df.insert(1, "ticker", "VNINDEX")
        benchmark_df["volume"] = np.nan
        benchmark_df = self._validate_df(benchmark_df, "benchmark_vnindex_proxy")

        if benchmark_df.empty:
            return benchmark_df

        self._save_csv(benchmark_df, "benchmark_VNINDEX_PROXY.csv")
        logger.warning(
            "Using VNINDEX proxy benchmark built from %d components: %s",
            len(used_components),
            ", ".join(used_components),
        )
        return benchmark_df

    def _fetch_vnindex_official(self) -> pd.DataFrame:
        """Fetch VNINDEX EOD from vnstock (VCI source) as the primary VN benchmark."""
        if VnQuote is None:
            logger.warning("vnstock not available; skip official VNINDEX fetch.")
            return pd.DataFrame()

        try:
            quote = VnQuote(symbol="VNINDEX", source="VCI")
            raw = quote.history(start=self.start_date, end=self.end_date, interval="1D")
            if raw is None or raw.empty:
                return pd.DataFrame()

            df = raw.copy()
            rename_map = {
                "time": "date",
                "tradingDate": "date",
            }
            df = df.rename(columns=rename_map)
            if "date" not in df.columns:
                return pd.DataFrame()

            if "volume" not in df.columns:
                df["volume"] = np.nan

            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in keep_cols if c in df.columns]].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            start_ts = pd.to_datetime(self.start_date, errors="coerce")
            end_ts = pd.to_datetime(self.end_date, errors="coerce")
            if pd.notna(start_ts):
                df = df[df["date"] >= start_ts]
            if pd.notna(end_ts):
                df = df[df["date"] <= end_ts]
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            df.insert(1, "ticker", "VNINDEX")
            df = self._validate_df(df, "benchmark_vnindex_official")
            if df.empty:
                return df

            self._save_csv(df, "benchmark_VNINDEX_OFFICIAL.csv")
            logger.info("Fetched %d rows for official VNINDEX benchmark (vnstock:VCI).", len(df))
            return df
        except Exception as e:
            logger.error("Official VNINDEX fetch failed (vnstock:VCI): %s", e)
            return pd.DataFrame()

    def _fetch_vn_price_official(self, ticker: str) -> pd.DataFrame:
        """Fetch VN ticker EOD from vnstock (VCI source) as a price fallback."""
        if VnQuote is None:
            logger.warning("vnstock not available; skip official price fetch for %s.", ticker)
            return pd.DataFrame()

        try:
            quote = VnQuote(symbol=ticker, source="VCI")
            raw = quote.history(start=self.start_date, end=self.end_date, interval="1D")
            if raw is None or raw.empty:
                return pd.DataFrame()

            df = raw.copy()
            rename_map = {"time": "date", "tradingDate": "date"}
            df = df.rename(columns=rename_map)
            if "date" not in df.columns:
                return pd.DataFrame()

            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in keep_cols if c in df.columns]].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            start_ts = pd.to_datetime(self.start_date, errors="coerce")
            end_ts = pd.to_datetime(self.end_date, errors="coerce")
            if pd.notna(start_ts):
                df = df[df["date"] >= start_ts]
            if pd.notna(end_ts):
                df = df[df["date"] <= end_ts]

            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            df.insert(1, "ticker", ticker)
            df["adj_close"] = df["close"]
            cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
            df = df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)
            df = self._validate_df(df, f"{ticker}_price_vnstock")
            if df.empty:
                return df

            self._save_csv(df, f"{ticker}_prices.csv")
            logger.info("Fetched %d rows for %s from vnstock price fallback.", len(df), ticker)
            return df
        except Exception as e:
            logger.error("vnstock price fetch failed for %s: %s", ticker, e)
            return pd.DataFrame()

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
            return alias_match or (ticker_match and finance_context)

        if strong_terms:
            return alias_match or ticker_match
        return ticker_match

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

    def fetch_stock_prices(self):
        """
        Schema (price_df): date, ticker, open, high, low, close, adj_close, volume
        """
        data_map = {}
        for ticker in self.tickers:
            symbol = self._yf_symbol(ticker)
            logger.info("Fetching price data for %s (symbol=%s) ...", ticker, symbol)
            for attempt in range(1, 4):
                try:
                    raw = yf.download(symbol, start=self.start_date, end=self.end_date,
                                      auto_adjust=True, progress=False)
                    if raw is not None and not raw.empty:
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
                        if ticker in self._VN_TICKERS:
                            logger.warning("No yfinance price data for %s. Trying vnstock fallback ...", ticker)
                            vn_df = self._fetch_vn_price_official(ticker)
                            if vn_df is not None and not vn_df.empty:
                                data_map[ticker] = vn_df
                            else:
                                logger.warning("No price data for %s after vnstock fallback.", ticker)
                        else:
                            logger.warning("No price data for %s.", ticker)
                    break
                except Exception as e:
                    logger.error("Attempt %d/3 failed for %s: %s", attempt, ticker, e)
                    if attempt < 3:
                        time.sleep(3)
                    elif ticker in self._VN_TICKERS:
                        logger.warning("yfinance failed for %s after retries. Trying vnstock fallback ...", ticker)
                        vn_df = self._fetch_vn_price_official(ticker)
                        if vn_df is not None and not vn_df.empty:
                            data_map[ticker] = vn_df
        return data_map

    def fetch_benchmark(self):
        """
        Schema (benchmark_df): date, ticker, open, high, low, close, volume
        GLOBAL -> ^GSPC | VN -> vnstock official VNINDEX, then yfinance candidates, then proxy
        """
        if self.market == "VN":
            official_df = self._fetch_vnindex_official()
            if official_df is not None and not official_df.empty:
                return official_df

        for symbol in self._benchmark_candidates():
            logger.info("Fetching benchmark %s for market=%s ...", symbol, self.market)
            try:
                raw = yf.download(symbol, start=self.start_date, end=self.end_date,
                                  auto_adjust=True, progress=False)
                if raw is None or raw.empty:
                    continue
                df = self._flatten(raw)
                df.insert(1, "ticker", symbol)
                df = df[[c for c in ["date", "ticker", "open", "high", "low", "close", "volume"] if c in df.columns]]
                df = df.sort_values("date").reset_index(drop=True)
                df = self._validate_df(df, "benchmark_df")
                self._save_csv(df, f"benchmark_{symbol.replace('^', '')}.csv")
                logger.info("Fetched %d rows for benchmark %s.", len(df), symbol)
                return df
            except Exception as e:
                logger.error("Failed benchmark %s: %s", symbol, e)

        if self.market == "VN":
            proxy_df = self._build_vnindex_proxy()
            if proxy_df is not None and not proxy_df.empty:
                logger.info("Fetched %d rows for VNINDEX proxy benchmark.", len(proxy_df))
                return proxy_df

        logger.warning("No benchmark data available for market=%s from configured candidates.", self.market)
        return pd.DataFrame()

    def fetch_peers(self, peers=None):
        """
        Schema per ticker (peer_df): date, ticker, open, high, low, close, adj_close, volume
        """
        if peers is None:
            primary = self.tickers[0] if self.tickers else ""
            peers = self._resolve_dynamic_peers(primary)
        if not peers:
            logger.warning("No peers resolved.")
            return {}
        logger.info("Fetching peer data for: %s ...", peers)
        peer_map = {}
        for ticker in peers:
            try:
                symbol = self._yf_symbol(ticker)
                raw = yf.download(symbol, start=self.start_date, end=self.end_date,
                                  auto_adjust=True, progress=False)
                if raw is None or raw.empty:
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

    def _market_cap_bucket(self, market_cap):
        if market_cap is None or pd.isna(market_cap):
            return None
        if market_cap >= 10_000_000_000:
            return "large"
        if market_cap >= 2_000_000_000:
            return "mid"
        return "small"

    def _resolve_dynamic_peers(self, primary: str) -> list[str]:
        """Resolve peers dynamically by sector and market-cap bucket with static fallback."""
        primary = (primary or "").upper()
        if not primary:
            return []

        fallback = self._PEER_MAP.get(primary, [t for t in self.tickers if t != primary])

        if primary in self._VN_TICKERS:
            logger.debug("VN ticker %s: using static _PEER_MAP (dynamic lookup skipped).", primary)
            return fallback

        try:
            p_info = yf.Ticker(primary).info or {}
            p_sector = p_info.get("sector")
            p_cap = p_info.get("marketCap")
            p_bucket = self._market_cap_bucket(p_cap)
            if not p_sector:
                return fallback

            candidates = set(self._PEER_MAP.keys())
            for vals in self._PEER_MAP.values():
                candidates.update(vals)
            candidates.update(t.upper() for t in self.tickers)
            candidates.discard(primary)

            scored = []
            for candidate in sorted(candidates):
                try:
                    info = yf.Ticker(candidate).info or {}
                    c_sector = info.get("sector")
                    c_cap = info.get("marketCap")
                    c_bucket = self._market_cap_bucket(c_cap)
                    if c_sector == p_sector and c_bucket == p_bucket and c_cap is not None and p_cap is not None:
                        scored.append((abs(float(c_cap) - float(p_cap)), candidate))
                except Exception:
                    continue

            dynamic = [ticker for _, ticker in sorted(scored, key=lambda x: x[0])][:1]
            if not dynamic:
                for ticker in fallback:
                    if ticker not in dynamic:
                        dynamic.append(ticker)
                    if len(dynamic) >= 1:
                        break
            return dynamic
        except Exception as e:
            logger.warning("Dynamic peer resolution failed for %s: %s", primary, e)
            return fallback

    @staticmethod
    def _quarter_label_to_date(label: str) -> str | None:
        text = str(label or "").strip().upper()
        m = re.match(r"^(\d{4})-Q([1-4])$", text)
        if not m:
            return None
        year = int(m.group(1))
        quarter = int(m.group(2))
        month = quarter * 3
        period = pd.Period(f"{year}-{month:02d}", freq="M")
        return period.end_time.strftime("%Y-%m-%d")

    @staticmethod
    def _statement_date_columns(df: pd.DataFrame) -> list[str]:
        if df is None or df.empty:
            return []
        cols: list[str] = []
        for c in df.columns:
            if re.match(r"^\d{4}-Q[1-4]$", str(c).strip().upper()):
                cols.append(str(c))
        return cols

    @staticmethod
    def _pick_statement_row(df: pd.DataFrame, keywords: list[str]) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        source_col = "item_en" if "item_en" in df.columns else ("item" if "item" in df.columns else None)
        if source_col is None:
            return pd.Series(dtype=float)

        raw = df[source_col].fillna("").astype(str).str.lower()
        for kw in keywords:
            mask = raw.str.contains(kw.lower(), regex=False)
            if mask.any():
                return df.loc[mask].iloc[0]
        return pd.Series(dtype=float)

    def _row_to_timeseries(self, row: pd.Series, date_cols: list[str]) -> dict[str, float | None]:
        if row is None or row.empty:
            return {}
        out: dict[str, float | None] = {}
        for c in date_cols:
            date_key = self._quarter_label_to_date(c)
            if not date_key:
                continue
            val = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            out[date_key] = float(val) if pd.notna(val) else None
        return out

    def _build_fundamental_from_vnstock(self, ticker: str, info: dict | None = None) -> pd.DataFrame:
        if VnFinance is None:
            return pd.DataFrame()

        try:
            fin = VnFinance(source="VCI", symbol=ticker, period="quarter", get_all=False, show_log=False)
            income = fin.income_statement()
            balance = fin.balance_sheet()
            cash_flow = fin.cash_flow()
        except Exception as e:
            logger.warning("vnstock fundamental fetch failed for %s: %s", ticker, e)
            return pd.DataFrame()

        date_cols = sorted(
            set(self._statement_date_columns(income))
            | set(self._statement_date_columns(balance))
            | set(self._statement_date_columns(cash_flow))
        )
        if not date_cols:
            logger.warning("vnstock returned no quarterly columns for %s", ticker)
            return pd.DataFrame()

        mapped = {
            "revenue": self._row_to_timeseries(self._pick_statement_row(income, ["net sales", "sales"]), date_cols),
            "gross_profit": self._row_to_timeseries(self._pick_statement_row(income, ["gross profit"]), date_cols),
            "operating_profit": self._row_to_timeseries(self._pick_statement_row(income, ["operating profit"]), date_cols),
            "net_income": self._row_to_timeseries(self._pick_statement_row(income, ["net profit/(loss) after tax", "net profit"]), date_cols),
            "interest_expense": self._row_to_timeseries(self._pick_statement_row(income, ["interest expenses", "interest expense"]), date_cols),
            "pre_tax_income": self._row_to_timeseries(self._pick_statement_row(income, ["before tax"]), date_cols),
            "income_tax_expense": self._row_to_timeseries(self._pick_statement_row(income, ["income tax expenses", "corporate income tax"]), date_cols),
            "cogs": self._row_to_timeseries(self._pick_statement_row(income, ["cost of sales"]), date_cols),
            "total_assets": self._row_to_timeseries(self._pick_statement_row(balance, ["total assets"]), date_cols),
            "equity": self._row_to_timeseries(self._pick_statement_row(balance, ["owner's equity", "owners equity"]), date_cols),
            "total_debt": self._row_to_timeseries(self._pick_statement_row(balance, ["total debt", "borrowings"]), date_cols),
            "cash": self._row_to_timeseries(self._pick_statement_row(balance, ["cash and cash equivalents", "cash equivalents", "cash"]), date_cols),
            "current_assets": self._row_to_timeseries(self._pick_statement_row(balance, ["current assets"]), date_cols),
            "current_liabilities": self._row_to_timeseries(self._pick_statement_row(balance, ["current liabilities"]), date_cols),
            "accounts_receivable": self._row_to_timeseries(self._pick_statement_row(balance, ["accounts receivable", "trade accounts receivable", "receivables"]), date_cols),
            "inventory": self._row_to_timeseries(self._pick_statement_row(balance, ["inventories", "inventory"]), date_cols),
            "accounts_payable": self._row_to_timeseries(self._pick_statement_row(balance, ["trade accounts payable", "payables"]), date_cols),
            "retained_earnings": self._row_to_timeseries(self._pick_statement_row(balance, ["undistributed earnings", "retained earnings"]), date_cols),
            "operating_cash_flow": self._row_to_timeseries(self._pick_statement_row(cash_flow, ["operating activities"]), date_cols),
            "capex": self._row_to_timeseries(self._pick_statement_row(cash_flow, ["purchases of fixed assets", "fixed assets and other long term assets"]), date_cols),
        }

        all_dates = sorted({d for values in mapped.values() for d in values.keys()})
        rows = []
        for date in all_dates:
            revenue = mapped["revenue"].get(date)
            net_income = mapped["net_income"].get(date)
            total_assets = mapped["total_assets"].get(date)
            equity = mapped["equity"].get(date)
            total_debt = mapped["total_debt"].get(date)

            if revenue is None and net_income is None and total_assets is None and equity is None:
                continue

            roe = (net_income / equity) if (net_income is not None and equity not in (None, 0)) else None
            roa = (net_income / total_assets) if (net_income is not None and total_assets not in (None, 0)) else None
            margin = (net_income / revenue) if (net_income is not None and revenue not in (None, 0)) else None
            debt_to_equity = (total_debt / equity) if (total_debt is not None and equity not in (None, 0)) else None

            rows.append({
                "date": date,
                "ticker": ticker,
                "revenue": revenue,
                "gross_profit": mapped["gross_profit"].get(date),
                "operating_profit": mapped["operating_profit"].get(date),
                "net_income": net_income,
                "total_assets": total_assets,
                "equity": equity,
                "total_debt": total_debt,
                "cash": mapped["cash"].get(date),
                "operating_cash_flow": mapped["operating_cash_flow"].get(date),
                "capex": mapped["capex"].get(date),
                "accounts_receivable": mapped["accounts_receivable"].get(date),
                "inventory": mapped["inventory"].get(date),
                "accounts_payable": mapped["accounts_payable"].get(date),
                "cogs": mapped["cogs"].get(date),
                "income_tax_expense": mapped["income_tax_expense"].get(date),
                "pre_tax_income": mapped["pre_tax_income"].get(date),
                "current_assets": mapped["current_assets"].get(date),
                "current_liabilities": mapped["current_liabilities"].get(date),
                "retained_earnings": mapped["retained_earnings"].get(date),
                "interest_expense": mapped["interest_expense"].get(date),
                "ebitda": None,
                "roe": roe,
                "roa": roa,
                "margin": margin,
                "debt_to_equity": debt_to_equity,
                "shares_outstanding": None,
                "bvps": None,
                "eps": None,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

        info = info or {}
        def _info_float(key: str) -> float | None:
            val = info.get(key)
            return float(val) if val is not None else None

        df["pe"] = _info_float("trailingPE")
        df["pb"] = _info_float("priceToBook")
        df["dividend"] = _info_float("dividendRate")
        df["market_cap"] = _info_float("marketCap")

        cols = [
            "date", "ticker",
            "revenue", "gross_profit", "operating_profit", "net_income", "eps",
            "total_assets", "total_liabilities", "equity", "total_debt", "cash", "operating_cash_flow",
            "capex", "accounts_receivable", "inventory", "accounts_payable", "cogs",
            "short_term_investments", "income_tax_expense", "pre_tax_income",
            "current_assets", "current_liabilities", "retained_earnings", "interest_expense", "ebitda", "market_cap",
            "roe", "roa", "pe", "pb", "margin", "debt_to_equity",
            "shares_outstanding", "bvps", "dividend",
        ]
        return df[[c for c in cols if c in df.columns]]

    @staticmethod
    def _merge_fundamental_frames(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
        if primary is None or primary.empty:
            return secondary.copy() if secondary is not None else pd.DataFrame()
        if secondary is None or secondary.empty:
            return primary.copy()

        p = primary.copy()
        s = secondary.copy()
        p["date"] = pd.to_datetime(p["date"], errors="coerce")
        s["date"] = pd.to_datetime(s["date"], errors="coerce")

        key_cols = ["date", "ticker"]
        all_cols = sorted(set(p.columns) | set(s.columns))
        merged = p[key_cols].drop_duplicates().merge(s[key_cols].drop_duplicates(), on=key_cols, how="outer")
        for col in all_cols:
            if col in key_cols:
                continue
            p_col = p[[*key_cols, col]] if col in p.columns else pd.DataFrame(columns=[*key_cols, col])
            s_col = s[[*key_cols, col]] if col in s.columns else pd.DataFrame(columns=[*key_cols, col])
            tmp = merged.merge(p_col, on=key_cols, how="left", suffixes=("", "_p"))
            tmp = tmp.merge(s_col, on=key_cols, how="left", suffixes=("_p", "_s"))
            val_p = f"{col}_p" if f"{col}_p" in tmp.columns else col
            val_s = f"{col}_s" if f"{col}_s" in tmp.columns else col
            merged[col] = tmp[val_p].combine_first(tmp[val_s])

        merged = merged.sort_values("date").reset_index(drop=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return merged

    def fetch_financial_statements(self):
        result = {}
        for ticker in self.tickers:
            symbol = self._yf_symbol(ticker)
            logger.info("Fetching financial statements for %s (symbol=%s) ...", ticker, symbol)
            t = yf.Ticker(symbol)
            income = t.quarterly_financials
            balance = t.quarterly_balance_sheet
            cash_flow = t.quarterly_cashflow
            info = t.info
            df_yf = self._build_fundamental(income, balance, cash_flow, info, ticker)

            df_vn = pd.DataFrame()
            if ticker in self._VN_TICKERS:
                logger.info("Applying VN fundamental fallback order for %s: yfinance -> vnstock -> merge", ticker)
                df_vn = self._build_fundamental_from_vnstock(ticker=ticker, info=info)

            df = self._merge_fundamental_frames(df_yf, df_vn)
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
            ebit = _v(_row(income, "EBIT", "Operating Income", "Total Operating Income As Reported"))
            pretax_income = _v(_row(income, "Pretax Income", "Income Before Tax"))
            assets = _v(_row(balance, "Total Assets"))
            equity = _v(_row(balance, "Stockholders Equity", "Common Stock Equity"))
            debt = _v(_row(balance, "Total Debt"))
            current_assets = _v(_row(balance, "Current Assets"))
            current_liabilities = _v(_row(balance, "Current Liabilities"))
            retained_earnings = _v(_row(balance, "Retained Earnings"))
            interest_expense = _v(_row(income, "Interest Expense", "Interest Expenses", "Interest Expense And Other"))
            if interest_expense is None and ebit is not None and pretax_income is not None:
                interest_expense = abs(ebit - pretax_income)
            ebitda = _v(_row(income, "EBITDA"))
            
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
                eps = net_inc / shares
            cfo = _v(_row(cash_flow, "Operating Cash Flow", "Total Cash From Operating Activities"))
            capex = _v(_row(cash_flow, "Capital Expenditure", "Capital Expenditures"))
            receivables = _v(_row(balance, "Accounts Receivable", "Receivables"))
            inventory = _v(_row(balance, "Inventory"))
            payables = _v(_row(balance, "Accounts Payable", "Payables"))
            cogs = _v(_row(income, "Cost Of Revenue", "Cost Of Goods And Services Sold"))
            short_term_investments = _v(_row(balance, "Other Short Term Investments", "Cash Cash Equivalents And Short Term Investments"))
            income_tax_expense = _v(_row(income, "Tax Provision", "Income Tax Expense"))
            pre_tax_income = _v(_row(income, "Pretax Income", "Income Before Tax"))
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
                "capex":              capex,
                "accounts_receivable": receivables,
                "inventory":          inventory,
                "accounts_payable":   payables,
                "cogs":               cogs,
                "short_term_investments": short_term_investments,
                "income_tax_expense": income_tax_expense,
                "pre_tax_income":     pre_tax_income,
                "current_assets":      current_assets,
                "current_liabilities": current_liabilities,
                "retained_earnings":   retained_earnings,
                "interest_expense":    interest_expense,
                "ebitda":              ebitda,
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
        df["pe"]                 = _i("trailingPE")
        df["pb"]                 = _i("priceToBook")
        df["dividend"]           = _i("dividendRate")
        df["market_cap"]         = _i("marketCap")

        cols = [
            "date", "ticker",
            "revenue", "gross_profit", "operating_profit", "net_income", "eps",
            "total_assets", "total_liabilities", "equity", "total_debt", "cash", "operating_cash_flow",
            "capex", "accounts_receivable", "inventory", "accounts_payable", "cogs",
            "short_term_investments", "income_tax_expense", "pre_tax_income",
            "current_assets", "current_liabilities", "retained_earnings", "interest_expense", "ebitda", "market_cap",
            "roe", "roa", "pe", "pb", "margin", "debt_to_equity",
            "shares_outstanding", "bvps", "dividend",
        ]
        return df[[c for c in cols if c in df.columns]].sort_values("date").reset_index(drop=True)

    def fetch_news(self, query, page_size=50):
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
                    "q": ticker_query, "pageSize": min(int(page_size), 100),
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

    def fetch_macro_indicators(self):
        """
        Schema (macro_df_wide): date, fed_funds_rate, us_10y_yield, us_cpi, dxy, gold_price, oil_price
        """
        import requests
        data_frames = []

        fred_indicators = {
            "fed_funds_rate": "FEDFUNDS",
            "us_10y_yield": "DGS10",  
            "us_cpi": "CPIAUCSL",
            "gdp": "GDP",
            "unemployment_rate": "UNRATE",
        }
        
        for name, series_id in fred_indicators.items():
            logger.info("Fetching macro from FRED API: %s (%s) ...", name, series_id)
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": getattr(self, 'fred_api_key', None),
                    "file_type": "json",
                    "observation_start": self.start_date,
                    "observation_end": self.end_date
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                
                observations = data.get("observations", [])
                if observations:
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

        yf_indicators = {
            "dxy": "DX-Y.NYB",       
            "gold_price": "GC=F",    
            "oil_price": "CL=F",
            "fx_rate": "USDVND=X",
        }

        for name, symbol in yf_indicators.items():
            logger.info("Fetching macro from yfinance: %s (%s) ...", name, symbol)
            try:
                raw = yf.download(symbol, start=self.start_date, end=self.end_date, auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    df = self._flatten(raw)
                    df = df[["date", "close"]].rename(columns={"close": name})
                    df["date"] = pd.to_datetime(df["date"]) 
                    df.set_index("date", inplace=True)
                    data_frames.append(df)
            except Exception as e:
                logger.error("Failed yfinance macro %s: %s", name, e)

        try:
            start_year = pd.Timestamp(self.start_date).year
            end_year   = pd.Timestamp(self.end_date).year
            wb_url = (
                f"https://api.worldbank.org/v2/country/US/indicator/"
                f"BX.KLT.DINV.CD.WD?format=json&date={start_year}:{end_year}&per_page=50"
            )
            resp = requests.get(wb_url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list) and len(payload) > 1 and payload[1]:
                wb_rows = [
                    {"date": pd.Timestamp(f"{r['date']}-01-01"), "fdi_inflow": r["value"]}
                    for r in payload[1] if r.get("value") is not None
                ]
                if wb_rows:
                    fdi_df = pd.DataFrame(wb_rows).set_index("date").sort_index()
                    data_frames.append(fdi_df)
                    logger.info("FDI inflow fetched from World Bank: %d annual rows.", len(fdi_df))
                else:
                    logger.warning("World Bank FDI: no valid rows returned.")
            else:
                logger.warning("World Bank FDI: unexpected response format.")
        except Exception as e:
            logger.error("Failed World Bank FDI fetch: %s", e)

        if not data_frames:
            logger.warning("No macro data fetched.")
            return pd.DataFrame(columns=["date"])

        macro_df = pd.concat(data_frames, axis=1).reset_index()
        macro_df["date"] = pd.to_datetime(macro_df["date"])
        macro_df = macro_df.sort_values("date").reset_index(drop=True)

        value_cols = [c for c in macro_df.columns if c != "date"]
        macro_df[value_cols] = macro_df[value_cols].ffill()

        macro_df = macro_df.dropna(how='all', subset=value_cols).reset_index(drop=True)

        macro_df["date"] = macro_df["date"].dt.strftime("%Y-%m-%d")

        expected_cols = [
            "date", "fed_funds_rate", "us_10y_yield", "us_cpi",
            "dxy", "gold_price", "oil_price", "gdp", "unemployment_rate",
            "domestic_interest_rate", "fx_rate", "fdi_inflow",
        ]
        for col in expected_cols:
            if col not in macro_df.columns:
                macro_df[col] = np.nan
        if "domestic_interest_rate" in macro_df.columns:
            macro_df["domestic_interest_rate"] = macro_df["domestic_interest_rate"].fillna(macro_df.get("fed_funds_rate"))
        macro_df = macro_df[expected_cols]

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
                symbol = self._yf_symbol(ticker)
                t = yf.Ticker(symbol)
                inc = t.quarterly_financials
                bal = t.quarterly_balance_sheet

                try:
                    price_hist = t.history(start=self.start_date, end=self.end_date, interval="1d")
                    if isinstance(price_hist.index, pd.DatetimeIndex):
                        if price_hist.index.tz is not None:
                            price_hist.index = price_hist.index.tz_localize(None)
                    else:
                        price_hist.index = pd.to_datetime(price_hist.index, errors='coerce')
                    price_series = price_hist["Close"] if "Close" in price_hist.columns else None
                except Exception:
                    price_series = None

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
                    shares = _v(bal, "Ordinary Shares Number", "Share Issued")
                        
                    roe = (net_inc / eq) if (net_inc and eq and eq != 0) else None
                    margin = (net_inc / rev) if (net_inc and rev and rev != 0) else None

                    price_at_date = None
                    if price_series is not None and not price_series.empty:
                        d_ts = pd.Timestamp(d).tz_localize(None)
                        eligible = price_series[price_series.index <= d_ts]
                        if not eligible.empty:
                            price_at_date = float(eligible.iloc[-1])

                    pe = None
                    if price_at_date and net_inc and shares and shares > 0:
                        eps = net_inc / shares
                        if eps > 0:
                            pe = price_at_date / eps

                    pb = None
                    if price_at_date and eq and shares and shares > 0:
                        bvps = eq / shares
                        if bvps > 0:
                            pb = price_at_date / bvps

                    if roe is not None or margin is not None:
                        all_peer_data.append({
                            "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "roe": roe,
                            "margin": margin,
                            "pe": pe,
                            "pb": pb,
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
            "margin": "industry_margin",
            "pe": "industry_pe",
            "pb": "industry_pb",
        })
        
        industry_df = industry_df.sort_values("date").reset_index(drop=True)
        df = self._validate_df(industry_df, "industry_df")
        self._save_csv(df, "industry_data.csv")
        logger.info("Historical Industry data compiled: %d quarters.", len(df))
        
        return df

    def fetch_intraday(self, interval="5m", period="5d"):
        """
        Schema (intraday_df): timestamp, ticker, price, volume
        interval : "1m" | "5m" | "15m" | "30m" | "60m" | "90m"
        period   : "1d" | "5d" | "1mo"
        """
        data_map = {}
        for ticker in self.tickers:
            symbol = self._yf_symbol(ticker)
            logger.info("Fetching intraday %s for %s (symbol=%s, period=%s)...", interval, ticker, symbol, period)
            try:
                raw = yf.download(symbol, period=period, interval=interval,
                                  auto_adjust=True, progress=False)
                if raw is None or raw.empty:
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

