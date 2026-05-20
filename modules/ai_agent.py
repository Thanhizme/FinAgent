"""
ai_agent.py
-----------
Implements Module 4: AI Analysis Specification per module4.md.

Generates a professional-grade financial analysis report with 5 main sections:
1. Executive Summary
2. Macro Analysis
3. Financial Health
4. Valuation Analysis
5. Peer Comparison

Supported provider:
    - Google Gemini     (GEMINI_API_KEY)

Output format: {Indicator}: {Value} → {2-3 sentence interpretation}
Every metric includes analytical interpretation, never standalone numbers.
"""

import os
import json
import re
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Orchestrates LLM calls to generate comprehensive financial analysis per Module 4.

    Parameters
    ----------
    provider : {'gemini'}
        LLM provider to use. Default: 'gemini'.
    model : str, optional
        Model identifier. Default: 'gemini-2.5-flash'.
    max_tokens : int
        Maximum tokens to generate. Default: 8192 (per module4.md requirement).
    temperature : float
        LLM temperature (0.0-1.0). Default: 0.2 (factual, data-grounded).
    """

    _DEFAULT_MODELS = {
        "gemini": "gemini-2.5-flash",
    }

    _MODEL_FALLBACKS = {
        "gemini": ["gemini-2.5-flash"],
    }

    _ENV_KEYS = {
        "gemini": "GEMINI_API_KEY",
    }

    _GEMINI_MULTI_KEY_ENV = "GEMINI_API_KEYS"
    _GEMINI_INDEXED_KEY_PATTERN = re.compile(r"^GEMINI_API_KEY_(\d+)$")

    _SYSTEM_INSTRUCTION = (
        "You are a professional financial analyst. "
        "Output language must be English only. "
        "Never switch to Vietnamese or any other language. "
        "Do not output standalone raw numbers without interpretation."
    )

    _MAX_CALL_RETRIES = 3
    _RETRY_WAIT_SECONDS = 1.2
    _TRANSIENT_ERROR_MARKERS = (
        "503", "unavailable", "deadline exceeded",
        "internal", "timeout",
    )

    _FAILOVER_ERROR_MARKERS = (
        "404", "not_found", "not found", "resource_exhausted",
        "quota", "unsupported", "permission denied",
    )

    _TREND_EPSILON = 1e-6

    _COMPANY_METADATA = {
        # ── US Large Cap ─────────────────────────────────────────────────────────
        "AAPL":  {"company_name": "Apple Inc.",                   "exchange": "NASDAQ", "industry": "Information Technology",  "sub_sector": "Technology Hardware, Storage & Peripherals", "market_cap_usd": 4_450_000_000_000},
        "MSFT":  {"company_name": "Microsoft Corporation",        "exchange": "NASDAQ", "industry": "Information Technology",  "sub_sector": "Systems Software",                           "market_cap_usd": 3_120_000_000_000},
        "GOOGL": {"company_name": "Alphabet Inc. (Class A)",      "exchange": "NASDAQ", "industry": "Communication Services",  "sub_sector": "Interactive Media & Services",               "market_cap_usd": 2_200_000_000_000},
        "META":  {"company_name": "Meta Platforms, Inc.",         "exchange": "NASDAQ", "industry": "Communication Services",  "sub_sector": "Interactive Media & Services",               "market_cap_usd": 1_250_000_000_000},
        "AMZN":  {"company_name": "Amazon.com Inc.",              "exchange": "NASDAQ", "industry": "Consumer Discretionary",  "sub_sector": "Broadline Retail",                           "market_cap_usd": 1_950_000_000_000},
        "TSLA":  {"company_name": "Tesla, Inc.",                  "exchange": "NASDAQ", "industry": "Consumer Discretionary",  "sub_sector": "Automobile Manufacturers",                   "market_cap_usd":   620_000_000_000},
        "JPM":   {"company_name": "JPMorgan Chase & Co.",         "exchange": "NYSE",   "industry": "Financials",              "sub_sector": "Diversified Banks",                          "market_cap_usd":   831_000_000_000},
        "V":     {"company_name": "Visa Inc.",                    "exchange": "NYSE",   "industry": "Financials",              "sub_sector": "Transaction & Payment Processing Services",  "market_cap_usd":   584_000_000_000},
        "XOM":   {"company_name": "Exxon Mobil Corporation",      "exchange": "NYSE",   "industry": "Energy",                  "sub_sector": "Integrated Oil & Gas",                       "market_cap_usd":   510_000_000_000},
        "CVX":   {"company_name": "Chevron Corporation",          "exchange": "NYSE",   "industry": "Energy",                  "sub_sector": "Integrated Oil & Gas",                       "market_cap_usd":   300_000_000_000},
        "LIN":   {"company_name": "Linde plc",                    "exchange": "NYSE",   "industry": "Materials",               "sub_sector": "Industrial Gases",                           "market_cap_usd":   215_000_000_000},
        "NEE":   {"company_name": "NextEra Energy Inc.",          "exchange": "NYSE",   "industry": "Utilities",               "sub_sector": "Electric Utilities",                         "market_cap_usd":   155_000_000_000},
        "LMT":   {"company_name": "Lockheed Martin Corporation",  "exchange": "NYSE",   "industry": "Industrials",             "sub_sector": "Aerospace & Defense",                        "market_cap_usd":   132_000_000_000},
        "AMGN":  {"company_name": "Amgen Inc.",                   "exchange": "NASDAQ", "industry": "Health Care",             "sub_sector": "Biotechnology",                              "market_cap_usd":   165_000_000_000},
        "PLD":   {"company_name": "Prologis Inc.",                "exchange": "NYSE",   "industry": "Real Estate",             "sub_sector": "Industrial REITs",                           "market_cap_usd":   102_000_000_000},
        "MDLZ":  {"company_name": "Mondelez International Inc.",  "exchange": "NASDAQ", "industry": "Consumer Staples",        "sub_sector": "Packaged Foods & Meats",                     "market_cap_usd":    92_000_000_000},
        "EQIX":  {"company_name": "Equinix Inc.",                 "exchange": "NASDAQ", "industry": "Real Estate",             "sub_sector": "Data Center REITs",                          "market_cap_usd":    82_000_000_000},
        "SHW":   {"company_name": "The Sherwin-Williams Company", "exchange": "NYSE",   "industry": "Materials",               "sub_sector": "Specialty Chemicals",                        "market_cap_usd":    85_000_000_000},
        "ELV":   {"company_name": "Elevance Health Inc.",         "exchange": "NYSE",   "industry": "Health Care",             "sub_sector": "Managed Healthcare",                         "market_cap_usd":   115_000_000_000},
        "KMB":   {"company_name": "Kimberly-Clark Corporation",   "exchange": "NYSE",   "industry": "Consumer Staples",        "sub_sector": "Household Products",                         "market_cap_usd":    48_000_000_000},
        "GE":    {"company_name": "General Electric Aerospace",   "exchange": "NYSE",   "industry": "Industrials",             "sub_sector": "Industrial Conglomerates",                   "market_cap_usd":    19_000_000_000},
        "D":     {"company_name": "Dominion Energy Inc.",         "exchange": "NYSE",   "industry": "Utilities",               "sub_sector": "Multi-Utilities",                            "market_cap_usd":    43_000_000_000},

        # ── US Mid Cap ────────────────────────────────────────────────────────────
        "DECK":  {"company_name": "Deckers Outdoor Corporation",  "exchange": "NYSE",   "industry": "Consumer Discretionary",  "sub_sector": "Textiles, Apparel & Luxury Goods",           "market_cap_usd":    24_500_000_000},
        "CROX":  {"company_name": "Crocs Inc.",                   "exchange": "NASDAQ", "industry": "Consumer Discretionary",  "sub_sector": "Textiles, Apparel & Luxury Goods",           "market_cap_usd":     8_100_000_000},
        "DE":    {"company_name": "Deere & Co.",                  "exchange": "NYSE",   "industry": "Industrials",             "sub_sector": "Machinery",                                  "market_cap_usd":   112_000_000_000},
        "UPS":   {"company_name": "United Parcel Service Inc.",   "exchange": "NYSE",   "industry": "Industrials",             "sub_sector": "Air Freight & Logistics",                    "market_cap_usd":   118_000_000_000},
        "OVV":   {"company_name": "Ovintiv Inc.",                 "exchange": "NYSE",   "industry": "Energy",                  "sub_sector": "Oil, Gas & Consumable Fuels",                "market_cap_usd":    12_100_000_000},
        "APA":   {"company_name": "APA Corporation",              "exchange": "NASDAQ", "industry": "Energy",                  "sub_sector": "Oil, Gas & Consumable Fuels",                "market_cap_usd":     8_900_000_000},
        "VST":   {"company_name": "Vistra Corp.",                 "exchange": "NYSE",   "industry": "Utilities",               "sub_sector": "Independent Power & Renewable Producers",    "market_cap_usd":    32_000_000_000},
        "NRG":   {"company_name": "NRG Energy Inc.",              "exchange": "NYSE",   "industry": "Utilities",               "sub_sector": "Electric Utilities",                         "market_cap_usd":    17_400_000_000},
        "REXR":  {"company_name": "Rexford Industrial Realty",    "exchange": "NYSE",   "industry": "Real Estate",             "sub_sector": "Industrial REITs",                           "market_cap_usd":     9_800_000_000},
        "OHI":   {"company_name": "Omega Healthcare Investors",   "exchange": "NYSE",   "industry": "Real Estate",             "sub_sector": "Health Care REITs",                          "market_cap_usd":     8_100_000_000},
        "RS":    {"company_name": "Reliance Inc.",                "exchange": "NYSE",   "industry": "Materials",               "sub_sector": "Metals & Mining",                            "market_cap_usd":    16_800_000_000},
        "STLD":  {"company_name": "Steel Dynamics Inc.",          "exchange": "NASDAQ", "industry": "Materials",               "sub_sector": "Metals & Mining",                            "market_cap_usd":    20_400_000_000},
        "PINS":  {"company_name": "Pinterest Inc.",               "exchange": "NYSE",   "industry": "Communication Services",  "sub_sector": "Interactive Media & Services",               "market_cap_usd":    21_500_000_000},
        "TTWO":  {"company_name": "Take-Two Interactive",         "exchange": "NASDAQ", "industry": "Communication Services",  "sub_sector": "Entertainment",                              "market_cap_usd":    26_800_000_000},
        "MANH":  {"company_name": "Manhattan Associates Inc.",    "exchange": "NASDAQ", "industry": "Information Technology",  "sub_sector": "Software",                                   "market_cap_usd":    16_500_000_000},
        "TER":   {"company_name": "Teradyne Inc.",                "exchange": "NASDAQ", "industry": "Information Technology",  "sub_sector": "Semiconductors & Equipment",                 "market_cap_usd":    18_200_000_000},
        "SF":    {"company_name": "Stifel Financial Corp.",       "exchange": "NYSE",   "industry": "Financials",              "sub_sector": "Capital Markets",                            "market_cap_usd":     9_100_000_000},
        "JEF":   {"company_name": "Jefferies Financial Group",    "exchange": "NYSE",   "industry": "Financials",              "sub_sector": "Capital Markets",                            "market_cap_usd":    11_400_000_000},
        "HALO":  {"company_name": "Halozyme Therapeutics Inc.",   "exchange": "NASDAQ", "industry": "Health Care",             "sub_sector": "Biotechnology",                              "market_cap_usd":     6_800_000_000},
        "EHC":   {"company_name": "Encompass Health Corp.",       "exchange": "NYSE",   "industry": "Health Care",             "sub_sector": "Health Care Providers & Services",           "market_cap_usd":     8_400_000_000},
        "CASY":  {"company_name": "Casey's General Stores Inc.",  "exchange": "NASDAQ", "industry": "Consumer Staples",        "sub_sector": "Consumer Staples Distribution & Retail",     "market_cap_usd":    14_100_000_000},
        "CELH":  {"company_name": "Celsius Holdings Inc.",        "exchange": "NASDAQ", "industry": "Consumer Staples",        "sub_sector": "Beverages",                                  "market_cap_usd":    12_500_000_000},
        "BOOT":  {"company_name": "Boot Barn Holdings Inc.",      "exchange": "NYSE",   "industry": "Consumer Discretionary",  "sub_sector": "Apparel Retail",                             "market_cap_usd":     3_100_000_000},
        "AWR":   {"company_name": "American States Water Co.",    "exchange": "NYSE",   "industry": "Utilities",               "sub_sector": "Water Utilities",                            "market_cap_usd":     2_800_000_000},
        "AVA":   {"company_name": "Avista Corporation",           "exchange": "NYSE",   "industry": "Utilities",               "sub_sector": "Multi-Utilities",                            "market_cap_usd":     2_900_000_000},
        "MLI":   {"company_name": "Mueller Industries Inc.",      "exchange": "NYSE",   "industry": "Materials",               "sub_sector": "Industrial Machinery & Metal Fabrication",   "market_cap_usd":     5_800_000_000},

        # ── US Small Cap ──────────────────────────────────────────────────────────
        "IDCC":  {"company_name": "InterDigital Inc.",            "exchange": "NASDAQ", "industry": "Information Technology",  "sub_sector": "Communications Equipment",                   "market_cap_usd":     3_200_000_000},
        "KLIC":  {"company_name": "Kulicke & Soffa Industries",   "exchange": "NASDAQ", "industry": "Information Technology",  "sub_sector": "Semiconductor Materials & Equipment",        "market_cap_usd":     2_600_000_000},
        "DFIN":  {"company_name": "Donnelley Financial Solutions","exchange": "NYSE",   "industry": "Financials",              "sub_sector": "Capital Markets",                            "market_cap_usd":     1_800_000_000},
        "VBTX":  {"company_name": "Veritex Holdings Inc.",        "exchange": "NASDAQ", "industry": "Financials",              "sub_sector": "Regional Banks",                             "market_cap_usd":     1_100_000_000},
        "HIMS":  {"company_name": "Hims & Hers Health Inc.",      "exchange": "NYSE",   "industry": "Health Care",             "sub_sector": "Health Care Technology",                     "market_cap_usd":     3_500_000_000},
        "NSTG":  {"company_name": "NanoString Technologies Inc.", "exchange": "NASDAQ", "industry": "Health Care",             "sub_sector": "Life Sciences Tools & Services",             "market_cap_usd":        50_000_000},
        "SONO":  {"company_name": "Sonos, Inc.",                  "exchange": "NASDAQ", "industry": "Consumer Discretionary",  "sub_sector": "Household Durables",                         "market_cap_usd":     1_900_000_000},
        "CALM":  {"company_name": "Cal-Maine Foods Inc.",         "exchange": "NASDAQ", "industry": "Consumer Staples",        "sub_sector": "Packaged Foods & Meats",                     "market_cap_usd":     2_900_000_000},
        "JJSF":  {"company_name": "J&J Snack Foods Corp.",        "exchange": "NASDAQ", "industry": "Consumer Staples",        "sub_sector": "Packaged Foods & Meats",                     "market_cap_usd":     3_100_000_000},
        "BYRN":  {"company_name": "Byrna Technologies Inc.",      "exchange": "NASDAQ", "industry": "Industrials",             "sub_sector": "Aerospace & Defense",                        "market_cap_usd":       350_000_000},
        "MLKN":  {"company_name": "MillerKnoll Inc.",             "exchange": "NASDAQ", "industry": "Industrials",             "sub_sector": "Commercial Services & Supplies",             "market_cap_usd":     1_500_000_000},
        "REPX":  {"company_name": "Riley Exploration Permian",    "exchange": "NYSE",   "industry": "Energy",                  "sub_sector": "Oil, Gas & Consumable Fuels",                "market_cap_usd":       650_000_000},
        "PARR":  {"company_name": "Par Pacific Holdings Inc.",    "exchange": "NYSE",   "industry": "Energy",                  "sub_sector": "Oil, Gas & Consumable Fuels",                "market_cap_usd":     1_400_000_000},
        "LGIH":  {"company_name": "LGI Homes Inc.",               "exchange": "NASDAQ", "industry": "Real Estate",             "sub_sector": "Household Durables / Real Estate Development","market_cap_usd":    2_200_000_000},
        "UTL":   {"company_name": "Unitil Corporation",           "exchange": "NYSE",   "industry": "Utilities",               "sub_sector": "Multi-Utilities",                            "market_cap_usd":       850_000_000},
        "IOSP":  {"company_name": "Innospec Inc.",                "exchange": "NASDAQ", "industry": "Materials",               "sub_sector": "Specialty Chemicals",                        "market_cap_usd":     2_900_000_000},
        "CNK":   {"company_name": "Cinemark Holdings Inc.",       "exchange": "NYSE",   "industry": "Communication Services",  "sub_sector": "Entertainment",                              "market_cap_usd":     2_200_000_000},
        "YELP":  {"company_name": "Yelp Inc.",                    "exchange": "NYSE",   "industry": "Communication Services",  "sub_sector": "Interactive Media & Services",               "market_cap_usd":     2_400_000_000},

        # ── VN Large Cap ──────────────────────────────────────────────────────────
        "VCB":   {"company_name": "Vietcombank",                  "exchange": "HOSE",   "industry": "Financials",              "sub_sector": "Diversified Banks"},
        "BID":   {"company_name": "BIDV",                         "exchange": "HOSE",   "industry": "Financials",              "sub_sector": "Diversified Banks"},
        "VHM":   {"company_name": "Vinhomes",                     "exchange": "HOSE",   "industry": "Real Estate",             "sub_sector": "Real Estate Development"},
        "VIC":   {"company_name": "Vingroup",                     "exchange": "HOSE",   "industry": "Industrials",             "sub_sector": "Industrial Conglomerates"},
        "VNM":   {"company_name": "Vinamilk",                     "exchange": "HOSE",   "industry": "Consumer Staples",        "sub_sector": "Packaged Foods & Beverages"},
        "GAS":   {"company_name": "PV Gas",                       "exchange": "HOSE",   "industry": "Energy",                  "sub_sector": "Oil, Gas & Consumable Fuels"},

        # ── VN Mid Cap ────────────────────────────────────────────────────────────
        "KDH":   {"company_name": "Kien Hung Development",        "exchange": "HOSE",   "industry": "Real Estate",             "sub_sector": "Real Estate Development"},
        "NLG":   {"company_name": "Nam Long Group",               "exchange": "HOSE",   "industry": "Real Estate",             "sub_sector": "Real Estate Development"},
        "HPG":   {"company_name": "Hoa Phat Group",               "exchange": "HOSE",   "industry": "Materials",               "sub_sector": "Steel"},
        "GVR":   {"company_name": "Vietnam Rubber Group",         "exchange": "HOSE",   "industry": "Materials",               "sub_sector": "Agricultural Commodities"},
        "ADS":   {"company_name": "Adidas AG (VN-listed)",        "exchange": "NYSE/NASDAQ", "industry": "Consumer Discretionary", "sub_sector": "Textiles, Apparel & Luxury Goods"},
        "TCM":   {"company_name": "Thanh Cong Textile",           "exchange": "HOSE",   "industry": "Consumer Discretionary",  "sub_sector": "Textiles, Apparel & Luxury Goods"},
    }

    _METADATA_CACHE: Dict[str, Dict[str, str]] = {}

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = "gemini-2.5-flash",
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> None:
        provider_lower = provider.lower()
        if provider_lower not in self._DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {list(self._DEFAULT_MODELS.keys())}"
            )
        
        self.provider = provider_lower
        requested_model = (model or self._DEFAULT_MODELS[self.provider]).strip()
        self._model_candidates = self._build_model_candidates(requested_model)
        self.model = self._model_candidates[0]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._gemini_api_keys: List[str] = []
        self._active_api_key_index = 0
        self._client = self._init_client()
        logger.info(
            f"[{self.provider.upper()}] AnalysisAgent initialized with model {self.model} "
            f"(candidates={self._model_candidates}) "
            f"(temp={temperature}, tokens={max_tokens})"
        )

    def _build_model_candidates(self, requested_model: str) -> list[str]:
        """Resolve model priority list for automatic failover."""
        if self.provider != "gemini":
            return [requested_model]

        normalized = requested_model.lower()
        if normalized in ("auto", "gemini-2.5-flash"):
            return list(self._MODEL_FALLBACKS["gemini"])

        return [requested_model]

    def _init_client(self) -> Any:
        """Initialize SDK client for configured provider."""
        env_var = self._ENV_KEYS[self.provider]

        if self.provider == "gemini":
            try:
                from google import genai
                self._gemini_api_keys = self._load_gemini_api_keys()
                self._active_api_key_index = 0
                logger.info(
                    "Loaded %d Gemini API key(s) for automatic failover.",
                    len(self._gemini_api_keys),
                )
                return genai.Client(api_key=self._gemini_api_keys[self._active_api_key_index])
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )

        api_key = os.getenv(env_var)
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Please set '{env_var}' in your .env file."
            )

        raise NotImplementedError(f"Provider '{self.provider}' not implemented yet.")

    def _load_gemini_api_keys(self) -> List[str]:
        """Load and de-duplicate Gemini API keys from .env/environment."""
        keys: List[str] = []

        def add_key(raw: Optional[str]) -> None:
            value = (raw or "").strip()
            if not value:
                return
            if value.lower().startswith("your_"):
                return
            if value not in keys:
                keys.append(value)

        # Primary key.
        add_key(os.getenv("GEMINI_API_KEY"))

        # Bulk key list: comma/semicolon/newline separated.
        bulk = os.getenv(self._GEMINI_MULTI_KEY_ENV, "")
        if bulk:
            for item in re.split(r"[,;\r\n]+", bulk):
                add_key(item)

        # Indexed keys: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
        indexed_names = []
        for name in os.environ.keys():
            match = self._GEMINI_INDEXED_KEY_PATTERN.match(name)
            if match:
                indexed_names.append((int(match.group(1)), name))
        indexed_names.sort(key=lambda x: x[0])

        for _, name in indexed_names:
            add_key(os.getenv(name))

        if not keys:
            raise EnvironmentError(
                "No valid Gemini API key found. Configure at least one of: "
                "GEMINI_API_KEY, GEMINI_API_KEYS, GEMINI_API_KEY_1..N."
            )

        return keys

    def _switch_to_next_gemini_key(self) -> bool:
        """Rotate to the next configured Gemini key. Returns False if exhausted."""
        if self.provider != "gemini":
            return False
        if len(self._gemini_api_keys) <= 1:
            return False
        if self._active_api_key_index >= len(self._gemini_api_keys) - 1:
            return False

        self._active_api_key_index += 1
        from google import genai

        self._client = genai.Client(api_key=self._gemini_api_keys[self._active_api_key_index])
        logger.warning(
            "Switched to Gemini API key %d/%d after failure.",
            self._active_api_key_index + 1,
            len(self._gemini_api_keys),
        )
        return True

    def _is_transient_error(self, error_text: str) -> bool:
        """Check if error is transient (retry-worthy)."""
        lower = error_text.lower()
        return any(marker in lower for marker in self._TRANSIENT_ERROR_MARKERS)

    def _is_failover_error(self, error_text: str) -> bool:
        """Check if current model should fallback to next candidate."""
        lower = error_text.lower()
        return any(marker in lower for marker in self._FAILOVER_ERROR_MARKERS)

    def _compute_trend_direction(self, values: pd.Series) -> str:
        """Compute trend direction from the last two valid observations."""
        series = pd.to_numeric(values, errors='coerce').dropna()
        if len(series) < 2:
            return "not_available"

        latest = float(series.iloc[-1])
        previous = float(series.iloc[-2])
        delta = latest - previous

        if abs(delta) <= self._TREND_EPSILON:
            return "flat"
        return "up" if delta > 0 else "down"

    def _sanitize_llm_output(self, text: str) -> str:
        """Normalize malformed spacing artifacts while preserving markdown readability."""
        if not text:
            return ""

        cleaned = str(text)
        # Drop invisible zero-width chars that can fragment words in rendering.
        cleaned = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", cleaned)

        # Join fragmented words like "b i l l i o n" (5+ single-letter tokens).
        word_frag_pattern = re.compile(r"\b(?:[A-Za-z]\s+){5,}[A-Za-z]\b")
        cleaned = word_frag_pattern.sub(lambda m: m.group(0).replace(" ", ""), cleaned)

        # Join fragmented numbers like "1 . 5 2" or "1 0 2 . 2 9".
        cleaned = re.sub(r"(?<=\d)\s+(?=[\d.,%-])", "", cleaned)
        cleaned = re.sub(r"(?<=[\d.,%-])\s+(?=\d)", "", cleaned)
        cleaned = re.sub(r"(?<=\d)\s*\n\s*(?=[\d.,%-])", "", cleaned)
        cleaned = re.sub(r"(?<=[\d.,%-])\s*\n\s*(?=\d)", "", cleaned)

        # Standardize arrows and spacing around them.
        cleaned = re.sub(r"\s*(?:->|→)\s*", " -> ", cleaned)

        # Remove stray bold markers that can remain around headings/items.
        cleaned = cleaned.replace("**", "")

        # Force numbered list items onto their own lines when the model jams them together.
        # Restrict this to clear list starts and avoid matching decimal tails like "36.62. This".
        cleaned = re.sub(r"(?<!\n)(?<=:)\s*(\d{1,2})\.\s+(?=[A-Z(])", r"\n\1. ", cleaned)

        # Split cases like "Apple's 36.10.2. Current P/B..." into "36.10." + next numbered item.
        cleaned = re.sub(r"(\d+\.\d{1,2})\.(\d{1,2})\.\s+(?=[A-Z(])", r"\1.\n\2. ", cleaned)
        cleaned = re.sub(r"(\d+\.\d{1,2})\.(\d{1,2})\.\s+(?=[a-z_])", r"\1.\n\2. ", cleaned)

        # Split concatenated numbered items like "... trends.2. ROE: ..." while leaving decimals intact.
        cleaned = re.sub(r"(?<=[A-Za-z%)])\.(\d{1,2})\.\s+(?=[A-Z(])", r".\n\1. ", cleaned)
        cleaned = re.sub(r"(?<=[A-Za-z%)])\.(\d{1,2})\.\s+(?=[a-z_])", r".\n\1. ", cleaned)
        cleaned = re.sub(r"(?<=\+)\.(\d{1,2})\.\s+(?=[A-Za-z_])", r".\n\1. ", cleaned)
        cleaned = re.sub(r"(?<=[A-Za-z)])\.\s+(\d{1,2})\.\s+(?=[A-Z(])", r".\n\1. ", cleaned)
        cleaned = re.sub(r"(?<!\n)(?<=[a-z%])\s+(\d{1,2})\.\s+(?=[A-Z(])", r"\n\1. ", cleaned)

        broken_heading_prefixes = (
            "Basic Information:",
            "Summary Statement:",
            "Comparison Snippet:",
            "Call to Action:",
            "Financial Health Comparison",
            "Fundamental Valuation Comparison",
            "Technical & Risk Profile Comparison",
            "Comparison Summary",
            "Revenue Growth (YoY):",
            "ROE:",
            "Current Ratio:",
            "Debt-to-Equity (D/E):",
            "FCFE:",
            "Current P/E vs Industry avg:",
            "Current P/B vs Industry avg:",
            "DCF Valuation (Intrinsic Price vs Market Price, Upside/Downside %):",
            "Current Price:",
            "1W Return %:",
            "1M Return %:",
            "3M Return %:",
            "YTD Return % vs Index:",
            "YTD Return %:",
            "MA20:",
            "MA50:",
            "MA200:",
            "RSI(14):",
            "MACD:",
            "Bollinger Bands (price vs upper/middle/lower):",
            "Volume Spike:",
            "Gap Up/Gap Down:",
            "Sudden Price Movement:",
            "Historical Volatility 30D %:",
            "Historical Volatility 60D %:",
            "Beta vs Index:",
            "VaR 95% daily:",
            "VaR 99% daily:",
            "Max Drawdown %:",
            "Sharpe Ratio:",
        )

        # Repair decimals that were previously split across lines, e.g. "0.\n92" -> "0.92",
        # but keep numbered items like "36.10.\n2. Current P/B" on separate lines.
        cleaned = re.sub(r"(?<=\d)\.\s*\n\s*(?=\d(?!\.))", ".", cleaned)

        # Repair broken numbered headings line-by-line, e.g. "1." + next heading line,
        # or a trailing "5." left at the end of the prior line.
        normalized_lines: list[str] = []
        lines = cleaned.splitlines()
        index = 0
        while index < len(lines):
            current_line = lines[index].strip()
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""

            if re.fullmatch(r"\d{1,2}\.", current_line) and next_line.startswith(broken_heading_prefixes):
                normalized_lines.append(f"{current_line} {next_line}")
                index += 2
                continue

            trailing_item = re.match(r"^(.*\S)\s+(\d{1,2})\.$", current_line)
            if trailing_item and next_line.startswith(broken_heading_prefixes):
                normalized_lines.append(trailing_item.group(1))
                normalized_lines.append(f"{trailing_item.group(2)}. {next_line}")
                index += 2
                continue

            normalized_lines.append(current_line)
            index += 1

        cleaned = "\n".join(normalized_lines)

        inline_heading_labels = (
            r"Basic Information:|"
            r"Summary Statement:|"
            r"Comparison Snippet:|"
            r"Call to Action:|"
            r"Financial Health Comparison|"
            r"Fundamental Valuation Comparison|"
            r"Technical & Risk Profile Comparison|"
            r"Comparison Summary|"
            r"Revenue Growth \(YoY\):|ROE:|Current Ratio:|Debt-to-Equity \(D/E\):|FCFE:|"
            r"Current P/E vs Industry avg:|Current P/B vs Industry avg:|"
            r"DCF Valuation \(Intrinsic Price vs Market Price, Upside/Downside %\):|"
            r"Current Price:|"
            r"1W Return %:|"
            r"1M Return %:|"
            r"3M Return %:|"
            r"YTD Return %(?: vs Index)?:|"
            r"MA20:|MA50:|MA200:|RSI\(14\):|MACD:|"
            r"Bollinger Bands \(price vs upper/middle/lower\):|"
            r"Volume Spike:|Gap Up/Gap Down:|Sudden Price Movement:|"
            r"Historical Volatility 30D %:|Historical Volatility 60D %:|"
            r"Beta vs Index:|VaR 95% daily:|VaR 99% daily:|Max Drawdown %:|Sharpe Ratio:"
        )
        cleaned = re.sub(
            rf"(?<!\n)(?<=\S)(\d{{1,2}})\.(?=(?:{inline_heading_labels}))",
            r"\n\1. ",
            cleaned,
        )
        cleaned = re.sub(r"(?m)^(\d{1,2})\.(?=[A-Za-z_])", r"\1. ", cleaned)
        cleaned = re.sub(r"(?m)^(\d{1,2})\.(?=(?:1W Return %:|1M Return %:|3M Return %:))", r"\1. ", cleaned)

        # Executive Summary section titles should not be numbered in final output.
        cleaned = re.sub(
            r"(?m)^\d{1,2}\.\s+(Basic Information:|Summary Statement:|Comparison Snippet:|Call to Action:)",
            r"\1",
            cleaned,
        )

        # Normalise Short-term/Long-term View blocks:
        # Collect all sub-sentences and merge them onto the header line (inline after the colon).
        # Removes any separate bullet lines under the header.
        _view_header_re = re.compile(
            r"^[ \t]*[\u2022\-\*]?[ \t]*(Short-term View|Long-term View)\s*\(", re.IGNORECASE
        )
        _label_re = re.compile(r"^\[(?:Fundamental|Technical(?:/Valuation)?)\][ \t]*")
        _any_bullet_re = re.compile(r"^[ \t]*[\u2022\-\*][ \t]+")
        _cleaned_lines: list[str] = []
        _view_header_line: str | None = None
        _view_sentences: list[str] = []
        _in_view_block = False

        def _flush_view() -> None:
            """Append accumulated view header + sentences as a single line."""
            if _view_header_line is not None:
                # Strip trailing colon from header to re-add cleanly.
                header = _view_header_line.rstrip(":")
                combined = (header + ": " + " ".join(_view_sentences)).strip() if _view_sentences else _view_header_line
                _cleaned_lines.append(combined)

        for _line in cleaned.splitlines():
            _stripped_line = _line.strip()
            if _view_header_re.match(_stripped_line):
                # Flush previous view block if any.
                if _in_view_block:
                    _flush_view()
                _in_view_block = True
                # Build clean header with • prefix, strip any existing bullet.
                bare = _any_bullet_re.sub("", _stripped_line)
                # If content already follows the colon on the same line, split it off.
                _colon_match = re.match(r"^((?:Short-term View|Long-term View)[^:]*:)[ \t]*(.+)", bare, re.IGNORECASE)
                if _colon_match:
                    _view_header_line = "\u2022 " + _colon_match.group(1)
                    _view_sentences = [_colon_match.group(2).strip()]
                else:
                    _view_header_line = "\u2022 " + bare.rstrip(":")
                    _view_sentences = []
                continue
            if _in_view_block:
                # Exit block on blank line or next major section header.
                if _stripped_line == "" or re.match(r"^(?:Comparison Snippet|Call to Action)[\.:]?", _stripped_line):
                    _flush_view()
                    _view_header_line = None
                    _view_sentences = []
                    _in_view_block = False
                    _cleaned_lines.append(_line)
                    continue
                # Skip if it's another view header (handled at top of loop).
                if _view_header_re.match(_stripped_line):
                    _flush_view()
                    _in_view_block = True
                    bare = _any_bullet_re.sub("", _stripped_line)
                    _colon_match = re.match(r"^((?:Short-term View|Long-term View)[^:]*:)[ \t]*(.+)", bare, re.IGNORECASE)
                    if _colon_match:
                        _view_header_line = "\u2022 " + _colon_match.group(1)
                        _view_sentences = [_colon_match.group(2).strip()]
                    else:
                        _view_header_line = "\u2022 " + bare.rstrip(":")
                        _view_sentences = []
                    continue
                # Collect sentence: strip bullet/label prefixes.
                sentence = _any_bullet_re.sub("", _stripped_line)
                sentence = _label_re.sub("", sentence).strip()
                if sentence:
                    _view_sentences.append(sentence)
                continue
            _cleaned_lines.append(_line)

        # Flush any open view block at end of text.
        if _in_view_block:
            _flush_view()
        cleaned = "\n".join(_cleaned_lines)
        cleaned = re.sub(r"(?m)^Technical Conclusion\s*$", "Technical Conclusion:", cleaned)
        cleaned = re.sub(r"(?m)^Overall Financial Health Conclusion\s*$", "Overall Financial Health Conclusion:", cleaned)

        # Ensure non-numbered Technical Trend labels are on separate lines.
        trend_labels = (
            r"Current Price:|"
            r"1W Return %:|"
            r"1M Return %:|"
            r"3M Return %:|"
            r"YTD Return %(?: vs Index)?:"
        )
        cleaned = re.sub(
            rf"(?<!\n)(?<!\d\.)(?<=[.!?])\s*(?=(?:{trend_labels}))",
            "\n",
            cleaned,
        )

        # Ensure common section headings begin on their own line when merged with prior text.
        section_labels = (
            r"Basic Information:|"
            r"Summary Statement:|"
            r"Comparison Snippet:|"
            r"Call to Action:|"
            r"GLOBAL & US INDICATORS|"
            r"GLOBAL INDICATORS|"
            r"VIETNAM DOMESTIC INDICATORS|"
            r"INDUSTRY VALUATION & PROFITABILITY|"
            r"CORPORATE EVENTS & NEWS|"
            r"Overall Financial Health Conclusion:|"
            r"Technical Conclusion:|"
            r"Financial Health Comparison|"
            r"Fundamental Valuation Comparison|"
            r"Technical & Risk Profile Comparison|"
            r"Comparison Summary"
        )
        cleaned = re.sub(rf"(?<!\n)(?<!\d\.)(?<=[.!?])\s*(?=(?:{section_labels}))", "\n", cleaned)

        # Add a blank line before each numbered metric for readability.
        cleaned = re.sub(r"(?<!\n\n)(?<!^)\n(\d{1,2}\.\s+)", r"\n\n\1", cleaned)

        # Clean up accidental double spaces introduced by the normalizer.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)

        # Keep output compact but readable.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _normalize_coverage_value(self, value: Any, fallback: str = "N/A") -> Any:
        """Return a display-safe value for auto-patched coverage lines."""
        if value is None:
            return fallback
        if isinstance(value, str) and not value.strip():
            return fallback
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return fallback
        return value

    def _format_currency_compact(self, value: Any) -> str:
        """Format large currency values into readable USD strings."""
        normalized = self._normalize_coverage_value(value)
        if isinstance(normalized, str):
            return normalized

        amount = float(normalized)
        absolute = abs(amount)
        sign = "-" if amount < 0 else ""

        if absolute >= 1_000_000_000_000:
            return f"{sign}${absolute / 1_000_000_000_000:.2f} trillion"
        if absolute >= 1_000_000_000:
            return f"{sign}${absolute / 1_000_000_000:.2f} billion"
        if absolute >= 1_000_000:
            return f"{sign}${absolute / 1_000_000:.2f} million"
        return f"{sign}${absolute:,.2f}"

    def _format_metric_display(self, value: Any, style: str = "number") -> str:
        """Format metric values for prompt display to reduce raw-ratio outputs."""
        normalized = self._normalize_coverage_value(value)
        if isinstance(normalized, str):
            return normalized

        number = float(normalized)

        if style == "ratio_pct":
            return f"{number * 100:.2f}%"
        if style == "pct":
            return f"{number:.2f}%"
        if style == "currency":
            return self._format_currency_compact(number)
        if style == "price":
            return f"{number:.2f}"
        if style == "days":
            return f"{number:.2f} days"
        return f"{number:.2f}"

    def _classify_market_cap(self, market_cap_usd: Optional[float]) -> Dict[str, Any]:
        """Classify market cap into Large/Mid/Small buckets."""
        if market_cap_usd is None or not np.isfinite(market_cap_usd):
            return {
                "market_cap_usd": None,
                "market_cap_classification": "Not available",
            }

        if market_cap_usd >= 10_000_000_000:
            cap_class = "Large Cap"
        elif market_cap_usd >= 2_000_000_000:
            cap_class = "Mid Cap"
        else:
            cap_class = "Small Cap"

        return {
            "market_cap_usd": float(market_cap_usd),
            "market_cap_classification": cap_class,
        }

    def _ensure_macro_metric_coverage(
        self,
        text: str,
        ticker: str,
        market: str,
        macro_metrics: Dict[str, Any],
    ) -> str:
        """Ensure every required macro indicator appears with trend and interpretation."""
        if not macro_metrics:
            return text

        required = [
            'imf_global_growth', 'fed_funds_rate', 'oil_price',
            'vn_gdp_growth', 'vn_cpi',
        ] if market == "VN" else [
            'imf_global_growth', 'fed_funds_rate', 'oil_price',
            'us_gdp_growth', 'us_cpi',
        ]

        metric_aliases = {
            "imf_global_growth": ["imf_global_growth", "imf global gdp growth forecast", "global gdp growth forecast"],
            "fed_funds_rate": ["fed_funds_rate", "us federal reserve policy rate", "federal reserve policy rate", "fed funds rate"],
            "oil_price": ["oil_price", "global crude oil price level", "crude oil price", "oil price"],
            "us_gdp_growth": ["us_gdp_growth", "us gdp growth rate", "gdp growth rate"],
            "us_cpi": ["us_cpi", "us inflation level", "consumer price index", "cpi"],
            "vn_gdp_growth": ["vn_gdp_growth", "vietnam gdp growth rate"],
            "vn_cpi": ["vn_cpi", "vietnam inflation rate", "vn inflation rate"],
        }

        lower_text = text.lower()
        missing = [
            metric
            for metric in required
            if not any(alias in lower_text for alias in metric_aliases.get(metric, [metric]))
        ]
        if not missing:
            return text

        trend_impact = {
            "up": "a potential tailwind if demand-led, but a headwind if cost-led",
            "down": "a potential headwind if demand weakens, but a tailwind for rate-sensitive activity",
            "flat": "a neutral signal with limited incremental macro impulse",
            "not_available": "an uncertain signal due to limited trend data",
        }

        lines = ["", "Structured Macro Metric Coverage (Auto-Patch):"]
        for idx, metric in enumerate(missing, start=1):
            payload = macro_metrics.get(metric, {})
            if isinstance(payload, dict):
                value = self._normalize_coverage_value(payload.get("value"), "N/A")
                trend = str(self._normalize_coverage_value(payload.get("trend"), "not_available"))
            elif payload not in (None, ""):
                value = self._normalize_coverage_value(payload, "N/A")
                trend = "not_available"
            else:
                value = "N/A"
                trend = "not_available"
            impact = trend_impact.get(str(trend), trend_impact["not_available"])
            lines.append(
                f"{idx}. {metric}: {value} | Trend: {trend} -> "
                f"The latest reading indicates {trend} dynamics in this indicator. "
                f"For {ticker}, this suggests {impact} and should be assessed with sector-specific sensitivity."
            )

        logger.info(
            "Auto-patch macro coverage for %s (%s): %s",
            ticker,
            market,
            ", ".join(missing),
        )
        patched_text = f"{text.rstrip()}\n\n" + "\n".join(lines)
        return self._sanitize_llm_output(patched_text)

    def _ensure_valuation_structure(self, text: str) -> str:
        """Inject Module 4 valuation sub-section headers when the model omits them."""
        if not text:
            return ""

        structured = text
        headers = {
            "FUNDAMENTAL VALUATION": r"(?m)^1\.\s+Current P/E:",
            "TECHNICAL ANALYSIS - TREND SUMMARY": r"(?m)^4\.\s+Current Price:",
            "MOVING AVERAGES & OSCILLATORS": r"(?m)^9\.\s+MA20:",
            "PRICE & VOLUME ANOMALIES": r"(?m)^15\.\s+Volume Spike:",
            "RISK METRICS": r"(?m)^18\.\s+Historical Volatility 30D %:",
        }

        for heading, marker in headers.items():
            if heading in structured:
                continue
            structured = re.sub(marker, f"{heading}\n\\g<0>", structured, count=1)

        return self._sanitize_llm_output(structured)

    def _call_gemini(self, system: str, user_prompt: str) -> str:
        """Call Gemini API."""
        try:
            combined_prompt = f"{system}\n\n{user_prompt}"
            client: Any = self._client
            response = client.models.generate_content(
                model=self.model,
                contents=combined_prompt,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            return response.text or ""
        except Exception as exc:
            logger.error(f"Gemini API call failed: {exc}")
            raise

    def _call_llm(self, system: str, user_prompt: str) -> str:
        """Dispatch LLM call with retry, model failover, and API-key failover logic."""
        last_error: Optional[Exception] = None

        total_key_attempts = len(self._gemini_api_keys) if self.provider == "gemini" else 1

        for key_attempt in range(total_key_attempts):
            for model_index, candidate_model in enumerate(self._model_candidates):
                self.model = candidate_model

                for attempt in range(1, self._MAX_CALL_RETRIES + 1):
                    try:
                        if self.provider == "gemini":
                            return self._call_gemini(system, user_prompt)
                        raise NotImplementedError(f"Provider '{self.provider}' not implemented.")
                    except Exception as exc:
                        last_error = exc
                        error_text = str(exc)

                        if attempt < self._MAX_CALL_RETRIES and self._is_transient_error(error_text):
                            logger.warning(
                                "Transient LLM error on model %s (key %d/%d) attempt %d/%d: %s",
                                candidate_model,
                                self._active_api_key_index + 1,
                                max(1, len(self._gemini_api_keys)),
                                attempt,
                                self._MAX_CALL_RETRIES,
                                exc,
                            )
                            time.sleep(self._RETRY_WAIT_SECONDS * attempt)
                            continue

                        logger.warning(
                            "Model %s failed on key %d/%d after %d attempt(s): %s",
                            candidate_model,
                            self._active_api_key_index + 1,
                            max(1, len(self._gemini_api_keys)),
                            attempt,
                            exc,
                        )
                        break

                # Try next candidate model on the same key.
                if model_index < len(self._model_candidates) - 1:
                    logger.info(
                        "Switching model from %s to %s on key %d/%d.",
                        candidate_model,
                        self._model_candidates[model_index + 1],
                        self._active_api_key_index + 1,
                        max(1, len(self._gemini_api_keys)),
                    )

            # All model candidates failed on this key; rotate key and retry from first model.
            if key_attempt < total_key_attempts - 1 and self._switch_to_next_gemini_key():
                continue
            break

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM call failed after retries, model failover, and API-key failover attempts.")

    # ============================================================================
    # DATA EXTRACTION & PREPARATION
    # ============================================================================

    def _resolve_company_metadata(self, ticker: str, market: str) -> Dict[str, str]:
        """Resolve company metadata from static map first, then yfinance fallback."""
        ticker_upper = ticker.upper()

        if ticker_upper in self._METADATA_CACHE:
            return dict(self._METADATA_CACHE[ticker_upper])

        defaults = {
            "company_name": "Not available",
            "exchange": "HOSE/HNX/UPCoM" if market == "VN" else "NYSE/NASDAQ",
            "industry": "Not available",
            "sub_sector": "Not available",
        }

        static_meta = self._COMPANY_METADATA.get(ticker_upper)
        if static_meta:
            merged = {**defaults, **static_meta}
            self._METADATA_CACHE[ticker_upper] = merged
            return dict(merged)

        dynamic_meta: Dict[str, str] = {}
        # Prefer yfinance for GLOBAL tickers; fallback silently on network/data issues.
        try:
            import yfinance as yf

            candidates = [ticker_upper]
            if market == "VN":
                candidates.insert(0, f"{ticker_upper}.VN")

            info: Dict[str, Any] = {}
            for symbol in candidates:
                try:
                    info = yf.Ticker(symbol).info or {}
                except Exception:
                    info = {}
                if info:
                    break

            if info:
                dynamic_meta["company_name"] = str(
                    info.get("longName") or info.get("shortName") or defaults["company_name"]
                )
                dynamic_meta["exchange"] = str(
                    info.get("fullExchangeName") or info.get("exchange") or defaults["exchange"]
                )
                dynamic_meta["industry"] = str(
                    info.get("sectorDisp") or info.get("sector") or defaults["industry"]
                )
                dynamic_meta["sub_sector"] = str(
                    info.get("industryDisp") or info.get("industry") or defaults["sub_sector"]
                )
        except Exception as exc:
            logger.debug("Metadata lookup skipped for %s: %s", ticker_upper, exc)

        # Normalize empty string values.
        merged = {**defaults, **dynamic_meta}
        for key, value in list(merged.items()):
            if value is None or (isinstance(value, str) and not value.strip()):
                merged[key] = defaults[key]

        self._METADATA_CACHE[ticker_upper] = merged
        return dict(merged)

    def _extract_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Extract basic ticker info and determine market (VN vs GLOBAL)."""
        vn_tickers = [
            # Large Cap VN
            "VCB", "BID", "VHM", "VIC", "VNM", "MSN", "SSI", "VND", "HPG", "GVR",
            "GAS", "PLX", "MWG", "PNJ", "DGC", "DPM",
            # Mid Cap VN
            "LPB", "MSB", "VCI", "HCM", "HSG", "PVS", "PVD", "FRT", "DGW", "DCM",
            "CSV", "KDH", "NLG", "TCM",
            # Small Cap VN
            "BVB", "ABB", "DRH", "IDI", "BSI", "FTS", "TVN", "VGS", "PVC", "ASG",
            "BFC", "LAS",
        ]
        ticker_upper = ticker.upper()
        market = "VN" if ticker_upper in vn_tickers else "GLOBAL"

        metadata = self._resolve_company_metadata(ticker_upper, market)

        info: Dict[str, Any] = {
            "ticker": ticker_upper,
            "market": market,
            "exchange": metadata.get("exchange", "NYSE/NASDAQ"),
            "company_name": metadata.get("company_name", "Not available"),
            "industry": metadata.get("industry", "Not available"),
            "sub_sector": metadata.get("sub_sector", "Not available"),
        }
        # Carry static market_cap_usd so generate_full_analysis can use it as fallback.
        static_cap = metadata.get("market_cap_usd")
        if static_cap is not None:
            info["static_market_cap_usd"] = float(static_cap)
        static_class = metadata.get("market_cap_classification")
        if static_class:
            info["static_market_cap_classification"] = static_class
        return info

    def _extract_price_metrics(self, price_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Extract price-based metrics from price DataFrame."""
        if price_df is None or price_df.empty:
            return {}

        df = price_df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

        if df.empty:
            return {}

        latest = df.iloc[-1]
        current_price = float(latest['close']) if 'close' in df.columns else None

        metrics = {
            "current_price": current_price,
            "latest_date": str(latest['date'].date()) if 'date' in df.columns else None,
        }

        # Returns
        if 'daily_return' in df.columns:
            daily_ret = pd.to_numeric(df['daily_return'], errors='coerce').dropna()
            if len(daily_ret) >= 5:
                window = daily_ret.iloc[-5:].to_numpy(dtype=float)
                metrics['return_1w'] = float(np.prod(1.0 + window) - 1.0) * 100.0
            if len(daily_ret) >= 20:
                window = daily_ret.iloc[-20:].to_numpy(dtype=float)
                metrics['return_1m'] = float(np.prod(1.0 + window) - 1.0) * 100.0
            if len(daily_ret) >= 60:
                window = daily_ret.iloc[-60:].to_numpy(dtype=float)
                metrics['return_3m'] = float(np.prod(1.0 + window) - 1.0) * 100.0
            if len(daily_ret) >= 250:
                window = daily_ret.iloc[-250:].to_numpy(dtype=float)
                metrics['return_ytd'] = float(np.prod(1.0 + window) - 1.0) * 100.0

        # Technical indicators
        for col in ['ma20', 'ma50', 'ma200', 'rsi_14', 'volatility_30', 'volatility_60', 'beta', 'var_95', 'var_99', 'max_drawdown', 'sharpe_ratio', 'macd_line', 'bb_upper', 'bb_middle', 'bb_lower']:
            if col in df.columns:
                val = pd.to_numeric(latest[col], errors='coerce')
                if pd.notna(val):
                    metrics[col] = float(val)

        # Price position vs MAs
        if current_price:
            if 'ma20' in metrics:
                metrics['price_vs_ma20'] = "above" if current_price > metrics['ma20'] else "below"
            if 'ma50' in metrics:
                metrics['price_vs_ma50'] = "above" if current_price > metrics['ma50'] else "below"
            if 'ma200' in metrics:
                metrics['price_vs_ma200'] = "above" if current_price > metrics['ma200'] else "below"
            if 'bb_upper' in metrics and 'bb_lower' in metrics:
                if current_price > metrics['bb_upper']:
                    metrics['price_vs_bollinger'] = "above_upper_band"
                elif current_price < metrics['bb_lower']:
                    metrics['price_vs_bollinger'] = "below_lower_band"
                else:
                    metrics['price_vs_bollinger'] = "within_bands"

        return metrics

    def _extract_fundamental_metrics(self, fund_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract 7 core fundamental metrics per Module 4 spec."""
        if fund_df is None or fund_df.empty:
            return {}

        df = fund_df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

        if df.empty:
            return {}

        latest = df.iloc[-1]
        metrics = {}

        # PROFITABILITY (2 metrics)
        for col in ['revenue_growth', 'roe']:
            if col in df.columns:
                val = pd.to_numeric(latest[col], errors='coerce')
                if pd.notna(val):
                    metrics[col] = float(val)

        # ACTIVITY - Cash Conversion Cycle (1 metric)
        if 'cash_conversion_cycle' in df.columns:
            val = pd.to_numeric(latest['cash_conversion_cycle'], errors='coerce')
            if pd.notna(val):
                metrics['cash_conversion_cycle'] = float(val)

        # LIQUIDITY & SOLVENCY (2 metrics)
        for col in ['current_ratio', 'debt_to_equity']:
            if col in df.columns:
                val = pd.to_numeric(latest[col], errors='coerce')
                if pd.notna(val):
                    metrics[col] = float(val)

        # CASH FLOW (2 metrics)
        for col in ['fcff', 'fcfe']:
            if col in df.columns:
                val = pd.to_numeric(latest[col], errors='coerce')
                if pd.notna(val):
                    metrics[col] = float(val)

        # CFO TREND (for Financial Health coverage patch)
        if 'operating_cash_flow' in df.columns:
            cfo_series = pd.to_numeric(df['operating_cash_flow'], errors='coerce').dropna()
            if len(cfo_series) >= 1:
                metrics['cfo_latest'] = float(cfo_series.iloc[-1])
            if len(cfo_series) >= 2:
                prev = float(cfo_series.iloc[-2])
                curr = float(cfo_series.iloc[-1])
                if np.isfinite(prev) and abs(prev) > self._TREND_EPSILON and np.isfinite(curr):
                    metrics['cfo_trend_change_pct'] = ((curr - prev) / abs(prev)) * 100.0
                metrics['cfo_trend_direction'] = self._compute_trend_direction(cfo_series)

        # VALUATION (for Valuation Analysis section - not Financial Health)
        for col in ['pe', 'pb', 'pe_1y_avg', 'pe_5y_avg', 'pb_1y_avg', 'pb_5y_avg', 
                    'pe_industry', 'pb_industry', 'dcf_intrinsic_price', 'dcf_upside']:
            if col in df.columns:
                val = pd.to_numeric(latest[col], errors='coerce')
                if pd.notna(val):
                    metrics[col] = float(val)

        # MARKET INFO (for market cap classification)
        for col in ['market_cap', 'shares_outstanding']:
            if col in df.columns:
                val = pd.to_numeric(latest[col], errors='coerce')
                if pd.notna(val):
                    metrics[col] = float(val)

        return metrics

    def _extract_macro_metrics(self, macro_df: pd.DataFrame, market: str = "VN") -> Dict[str, Any]:
        """Extract macro indicators based on market type."""
        if macro_df is None or macro_df.empty:
            return {}

        df = macro_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)

        if df.empty:
            return {}

        latest = df.iloc[-1]
        metrics = {}

        if market == "VN":
            cols = ['imf_global_growth', 'fed_funds_rate', 'oil_price', 
                   'vn_gdp_growth', 'vn_interest_rate', 'vn_fx_rate', 
                   'vn_fdi_inflow', 'vn_cpi', 'vn_unemployment']
        else:  # GLOBAL
            cols = ['imf_global_growth', 'fed_funds_rate', 'oil_price',
                   'us_gdp_growth', 'us_interest_rate', 'us_fx_rate',
                   'us_fdi_inflow', 'us_cpi', 'us_unemployment']

        for col in cols:
            if col in df.columns:
                val = pd.to_numeric(latest[col], errors='coerce')
                if pd.notna(val):
                    col_series = pd.to_numeric(df[col], errors='coerce')
                    metrics[col] = {
                        "value": float(val),
                        "trend": self._compute_trend_direction(col_series),
                    }

        return metrics

    def _extract_industry_metrics(self, industry_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract industry valuation and profitability metrics."""
        if industry_df is None or industry_df.empty:
            return {}

        df = industry_df.copy()
        if df.empty:
            return {}

        latest = df.iloc[-1] if len(df) > 0 else {}
        metrics = {}

        for col in df.columns:
            if col not in ['date', 'industry', 'ticker']:
                try:
                    val = pd.to_numeric(latest[col], errors='coerce')
                    if pd.notna(val):
                        metrics[col] = float(val)
                except:
                    pass

        return metrics

    def _extract_news_summary(self, news_df: pd.DataFrame, ticker: str) -> Dict[str, list]:
        """Extract recent news/events for ticker."""
        if news_df is None or news_df.empty:
            return {"events": []}

        df = news_df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        if 'ticker' in df.columns:
            ticker_news = df[df['ticker'].str.upper() == ticker.upper()]
        else:
            ticker_news = df

        if ticker_news.empty:
            return {"events": []}

        # Prefer material company news over deal spam, package releases, and ambiguous brand mentions.
        low_signal_patterns = [
            r"\bdeal(s)?\b",
            r"\bfree shipping\b",
            r"\bopen-box\b",
            r"\bdiscount\b",
            r"\ball-time low\b",
            r"\b\d+(?:\.\d+)?\b",
            r"\bpypi\b",
            r"\bsdk\b",
            r"\bcli\b",
            r"\bpackage\b",
            r"\bdashboard\b",
            r"\bplugin\b",
            r"\bmcp\b",
            r"\bapple music\b",
            r"\bmrs\. green apple\b",
        ]
        business_signal_patterns = [
            r"\biphone\b",
            r"\bipad\b",
            r"\bmac(book)?\b",
            r"\bsiri\b",
            r"\bmodem\b",
            r"\bfoldable\b",
            r"\bfoxconn\b",
            r"\btsmc\b",
            r"\bchip(s)?\b",
            r"\bprivacy\b",
            r"\bstreaming\b",
            r"\bapple tv\b",
            r"\barcade\b",
            r"\bransomware\b",
            r"\bmanufacturing\b",
            r"\bsupply chain\b",
            r"\blaunch\b",
            r"\bdelay(ed)?\b",
            r"\bforecast\b",
        ]

        def _news_priority(row: pd.Series) -> tuple:
            headline = str(row.get('headline', '') or '')
            summary = str(row.get('summary', '') or '')
            source = str(row.get('source', '') or '')
            sentiment = str(row.get('sentiment', 'neutral') or 'neutral').lower()
            event_type = str(row.get('event_type', 'general') or 'general').lower()
            text_blob = f"{headline} {summary}".lower()

            is_low_signal = any(re.search(pattern, text_blob) for pattern in low_signal_patterns)
            business_hits = sum(bool(re.search(pattern, text_blob)) for pattern in business_signal_patterns)
            sentiment_bonus = 1 if sentiment in {"positive", "negative"} else 0
            event_bonus = 1 if event_type in {"earnings", "legal", "management_change", "expansion", "m&a"} else 0
            source_penalty = 1 if source.lower() in {"pypi.org", "slickdeals.net", "dealnews.com", "woot", "katalogpromosi.com"} else 0
            low_signal_penalty = 2 if is_low_signal else 0
            timestamp = row.get('date')
            timestamp_value = timestamp.value if pd.notna(timestamp) else 0

            return (
                business_hits + event_bonus + sentiment_bonus - source_penalty - low_signal_penalty,
                event_bonus,
                sentiment_bonus,
                0 if is_low_signal else 1,
                timestamp_value,
            )

        ticker_news = ticker_news.drop_duplicates(subset=['date', 'headline', 'source'], keep='first')
        ticker_news = ticker_news.assign(_priority=ticker_news.apply(_news_priority, axis=1))
        ticker_news = ticker_news.sort_values(by=['_priority', 'date'], ascending=[False, False]).head(10)

        events = []
        for _, row in ticker_news.iterrows():
            event = {
                "date": str(row['date'].date()) if 'date' in row and pd.notna(row['date']) else None,
                "event_type": str(row.get('event_type', 'general')),
                "description": str(row.get('summary') or row.get('description') or row.get('headline', '')),
                "headline": str(row.get('headline', '')),
                "sentiment": str(row.get('sentiment', 'neutral')),
            }
            events.append(event)

        return {"events": events}

    # ============================================================================
    # PROMPT BUILDING PER MODULE 4 SPECIFICATION
    # ============================================================================

    def _build_executive_summary_prompt(
        self,
        ticker: str,
        ticker_b: Optional[str],
        ticker_info: Dict[str, Any],
        price_metrics: Dict[str, Any],
        fundamental_metrics: Dict[str, Any],
        comparison_ticker_info: Optional[Dict[str, Any]] = None,
        comparison_price_metrics: Optional[Dict[str, Any]] = None,
        comparison_fundamental_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build Executive Summary prompt per Module 4."""
        context = {
            "ticker": ticker,
            "basic_info": ticker_info,
            "price_metrics": price_metrics,
            "fundamental_metrics": fundamental_metrics,
            "comparison_ticker": ticker_b,
            "comparison_basic_info": comparison_ticker_info,
            "comparison_price_metrics": comparison_price_metrics,
            "comparison_fundamental_metrics": comparison_fundamental_metrics,
        }

        return f"""You are a professional financial analyst. Generate an Executive Summary for {ticker} following Module 4 specification.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

REQUIRED SECTIONS:

Basic Information:
   Output as labelled lines:
   - Ticker & Exchange: <ticker> / <exchange>
   - Company Name: <name or "Not available">
   - Industry & Sub-sector: <industry> / <subsector>
   - Market Cap Classification: <Large Cap | Mid Cap | Small Cap> (~<USD value> if available)

Summary Statement:
   • Short-term View (1–3 months): [1 sentence on Fundamental signal (revenue trend, profitability, cash flow quality).] [1 sentence on Technical/Valuation signal (momentum, moving averages, RSI, DCF gap).]
   • Long-term View (1–3 years): [1 sentence on Fundamental signal (competitive moat, earnings trajectory, growth outlook).] [1 sentence on Technical/Valuation signal (re-rating potential, valuation vs history and peers).]

   Both views must be written as a SINGLE line each: the header followed immediately by the two sentences on the same line. Do NOT put the sentences on separate lines.

Comparison Snippet (if comparing to {ticker_b}): 2–3 sentences comparing investment profiles.
    - Use the comparison company data in the context when available.
    - Do not say the comparison company has no data if comparison fields are present.
    - Compare fundamentals, valuation, and overall investment profile directly.
     - Omit this section entirely if no comparison ticker or comparison data is available.

Call to Action:
   - Strategic recommendation: Accumulate / Hold / Wait
   - 2–3 sentences on investor suitability.
   - Include disclaimer: "This analysis is AI-generated for informational purposes only and is not investment advice."

RULES:
- Follow the section order exactly: Basic Information -> Summary Statement -> Comparison Snippet (only if available) -> Call to Action.
- Use clean markdown.
- Be data-driven and specific.
- Output language must be English only.
- Use plain ASCII only.
- Put each bullet or labelled line on its own line. Do not merge multiple items into one paragraph.
- Do not number the four Executive Summary section titles.
- Do not add any prefatory sentence before "Basic Information:".
- Prefix Short-term View and Long-term View lines with a bullet character •.
- Write both sentences for each view on the SAME line as the header, after the colon. Do NOT split them to separate lines.
- Do NOT include [Fundamental] or [Technical/Valuation] labels in the output sentences.
"""

    def _build_macro_analysis_prompt(
        self,
        ticker: str,
        market: str,
        macro_metrics: Dict[str, Any],
        industry_metrics: Dict[str, Any],
        news_summary: Dict[str, Any],
    ) -> str:
        """Build Macro Analysis prompt per Module 4."""
        display_macro_metrics = {
            "IMF global GDP growth forecast": {
                "value": self._format_metric_display(macro_metrics.get("imf_global_growth", {}).get("value"), "pct"),
                "trend": macro_metrics.get("imf_global_growth", {}).get("trend", "not_available"),
            },
            "US Federal Reserve policy rate": {
                "value": self._format_metric_display(macro_metrics.get("fed_funds_rate", {}).get("value"), "pct"),
                "trend": macro_metrics.get("fed_funds_rate", {}).get("trend", "not_available"),
            },
            "Global crude oil price level": {
                "value": self._format_metric_display(macro_metrics.get("oil_price", {}).get("value"), "currency"),
                "trend": macro_metrics.get("oil_price", {}).get("trend", "not_available"),
            },
            "US GDP growth rate": {
                "value": self._format_metric_display(macro_metrics.get("us_gdp_growth", {}).get("value"), "pct"),
                "trend": macro_metrics.get("us_gdp_growth", {}).get("trend", "not_available"),
            },
            "US inflation level (CPI)": {
                "value": self._format_metric_display(macro_metrics.get("us_cpi", {}).get("value")),
                "trend": macro_metrics.get("us_cpi", {}).get("trend", "not_available"),
            },
            "Vietnam GDP growth rate": {
                "value": self._format_metric_display(macro_metrics.get("vn_gdp_growth", {}).get("value"), "pct"),
                "trend": macro_metrics.get("vn_gdp_growth", {}).get("trend", "not_available"),
            },
            "Vietnam inflation rate": {
                "value": self._format_metric_display(macro_metrics.get("vn_cpi", {}).get("value"), "pct"),
                "trend": macro_metrics.get("vn_cpi", {}).get("trend", "not_available"),
            },
        }
        display_industry_metrics = {
            "Industry PE": self._format_metric_display(industry_metrics.get("industry_pe")),
            "Industry PB": self._format_metric_display(industry_metrics.get("industry_pb")),
            "Industry PE 1Y avg": self._format_metric_display(industry_metrics.get("industry_pe_1y")),
            "Industry PE 5Y avg": self._format_metric_display(industry_metrics.get("industry_pe_5y")),
            "Industry PB 1Y avg": self._format_metric_display(industry_metrics.get("industry_pb_1y")),
            "Industry PB 5Y avg": self._format_metric_display(industry_metrics.get("industry_pb_5y")),
        }
        context = {
            "ticker": ticker,
            "market": market,
            "display_macro_metrics": display_macro_metrics,
            "display_industry_metrics": display_industry_metrics,
            "recent_news_events": news_summary.get("events", [])[:5],
        }

        if market == "VN":
            indicators_spec = """
OUTPUT FORMAT FOR EACH MACRO METRIC (MANDATORY):
`1. <Metric>: <Value> | Trend: <up/down/flat/not_available> -> <2-3 sentence interpretation>`

GLOBAL INDICATORS (3 metrics):
1. IMF global GDP growth forecast
2. US Federal Reserve policy rate
3. Global crude oil price level

VIETNAM DOMESTIC INDICATORS (2 metrics):
4. Vietnam GDP growth rate
5. Vietnam inflation rate

RULES:
- Every metric must have 2-3 sentence interpretation.
- No standalone numbers.
- Be factual, specific, and data-driven.
- Output language must be English only.
- Keep the numbered order exactly as listed.
"""
        else:  # GLOBAL
            indicators_spec = """
OUTPUT FORMAT FOR EACH MACRO METRIC (MANDATORY):
`1. <Metric>: <Value> | Trend: <up/down/flat/not_available> -> <2-3 sentence interpretation>`

GLOBAL & US INDICATORS (5 metrics):
1. IMF global GDP growth forecast
2. US Federal Reserve policy rate
3. Global crude oil price level
4. US GDP growth rate
5. US inflation level (CPI)

INDUSTRY VALUATION & PROFITABILITY:
Output as bullet points followed by a 1-sentence comparative conclusion:

- P/E Industry 1Y avg: {Industry PE 1Y avg from display_industry_metrics}
- P/E Industry 5Y avg: {Industry PE 5Y avg from display_industry_metrics}
  -> 1 sentence comparing P/E 1Y vs P/E 5Y and concluding which valuation stage the industry is currently in.

- P/B Industry 1Y avg: {Industry PB 1Y avg from display_industry_metrics}
- P/B Industry 5Y avg: {Industry PB 5Y avg from display_industry_metrics}
  -> 1 sentence comparing P/B 1Y vs P/B 5Y and concluding which valuation stage the industry is currently in.

CORPORATE EVENTS & NEWS:
- For each detected event in recent_news_events, output a numbered line with the event type and a 2-3 sentence interpretation of its likely impact on stock price, sentiment, or fundamentals.
- If no events are available, state that no material recent events were provided.
- If recent_news_events is non-empty, do not say that no material recent events were provided.
- Use the supplied headline and description text to assess materiality; prioritize company-specific developments over generic promotions, shopping deals, and package release notices.

RULES:
- Every metric must have 2-3 sentence interpretation.
- No standalone numbers.
- Be factual, specific, and data-driven.
- Output language must be English only.
- Keep the numbered order exactly as listed.
- Use the metric names exactly as written above, not raw source keys.
- Use the rounded values from CONTEXT display_macro_metrics and display_industry_metrics.
"""

        return f"""You are a macroeconomic analyst. Generate Macro Analysis for {ticker} ({market} market) per Module 4 spec.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

{indicators_spec}

RULES:
- Every value must have 2-3 sentence interpretation.
- No standalone numbers.
- Be factual, specific, and data-driven.
- Output language must be English only.
- Do not use markdown math, LaTeX, or special unicode separators.
- Use plain ASCII text only.
- Output in clean markdown.
- Put every numbered metric on its own line, followed by its interpretation on the same line or the next sentence block.
- Separate major sub-sections with a blank line.
- Use the friendly labels and rounded values from CONTEXT rather than raw metric keys or long float strings.
"""

    def _build_financial_health_prompt(
        self,
        ticker: str,
        fundamental_metrics: Dict[str, Any],
    ) -> str:
        """Build Financial Health prompt (7 core metrics per Module 4)."""
        exact_metrics = {
            "Revenue Growth (YoY)": fundamental_metrics.get("revenue_growth"),
            "ROE": fundamental_metrics.get("roe"),
            "Cash Conversion Cycle (CCC)": fundamental_metrics.get("cash_conversion_cycle"),
            "Current Ratio": fundamental_metrics.get("current_ratio"),
            "Debt-to-Equity (D/E)": fundamental_metrics.get("debt_to_equity"),
            "FCFF": fundamental_metrics.get("fcff"),
            "FCFE": fundamental_metrics.get("fcfe"),
        }
        display_metrics = {
            "Revenue Growth (YoY)": self._format_metric_display(fundamental_metrics.get("revenue_growth"), "ratio_pct"),
            "ROE": self._format_metric_display(fundamental_metrics.get("roe"), "ratio_pct"),
            "Cash Conversion Cycle (CCC)": self._format_metric_display(fundamental_metrics.get("cash_conversion_cycle"), "days"),
            "Current Ratio": self._format_metric_display(fundamental_metrics.get("current_ratio")),
            "Debt-to-Equity (D/E)": self._format_metric_display(fundamental_metrics.get("debt_to_equity")),
            "FCFF": self._format_metric_display(fundamental_metrics.get("fcff"), "currency"),
            "FCFE": self._format_metric_display(fundamental_metrics.get("fcfe"), "currency"),
        }
        context = {
            "ticker": ticker,
            "exact_metrics": exact_metrics,
            "display_metrics": display_metrics,
        }

        return f"""You are a fundamental analyst. Analyze {ticker}'s Financial Health per Module 4 spec.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

STRICT OUTPUT FORMAT:
- Keep numbering from 1 to 24.
- For items 1 and 2 (P/E and P/B), use the nested bullet format exactly as specified below.
- For items 3 to 24, use: `N. <Metric>: <Value> -> <2-3 sentence interpretation>`

PROFITABILITY:
1. Revenue Growth (YoY) - Interpret acceleration/deceleration vs industry

2. ROE - Interpret capital efficiency and sustainability

ACTIVITY RATIOS:
3. Cash Conversion Cycle (CCC) - Interpret working capital efficiency vs industry norms

LIQUIDITY & SOLVENCY:
4. Current Ratio - Interpret short-term liquidity adequacy and buffer against operational stress

5. Debt-to-Equity (D/E) - Interpret leverage risk and impact on cost of capital

CASH FLOW:
6. FCFF - Interpret firm-level free cash flow capacity and reinvestment headroom

7. FCFE - Interpret equity cash flow available for dividends, buybacks, or growth

CONCLUSION:
End with "Overall Financial Health Conclusion" (2-3 sentences on balance sheet strength, earnings quality, cash flow profile vs peers).

RULES:
- Every metric must have analytical interpretation.
- Never output values without interpretation.
- Be specific and numerical.
- Output language must be English only.
- Use plain ASCII only.
- Keep the numbered order exactly as listed for the seven metrics.
- Write the conclusion as a separate labelled paragraph after the seven numbered metrics.
- Do not add extra metrics outside these seven items and the conclusion.
- Put each numbered metric on its own line.
- Use the values in display_metrics verbatim for the metric labels unless the spec explicitly asks for a different unit.
- Reproduce the exact numeric meaning from CONTEXT. Do not substitute or invent alternative values.
- Do not add a preamble before metric 1.
"""

    def _build_valuation_analysis_prompt(
        self,
        ticker: str,
        price_metrics: Dict[str, Any],
        fundamental_metrics: Dict[str, Any],
        industry_metrics: Dict[str, Any],
    ) -> str:
        """Build Valuation Analysis prompt (24 metrics per Module 4)."""
        sanitized_fundamental_metrics = {
            key: value
            for key, value in fundamental_metrics.items()
            if key not in {"pe_industry", "pb_industry"}
        }
        exact_valuation_metrics = {
            "Current P/E": fundamental_metrics.get("pe"),
            "P/E 1Y avg": fundamental_metrics.get("pe_1y_avg"),
            "P/E 5Y avg": fundamental_metrics.get("pe_5y_avg"),
            "Industry P/E": industry_metrics.get("industry_pe", fundamental_metrics.get("pe_industry")),
            "Current P/B": fundamental_metrics.get("pb"),
            "P/B 1Y avg": fundamental_metrics.get("pb_1y_avg"),
            "P/B 5Y avg": fundamental_metrics.get("pb_5y_avg"),
            "Industry P/B": industry_metrics.get("industry_pb", fundamental_metrics.get("pb_industry")),
            "DCF Intrinsic Price": fundamental_metrics.get("dcf_intrinsic_price"),
            "DCF Market Price": price_metrics.get("current_price"),
            "DCF Upside/Downside %": fundamental_metrics.get("dcf_upside"),
            "DCF Valuation Status": "Overvalued" if self._normalize_coverage_value(fundamental_metrics.get("dcf_upside")) not in {"N/A", None} and float(fundamental_metrics.get("dcf_upside")) < -0.05 else "Undervalued" if self._normalize_coverage_value(fundamental_metrics.get("dcf_upside")) not in {"N/A", None} and float(fundamental_metrics.get("dcf_upside")) > 0.05 else "Fairly Valued",
        }
        display_valuation_metrics = {
            "Current P/E": self._format_metric_display(exact_valuation_metrics["Current P/E"]),
            "P/E 1Y avg": self._format_metric_display(exact_valuation_metrics["P/E 1Y avg"]),
            "P/E 5Y avg": self._format_metric_display(exact_valuation_metrics["P/E 5Y avg"]),
            "Industry P/E": self._format_metric_display(exact_valuation_metrics["Industry P/E"]),
            "Current P/B": self._format_metric_display(exact_valuation_metrics["Current P/B"]),
            "P/B 1Y avg": self._format_metric_display(exact_valuation_metrics["P/B 1Y avg"]),
            "P/B 5Y avg": self._format_metric_display(exact_valuation_metrics["P/B 5Y avg"]),
            "Industry P/B": self._format_metric_display(exact_valuation_metrics["Industry P/B"]),
            "DCF Intrinsic Price": self._format_metric_display(exact_valuation_metrics["DCF Intrinsic Price"], "price"),
            "DCF Market Price": self._format_metric_display(exact_valuation_metrics["DCF Market Price"], "price"),
            "DCF Upside/Downside %": self._format_metric_display(exact_valuation_metrics["DCF Upside/Downside %"], "ratio_pct"),
            "DCF Valuation Status": exact_valuation_metrics["DCF Valuation Status"],
            "Current Price": self._format_metric_display(price_metrics.get("current_price"), "price"),
            "1W Return %": self._format_metric_display(price_metrics.get("return_1w"), "pct"),
            "1M Return %": self._format_metric_display(price_metrics.get("return_1m"), "pct"),
            "3M Return %": self._format_metric_display(price_metrics.get("return_3m"), "pct"),
            "YTD Return %": self._format_metric_display(price_metrics.get("return_ytd"), "pct"),
            "MA20": self._format_metric_display(price_metrics.get("ma20"), "price"),
            "MA50": self._format_metric_display(price_metrics.get("ma50"), "price"),
            "MA200": self._format_metric_display(price_metrics.get("ma200"), "price"),
            "RSI(14)": self._format_metric_display(price_metrics.get("rsi_14")),
            "MACD": self._format_metric_display(price_metrics.get("macd_line")),
            "Bollinger Upper": self._format_metric_display(price_metrics.get("bb_upper"), "price"),
            "Bollinger Middle": self._format_metric_display(price_metrics.get("bb_middle"), "price"),
            "Bollinger Lower": self._format_metric_display(price_metrics.get("bb_lower"), "price"),
            "Historical Volatility 30D %": self._format_metric_display(price_metrics.get("volatility_30"), "ratio_pct"),
            "Historical Volatility 60D %": self._format_metric_display(price_metrics.get("volatility_60"), "ratio_pct"),
            "Beta vs Index": self._format_metric_display(price_metrics.get("beta")),
            "VaR 95% daily": self._format_metric_display(price_metrics.get("var_95"), "ratio_pct"),
            "VaR 99% daily": self._format_metric_display(price_metrics.get("var_99"), "ratio_pct"),
            "Max Drawdown %": self._format_metric_display(price_metrics.get("max_drawdown"), "ratio_pct"),
            "Sharpe Ratio": self._format_metric_display(price_metrics.get("sharpe_ratio")),
        }
        context = {
            "ticker": ticker,
            "price_metrics": price_metrics,
            "fundamental_metrics": sanitized_fundamental_metrics,
            "industry_metrics": industry_metrics,
            "exact_valuation_metrics": exact_valuation_metrics,
            "display_valuation_metrics": display_valuation_metrics,
        }

        return f"""You are a valuation analyst. Generate comprehensive Valuation Analysis for {ticker} per Module 4 spec.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

STRICT OUTPUT FORMAT FOR EACH METRIC:
`1. <Metric>: <Value> -> <2-3 sentence interpretation>`

FUNDAMENTAL VALUATION (3 metrics):
1. P/E
- P/E 1Y: use display_valuation_metrics["P/E 1Y avg"]
- P/E 5Y: use display_valuation_metrics["P/E 5Y avg"]
- P/E Industry: use display_valuation_metrics["Industry P/E"]
-> Write exactly 2 interpretation sentences: (a) compare P/E 1Y vs P/E 5Y, (b) compare P/E 1Y vs Industry P/E; then conclude with one of: Undervalued / Fairly Valued / Overvalued.

2. P/B
- P/B 1Y: use display_valuation_metrics["P/B 1Y avg"]
- P/B 5Y: use display_valuation_metrics["P/B 5Y avg"]
- P/B Industry: use display_valuation_metrics["Industry P/B"]
-> Write exactly 2 interpretation sentences: (a) compare P/B 1Y vs P/B 5Y, (b) compare P/B 1Y vs Industry P/B; then conclude with one of: Undervalued / Fairly Valued / Overvalued.

3. DCF Valuation (FCFE-Based): Intrinsic Price, Market Price, Upside/Downside %, Valuation Status (Undervalued / Fairly Valued / Overvalued) - Interpret reliability and margin of safety

TECHNICAL ANALYSIS - TREND SUMMARY (5 metrics):
4. Current Price - Interpret vs 12-month range and market structure
5. 1W Return % - Interpret short-term momentum
6. 1M Return % - Interpret 1-month price action and sentiment
7. 3M Return % - Interpret medium-term trend
8. YTD Return % vs Index - Interpret relative performance; if the benchmark return is missing, explicitly say the benchmark comparison is unavailable instead of inventing one

MOVING AVERAGES & OSCILLATORS (6 metrics):
9. MA20: Price vs MA20 - Interpret short-term momentum
10. MA50: Price vs MA50 - Interpret medium-term trend
11. MA200: Price vs MA200 - Interpret long-term trend status
12. RSI(14) - Interpret momentum condition (overbought/neutral/oversold)
13. MACD - Interpret bullish/bearish signal
14. Bollinger Bands (price vs upper/middle/lower) - Evaluate volatility state, potential mean reversion, and breakout risk

PRICE & VOLUME ANOMALIES (3 metrics):
15. Volume Spike - Explain likely cause and supply/demand implications
16. Gap Up/Gap Down - Explain trigger and fill status
17. Sudden Price Movement - Explain probable cause and trend implication

RISK METRICS (7 metrics):
18. Historical Volatility 30D % - Interpret risk vs sector peers
19. Historical Volatility 60D % - Interpret medium-term stability
20. Beta vs Index - Interpret market sensitivity
21. VaR 95% daily - Interpret expected max daily loss 95% of time
22. VaR 99% daily - Interpret tail-risk exposure
23. Max Drawdown % - Interpret historical worst-case loss
24. Sharpe Ratio - Interpret risk-adjusted return efficiency

CONCLUSION:
End with "Technical Conclusion" (2-3 sentences on trend structure, momentum, recommended approach).

RULES:
- Every metric MUST include analytical interpretation.
- No naked numbers.
- Keep the numbered order from 1 to 24 exactly as listed.
- Be specific and data-driven.
- Output language must be English only.
- Use plain ASCII only.
- Do not use markdown math, LaTeX, or fragmented digit formatting.
- If 12-month range, volume anomaly, gap, or sudden-move data is unavailable, state that explicitly and still provide interpretation of the missing-data implication.
- Write "Technical Conclusion" as a separate labelled paragraph after the numbered metrics.
- Do not merge numbered items. Keep each numbered metric block distinct.
- For items 1 and 2, keep the metric title plus its sub-bullets and interpretation together as one block.
- Separate the Fundamental Valuation, Technical Analysis, Price & Volume Anomalies, and Risk Metrics blocks with blank lines.
- Use the values in display_valuation_metrics verbatim for the valuation lines, including the explicit DCF Valuation Status field.
- Use the values in display_valuation_metrics verbatim for the technical and risk metric labels as well, especially percentage-formatted volatility, VaR, and max drawdown values.
- Reproduce valuation numbers exactly from CONTEXT, especially Current P/E, Current P/B, industry averages, DCF intrinsic price, current market price, and upside/downside.
- Use the industry averages from industry_metrics when present; do not invent or substitute alternative industry averages.
- Do not add a preamble such as "Here is a comprehensive Valuation Analysis".
"""

    def _build_peer_comparison_prompt(
        self,
        ticker_a: str,
        ticker_b: str,
        fund_a: Dict[str, Any],
        fund_b: Dict[str, Any],
        price_a: Dict[str, Any],
        price_b: Dict[str, Any],
        industry_metrics: Dict[str, Any],
    ) -> str:
        """Build Peer Comparison prompt per Module 4 spec (simplified)."""
        sanitized_fund_a = {
            key: value
            for key, value in fund_a.items()
            if key not in {"pe_industry", "pb_industry"}
        }
        sanitized_fund_b = {
            key: value
            for key, value in fund_b.items()
            if key not in {"pe_industry", "pb_industry"}
        }
        exact_comparison_metrics = {
            "industry_pe": industry_metrics.get("industry_pe", fund_a.get("pe_industry")),
            "industry_pb": industry_metrics.get("industry_pb", fund_a.get("pb_industry")),
            ticker_a: {
                "revenue_growth": fund_a.get("revenue_growth"),
                "roe": fund_a.get("roe"),
                "current_ratio": fund_a.get("current_ratio"),
                "debt_to_equity": fund_a.get("debt_to_equity"),
                "fcfe": fund_a.get("fcfe"),
                "pe": fund_a.get("pe"),
                "pb": fund_a.get("pb"),
                "dcf_intrinsic_price": fund_a.get("dcf_intrinsic_price"),
                "dcf_upside": fund_a.get("dcf_upside"),
                "market_price": price_a.get("current_price"),
                "macd": price_a.get("macd_line"),
                "rsi": price_a.get("rsi_14"),
                "volatility_30": price_a.get("volatility_30"),
                "max_drawdown": price_a.get("max_drawdown"),
                "sharpe_ratio": price_a.get("sharpe_ratio"),
            },
            ticker_b: {
                "revenue_growth": fund_b.get("revenue_growth"),
                "roe": fund_b.get("roe"),
                "current_ratio": fund_b.get("current_ratio"),
                "debt_to_equity": fund_b.get("debt_to_equity"),
                "fcfe": fund_b.get("fcfe"),
                "pe": fund_b.get("pe"),
                "pb": fund_b.get("pb"),
                "dcf_intrinsic_price": fund_b.get("dcf_intrinsic_price"),
                "dcf_upside": fund_b.get("dcf_upside"),
                "market_price": price_b.get("current_price"),
                "macd": price_b.get("macd_line"),
                "rsi": price_b.get("rsi_14"),
                "volatility_30": price_b.get("volatility_30"),
                "max_drawdown": price_b.get("max_drawdown"),
                "sharpe_ratio": price_b.get("sharpe_ratio"),
            },
        }
        display_comparison_metrics = {
            "industry_pe": self._format_metric_display(exact_comparison_metrics["industry_pe"]),
            "industry_pb": self._format_metric_display(exact_comparison_metrics["industry_pb"]),
            ticker_a: {
                "revenue_growth": self._format_metric_display(exact_comparison_metrics[ticker_a]["revenue_growth"], "ratio_pct"),
                "roe": self._format_metric_display(exact_comparison_metrics[ticker_a]["roe"], "ratio_pct"),
                "current_ratio": self._format_metric_display(exact_comparison_metrics[ticker_a]["current_ratio"]),
                "debt_to_equity": self._format_metric_display(exact_comparison_metrics[ticker_a]["debt_to_equity"]),
                "fcfe": self._format_metric_display(exact_comparison_metrics[ticker_a]["fcfe"], "currency"),
                "pe": self._format_metric_display(exact_comparison_metrics[ticker_a]["pe"]),
                "pb": self._format_metric_display(exact_comparison_metrics[ticker_a]["pb"]),
                "dcf_intrinsic_price": self._format_metric_display(exact_comparison_metrics[ticker_a]["dcf_intrinsic_price"], "price"),
                "dcf_upside": self._format_metric_display(exact_comparison_metrics[ticker_a]["dcf_upside"], "ratio_pct"),
                "market_price": self._format_metric_display(exact_comparison_metrics[ticker_a]["market_price"], "price"),
                "macd": self._format_metric_display(exact_comparison_metrics[ticker_a]["macd"]),
                "rsi": self._format_metric_display(exact_comparison_metrics[ticker_a]["rsi"]),
                "volatility_30": self._format_metric_display(exact_comparison_metrics[ticker_a]["volatility_30"], "ratio_pct"),
                "max_drawdown": self._format_metric_display(exact_comparison_metrics[ticker_a]["max_drawdown"], "ratio_pct"),
                "sharpe_ratio": self._format_metric_display(exact_comparison_metrics[ticker_a]["sharpe_ratio"]),
            },
            ticker_b: {
                "revenue_growth": self._format_metric_display(exact_comparison_metrics[ticker_b]["revenue_growth"], "ratio_pct"),
                "roe": self._format_metric_display(exact_comparison_metrics[ticker_b]["roe"], "ratio_pct"),
                "current_ratio": self._format_metric_display(exact_comparison_metrics[ticker_b]["current_ratio"]),
                "debt_to_equity": self._format_metric_display(exact_comparison_metrics[ticker_b]["debt_to_equity"]),
                "fcfe": self._format_metric_display(exact_comparison_metrics[ticker_b]["fcfe"], "currency"),
                "pe": self._format_metric_display(exact_comparison_metrics[ticker_b]["pe"]),
                "pb": self._format_metric_display(exact_comparison_metrics[ticker_b]["pb"]),
                "dcf_intrinsic_price": self._format_metric_display(exact_comparison_metrics[ticker_b]["dcf_intrinsic_price"], "price"),
                "dcf_upside": self._format_metric_display(exact_comparison_metrics[ticker_b]["dcf_upside"], "ratio_pct"),
                "market_price": self._format_metric_display(exact_comparison_metrics[ticker_b]["market_price"], "price"),
                "macd": self._format_metric_display(exact_comparison_metrics[ticker_b]["macd"]),
                "rsi": self._format_metric_display(exact_comparison_metrics[ticker_b]["rsi"]),
                "volatility_30": self._format_metric_display(exact_comparison_metrics[ticker_b]["volatility_30"], "ratio_pct"),
                "max_drawdown": self._format_metric_display(exact_comparison_metrics[ticker_b]["max_drawdown"], "ratio_pct"),
                "sharpe_ratio": self._format_metric_display(exact_comparison_metrics[ticker_b]["sharpe_ratio"]),
            },
        }
        context = {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "fundamental_a": sanitized_fund_a,
            "fundamental_b": sanitized_fund_b,
            "price_a": price_a,
            "price_b": price_b,
            "industry_metrics": industry_metrics,
            "exact_comparison_metrics": exact_comparison_metrics,
            "display_comparison_metrics": display_comparison_metrics,
        }

        return f"""You are a comparative equity analyst. Compare {ticker_a} vs {ticker_b} per Module 4 spec.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

GENERATE FOLLOWING SECTIONS:

1. Financial Health Comparison (5 metrics):
- Revenue Growth (YoY)
- ROE
- Current Ratio
- Debt-to-Equity (D/E)
- FCFE

Highlight which company has: better financial health, stronger profitability, better liquidity, healthier cash flow.

2. Fundamental Valuation Comparison (3 metrics):
- Current P/E vs Industry avg
- Current P/B vs Industry avg
- DCF Valuation (Intrinsic Price vs Market Price, Upside/Downside %)

Explain which stock appears undervalued/fairly valued/expensive. Discuss valuation attractiveness in current market.

3. Technical & Risk Profile Comparison (5 metrics):
- MACD
- RSI(14)
- Historical Volatility 30D
- Max Drawdown
- Sharpe Ratio

Explain differences in: price momentum, volatility/risk profile, defensive vs cyclical characteristics, trading sentiment.

4. Comparison Summary:
Position {ticker_a} vs {ticker_b} under current market conditions.
Identify which stock suits: growth-oriented investors / value-focused investors.
Summarize key differentiating factors driving recommendation.

RULES:
- Use concrete numbers from context.
- Use numbered lines for the comparison metrics and keep each metric on its own line.
- Use format: `1. <Metric>: <A> vs <B> -> <2-3 sentence interpretation>`
- Output language: English only.
- Use plain ASCII only.
- Keep the section order exactly as specified.
- Keep the section number and title on the same line, e.g. "1. Financial Health Comparison".
- Use the values in display_comparison_metrics verbatim for the side-by-side metric labels and shared industry averages.
- Reproduce the exact metric values from CONTEXT for both companies and for the shared industry averages.
- Do not add a preamble such as "Here is a comparative equity analysis".
"""

    # ============================================================================
    # MAIN ANALYSIS GENERATION
    # ============================================================================

    def generate_full_analysis(
        self,
        ticker_a: str,
        price_df_a: pd.DataFrame,
        fundamental_df_a: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        industry_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
        ticker_b: Optional[str] = None,
        price_df_b: Optional[pd.DataFrame] = None,
        fundamental_df_b: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive 5-section financial analysis per Module 4.

        Parameters
        ----------
        ticker_a : str
            Primary ticker to analyze
        price_df_a : pd.DataFrame
            Price data for ticker_a
        fundamental_df_a : pd.DataFrame
            Fundamental metrics for ticker_a
        macro_df : pd.DataFrame, optional
            Macroeconomic indicators
        industry_df : pd.DataFrame, optional
            Industry metrics
        news_df : pd.DataFrame, optional
            News/events data
        ticker_b : str, optional
            Comparison ticker (peer)
        price_df_b : pd.DataFrame, optional
            Price data for ticker_b
        fundamental_df_b : pd.DataFrame, optional
            Fundamental metrics for ticker_b

        Returns
        -------
        dict
            Report with 5 main sections + metadata
        """
        logger.info(f"Generating Module 4 full analysis for {ticker_a}...")

        # Extract all metrics
        ticker_info_a = self._extract_ticker_info(ticker_a)
        market = ticker_info_a.get("market", "GLOBAL")
        
        price_metrics_a = self._extract_price_metrics(price_df_a, ticker_a)
        fundamental_metrics_a = self._extract_fundamental_metrics(fundamental_df_a)

        market_cap_usd = None
        if 'market_cap' in fundamental_metrics_a:
            market_cap_usd = float(fundamental_metrics_a['market_cap'])
        elif 'shares_outstanding' in fundamental_metrics_a and 'current_price' in price_metrics_a:
            market_cap_usd = float(fundamental_metrics_a['shares_outstanding']) * float(price_metrics_a['current_price'])
        # Fallback to static map value when CSV data is missing.
        if market_cap_usd is None:
            market_cap_usd = ticker_info_a.get('static_market_cap_usd')

        ticker_info_a.update(self._classify_market_cap(market_cap_usd))
        macro_metrics = self._extract_macro_metrics(macro_df, market) if macro_df is not None else {}
        industry_metrics = self._extract_industry_metrics(industry_df) if industry_df is not None else {}
        news_summary = self._extract_news_summary(news_df, ticker_a) if news_df is not None else {"events": []}

        price_metrics_b = {}
        fundamental_metrics_b = {}
        if ticker_b and price_df_b is not None and fundamental_df_b is not None:
            price_metrics_b = self._extract_price_metrics(price_df_b, ticker_b)
            fundamental_metrics_b = self._extract_fundamental_metrics(fundamental_df_b)

        report = {}

        # 1. EXECUTIVE SUMMARY
        logger.info("Generating Executive Summary...")
        try:
            ticker_info_b = None
            if ticker_b and price_df_b is not None and fundamental_df_b is not None:
                ticker_info_b = self._extract_ticker_info(ticker_b)

            prompt = self._build_executive_summary_prompt(
                ticker_a,
                ticker_b,
                ticker_info_a,
                price_metrics_a,
                fundamental_metrics_a,
                ticker_info_b,
                price_metrics_b if price_metrics_b else None,
                fundamental_metrics_b if fundamental_metrics_b else None,
            )
            exec_summary = self._call_llm(self._SYSTEM_INSTRUCTION, prompt)
            report['executive_summary'] = self._sanitize_llm_output(exec_summary)
            logger.info("✓ Executive Summary generated")
        except Exception as exc:
            logger.error(f"Executive Summary generation failed: {exc}")
            report['executive_summary'] = f"Error generating Executive Summary: {exc}"

        # 2. MACRO ANALYSIS
        logger.info("Generating Macro Analysis...")
        try:
            prompt = self._build_macro_analysis_prompt(
                ticker_a, market, macro_metrics, industry_metrics, news_summary
            )
            macro_analysis = self._call_llm(self._SYSTEM_INSTRUCTION, prompt)
            cleaned_macro = self._sanitize_llm_output(macro_analysis)
            report['macro_analysis'] = self._ensure_macro_metric_coverage(
                cleaned_macro, ticker_a, market, macro_metrics
            )
            logger.info("✓ Macro Analysis generated")
        except Exception as exc:
            logger.error(f"Macro Analysis generation failed: {exc}")
            report['macro_analysis'] = f"Error generating Macro Analysis: {exc}"

        # 3. FINANCIAL HEALTH
        logger.info("Generating Financial Health...")
        try:
            prompt = self._build_financial_health_prompt(ticker_a, fundamental_metrics_a)
            financial_health = self._call_llm(self._SYSTEM_INSTRUCTION, prompt)
            report['financial_health'] = self._sanitize_llm_output(financial_health)
            logger.info("✓ Financial Health generated")
        except Exception as exc:
            logger.error(f"Financial Health generation failed: {exc}")
            report['financial_health'] = f"Error generating Financial Health: {exc}"

        # 4. VALUATION ANALYSIS
        logger.info("Generating Valuation Analysis...")
        try:
            prompt = self._build_valuation_analysis_prompt(
                ticker_a, price_metrics_a, fundamental_metrics_a, industry_metrics
            )
            valuation_analysis = self._call_llm(self._SYSTEM_INSTRUCTION, prompt)
            report['valuation_analysis'] = self._ensure_valuation_structure(
                self._sanitize_llm_output(valuation_analysis)
            )
            logger.info("✓ Valuation Analysis generated")
        except Exception as exc:
            logger.error(f"Valuation Analysis generation failed: {exc}")
            report['valuation_analysis'] = f"Error generating Valuation Analysis: {exc}"

        # 5. PEER COMPARISON (if ticker_b provided)
        if ticker_b and price_df_b is not None and fundamental_df_b is not None:
            logger.info(f"Generating Peer Comparison: {ticker_a} vs {ticker_b}...")
            try:
                prompt = self._build_peer_comparison_prompt(
                    ticker_a, ticker_b, fundamental_metrics_a, fundamental_metrics_b,
                    price_metrics_a, price_metrics_b, industry_metrics
                )
                peer_comparison = self._call_llm(self._SYSTEM_INSTRUCTION, prompt)
                report['peer_comparison'] = self._sanitize_llm_output(peer_comparison)
                logger.info("✓ Peer Comparison generated")
            except Exception as exc:
                logger.error(f"Peer Comparison generation failed: {exc}")
                report['peer_comparison'] = f"Error generating Peer Comparison: {exc}"
        else:
            report['peer_comparison'] = "Peer comparison skipped because ticker_b or comparison data is missing."

        report['ticker_a'] = ticker_a
        report['market'] = market
        report['temperature'] = self.temperature
        report['max_tokens'] = self.max_tokens

        logger.info(f"✓ Full Module 4 analysis for {ticker_a} completed successfully.")
        return report

    def run_full_analysis(
        self,
        ticker_a: str,
        price_df_a: pd.DataFrame,
        fundamental_df_a: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        industry_df: Optional[pd.DataFrame] = None,
        news_df: Optional[pd.DataFrame] = None,
        ticker_b: Optional[str] = None,
        price_df_b: Optional[pd.DataFrame] = None,
        fundamental_df_b: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper for Module 4 analysis."""
        return self.generate_full_analysis(
            ticker_a, price_df_a, fundamental_df_a, macro_df, industry_df, news_df,
            ticker_b, price_df_b, fundamental_df_b
        )


# Backward-compatible alias
AIAgent = AnalysisAgent

