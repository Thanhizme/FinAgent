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
        # US Large Cap
        "AAPL": {"company_name": "Apple Inc.", "exchange": "NASDAQ", "industry": "Information Technology", "sub_sector": "Technology Hardware"},
        "MSFT": {"company_name": "Microsoft Corporation", "exchange": "NASDAQ", "industry": "Information Technology", "sub_sector": "Systems Software"},
        "AMZN": {"company_name": "Amazon.com, Inc.", "exchange": "NASDAQ", "industry": "Consumer Discretionary", "sub_sector": "Internet Retail"},
        "TSLA": {"company_name": "Tesla, Inc.", "exchange": "NASDAQ", "industry": "Consumer Discretionary", "sub_sector": "Automobile Manufacturers"},
        "JPM": {"company_name": "JPMorgan Chase & Co.", "exchange": "NYSE", "industry": "Financials", "sub_sector": "Diversified Banks"},
        "V": {"company_name": "Visa Inc.", "exchange": "NYSE", "industry": "Financials", "sub_sector": "Payment Processing"},
        "GOOGL": {"company_name": "Alphabet Inc.", "exchange": "NASDAQ", "industry": "Communication Services", "sub_sector": "Internet Services"},
        "META": {"company_name": "Meta Platforms Inc.", "exchange": "NASDAQ", "industry": "Communication Services", "sub_sector": "Internet Services"},
        "XOM": {"company_name": "Exxon Mobil Corporation", "exchange": "NYSE", "industry": "Energy", "sub_sector": "Oil & Gas"},
        "CVX": {"company_name": "Chevron Corporation", "exchange": "NYSE", "industry": "Energy", "sub_sector": "Oil & Gas"},
        
        # US Mid Cap
        "CROX": {"company_name": "Crocs Inc.", "exchange": "NASDAQ", "industry": "Consumer Discretionary", "sub_sector": "Footwear"},
        "DECK": {"company_name": "Deckers Outdoor Corporation", "exchange": "NYSE", "industry": "Consumer Discretionary", "sub_sector": "Footwear"},
        "BOOT": {"company_name": "Boot Barn Holdings Inc.", "exchange": "NYSE", "industry": "Consumer Discretionary", "sub_sector": "Specialty Retail"},
        "OVV": {"company_name": "Ovintiv Inc.", "exchange": "NYSE", "industry": "Energy", "sub_sector": "Oil & Gas"},
        "APA": {"company_name": "Apache Corporation", "exchange": "NASDAQ", "industry": "Energy", "sub_sector": "Oil & Gas"},
        "AWR": {"company_name": "American Water Works Company Inc.", "exchange": "NYSE", "industry": "Utilities", "sub_sector": "Water Utilities"},
        "AVA": {"company_name": "Avista Corporation", "exchange": "NYSE", "industry": "Utilities", "sub_sector": "Diversified Utilities"},
        
        # US Small Cap
        "SONO": {"company_name": "Sonos Inc.", "exchange": "NASDAQ", "industry": "Consumer Discretionary", "sub_sector": "Consumer Electronics"},
        "KLIC": {"company_name": "Kulicke & Soffa Industries Inc.", "exchange": "NASDAQ", "industry": "Information Technology", "sub_sector": "Semiconductors"},
        "HIMS": {"company_name": "Hims & Hers Health Inc.", "exchange": "NYSE", "industry": "Health Care", "sub_sector": "Health Care Services"},
        
        # VN Large Cap
        "VCB": {"company_name": "Vietcombank", "exchange": "HOSE", "industry": "Ngân hàng", "sub_sector": "Ngân hàng Thương mại"},
        "BID": {"company_name": "BIDV", "exchange": "HOSE", "industry": "Ngân hàng", "sub_sector": "Ngân hàng Thương mại"},
        "VHM": {"company_name": "Vinhomes", "exchange": "HOSE", "industry": "Bất động sản", "sub_sector": "Phát triển Bất động sản"},
        "VIC": {"company_name": "Vingroup", "exchange": "HOSE", "industry": "Conglomerate", "sub_sector": "Đa ngành"},
        "VNM": {"company_name": "Vinamilk", "exchange": "HOSE", "industry": "Thực phẩm & Đồ uống", "sub_sector": "Thực phẩm Đã chế biến"},
        "GAS": {"company_name": "PV Gas", "exchange": "HOSE", "industry": "Dầu khí", "sub_sector": "Dầu khí & Gas"},
        
        # VN Mid Cap
        "KDH": {"company_name": "Kien Hung Development", "exchange": "HOSE", "industry": "Bất động sản", "sub_sector": "Phát triển Bất động sản"},
        "NLG": {"company_name": "Nha Trang Land Group", "exchange": "HOSE", "industry": "Bất động sản", "sub_sector": "Phát triển Bất động sản"},
        "HPG": {"company_name": "Hoa Phat Group", "exchange": "HOSE", "industry": "Tài nguyên cơ bản", "sub_sector": "Thép"},
        "GVR": {"company_name": "GEOVANCOUVER", "exchange": "HOSE", "industry": "Tài nguyên cơ bản", "sub_sector": "Khai thác"},
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

        # Standardize arrows and spacing around them.
        cleaned = re.sub(r"\s*(?:->|→)\s*", " -> ", cleaned)

        # Remove stray bold markers that can remain around headings/items.
        cleaned = cleaned.replace("**", "")

        # Force numbered list items onto their own lines when the model jams them together.
        # Restrict this to cases after a colon to avoid false positives like "TCM's 1.".
        cleaned = re.sub(r"(?<!\n)(?<=:)\s*(\d{1,2})\.\s+(?=[A-Z(])", r"\n\1. ", cleaned)

        # Also split concatenated section items like "... trends.2. ROE: ...".
        cleaned = re.sub(r"(?<!\n)\.\s*(\d{1,2})\.\s+(?=[A-Z(])", r".\n\1. ", cleaned)

        # Ensure non-numbered Technical Trend labels are on separate lines.
        trend_labels = (
            r"Current Price:|"
            r"1W Return %:|"
            r"1M Return %:|"
            r"3M Return %:|"
            r"YTD Return %(?: vs Index)?:"
        )
        cleaned = re.sub(
            rf"(?<!\n)(?<=[.!?])\s*(?=(?:{trend_labels}))",
            "\n",
            cleaned,
        )

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

        lower_text = text.lower()
        missing = [metric for metric in required if metric not in lower_text]
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

    def _ensure_cfo_trend_coverage(self, text: str, fundamental_metrics: Dict[str, Any]) -> str:
        """Ensure Financial Health includes CFO trend with value and direction."""
        if "cfo trend" in text.lower():
            return text

        cfo_latest = self._normalize_coverage_value(fundamental_metrics.get("cfo_latest"), "N/A")
        cfo_trend = str(self._normalize_coverage_value(fundamental_metrics.get("cfo_trend_direction"), "not_available"))
        cfo_change = fundamental_metrics.get("cfo_trend_change_pct")

        # Avoid adding a noisy placeholder when no real CFO signal exists.
        if cfo_latest == "N/A" and cfo_trend == "not_available" and not isinstance(cfo_change, (int, float)):
            return text

        change_text = f", change {cfo_change:.2f}% vs prior period" if isinstance(cfo_change, (int, float)) else ""

        logger.info(
            "Auto-patch CFO trend coverage: latest=%s trend=%s change=%s",
            cfo_latest,
            cfo_trend,
            cfo_change,
        )
        patch_line = (
            f"CFO Trend: {cfo_latest} | Trend: {cfo_trend}{change_text} -> "
            "Operating cash-flow momentum is a key quality signal for earnings sustainability. "
            "A stable or improving CFO trend supports internal funding capacity, while a weakening trend raises financing and execution risk."
        )
        patched_text = f"{text.rstrip()}\n\n{patch_line}"
        return self._sanitize_llm_output(patched_text)

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
        vn_tickers = ["VCB", "BID", "VNM", "VHM", "VIC", "KDH", "NLG", "DRH", "HQC"]
        ticker_upper = ticker.upper()
        market = "VN" if ticker_upper in vn_tickers else "GLOBAL"

        metadata = self._resolve_company_metadata(ticker_upper, market)

        return {
            "ticker": ticker_upper,
            "market": market,
            "exchange": metadata.get("exchange", "NYSE/NASDAQ"),
            "company_name": metadata.get("company_name", "Not available"),
            "industry": metadata.get("industry", "Not available"),
            "sub_sector": metadata.get("sub_sector", "Not available"),
        }

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

        ticker_news = ticker_news.sort_values('date', ascending=False).head(10)

        events = []
        for _, row in ticker_news.iterrows():
            event = {
                "date": str(row['date'].date()) if 'date' in row and pd.notna(row['date']) else None,
                "event_type": str(row.get('event_type', 'general')),
                "description": str(row.get('description', row.get('headline', ''))),
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

1. **Basic Information:**
   Output as labelled lines:
   - Ticker & Exchange: <ticker> / <exchange>
   - Company Name: <name or "Not available">
   - Industry & Sub-sector: <industry> / <subsector>
   - Market Cap Classification: <Large Cap | Mid Cap | Small Cap> (~<USD value> if available)

2. **Summary Statement:**
   - Short-term View (1–3 months): Combine Fundamental + Technical signals. 2–3 sentences.
   - Long-term View (1–3 years): Structural growth drivers and risk profile. 2–3 sentences.

3. **Comparison Snippet** (if comparing to {ticker_b}): 2–3 sentences comparing investment profiles.
    - Use the comparison company data in the context when available.
    - Do not say the comparison company has no data if comparison fields are present.
    - Compare fundamentals, valuation, and overall investment profile directly.

4. **Call to Action:**
   - Strategic recommendation: Accumulate / Hold / Wait
   - 2–3 sentences on investor suitability.
   - Include disclaimer: "This analysis is AI-generated for informational purposes only and is not investment advice."

Output clean markdown. Be data-driven and specific.
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
        context = {
            "ticker": ticker,
            "market": market,
            "macro_metrics": macro_metrics,
            "industry_metrics": industry_metrics,
            "recent_news_events": news_summary.get("events", [])[:5],
        }

        if market == "VN":
            indicators_spec = """
OUTPUT FORMAT FOR EACH MACRO METRIC (MANDATORY):
`<Metric>: <Value> | Trend: <up/down/flat/not_available> -> <2-3 sentence interpretation>`

GLOBAL INDICATORS (3 metrics):
1. imf_global_growth - IMF global GDP growth forecast  
2. fed_funds_rate - US Federal Reserve policy rate
3. oil_price - Global crude oil price level

VIETNAM DOMESTIC INDICATORS (2 metrics):
4. vn_gdp_growth - Vietnam GDP growth rate
5. vn_cpi - Vietnam inflation rate

RULES:
- Every metric must have 2-3 sentence interpretation.
- No standalone numbers.
- Be factual, specific, and data-driven.
- Output language must be English only.
"""
        else:  # GLOBAL
            indicators_spec = """
OUTPUT FORMAT FOR EACH MACRO METRIC (MANDATORY):
`<Metric>: <Value> | Trend: <up/down/flat/not_available> -> <2-3 sentence interpretation>`

GLOBAL & US INDICATORS (5 metrics):
1. imf_global_growth - IMF global GDP growth forecast
2. fed_funds_rate - US Federal Reserve policy rate
3. oil_price - Global crude oil price level
4. us_gdp_growth - US GDP growth rate
5. us_cpi - US inflation level

INDUSTRY VALUATION & PROFITABILITY (use the most prominent available industry metrics):
- Output each available industry metric as a labelled line with a 2-3 sentence interpretation.
- Explain whether the industry is cheap or expensive versus history and what that means for Stock A.

CORPORATE EVENTS & NEWS:
- For each detected event in recent_news_events, output the event type and a 2-3 sentence interpretation of its likely impact on stock price, sentiment, or fundamentals.
- If no events are available, state that no material recent events were provided.

RULES:
- Every metric must have 2-3 sentence interpretation.
- No standalone numbers.
- Be factual, specific, and data-driven.
- Output language must be English only.
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
"""

    def _build_financial_health_prompt(
        self,
        ticker: str,
        fundamental_metrics: Dict[str, Any],
    ) -> str:
        """Build Financial Health prompt (7 core metrics per Module 4)."""
        context = {
            "ticker": ticker,
            "metrics": fundamental_metrics,
        }

        return f"""You are a fundamental analyst. Analyze {ticker}'s Financial Health per Module 4 spec.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

STRICT OUTPUT FORMAT FOR EACH METRIC:
`1. <Metric>: <Value> -> <2-3 sentence interpretation>`

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
"""

    def _build_valuation_analysis_prompt(
        self,
        ticker: str,
        price_metrics: Dict[str, Any],
        fundamental_metrics: Dict[str, Any],
    ) -> str:
        """Build Valuation Analysis prompt (24 metrics per Module 4)."""
        context = {
            "ticker": ticker,
            "price_metrics": price_metrics,
            "fundamental_metrics": fundamental_metrics,
        }

        return f"""You are a valuation analyst. Generate comprehensive Valuation Analysis for {ticker} per Module 4 spec.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

STRICT OUTPUT FORMAT FOR EACH METRIC:
'Metric: Value -> 2-3 sentence interpretation'

FUNDAMENTAL VALUATION (3 metrics):
1. Current P/E vs 1Y avg vs 5Y avg vs Industry avg - Interpret cheap/expensive positioning
2. Current P/B vs 1Y avg vs 5Y avg vs Industry avg - Interpret asset-based valuation
3. DCF Valuation: Intrinsic Price, Market Price, Upside/Downside % - Interpret reliability and margin of safety

TECHNICAL ANALYSIS - TREND SUMMARY (5 metrics):
- Current Price - Interpret vs 12-month range and market structure
- 1W Return % - Interpret short-term momentum
- 1M Return % - Interpret 1-month price action and sentiment
- 3M Return % - Interpret medium-term trend
- YTD Return % vs Index - Interpret relative performance

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
- Do not prefix metric names with numeric labels.
- Be specific and data-driven.
- Output language must be English only.
- Use plain ASCII only.
- Do not use markdown math, LaTeX, or fragmented digit formatting.
"""

    def _build_peer_comparison_prompt(
        self,
        ticker_a: str,
        ticker_b: str,
        fund_a: Dict[str, Any],
        fund_b: Dict[str, Any],
        price_a: Dict[str, Any],
        price_b: Dict[str, Any],
    ) -> str:
        """Build Peer Comparison prompt per Module 4 spec (simplified)."""
        context = {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "fundamental_a": fund_a,
            "fundamental_b": fund_b,
            "price_a": price_a,
            "price_b": price_b,
        }

        return f"""You are a comparative equity analyst. Compare {ticker_a} vs {ticker_b} per Module 4 spec.

CONTEXT (JSON):
{json.dumps(context, indent=2, default=str)}

GENERATE FOLLOWING SECTIONS:

**1. Financial Health Comparison** (5 metrics):
- Revenue Growth (YoY)
- ROE
- Current Ratio
- Debt-to-Equity (D/E)
- FCFE

Highlight which company has: better financial health, stronger profitability, better liquidity, healthier cash flow.

**2. Fundamental Valuation Comparison** (3 metrics):
- Current P/E vs Industry avg
- Current P/B vs Industry avg
- DCF Valuation (Intrinsic Price vs Market Price, Upside/Downside %)

Explain which stock appears undervalued/fairly valued/expensive. Discuss valuation attractiveness in current market.

**3. Technical & Risk Profile Comparison** (5 metrics):
- MACD
- RSI(14)
- Historical Volatility 30D
- Max Drawdown
- Sharpe Ratio

Explain differences in: price momentum, volatility/risk profile, defensive vs cyclical characteristics, trading sentiment.

**4. Comparison Summary**:
Position {ticker_a} vs {ticker_b} under current market conditions.
Identify which stock suits: growth-oriented investors / value-focused investors.
Summarize key differentiating factors driving recommendation.

RULES:
- Use concrete numbers from context.
- Use format: `<Metric>: <A> vs <B> -> <2-3 sentence interpretation>`
- Output language: English only.
- Use plain ASCII only.
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
            cleaned_financial = self._sanitize_llm_output(financial_health)
            report['financial_health'] = self._ensure_cfo_trend_coverage(
                cleaned_financial, fundamental_metrics_a
            )
            logger.info("✓ Financial Health generated")
        except Exception as exc:
            logger.error(f"Financial Health generation failed: {exc}")
            report['financial_health'] = f"Error generating Financial Health: {exc}"

        # 4. VALUATION ANALYSIS
        logger.info("Generating Valuation Analysis...")
        try:
            prompt = self._build_valuation_analysis_prompt(
                ticker_a, price_metrics_a, fundamental_metrics_a
            )
            valuation_analysis = self._call_llm(self._SYSTEM_INSTRUCTION, prompt)
            report['valuation_analysis'] = self._sanitize_llm_output(valuation_analysis)
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
                    price_metrics_a, price_metrics_b
                )
                peer_comparison = self._call_llm(self._SYSTEM_INSTRUCTION, prompt)
                report['peer_comparison'] = self._sanitize_llm_output(peer_comparison)
                logger.info("✓ Peer Comparison generated")
            except Exception as exc:
                logger.error(f"Peer Comparison generation failed: {exc}")
                report['peer_comparison'] = f"Error generating Peer Comparison: {exc}"
        else:
            report['peer_comparison'] = "Peer comparison skipped because ticker_b or comparison data is missing."

        # Metadata
        report['analysis_mode'] = 'llm_module4'
        report['model_used'] = self.model
        report['provider'] = self.provider
        if self.provider == "gemini":
            report['api_key_slot_used'] = self._active_api_key_index + 1
            report['api_key_slots_total'] = len(self._gemini_api_keys)
        report['generated_at'] = datetime.now().isoformat()
        report['ticker_a'] = ticker_a
        report['ticker_b'] = ticker_b
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
