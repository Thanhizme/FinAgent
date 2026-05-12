"""
ai_agent.py
-----------
Integrates a Large Language Model (LLM) to produce automated natural language
analysis of processed financial data.

Supported providers (configured via .env):
    - Google Gemini     (GEMINI_API_KEY)     [PRIMARY]
    - Anthropic Claude  (ANTHROPIC_API_KEY)  [optional]
    - OpenAI GPT        (OPENAI_API_KEY)     [optional]

Analysis outputs (at minimum):
  1. Trend summary     : current trend and recent performance per asset.
  2. Anomaly report    : notable events or outliers detected in the dataset.
  3. Risk commentary   : volatility-based risk assessment.
  4. Comparative note  : side-by-side comparison of two or more assets.

The LLM receives structured JSON context built from processed DataFrames to
ensure grounded, data-referenced output and minimise hallucinations.
"""

import os
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Orchestrates LLM calls to generate natural language financial analysis.

    Parameters
    ----------
    provider : {'gemini', 'anthropic', 'openai'}
        LLM provider to use. The corresponding API key must be set in .env.
        Defaults to 'gemini'.
        model : str, optional
                Model identifier. Defaults per provider:
                    - anthropic : 'claude-3-5-sonnet-20241022'
                    - gemini    : 'gemini-2.5-flash'
                    - openai    : 'gpt-4o'
    max_tokens : int
        Maximum tokens to generate per response. Default: 1600.
    temperature : float
        LLM temperature (0.0-1.0). Default: 0.7.
    """

    _DEFAULT_MODELS = {
        "gemini":    "gemini-2.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai":    "gpt-4o",
    }

    _ENV_KEYS = {
        "gemini":    "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
    }

    _MAX_CALL_RETRIES = 3
    _RETRY_WAIT_SECONDS = 1.2
    _TRANSIENT_ERROR_MARKERS = (
        "503",
        "unavailable",
        "deadline exceeded",
        "429",
        "rate limit",
        "internal",
        "timeout",
    )

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        max_tokens: int = 2400,
        temperature: float = 0.7,
    ) -> None:
        provider_lower = provider.lower()
        if provider_lower not in self._DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {list(self._DEFAULT_MODELS.keys())}"
            )
        
        self.provider = provider_lower
        self.model = model or self._DEFAULT_MODELS[self.provider]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = self._init_client()
        logger.info(f"[{self.provider.upper()}] AnalysisAgent initialized with model {self.model}")

    def _init_client(self) -> Any:
        """
        Instantiate the appropriate SDK client based on self.provider.

        Returns
        -------
        object
            Authenticated SDK client/model object for configured provider.

        Raises
        ------
        EnvironmentError
            If the required API key environment variable is not set.
        ImportError
            If the required SDK is not installed.
        """
        env_var = self._ENV_KEYS[self.provider]
        api_key = os.getenv(env_var)

        if not api_key:
            raise EnvironmentError(
                f"API key not found. Please set '{env_var}' in your .env file."
            )

        if self.provider == "gemini":
            try:
                from google import genai
                return genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install it with: pip install google-genai"
                )

        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install it with: pip install anthropic"
                )

        raise NotImplementedError(
            f"Provider '{self.provider}' is not implemented yet."
        )

    def _extract_metrics(self, df: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
        """Extract key metrics from processed DataFrame."""
        if df is None or df.empty:
            return None

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

        if df.empty:
            return None

        latest_row = df.iloc[-1]
        
        # Latest price
        latest_price = float(latest_row['close']) if 'close' in df.columns else None
        latest_date = str(latest_row['date'].date()) if 'date' in df.columns else None

        # Returns
        if 'daily_return' in df.columns:
            daily_ret = pd.to_numeric(df['daily_return'], errors='coerce').dropna()
            ret_30d = float(((1 + daily_ret.iloc[-30:]).prod() - 1) * 100) if len(daily_ret) >= 30 else None  # type: ignore
            ret_90d = float(((1 + daily_ret.iloc[-90:]).prod() - 1) * 100) if len(daily_ret) >= 90 else None  # type: ignore
        else:
            ret_30d = ret_90d = None

        # Volatility & Sharpe
        volatility_20d = float(latest_row['volatility_20']) * 100 if 'volatility_20' in df.columns else None
        sharpe_ratio = float(latest_row['sharpe_ratio']) if 'sharpe_ratio' in df.columns else None

        # Max drawdown
        max_drawdown = float(latest_row['max_drawdown']) * 100 if 'max_drawdown' in df.columns else None

        # Moving averages
        ma_20 = float(latest_row['ma20']) if 'ma20' in df.columns else None
        ma_200 = float(latest_row['ma200']) if 'ma200' in df.columns else None

        # RSI
        rsi_14 = float(latest_row['rsi_14']) if 'rsi_14' in df.columns else None

        return {
            'ticker': ticker,
            'latest_price': latest_price,
            'latest_date': latest_date,
            'return_30d': ret_30d,
            'return_90d': ret_90d,
            'volatility_20d': volatility_20d,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'ma_20': ma_20,
            'ma_200': ma_200,
            'rsi_14': rsi_14,
            'row_count': len(df),
            'anomaly_events': self._extract_anomaly_events(df),
        }

    def _extract_anomaly_events(self, df: pd.DataFrame, z_threshold: float = 2.0) -> list[dict[str, Any]]:
        """Find top daily return outlier events from z-score ranking."""
        if 'daily_return' not in df.columns or 'date' not in df.columns:
            return []

        series = pd.to_numeric(df['daily_return'], errors='coerce')
        valid = series.dropna()
        if len(valid) < 20:
            return []

        std = float(valid.std(ddof=0))
        if std == 0:
            return []

        mean = float(valid.mean())
        z_scores = (valid - mean) / std
        candidates = z_scores.abs().sort_values(ascending=False).head(5)

        events: list[dict[str, Any]] = []
        for idx, abs_z in candidates.items():
            z_val = float(z_scores.loc[idx])  # type: ignore[index]
            if abs(z_val) < z_threshold:
                continue
            event_date = pd.to_datetime(df.loc[idx, 'date'], errors='coerce')  # type: ignore[index]
            event_ret = float(valid.loc[idx]) * 100  # type: ignore[index]
            events.append(
                {
                    'date': str(event_date.date()) if pd.notna(event_date) else None,
                    'daily_return_pct': event_ret,
                    'z_score': z_val,
                }
            )

        return events[:3]

    def _risk_level(self, metrics: Dict[str, Any]) -> str:
        """Classify risk level from volatility, Sharpe, and max drawdown."""
        score = 0
        vol = metrics.get('volatility_20d')
        sharpe = metrics.get('sharpe_ratio')
        drawdown = metrics.get('max_drawdown')

        if isinstance(vol, (int, float)):
            if vol > 30:
                score += 2
            elif vol > 20:
                score += 1
        if isinstance(sharpe, (int, float)):
            if sharpe < 0:
                score += 2
            elif sharpe < 0.5:
                score += 1
        if isinstance(drawdown, (int, float)):
            if drawdown < -25:
                score += 2
            elif drawdown < -15:
                score += 1

        if score >= 4:
            return "High"
        if score >= 2:
            return "Moderate"
        return "Low"

    def _build_analysis_context(self, data: Dict[str, pd.DataFrame]) -> str:
        """Build structured JSON context for LLM."""
        context = {}
        for ticker, df in data.items():
            metrics = self._extract_metrics(df, ticker)
            if metrics:
                metrics['risk_level'] = self._risk_level(metrics)
                context[ticker] = metrics

        return json.dumps(context, indent=2, default=str)

    def _is_transient_error(self, error_text: str) -> bool:
        lower = error_text.lower()
        return any(marker in lower for marker in self._TRANSIENT_ERROR_MARKERS)

    def _is_response_complete(self, text: str, required_keywords: list[str], min_chars: int = 220) -> bool:
        if not text or len(text.strip()) < min_chars:
            return False
        trimmed = text.strip()
        low = trimmed.lower()
        if "error generating" in low:
            return False
        if low.endswith((":", "-", " of", " and", " with", ",", ";")):
            return False
        # Some providers return markdown sections that may not end with punctuation.
        # Avoid false negatives here and rely on truncation check instead.
        if required_keywords:
            # Some models use company names instead of ticker symbols.
            # Require at least one marker, not all, to avoid false rejection.
            if not any(keyword.lower() in low for keyword in required_keywords):
                return False
        return True

    def _looks_truncated(self, text: str) -> bool:
        if not text:
            return True
        trimmed = text.strip().lower()
        suspicious_endings = (
            " of",
            " with",
            " and",
            " is",
            " are",
            " to",
            " for",
            " by",
            " from",
            " than",
            " a",
            " an",
            " the",
            ":",
            "-",
            ",",
            ";",
            "*",
            "(",
            "/",
        )
        return trimmed.endswith(suspicious_endings)

    def _validate_section_response(self, section: str, text: str, tickers: list[str], min_chars: int) -> bool:
        if not self._is_response_complete(text, [], min_chars=min_chars):
            return False
        low = text.lower()
        ticker_markers = [str(t).lower() for t in tickers if isinstance(t, str) and t]

        def has_ticker_coverage(required: int = 1) -> bool:
            if not ticker_markers:
                return True
            mentions = sum(1 for marker in ticker_markers if marker in low)
            return mentions >= min(required, len(ticker_markers))

        if section == "trend":
            trend_tokens = ("uptrend", "downtrend", "sideways", "bullish", "bearish", "consolidation")
            if any(token in low for token in trend_tokens) and has_ticker_coverage(required=max(1, len(ticker_markers))):
                return True
            # Accept narrative trend text even without explicit trend keyword.
            return (not self._looks_truncated(text)) and has_ticker_coverage(required=max(1, len(ticker_markers)))
        if section == "anomaly":
            anomaly_tokens = ("outlier", "anomal", "spike", "drop", "deviation", "irregular")
            if not any(token in low for token in anomaly_tokens):
                return False
            if self._looks_truncated(text):
                return False
            if not has_ticker_coverage(required=max(1, len(ticker_markers))):
                return False
            return True
        if section == "risk":
            core_tokens = ("risk", "volatility", "drawdown", "sharpe")
            level_tokens = ("low", "moderate", "high", "risk level")
            if any(token in low for token in core_tokens) and any(token in low for token in level_tokens):
                # Assignment requires risk commentary grounded in volatility metrics.
                if "volatility" not in low:
                    return False
                return has_ticker_coverage(required=max(1, len(ticker_markers)))
            return (not self._looks_truncated(text)) and has_ticker_coverage(required=max(1, len(ticker_markers)))
        if section == "comparison":
            if len(tickers) >= 2:
                comparison_tokens = ("winner", "better", "outperform", "verdict", "relative", "compare", "return", "volatility", "sharpe")
                if any(token in low for token in comparison_tokens):
                    # Assignment asks for comparison paragraph; reject table-only outputs.
                    pipe_count = text.count("|")
                    sentence_count = text.count(".")
                    if pipe_count >= 8 and sentence_count < 2:
                        return False
                    return has_ticker_coverage(required=2)
                return (not self._looks_truncated(text)) and has_ticker_coverage(required=2)
            return True
        return True

    def _validate_full_report_sections(
        self,
        report: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        comparison_tickers: Optional[list],
        primary_ticker: Optional[str] = None,
    ) -> bool:
        """Validate all mandatory assignment sections before accepting a full report."""
        _primary = primary_ticker if (primary_ticker and primary_ticker in data) else list(data.keys())[0]
        checks = {
            "trend_summary": ("trend", [_primary], 170),
            "anomaly_report": ("anomaly", [_primary], 170),
            "risk_commentary": ("risk", [_primary], 170),
            "comparison": ("comparison", comparison_tickers or list(data.keys()), 170),
        }
        for key, (section_name, tickers, min_chars) in checks.items():
            content = str(report.get(key, ""))
            if not self._validate_section_response(section_name, content, tickers, min_chars=min_chars):
                return False
        return True

    def _enforce_required_sections(
        self,
        report: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        comparison_tickers: Optional[list],
        section_sources: Dict[str, str],
        primary_ticker: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Ensure the 4 required assignment sections are complete; fallback only invalid sections."""
        _primary = primary_ticker if (primary_ticker and primary_ticker in data) else list(data.keys())[0]
        primary_data = {_primary: data[_primary]}
        checks = {
            "trend_summary": ("trend", [_primary], 170),
            "anomaly_report": ("anomaly", [_primary], 170),
            "risk_commentary": ("risk", [_primary], 170),
            "comparison": ("comparison", comparison_tickers or list(data.keys()), 170),
        }

        for section_key, (section_name, tickers, min_chars) in checks.items():
            text_val = str(report.get(section_key, ""))
            if self._validate_section_response(section_name, text_val, tickers, min_chars=min_chars):
                continue

            if section_key == "trend_summary":
                report[section_key] = self._fallback_trend_summary(primary_data)
            elif section_key == "anomaly_report":
                report[section_key] = self._fallback_anomaly_report(primary_data)
            elif section_key == "risk_commentary":
                report[section_key] = self._fallback_risk_commentary(primary_data)
            else:
                report[section_key] = self._fallback_comparison(data, comparison_tickers)

            section_sources[section_key] = "deterministic"

        return report, section_sources

    def _repair_weak_sections(
        self,
        report: Dict[str, Any],
        data: Dict[str, pd.DataFrame],
        comparison_tickers: Optional[list],
        section_sources: Dict[str, str],
        primary_ticker: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Repair weak or truncated sections before returning final report."""
        _primary = primary_ticker if (primary_ticker and primary_ticker in data) else list(data.keys())[0]
        primary_data = {_primary: data[_primary]}
        validators = {
            "trend_summary": ("trend", [_primary], 170),
            "anomaly_report": ("anomaly", [_primary], 170),
            "risk_commentary": ("risk", [_primary], 170),
            "comparison": ("comparison", comparison_tickers or list(data.keys()), 170),
        }

        for section_key, (validator_name, tickers, min_chars) in validators.items():
            text_val = str(report.get(section_key, ""))
            if self._validate_section_response(validator_name, text_val, tickers, min_chars=min_chars):
                continue

            if section_key == "trend_summary":
                repaired = self.generate_trend_summary(primary_data)
            elif section_key == "anomaly_report":
                repaired = self.generate_anomaly_report(primary_data)
            elif section_key == "risk_commentary":
                repaired = self.generate_risk_commentary(primary_data)
            else:
                repaired = self.generate_comparison(data, comparison_tickers)

            report[section_key] = repaired
            section_sources[section_key] = "deterministic" if self._is_deterministic_output(repaired) else "llm"

        return report, section_sources

    def _soft_accept_llm_section(self, text: str, min_chars: int = 140) -> bool:
        """Lenient acceptance gate to avoid unnecessary deterministic fallback."""
        if not text or len(text.strip()) < min_chars:
            return False
        if self._looks_truncated(text):
            return False
        if self._is_deterministic_output(text):
            return False
        return True

    def _format_pct(self, value: Any) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return "N/A"

    def _format_num(self, value: Any, decimals: int = 2) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}"
        except Exception:
            return "N/A"

    def _fallback_trend_summary(self, data: Dict[str, pd.DataFrame]) -> str:
        lines = ["Automated trend summary (fallback mode):"]
        for ticker, df in data.items():
            metrics = self._extract_metrics(df, ticker)
            if not metrics:
                lines.append(f"{ticker}: Insufficient data to determine trend reliably.")
                continue
            price = metrics.get('latest_price')
            ma20 = metrics.get('ma_20')
            ma200 = metrics.get('ma_200')
            trend = "sideways"
            if isinstance(price, (int, float)) and isinstance(ma20, (int, float)) and isinstance(ma200, (int, float)):
                if price > ma20 > ma200:
                    trend = "uptrend"
                elif price < ma20 < ma200:
                    trend = "downtrend"
            lines.append(
                f"{ticker} is currently trading at ${self._format_num(price, 2)}. "
                f"Its 30-day return is {self._format_pct(metrics.get('return_30d'))} and 90-day return is {self._format_pct(metrics.get('return_90d'))}. "
                f"RSI is {self._format_num(metrics.get('rsi_14'), 2)}, and the price-vs-moving-average structure suggests a {trend}."
            )
        return "\n\n".join(lines)

    def _fallback_anomaly_report(self, data: Dict[str, pd.DataFrame]) -> str:
        lines = ["Automated anomaly report (fallback mode):"]
        any_event = False
        for ticker, df in data.items():
            metrics = self._extract_metrics(df, ticker)
            events = metrics.get('anomaly_events', []) if metrics else []
            if events:
                any_event = True
                event_parts = []
                for ev in events:
                    event_parts.append(
                        f"{ev.get('date')} (daily return {self._format_num(ev.get('daily_return_pct'), 2)}%, z-score {self._format_num(ev.get('z_score'), 2)})"
                    )
                lines.append(
                    f"For {ticker}, notable outlier sessions were observed on "
                    + ", ".join(event_parts)
                    + "."
                )
            else:
                lines.append(f"For {ticker}, no significant z-score outlier (>= 2.0) was detected in daily returns.")
        if not any_event:
            lines.append("No significant return outliers found with z-score threshold >= 2.0.")
        return "\n\n".join(lines)

    def _fallback_risk_commentary(self, data: Dict[str, pd.DataFrame]) -> str:
        lines = ["Automated risk commentary (fallback mode):"]
        for ticker, df in data.items():
            metrics = self._extract_metrics(df, ticker)
            if not metrics:
                lines.append(f"{ticker}: Insufficient data to derive a robust risk profile.")
                continue
            risk_level = self._risk_level(metrics)
            suggestion = {
                "High": "Consider smaller position sizing and tighter risk controls.",
                "Moderate": "Balanced sizing with stop-loss discipline is recommended.",
                "Low": "Risk profile appears relatively stable, but monitoring is still necessary.",
            }.get(risk_level, "Maintain normal risk management practices.")
            lines.append(
                f"{ticker} is classified as {risk_level} risk based on current volatility ({self._format_pct(metrics.get('volatility_20d'))}), "
                f"max drawdown ({self._format_pct(metrics.get('max_drawdown'))}), and Sharpe ratio ({self._format_num(metrics.get('sharpe_ratio'), 3)}). "
                f"{suggestion}"
            )
        return "\n\n".join(lines)

    def _fallback_comparison(self, data: Dict[str, pd.DataFrame], tickers: Optional[list[str]] = None) -> str:
        selected = tickers or list(data.keys())
        selected = [t for t in selected if t in data]
        if len(selected) < 2:
            return "Insufficient data for comparison."
        a, b = selected[0], selected[1]
        ma = self._extract_metrics(data[a], a)
        mb = self._extract_metrics(data[b], b)
        if not ma or not mb:
            return "Insufficient data for comparison."

        return (
            f"Automated comparison fallback between {a} and {b}: "
            f"{a} shows 30d/90d returns of {self._format_pct(ma.get('return_30d'))}/{self._format_pct(ma.get('return_90d'))}, "
            f"while {b} shows {self._format_pct(mb.get('return_30d'))}/{self._format_pct(mb.get('return_90d'))}. "
            f"On risk-adjusted quality, Sharpe is {self._format_num(ma.get('sharpe_ratio'), 3)} for {a} versus {self._format_num(mb.get('sharpe_ratio'), 3)} for {b}. "
            f"Volatility is {self._format_pct(ma.get('volatility_20d'))} for {a} and {self._format_pct(mb.get('volatility_20d'))} for {b}. "
            f"Overall risk classification is {self._risk_level(ma)} for {a} and {self._risk_level(mb)} for {b}."
        )

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract first JSON object from potentially fenced model output."""
        if not text:
            return None
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            return None

    def _generate_full_analysis_once(
        self,
        data: Dict[str, pd.DataFrame],
        comparison_tickers: Optional[list] = None,
        primary_ticker: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single LLM call that returns all required sections in one JSON payload."""
        context_json = self._build_analysis_context(data)
        tickers = list(data.keys())
        compare_tickers = comparison_tickers or tickers
        _primary = primary_ticker if (primary_ticker and primary_ticker in tickers) else tickers[0]
        # Tickers expected in non-comparison sections (primary only).
        primary_tickers_list = [_primary]

        system_prompt = f"""You are a financial analyst assistant.
Return ONLY valid JSON with exactly these keys:
- trend_summary
- anomaly_report
- risk_commentary
- comparison

Requirements:
1) Each value must be a natural-language paragraph-style markdown text.
2) trend_summary, anomaly_report, and risk_commentary MUST cover ONLY the primary ticker: {_primary}. Do NOT mention any other ticker in these three sections.
3) comparison must cover ALL comparison tickers: {compare_tickers}. It must be a narrative paragraph-style comparison (NOT a markdown table).
4) Mention concrete numbers/dates from context.
5) Do not return bullet-only numeric dumps.
6) Ensure complete sentences and no truncation.
7) Each section should be at least 4 complete sentences.
8) risk_commentary must include explicit risk level (Low/Moderate/High) for {_primary}.
"""

        user_prompt = (
            "Create full analysis for the following structured data context.\n\n"
            f"PRIMARY_TICKER (for trend_summary, anomaly_report, risk_commentary): {_primary}\n"
            f"COMPARISON_TICKERS (for comparison section): {compare_tickers}\n\n"
            f"DATA_CONTEXT_JSON:\n{context_json}\n"
        )

        raw = self._call_llm(system_prompt, user_prompt)
        payload = self._extract_json_payload(raw)
        if not payload:
            return None

        required_keys = ["trend_summary", "anomaly_report", "risk_commentary", "comparison"]
        if any(k not in payload for k in required_keys):
            return None

        trend = str(payload.get("trend_summary", ""))
        anomaly = str(payload.get("anomaly_report", ""))
        risk = str(payload.get("risk_commentary", ""))
        comparison = str(payload.get("comparison", ""))

        if not self._validate_section_response("trend", trend, primary_tickers_list, min_chars=170):
            return None
        if not self._validate_section_response("anomaly", anomaly, primary_tickers_list, min_chars=170):
            return None
        if not self._validate_section_response("risk", risk, primary_tickers_list, min_chars=170):
            return None
        if not self._validate_section_response("comparison", comparison, compare_tickers, min_chars=170):
            return None

        return {
            "trend_summary": trend,
            "anomaly_report": anomaly,
            "risk_commentary": risk,
            "comparison": comparison,
        }

    def _deterministic_full_report(
        self,
        data: Dict[str, pd.DataFrame],
        comparison_tickers: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Fallback report that still satisfies required 4 narrative outputs."""
        return {
            "trend_summary": self._fallback_trend_summary(data),
            "anomaly_report": self._fallback_anomaly_report(data),
            "risk_commentary": self._fallback_risk_commentary(data),
            "comparison": self._fallback_comparison(data, comparison_tickers),
        }

    def _is_deterministic_output(self, text: str) -> bool:
        if not text:
            return True
        low = text.lower().strip()
        return (
            low.startswith("automated trend summary (fallback mode):")
            or low.startswith("automated anomaly report (fallback mode):")
            or low.startswith("automated risk commentary (fallback mode):")
            or low.startswith("automated comparison fallback")
            or low.startswith("insufficient data for comparison")
        )

    def _attempt_repair_deterministic_sections(
        self,
        data: Dict[str, pd.DataFrame],
        comparison_tickers: Optional[list],
        report: Dict[str, Any],
        section_sources: Dict[str, str],
        max_rounds: int = 2,
        primary_ticker: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Try to replace deterministic sections with LLM output to reach full-LLM report."""
        context_json = self._build_analysis_context(data)
        compare_tickers = comparison_tickers or list(data.keys())
        _primary = primary_ticker if (primary_ticker and primary_ticker in data) else list(data.keys())[0]

        for _ in range(max_rounds):
            missing = [k for k, v in section_sources.items() if v == "deterministic"]
            if not missing:
                break

            system_prompt = f"""You are a financial analyst assistant.
Return ONLY valid JSON and include ONLY the requested keys.
Each key value must be a natural-language narrative paragraph with concrete numbers/dates.
Do not include markdown code fences.
Do not omit requested keys.
For trend_summary, anomaly_report, risk_commentary: ONLY cover {_primary}.
For comparison: cover {compare_tickers}.
"""

            user_prompt = (
                "Repair missing sections for financial analysis.\n\n"
                f"REQUESTED_KEYS: {missing}\n"
                f"PRIMARY_TICKER (trend/anomaly/risk): {_primary}\n"
                f"COMPARISON_TICKERS (comparison section): {compare_tickers}\n\n"
                f"DATA_CONTEXT_JSON:\n{context_json}\n\n"
                "Return JSON object with only requested keys."
            )

            try:
                raw = self._call_llm(system_prompt, user_prompt)
                payload = self._extract_json_payload(raw)
                if not payload:
                    continue

                for section in missing:
                    if section not in payload:
                        continue
                    text_val = str(payload.get(section, ""))
                    tickers_for_check = [_primary] if section != "comparison" else compare_tickers
                    section_name_map = {
                        "trend_summary": "trend",
                        "anomaly_report": "anomaly",
                        "risk_commentary": "risk",
                        "comparison": "comparison",
                    }
                    validator_section = section_name_map.get(section, section)
                    if (
                        self._validate_section_response(validator_section, text_val, tickers_for_check, min_chars=180)
                        and not self._is_deterministic_output(text_val)
                    ):
                        report[section] = text_val
                        section_sources[section] = "llm"
            except Exception:
                continue

        return report, section_sources

    def _call_claude(self, system: str, user_prompt: str) -> str:
        """Call Claude API with specified prompts."""
        try:
            client: Any = self._client
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
            # Extract text from first text content block
            for content_block in message.content:
                if hasattr(content_block, 'text'):
                    return content_block.text  # type: ignore
            # Fallback: return string representation
            return str(message.content[0]) if message.content else ""
        except Exception as exc:
            logger.error(f"Claude API call failed: {exc}")
            raise

    def _call_gemini(self, system: str, user_prompt: str) -> str:
        """Call Gemini API with specified prompts."""
        try:
            combined_prompt = f"{system}\n\n{user_prompt}"
            client: Any = self._client
            candidates = [
                self.model,
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash-lite-001",
                "gemini-flash-latest",
                "gemini-flash-lite-latest",
                "gemini-3.1-flash-lite",
            ]

            if self.model.startswith("models/"):
                candidates.append(self.model.replace("models/", "", 1))
            else:
                candidates.append(f"models/{self.model}")

            seen = set()
            ordered_candidates = []
            for name in candidates:
                if name and name not in seen:
                    seen.add(name)
                    ordered_candidates.append(name)

            last_error: Exception | None = None
            for model_name in ordered_candidates:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=combined_prompt,
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens,
                        },
                    )
                    self.model = model_name
                    return response.text or ""
                except Exception as exc:
                    last_error = exc
                    err_text = str(exc).lower()
                    # Model-specific failures should move to the next candidate.
                    model_specific_failure = (
                        "not found" in err_text
                        or "404" in err_text
                        or "not supported" in err_text
                        or "resource_exhausted" in err_text
                        or "quota exceeded" in err_text
                        or "429" in err_text
                    )
                    if model_specific_failure:
                        logger.warning(
                            "Gemini model '%s' unavailable or rate-limited, trying next fallback.",
                            model_name,
                        )
                        continue
                    raise

            logger.error("Gemini API call failed for all fallback models: %s", last_error)
            if last_error is not None:
                raise last_error
            raise RuntimeError("Gemini API call failed with unknown error.")
        except Exception as exc:
            logger.error(f"Gemini API call failed: {exc}")
            raise

    def _call_llm(self, system: str, user_prompt: str) -> str:
        """Dispatch model call based on configured provider."""
        for attempt in range(1, self._MAX_CALL_RETRIES + 1):
            try:
                if self.provider == "gemini":
                    return self._call_gemini(system, user_prompt)
                if self.provider == "anthropic":
                    return self._call_claude(system, user_prompt)
                raise NotImplementedError(f"Provider '{self.provider}' is not implemented yet.")
            except Exception as exc:
                if attempt < self._MAX_CALL_RETRIES and self._is_transient_error(str(exc)):
                    logger.warning(
                        "Transient LLM error on attempt %d/%d: %s",
                        attempt,
                        self._MAX_CALL_RETRIES,
                        exc,
                    )
                    time.sleep(self._RETRY_WAIT_SECONDS * attempt)
                    continue
                raise
        raise RuntimeError("LLM call failed after retries.")

    def generate_trend_summary(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate trend summary for each asset."""
        context_json = self._build_analysis_context(data)
        tickers = list(data.keys())

        system_prompt = """You are a professional financial analyst.
    Return a complete markdown report and do not truncate sections.
    Output format for EACH ticker:
    ### <TICKER>
    1) Current Price & Recent Move
    2) 30d and 90d Performance
    3) Moving Average Position (vs MA20 and MA200)
    4) Trend Verdict (uptrend/downtrend/sideways)

    Rules:
    - Use exact numbers from context.
    - Mention RSI and latest date.
    - Avoid generic intros and avoid missing tickers."""

        user_prompt = f"""Analyze the following market data and provide a trend summary for each ticker:

{context_json}

Provide the analysis in a clear, structured format."""

        try:
            response = self._call_llm(system_prompt, user_prompt)
            if self._validate_section_response("trend", response, tickers, min_chars=240):
                return response
            if self._soft_accept_llm_section(response):
                return response

            # One recovery reprompt before deterministic fallback.
            retry_prompt = user_prompt + "\n\nRewrite as one complete narrative paragraph per ticker with explicit trend verdict."
            response_retry = self._call_llm(system_prompt, retry_prompt)
            if self._validate_section_response("trend", response_retry, tickers, min_chars=180) or self._soft_accept_llm_section(response_retry):
                return response_retry
            logger.warning("Trend summary incomplete, returning deterministic fallback.")
            return self._fallback_trend_summary(data)
        except Exception as exc:
            logger.error(f"Trend summary generation failed: {exc}")
            return self._fallback_trend_summary(data)

    def generate_anomaly_report(self, data: Dict[str, pd.DataFrame]) -> str:
        """Detect and report notable anomalies or events."""
        context_json = self._build_analysis_context(data)
        tickers = list(data.keys())

        system_prompt = """You are a financial data analyst.
    Produce a complete markdown anomaly report.
    For each ticker include:
    1) Top outlier events (date, daily return %, z-score)
    2) RSI extreme conditions (if any)
    3) Short impact commentary

    Rules:
    - If no anomaly exists, explicitly say none detected.
    - Never skip any ticker.
    - Use concrete numbers and dates only from context."""

        user_prompt = f"""Analyze the following data for anomalies and unusual events:

{context_json}

Report any notable findings with dates and impact assessment."""

        try:
            response = self._call_llm(system_prompt, user_prompt)
            if self._validate_section_response("anomaly", response, tickers, min_chars=260):
                return response
            if self._soft_accept_llm_section(response):
                return response

            retry_prompt = user_prompt + "\n\nRewrite as complete anomaly narratives for every ticker and include at least one explicit date/value mention per ticker."
            response_retry = self._call_llm(system_prompt, retry_prompt)
            if self._validate_section_response("anomaly", response_retry, tickers, min_chars=190) or self._soft_accept_llm_section(response_retry):
                return response_retry
            logger.warning("Anomaly report incomplete, returning deterministic fallback.")
            return self._fallback_anomaly_report(data)
        except Exception as exc:
            logger.error(f"Anomaly report generation failed: {exc}")
            return self._fallback_anomaly_report(data)

    def generate_risk_commentary(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate risk assessment based on volatility and drawdown metrics."""
        context_json = self._build_analysis_context(data)
        tickers = list(data.keys())

        system_prompt = """You are a risk management specialist.
    Produce a complete markdown risk report for each ticker.
    For each ticker include:
    1) Volatility assessment (20d annualized)
    2) Drawdown severity
    3) Sharpe quality
    4) Final risk level: Low / Moderate / High
    5) One concise action recommendation

    Rules:
    - Must include every ticker.
    - Must include numeric values and threshold-based conclusion."""

        user_prompt = f"""Analyze risk metrics for the following assets:

{context_json}

Provide a risk commentary with classifications and recommendations."""

        try:
            response = self._call_llm(system_prompt, user_prompt)
            if self._validate_section_response("risk", response, tickers, min_chars=220):
                return response
            if self._soft_accept_llm_section(response):
                return response

            retry_prompt = user_prompt + "\n\nRewrite as complete risk commentary for each ticker with numeric references and a clear risk level."
            response_retry = self._call_llm(system_prompt, retry_prompt)
            if self._validate_section_response("risk", response_retry, tickers, min_chars=180) or self._soft_accept_llm_section(response_retry):
                return response_retry
            logger.warning("Risk commentary incomplete, returning deterministic fallback.")
            return self._fallback_risk_commentary(data)
        except Exception as exc:
            logger.error(f"Risk commentary generation failed: {exc}")
            return self._fallback_risk_commentary(data)

    def generate_comparison(
        self,
        data: Dict[str, pd.DataFrame],
        tickers: Optional[list] = None,
    ) -> str:
        """Write comparative analysis for selected assets."""
        if tickers is None:
            tickers = list(data.keys())
        
        # Filter to only selected tickers
        filtered_data = {t: data[t] for t in tickers if t in data}
        
        if len(filtered_data) < 2:
            logger.warning(f"Need at least 2 tickers for comparison. Got: {len(filtered_data)}")
            return "Insufficient data for comparison. Please provide at least 2 tickers."

        context_json = self._build_analysis_context(filtered_data)

        system_prompt = """You are a comparative financial analyst.
    Produce a complete side-by-side comparison as narrative paragraphs (no markdown table).
    Required sections:
    1) Performance (30d/90d)
    2) Risk profile (volatility, drawdown, Sharpe)
    3) Technical state (RSI + MA relation)
    4) Relative winner by investor style (risk-averse vs growth)

    Rules:
    - Always compare all selected tickers.
    - Always include a final verdict paragraph.
    - Output must be paragraph style; do NOT use table format with pipe separators.
    - Do not stop mid-sentence."""

        user_prompt = f"""Compare the following {len(filtered_data)} assets:

{context_json}

Provide a comprehensive comparison highlighting relative strengths and weaknesses."""

        try:
            response = self._call_llm(system_prompt, user_prompt)
            if self._validate_section_response("comparison", response, list(filtered_data.keys()), min_chars=240):
                return response
            if self._soft_accept_llm_section(response):
                return response

            retry_prompt = user_prompt + "\n\nRewrite as paragraph-only comparative narrative (no table), with a final verdict and no truncation."
            response_retry = self._call_llm(system_prompt, retry_prompt)
            if self._validate_section_response("comparison", response_retry, list(filtered_data.keys()), min_chars=180) or self._soft_accept_llm_section(response_retry):
                return response_retry
            logger.warning("Comparison output incomplete, returning deterministic fallback.")
            return self._fallback_comparison(filtered_data, tickers)
        except Exception as exc:
            logger.error(f"Comparison generation failed: {exc}")
            return self._fallback_comparison(filtered_data, tickers)

    def generate_full_analysis(
        self,
        data: Dict[str, pd.DataFrame],
        comparison_tickers: Optional[list] = None,
        primary_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate all required sections, preferring one structured LLM call.

        ``primary_ticker`` controls which ticker is covered in trend_summary,
        anomaly_report, and risk_commentary.  The comparison section always
        covers both tickers.  When omitted, the first key in ``data`` is used.
        """
        logger.info(f"Generating full analysis for {len(data)} ticker(s)...")

        # Derive primary-only subset used for non-comparison sections.
        _primary = primary_ticker if (primary_ticker and primary_ticker in data) else list(data.keys())[0]
        primary_data: Dict[str, pd.DataFrame] = {_primary: data[_primary]}

        report: Optional[Dict[str, Any]] = None
        fallback_reason = ""
        try:
            report = self._generate_full_analysis_once(data, comparison_tickers, primary_ticker=_primary)
            if report is None:
                fallback_reason = "single_call_invalid_or_incomplete"
                logger.warning("Single-call LLM analysis invalid/incomplete. Trying sectional generation.")
        except Exception as exc:
            fallback_reason = f"single_call_error: {exc}"
            logger.warning("Single-call LLM analysis failed: %s", exc)

        if report is None:
            sectional: Dict[str, Any] = {
                "trend_summary": self.generate_trend_summary(primary_data),
                "anomaly_report": self.generate_anomaly_report(primary_data),
                "risk_commentary": self.generate_risk_commentary(primary_data),
                "comparison": self.generate_comparison(data, comparison_tickers),
            }
            section_sources = {
                "trend_summary": "deterministic" if self._is_deterministic_output(sectional.get("trend_summary", "")) else "llm",
                "anomaly_report": "deterministic" if self._is_deterministic_output(sectional.get("anomaly_report", "")) else "llm",
                "risk_commentary": "deterministic" if self._is_deterministic_output(sectional.get("risk_commentary", "")) else "llm",
                "comparison": "deterministic" if self._is_deterministic_output(sectional.get("comparison", "")) else "llm",
            }
            report = sectional
            report, section_sources = self._attempt_repair_deterministic_sections(
                data,
                comparison_tickers,
                report,
                section_sources,
                primary_ticker=_primary,
            )

            all_deterministic = all(v == "deterministic" for v in section_sources.values())
            all_llm = all(v == "llm" for v in section_sources.values())

            if not all_deterministic:
                report['analysis_mode'] = 'llm_recovered_full_sections' if all_llm else 'llm_sectional_fallback'
                report['fallback_reason'] = fallback_reason or "single_call_failed_sectional_used"
                report['section_sources'] = section_sources
            else:
                # Reuse already-produced deterministic sections; avoid recomputing fallback report.
                report['analysis_mode'] = 'deterministic_fallback'
                report['fallback_reason'] = fallback_reason or 'sectional_generation_failed_or_quota'
                report['section_sources'] = section_sources

        if report is not None:
            report['analysis_mode'] = report.get('analysis_mode', 'llm_single_call')
            report['fallback_reason'] = report.get('fallback_reason', '')
            report['section_sources'] = report.get('section_sources', {
                "trend_summary": "llm",
                "anomaly_report": "llm",
                "risk_commentary": "llm",
                "comparison": "llm",
            })

            report, report['section_sources'] = self._repair_weak_sections(
                report,
                data,
                comparison_tickers,
                report['section_sources'],
                primary_ticker=_primary,
            )

            report, report['section_sources'] = self._enforce_required_sections(
                report,
                data,
                comparison_tickers,
                report['section_sources'],
                primary_ticker=_primary,
            )

            if report.get('analysis_mode') == 'llm_single_call' and any(v != 'llm' for v in report['section_sources'].values()):
                report['analysis_mode'] = 'llm_recovered_full_sections'
                if not report.get('fallback_reason'):
                    report['fallback_reason'] = 'single_call_quality_repair'

            if report.get('analysis_mode') == 'llm_recovered_full_sections' and all(v == 'llm' for v in report['section_sources'].values()):
                report['fallback_reason'] = ''
        else:
            report = self._deterministic_full_report(data, comparison_tickers)
            report['analysis_mode'] = 'deterministic_fallback'
            report['fallback_reason'] = fallback_reason or 'sectional_generation_failed_or_quota'
            report['section_sources'] = {
                "trend_summary": "deterministic",
                "anomaly_report": "deterministic",
                "risk_commentary": "deterministic",
                "comparison": "deterministic",
            }

        report['model_used'] = self.model
        report['generated_at'] = datetime.now().isoformat()
        return report

    def run_full_analysis(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Backward-compatible wrapper for older pipeline code."""
        return self.generate_full_analysis(data)


# Backward-compatible class name used by existing imports in main.py
AIAgent = AnalysisAgent
