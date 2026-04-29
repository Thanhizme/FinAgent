"""
ai_agent.py
-----------
Integrates a Large Language Model (LLM) to produce automated natural language
analysis of processed financial data.

Supported providers (configured via .env):
  - Google Gemini     (GEMINI_API_KEY)       [default]
  - Anthropic Claude  (ANTHROPIC_API_KEY)
  - OpenAI GPT        (OPENAI_API_KEY)

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
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AIAgent:
    """
    Orchestrates LLM calls to generate natural language financial analysis.

    Parameters
    ----------
    provider : {'gemini', 'anthropic', 'openai'}
        LLM provider to use. The corresponding API key must be set in .env.
        Defaults to 'gemini'.
    model : str, optional
        Model identifier. Defaults per provider:
          - gemini    : 'gemini-1.5-flash'
          - anthropic : 'claude-3-5-sonnet-20241022'
          - openai    : 'gpt-4o'
    max_tokens : int
        Maximum tokens to generate per response.
    """

    _DEFAULT_MODELS: dict[str, str] = {
        "gemini":    "gemini-1.5-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai":    "gpt-4o",
    }

    _ENV_KEYS: dict[str, str] = {
        "gemini":    "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
    }

    def __init__(
        self,
        provider: str = "gemini",
        model: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        if provider.lower() not in self._DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Choose from: {list(self._DEFAULT_MODELS.keys())}"
            )
        self.provider = provider.lower()
        self.model = model or self._DEFAULT_MODELS[self.provider]
        self.max_tokens = max_tokens
        self._client = self._init_client()

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _init_client(self):
        """
        Instantiate the appropriate SDK client based on self.provider.

        Returns
        -------
        google.generativeai.GenerativeModel | anthropic.Anthropic | openai.OpenAI
            Authenticated SDK client.

        Raises
        ------
        EnvironmentError
            If the required API key environment variable is not set.
        ValueError
            If an unsupported provider is specified.
        """
        env_var = self._ENV_KEYS[self.provider]
        api_key = os.getenv(env_var)

        if not api_key:
            raise EnvironmentError(
                f"API key not found. Please set '{env_var}' in your .env file."
            )

        if self.provider == "gemini":
            # TODO: import google.generativeai as genai
            #       genai.configure(api_key=api_key)
            #       return genai.GenerativeModel(self.model)
            raise NotImplementedError

        if self.provider == "anthropic":
            # TODO: import anthropic
            #       return anthropic.Anthropic(api_key=api_key)
            raise NotImplementedError

        if self.provider == "openai":
            # TODO: import openai
            #       return openai.OpenAI(api_key=api_key)
            raise NotImplementedError

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _build_context(self, data: dict[str, pd.DataFrame]) -> str:
        """
        Serialise processed DataFrames into a compact JSON string suitable for
        inclusion in an LLM prompt.

        Included statistics per ticker:
          - Latest date and closing price
          - 7-day and 30-day rolling averages (last value)
          - 30-day annualised volatility (last value)
          - Min / max closing price over the entire period
          - Cumulative return (%)

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Processed DataFrames produced by DataProcessor.

        Returns
        -------
        str
            JSON-formatted string to embed in the prompt.
        """
        # TODO: iterate data, extract summary stats, return json.dumps(summary, indent=2)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def generate_trend_summary(self, data: dict[str, pd.DataFrame]) -> str:
        """
        Produce a concise trend and recent performance summary for each asset.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]

        Returns
        -------
        str
            LLM-generated narrative referencing specific prices and dates.
        """
        # TODO: build prompt with _build_context, call _call_llm, return text
        raise NotImplementedError

    def generate_anomaly_report(self, data: dict[str, pd.DataFrame]) -> str:
        """
        Identify and narrate notable events or data anomalies in the dataset.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]

        Returns
        -------
        str
            LLM-generated report citing flagged outlier dates and magnitudes.
        """
        # TODO: filter rows where is_outlier == True, embed in prompt
        raise NotImplementedError

    def generate_risk_commentary(self, data: dict[str, pd.DataFrame]) -> str:
        """
        Provide a risk commentary grounded in volatility and drawdown metrics.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]

        Returns
        -------
        str
            LLM-generated risk narrative with specific volatility figures.
        """
        # TODO: extract volatility_30 statistics, embed in prompt
        raise NotImplementedError

    def generate_comparison(
        self,
        data: dict[str, pd.DataFrame],
        tickers: Optional[list[str]] = None,
    ) -> str:
        """
        Write a comparative analysis paragraph for two or more selected assets.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
        tickers : list[str], optional
            Subset of tickers to compare. Defaults to all keys in data.

        Returns
        -------
        str
            LLM-generated comparison narrative.
        """
        # TODO: build side-by-side context for selected tickers, call _call_llm
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Full report runner
    # ------------------------------------------------------------------

    def run_full_analysis(self, data: dict[str, pd.DataFrame]) -> dict[str, str]:
        """
        Execute all four analysis tasks and return results as a dictionary.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]

        Returns
        -------
        dict[str, str]
            Keys: 'trend_summary', 'anomaly_report', 'risk_commentary', 'comparison'.
        """
        # TODO: call each generate_* method, collect results, log completion
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Low-level LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request to the configured LLM provider.

        Parameters
        ----------
        system_prompt : str
            System-level instruction that sets the model persona and constraints.
        user_prompt : str
            User message containing the structured data context and task.

        Returns
        -------
        str
            Raw text content of the model response.

        Notes
        -----
        - Always include explicit instructions to reference specific numbers.
        - Add a disclaimer that outputs are not financial advice.

        Provider call patterns:
          - gemini    : client.generate_content(system_prompt + user_prompt)
          - anthropic : client.messages.create(...)
          - openai    : client.chat.completions.create(...)
        """
        # TODO: branch on self.provider and call the appropriate SDK method
        raise NotImplementedError