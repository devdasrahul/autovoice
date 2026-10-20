"""LLM provider abstraction for AutoVoice RAG.

Supported backends (selected by LLM_PROVIDER):
    1. 'none': Pure extractive synthesizer (zero external API calls, zero cost, 100% reliable).
    2. 'gemini': Google Gemini free-tier with backoff retry and graceful fallback to 'none'.
    3. 'local': Local or rule-based fallback model.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from autovoice.config import get_settings
from autovoice.logging_conf import get_logger
from autovoice.rag.prompts import DISCLAIMER_TEXT, SYSTEM_PROMPT

log = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for RAG generation backends."""

    @abstractmethod
    def generate(self, prompt: str, passages: list[dict[str, Any]]) -> str:
        """Generate cited answer given prompt and retrieved evidence passages."""
        ...


class ExtractiveLLMProvider(BaseLLMProvider):
    """Extractive synthesizer that constructs a cited answer without an external LLM."""

    def generate(self, prompt: str, passages: list[dict[str, Any]]) -> str:
        """Format top retrieved passages into an objective, cited summary report."""
        if not passages:
            return (
                f"No matching complaint records were found in the database. \n\n{DISCLAIMER_TEXT}"
            )

        lines = ["### Complaint Intelligence Summary\n"]
        for i, p in enumerate(passages[:5], 1):
            cmplid = p.get("id") or p.get("metadata", {}).get("cmplid", "N/A")
            make = p.get("metadata", {}).get("make", "UNKNOWN")
            model = p.get("metadata", {}).get("model", "")
            year = p.get("metadata", {}).get("model_year", "")
            comp = p.get("metadata", {}).get("component", "General")
            sev = p.get("metadata", {}).get("severity", "LOW")
            text = p.get("text", "").strip()

            lines.append(
                f"**{i}. {make} {model} ({year}) — Component: {comp} | Severity: {sev}** "
                f"[CMPLID: {cmplid}]\n"
                f"> {text}\n"
            )

        lines.append(f"\n*{DISCLAIMER_TEXT}*")
        return "\n".join(lines)


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini API provider (free tier) with rate-limit handling and extractive fallback."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str | None = None):
        """Initialize Gemini provider."""
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._fallback = ExtractiveLLMProvider()

    def generate(self, prompt: str, passages: list[dict[str, Any]]) -> str:
        """Query Gemini API with system instructions and citations; fallback on error."""
        if not self.api_key:
            log.info("rag.gemini_key_missing_using_extractive")
            return self._fallback.generate(prompt, passages)

        # Build grounded context
        evidence_lines = []
        for p in passages:
            cmplid = p.get("id") or p.get("metadata", {}).get("cmplid", "")
            evidence_lines.append(f"[CMPLID: {cmplid}]\n{p.get('text', '')}")
        evidence_text = "\n\n".join(evidence_lines)

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User Question: {prompt}\n\n"
            f"Retrieved Complaint Evidence:\n{evidence_text}\n\n"
            f"Answer the question with specific [CMPLID: ...] citations for every claim:"
        )

        try:
            import httpx

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
            }

            resp = httpx.post(url, json=payload, timeout=12.0)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return f"{text}\n\n*{DISCLAIMER_TEXT}*"

            log.warning("rag.gemini_call_failed", status=resp.status_code, text=resp.text[:200])
        except Exception as exc:
            log.warning("rag.gemini_exception_falling_back", error=str(exc))

        # Graceful fallback to extractive
        return self._fallback.generate(prompt, passages)


def get_llm_provider(provider: str | None = None) -> BaseLLMProvider:
    """Factory to return configured LLM provider backend."""
    settings = get_settings()
    p = (provider or settings.llm_provider).lower()

    if p == "gemini":
        return GeminiLLMProvider(
            model_name=settings.gemini_model,
            api_key=os.environ.get(settings.gemini_api_key_env, ""),
        )
    return ExtractiveLLMProvider()
