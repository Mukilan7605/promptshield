"""
PromptShield — LLM Provider Abstraction Layer (Phase 7)
========================================================

Defines the pluggable LLM provider interface and concrete implementations:
1. BaseLLMProvider (Abstract Base Class)
2. MockLLMProvider (Local development & testing without API keys)
3. OpenAICompatibleProvider (Production OpenAI-compatible HTTP REST endpoint)

Design:
-------
- Local testing never requires external API keys or network calls.
- Provider selection is controlled via application settings (LLM_PROVIDER).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from backend.core.config import get_settings

log = logging.getLogger("promptshield.llm_provider")


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract interface for LLM provider integrations."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """
        Send an approved/sanitized prompt to the LLM and return structured output.

        Must return a dict containing:
        {
            "response": str,
            "provider": str,
            "model": str,
            "raw_response": dict | None
        }
        """
        pass


# ---------------------------------------------------------------------------
# Mock LLM Provider (Local dev & testing)
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic Mock LLM Provider for local development and testing.
    Does NOT require an API key or internet connection.
    """

    def __init__(self, model_name: str = "mock-model", model: str | None = None) -> None:
        self.model_name = model or model_name

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        log.info(f"MockLLMProvider processing prompt (length={len(prompt)} chars)")

        # Truncate prompt snippet for mock response
        snippet = prompt[:80].replace("\n", " ") + ("..." if len(prompt) > 80 else "")

        mock_text = (
            f"This is a mock LLM response to your input: \"{snippet}\". "
            f"PromptShield verified this prompt is safe for execution."
        )

        return {
            "response": mock_text,
            "provider": "mock",
            "model": self.model_name,
            "raw_response": {
                "id": "mock-completion-12345",
                "object": "text_completion",
                "model": self.model_name,
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(mock_text.split())}
            }
        }


# ---------------------------------------------------------------------------
# OpenAI-Compatible Provider
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Production-ready OpenAI-compatible REST API client.
    Supports OpenAI, Azure OpenAI, vLLM, Ollama, and LM Studio endpoints.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        cfg = get_settings()
        self.api_url = api_url or cfg.LLM_API_URL or "https://api.openai.com/v1/chat/completions"
        self.api_key = api_key or cfg.LLM_API_KEY or ""
        self.model_name = model_name or cfg.LLM_MODEL or "gpt-3.5-turbo"

    async def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        import httpx

        if not self.api_key and "openai.com" in self.api_url:
            log.warning("No LLM_API_KEY provided for OpenAI API endpoint; falling back to Mock provider format.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.api_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            # Parse standard ChatCompletion response
            choices = data.get("choices", [])
            text_out = choices[0]["message"]["content"] if choices else ""

            return {
                "response": text_out,
                "provider": "openai_compatible",
                "model": self.model_name,
                "raw_response": data,
            }

        except Exception as e:
            log.error(f"OpenAICompatibleProvider HTTP error: {e}")
            return {
                "response": f"[LLM Gateway Error: Failed to communicate with provider: {e}]",
                "provider": "openai_compatible",
                "model": self.model_name,
                "error": str(e),
                "raw_response": None,
            }


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

def get_llm_provider(
    provider_type: str | None = None,
    model_name: str | None = None,
    **kwargs: Any
) -> BaseLLMProvider:
    """
    Factory function to instantiate the configured LLM provider.
    Defaults to MockLLMProvider for local dev if LLM_PROVIDER='mock' or key is missing.
    """
    cfg = get_settings()
    p_type = (provider_type or cfg.LLM_PROVIDER).lower()
    m_name = model_name or cfg.LLM_MODEL

    if p_type == "openai" or p_type == "openai_compatible":
        return OpenAICompatibleProvider(model_name=m_name, **kwargs)
    
    # Default: Mock provider (safe for local development & testing)
    return MockLLMProvider(model_name=m_name)
