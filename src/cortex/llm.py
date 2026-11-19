"""LLM providers.

Agents decide with deterministic, testable rules; the LLM writes operator-facing
briefings and handles open-ended research. That split keeps money-moving logic
auditable and keeps the system runnable with no API key (the offline provider).
"""

from __future__ import annotations

import os
from typing import Protocol

from cortex.config import LLMSettings

BRIEFING_MARKER = "<<<BRIEFING>>>"


class LLM(Protocol):
    name: str

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 800) -> str: ...


class OfflineLLM:
    """Deterministic stand-in: returns the structured briefing embedded in the prompt."""

    name = "offline"

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 800) -> str:
        if BRIEFING_MARKER in prompt:
            return prompt.split(BRIEFING_MARKER, 1)[1].strip()
        return prompt.strip()[: max_tokens * 4]


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str, api_key: str) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("install the 'llm' extra: pip install 'cortex-agents[llm]'") from exc
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 800) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


def build_llm(settings: LLMSettings) -> LLM:
    if settings.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("llm.provider is 'anthropic' but ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(settings.model, api_key)
    return OfflineLLM()
