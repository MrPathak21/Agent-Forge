from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")

# Default models per provider
_DEFAULTS: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
}

# Env-var names per provider
_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_BASE_URL_ENV: dict[str, str] = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "gemini": "GEMINI_BASE_URL",
}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key: str
    model: str
    base_url: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


class Settings:
    """
    Central config resolver. Reads from environment / .env file.

    Usage:
        config = Settings.for_provider("openai")
        config = Settings.for_provider("anthropic", model="claude-opus-4-6")
    """

    @classmethod
    def for_provider(
        cls,
        provider: str = "openai",
        *,
        model: str | None = None,
    ) -> ProviderConfig:
        provider = provider.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider!r}. Choose from {SUPPORTED_PROVIDERS}."
            )

        api_key = os.getenv(_API_KEY_ENV[provider], "").strip()
        if not api_key:
            raise ValueError(
                f"{_API_KEY_ENV[provider]} is not set. Add it to your .env file."
            )

        base_url = os.getenv(_BASE_URL_ENV[provider], "").strip() or None
        resolved_model = model or _DEFAULTS[provider]

        return ProviderConfig(
            provider=provider,
            api_key=api_key,
            model=resolved_model,
            base_url=base_url,
        )
