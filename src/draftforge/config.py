"""Application configuration loaded from the environment.

`Settings.load()` reads configuration from environment variables (after pulling
in a local `.env` via python-dotenv, if present). The Anthropic API key is
required; everything else has a sensible default so the pipeline can run against
the documented model IDs without extra setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_MODEL_FAST = "claude-haiku-4-5-20251001"
DEFAULT_MODEL_SMART = "claude-opus-4-8"
DEFAULT_APP_PASSWORD = "change-me"


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    api_key: str
    model_fast: str
    model_smart: str
    app_password: str

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from the environment.

        `.env` values do not override variables already present in the
        environment (``load_dotenv`` defaults to ``override=False``), so process
        env and test monkeypatching stay authoritative.

        Raises:
            KeyError: if ``ANTHROPIC_API_KEY`` is not set.
        """
        load_dotenv()
        return cls(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_fast=os.environ.get("MODEL_FAST", DEFAULT_MODEL_FAST),
            model_smart=os.environ.get("MODEL_SMART", DEFAULT_MODEL_SMART),
            app_password=os.environ.get("APP_PASSWORD", DEFAULT_APP_PASSWORD),
        )
