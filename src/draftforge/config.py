"""Application configuration loaded from the environment.

`Settings.load()` reads configuration from environment variables (after pulling
in a local `.env` via python-dotenv, if present). The Anthropic API key is
required; the model IDs have documented defaults so the pipeline can run without
extra setup.

The web app's launch configuration (:func:`load_launch_config`) is stricter. It
requires a deliberately supplied ``APP_PASSWORD`` (there is no default), binds to
loopback unless the operator explicitly opts in to another address, and is
validated before any UI, store, or listener is created.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from dotenv import load_dotenv

DEFAULT_MODEL_FAST = "claude-haiku-4-5-20251001"
DEFAULT_MODEL_SMART = "claude-opus-4-8"

# Example passwords this repo's docs/templates have shown. They are public, so they
# are never accepted as a deployable APP_PASSWORD (compared case-insensitively,
# ignoring surrounding whitespace).
EXAMPLE_APP_PASSWORDS = frozenset({"change-me"})

LOOPBACK_HOST = "127.0.0.1"
# Explicit operator opt-in for a non-loopback bind (e.g. a hosted Space). Gradio's
# own GRADIO_SERVER_NAME is deliberately NOT consulted: the launcher always passes
# an explicit server_name.
HOST_ENV_VAR = "DRAFTFORGE_HOST"


class LaunchConfigError(ValueError):
    """The web app's launch configuration is missing, unsafe, or malformed.

    Messages describe the problem only; they never include a password or key.
    """


@dataclass(frozen=True)
class Settings:
    """Resolved application settings."""

    api_key: str = field(repr=False)
    model_fast: str
    model_smart: str
    # None when APP_PASSWORD is unset. There is no default: the web app validates
    # it with validate_app_password() before launching; the CLI never needs it.
    app_password: str | None = field(repr=False)

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
            app_password=os.environ.get("APP_PASSWORD"),
        )


def validate_app_password(value: str | None) -> str:
    """Return ``value`` unchanged if it is a deliberately supplied password.

    Rejects a missing, empty or whitespace-only value and any documented example
    value. This is a deny-list of known-unusable values, not a strength check:
    choosing a strong, unique password remains the operator's responsibility.

    Raises:
        LaunchConfigError: describing which rule failed (never the value).
    """
    if value is None:
        raise LaunchConfigError(
            "APP_PASSWORD is not set. The web app has no default password; set "
            "your own value (environment or a local .env) before launching."
        )
    if not isinstance(value, str):
        raise LaunchConfigError("APP_PASSWORD must be a string.")
    if not value.strip():
        raise LaunchConfigError(
            "APP_PASSWORD is empty or whitespace-only. Set your own password."
        )
    if value.strip().lower() in EXAMPLE_APP_PASSWORDS:
        raise LaunchConfigError(
            "APP_PASSWORD is a documented example value, which is public. Set "
            "your own password."
        )
    return value


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class LaunchConfig:
    """A validated web-app launch configuration (see :func:`load_launch_config`)."""

    app_password: str = field(repr=False)
    server_name: str
    share: bool

    @property
    def external(self) -> bool:
        """True when the app is reachable beyond this machine (bind or share link)."""
        return self.share or not _is_loopback(self.server_name)


def load_launch_config(
    *,
    server_name: str | None = None,
    share: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> LaunchConfig:
    """Validate the web app's launch configuration before anything is built.

    Args:
        server_name: Explicit bind address (``--host`` / ``launch(server_name=)``).
            When ``None``: ``DRAFTFORGE_HOST`` if set, else ``127.0.0.1``.
        share: Explicit Gradio share-link opt-in; ``None`` means off. Only a real
            ``bool`` is accepted.
        environ: Mapping to read (tests). Default: ``os.environ`` after loading a
            local ``.env``.

    Raises:
        LaunchConfigError: unusable ``APP_PASSWORD`` (see
            :func:`validate_app_password`), a blank/malformed bind host, a
            non-bool ``share``, or no ``ANTHROPIC_API_KEY``.
    """
    if environ is None:
        load_dotenv()
        environ = os.environ

    password = validate_app_password(environ.get("APP_PASSWORD"))

    if server_name is None:
        source = HOST_ENV_VAR if HOST_ENV_VAR in environ else None
        server_name = environ.get(HOST_ENV_VAR, LOOPBACK_HOST)
    else:
        source = "server_name"
    if (
        not isinstance(server_name, str)
        or not server_name
        or any(ch.isspace() for ch in server_name)
    ):
        raise LaunchConfigError(
            f"{source or 'the bind host'} must be a host name or IP address "
            "without whitespace (leave it unset for the 127.0.0.1 default)."
        )

    if share is None:
        share = False
    if not isinstance(share, bool):
        raise LaunchConfigError("share must be True or False.")

    if not (environ.get("ANTHROPIC_API_KEY") or "").strip():
        raise LaunchConfigError(
            "ANTHROPIC_API_KEY is not set. The web app needs it to generate drafts."
        )

    return LaunchConfig(app_password=password, server_name=server_name, share=share)
