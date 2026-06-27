"""Real Anthropic transport for :class:`draftforge.llm.client.LLMClient`.

This is the only place in the package that imports the ``anthropic`` SDK or
touches the network. It satisfies the transport protocol the client depends on
(``text(*, model, system, user, max_tokens) -> str``) by wrapping
``anthropic.Anthropic()`` and returning the first text block of the response.

Constructing :class:`AnthropicTransport` builds an ``anthropic.Anthropic``
client, which resolves credentials from the environment (``ANTHROPIC_API_KEY``).
Unit tests never construct this class — they inject a fake transport instead —
so the whole suite runs with no key and no network.
"""

from __future__ import annotations

import anthropic


class AnthropicTransport:
    """Transport backed by the Anthropic Messages API."""

    def __init__(self, client: "anthropic.Anthropic | None" = None) -> None:
        # Default client resolves ANTHROPIC_API_KEY from the environment.
        self._client = client if client is not None else anthropic.Anthropic()

    def text(self, *, model: str, system: str, user: str, max_tokens: int) -> str:
        """Send a single-turn request and return the response's first text block."""
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text
