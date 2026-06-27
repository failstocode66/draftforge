"""Live smoke test against the real Anthropic API.

Marked ``smoke`` and therefore excluded by the default ``-m "not smoke"`` addopts
in pytest.ini. It also skips when ``ANTHROPIC_API_KEY`` is unset, so opting in
without a key is a clean skip rather than a failure. Run it explicitly with::

    pytest -m smoke

This exercises the full stack — AnthropicTransport + LLMClient + schema
validation — end to end against the model IDs in the environment.
"""

import os

import pytest
from pydantic import BaseModel

from draftforge.config import Settings
from draftforge.llm.anthropic_transport import AnthropicTransport
from draftforge.llm.client import LLMClient

pytestmark = pytest.mark.smoke


class Out(BaseModel):
    label: str


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; live smoke test requires a real key",
)
def test_live_llm_returns_schema_shaped_object():
    settings = Settings.load()
    client = LLMClient(
        AnthropicTransport(),
        model_fast=settings.model_fast,
        model_smart=settings.model_smart,
    )

    result = client.complete_json(
        system="You are a JSON API. Respond with only the requested JSON object.",
        user='Return exactly this JSON object and nothing else: {"label": "ok"}',
        schema=Out,
        fast=True,
    )

    assert isinstance(result, Out)
    assert result.label == "ok"
