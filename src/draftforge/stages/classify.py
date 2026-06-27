"""Classify stage — pick the marketing *angle* of a piece of source text.

This is the first LLM stage and establishes the prompt-stage pattern the later
stages follow: a tiny output schema, a system prompt loaded from a committed
``prompts/*.md`` file at call time, and a single ``complete_json`` call routed
to the fast model (D8).

The angle returned here drives the *next* stage: ``extract`` selects its
per-angle prompt by this value, so the ``Literal`` set must stay in lockstep
with the prompt files under ``prompts/extract/``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from draftforge.llm.client import LLMClient
from draftforge.stages import load_prompt, wrap_untrusted

_PROMPT_FILE = "classify.md"


class Classification(BaseModel):
    """The classifier's structured output: a single content angle."""

    angle: Literal[
        "educational",
        "personal_story",
        "benefit_spotlight",
        "myth_buster",
        "offer_promo",
        "other",
    ]


def classify_content(text: str, llm: LLMClient) -> Classification:
    """Classify ``text`` into one of the known marketing angles.

    Loads the few-shot classifier prompt at call time and asks the *fast* model
    for a JSON ``{"angle": ...}``. The :class:`LLMClient` validates the reply
    against :class:`Classification` and retries on a bad/unknown angle, raising
    :class:`~draftforge.llm.client.LLMError` if it still cannot get a valid value.

    Args:
        text: The source material to classify.
        llm: The schema-validating LLM client.

    Returns:
        A validated :class:`Classification`.
    """
    system = load_prompt(_PROMPT_FILE)
    return llm.complete_json(system, wrap_untrusted(text), Classification, fast=True)
