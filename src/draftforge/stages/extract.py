"""Extract stage — pull structured marketing data for a given angle.

The angle decided by :func:`~draftforge.stages.classify.classify_content` selects
*which* extraction prompt runs here, because what you mine from a piece depends
on its angle: a ``personal_story`` yields a human hook and a before/after
benefit, while an ``offer_promo`` yields the offer and a call-to-action.

Prompt composition keeps things DRY: a shared base template
(``prompts/extract/_base.md``) defines the :class:`~draftforge.models.ExtractedItem`
JSON contract and field meanings once, and a per-angle file
(``prompts/extract/<angle>.md``) supplies extraction guidance specific to that
angle. The two are concatenated into the system prompt. Routed to the *fast*
model (D8).
"""

from __future__ import annotations

from typing import get_args

from draftforge.llm.client import LLMClient
from draftforge.models import ExtractedItem
from draftforge.stages import load_prompt, wrap_untrusted
from draftforge.stages.classify import Classification

# Angles that have a dedicated extraction-guidance prompt. SINGLE-SOURCED from
# the Classification angle Literal via typing.get_args, so the angle vocabulary
# has exactly one source of truth and cannot drift between classify and extract.
# (The per-angle prompt files under prompts/extract/ remain test-guarded.)
_ANGLES = frozenset(get_args(Classification.model_fields["angle"].annotation))

_BASE_PROMPT = ("extract", "_base.md")


def extract_marketing_data(text: str, angle: str, llm: LLMClient) -> ExtractedItem:
    """Extract a validated :class:`ExtractedItem` from ``text`` for ``angle``.

    Composes the shared base prompt with the angle-specific guidance prompt,
    then asks the *fast* model for the structured fields. Only ``hook`` and
    ``core_benefit`` are required; the model omits optionals it cannot ground.

    Args:
        text: The source material to mine.
        angle: One of the known angles (selects the per-angle prompt).
        llm: The schema-validating LLM client.

    Returns:
        A validated :class:`ExtractedItem`.

    Raises:
        ValueError: If ``angle`` is not a known angle.
    """
    if angle not in _ANGLES:
        raise ValueError(
            f"unknown angle {angle!r}; expected one of {sorted(_ANGLES)}"
        )

    base = load_prompt(*_BASE_PROMPT)
    guidance = load_prompt("extract", f"{angle}.md")
    system = f"{base}\n\n{guidance}"

    return llm.complete_json(system, wrap_untrusted(text), ExtractedItem, fast=True)
