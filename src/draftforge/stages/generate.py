"""Generate stage — write platform drafts, grounded in the business's voice.

This is the content core. It turns one :class:`~draftforge.models.ExtractedItem`
into ``n`` :class:`~draftforge.models.Draft` posts for a given platform, asking
the *smart* model (D8: ``fast=False``).

Grounding is everything here, so the system prompt injects four things, supplied
as **in-memory arguments** (D9 — never read from disk in this stage, never a
silent placeholder):

* ``voice_exemplars`` — few-shot examples of the brand's actual posts, so the
  model imitates the real voice rather than a generic one.
* ``corpus`` — excerpts of the business owner's real positions/voice, so claims
  and stance stay true to the business.
* ``guidance`` — the run's instruction prompt (tone, focus, do/don't).
* platform conventions — Facebook vs Instagram length/hashtag norms.

The model returns a JSON array validated against :class:`GeneratedBatch`; each
generated post is mapped to a :class:`Draft` with a **deterministic** id
``f"{id_prefix}-{i}"`` (no random/uuid, so the pipeline and tests are
reproducible).

The ``n`` count is a **best-effort** target, not a guarantee: we ask the model
for exactly ``n`` posts, but if it returns a different number we log a WARNING
and proceed with whatever it returned rather than hard-failing — the
over/under-supply is review-gated downstream anyway.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from draftforge.llm.client import LLMClient
from draftforge.models import Draft, ExtractedItem, Platform
from draftforge.stages import load_prompt

logger = logging.getLogger(__name__)

_PROMPT_FILE = "generate.md"

# Per-platform conventions injected into the prompt. Facebook tolerates a longer
# caption and fewer hashtags; Instagram wants a punchy caption and a hashtag set.
_PLATFORM_CONVENTIONS = {
    Platform.facebook: (
        "Platform: FACEBOOK. Write a slightly longer, conversational caption "
        "(2-4 sentences is fine). Use FEW hashtags (0-3), placed at the end. "
        "Facebook readers tolerate more prose; lead with the hook, then expand."
    ),
    Platform.instagram: (
        "Platform: INSTAGRAM. Write a punchy, scannable caption (1-2 short "
        "sentences or a tight line). Include a focused hashtag set (5-12 "
        "relevant tags). Instagram rewards immediacy and a strong first line."
    ),
}


class GeneratedPost(BaseModel):
    """One post as returned by the model (pre-:class:`Draft` shape)."""

    caption: str
    hashtags: list[str] = Field(default_factory=list)
    image_direction: str | None = None
    claims_used: list[str] = Field(default_factory=list)


class GeneratedBatch(BaseModel):
    """Wrapper schema: the model returns exactly this ``{"posts": [...]}``."""

    posts: list[GeneratedPost]


def generate_posts(
    item: ExtractedItem,
    *,
    guidance: str,
    voice_exemplars: str,
    corpus: str,
    platform: Platform,
    n: int,
    llm: LLMClient,
    angle: str,
    id_prefix: str = "post",
) -> list[Draft]:
    """Generate ``n`` grounded :class:`Draft` posts for ``platform``.

    Args:
        item: The extracted marketing material to write from.
        guidance: The run's instruction prompt (in-memory; D9).
        voice_exemplars: Few-shot brand-voice examples (in-memory; D9).
        corpus: Excerpts of the business's real positions/voice (in-memory; D9).
        platform: Target platform; selects the conventions block.
        n: Best-effort target post count (see note below). Communicated to the
            model and used to size the request.
        llm: The schema-validating LLM client (routed to the smart model).
        angle: The content angle (threaded from classify) stamped on each draft.
            Required (keyword-only) — there is no placeholder default, so
            ``Draft.angle`` is always the real classified angle.
        id_prefix: Prefix for the deterministic draft ids (``"{prefix}-{i}"``).

    Returns:
        A list of validated :class:`Draft` objects, ids ``f"{id_prefix}-{i}"``.
    """
    system = _build_system_prompt(
        item=item,
        guidance=guidance,
        voice_exemplars=voice_exemplars,
        corpus=corpus,
        platform=platform,
        n=n,
    )
    user = _build_user_prompt(item, n)

    # Generation can be long (n posts): give the smart model ample room.
    batch = llm.complete_json(
        system, user, GeneratedBatch, fast=False, max_tokens=4000
    )

    # Best-effort count contract: we ASK for exactly n, but the model may return
    # a different number. We do not hard-fail on a mismatch (re-prompting for an
    # exact count is wasteful and the surplus/deficit is review-gated anyway) —
    # we log a WARNING and proceed with whatever it returned.
    returned = len(batch.posts)
    if returned != n:
        logger.warning(
            "generate: requested %d post(s) but the model returned %d "
            "(platform=%s, angle=%s); proceeding with %d.",
            n,
            returned,
            platform,
            angle,
            returned,
        )

    return [
        Draft(
            id=f"{id_prefix}-{i}",
            platform=platform,
            angle=angle,
            caption=post.caption,
            hashtags=post.hashtags,
            image_direction=post.image_direction,
            claims_used=post.claims_used,
            status="draft",
        )
        for i, post in enumerate(batch.posts)
    ]


def _build_system_prompt(
    *,
    item: ExtractedItem,
    guidance: str,
    voice_exemplars: str,
    corpus: str,
    platform: Platform,
    n: int,
) -> str:
    """Compose the grounded system prompt from the template + injected context."""
    template = load_prompt(_PROMPT_FILE)
    conventions = _PLATFORM_CONVENTIONS[platform]
    return (
        f"{template}\n\n"
        f"## Run guidance\n{guidance}\n\n"
        f"## Brand voice — exemplars (imitate this voice)\n{voice_exemplars}\n\n"
        f"## Business corpus (the owner's real positions and voice)\n{corpus}\n\n"
        f"## Platform conventions\n{conventions}\n\n"
        f"## This run\n"
        f"Produce EXACTLY {n} post(s) as JSON: "
        f'{{"posts": [{{"caption": ..., "hashtags": [...], '
        f'"image_direction": ..., "claims_used": [...]}}]}}.\n'
        f"Source hook: {item.hook}\n"
        f"Core benefit: {item.core_benefit}"
    )


def _build_user_prompt(item: ExtractedItem, n: int) -> str:
    """The user turn carries the concrete extracted item to write from."""
    lines = [
        f"Write {n} post(s) from this extracted material:",
        f"- hook: {item.hook}",
        f"- core_benefit: {item.core_benefit}",
    ]
    if item.claim:
        lines.append(f"- claim: {item.claim} ({item.claim_type or 'unspecified'})")
    if item.supporting_source:
        lines.append(f"- supporting_source: {item.supporting_source}")
    if item.audience:
        lines.append(f"- audience: {item.audience}")
    if item.suggested_cta:
        lines.append(f"- suggested_cta: {item.suggested_cta}")
    return "\n".join(lines)
