"""Tests for output rendering (text + JSON).

Both renderers are pure and offline — no LLM, no network. We verify that:

* ``render_text`` produces a human-readable block carrying the load-bearing
  fields (caption, hashtags, platform, angle, image direction, claims), and is
  deterministic given an injected ``now`` (no wall-clock leak).
* ``render_json`` emits the model's JSON and round-trips back to an equal
  :class:`Draft`, so persisted output is loss-free.
"""

from draftforge.models import Draft, Platform
from draftforge.output.render import render_json, render_text


DRAFT = Draft(
    id="src0-instagram-1",
    platform=Platform.instagram,
    angle="benefit_spotlight",
    caption="Sixty minutes of weightless quiet. Your nervous system exhales.",
    hashtags=["#floattherapy", "#stillness", "#wellness"],
    image_direction="Overhead shot of calm water, soft warm light.",
    claims_used=["clients report deeper rest"],
)

FIXED_NOW = "2026-06-25T12:00:00Z"


def test_render_text_contains_caption_hashtags_and_platform():
    out = render_text(DRAFT, now=FIXED_NOW)

    assert DRAFT.caption in out
    for tag in DRAFT.hashtags:
        assert tag in out
    assert "instagram" in out.lower()


def test_render_text_includes_angle_image_direction_and_claims():
    out = render_text(DRAFT, now=FIXED_NOW)

    assert "benefit_spotlight" in out
    assert DRAFT.image_direction in out
    assert "clients report deeper rest" in out


def test_render_text_is_deterministic_with_injected_now():
    a = render_text(DRAFT, now=FIXED_NOW)
    b = render_text(DRAFT, now=FIXED_NOW)
    assert a == b
    # The injected timestamp appears (no hidden wall-clock).
    assert FIXED_NOW in a


def test_render_text_carries_draft_id_and_status():
    out = render_text(DRAFT, now=FIXED_NOW)
    assert DRAFT.id in out
    assert DRAFT.status in out


def test_render_text_handles_optional_fields_absent():
    bare = Draft(
        id="x-1",
        platform=Platform.facebook,
        angle="other",
        caption="Just a quiet afternoon.",
        hashtags=[],
    )
    # No image_direction, no claims, no hashtags — must not raise.
    out = render_text(bare, now=FIXED_NOW)
    assert "Just a quiet afternoon." in out
    assert "facebook" in out.lower()


def test_render_json_round_trips_to_equal_draft():
    payload = render_json(DRAFT)
    restored = Draft.model_validate_json(payload)
    assert restored == DRAFT
