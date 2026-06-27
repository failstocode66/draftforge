"""Unit tests for the Review-queue handlers + claim badge (Task 3.3 / M5).

Covers the pure ``_claim_badge`` helper (the Improvement-#1 advisory badge — the
``advisory`` lane is the gentle, non-blocking tier) and the store-backed review
handlers: edit / approve / reject, per-post media swap+remove, and regenerate.
The key compliance invariant under test: a draft with ANY claim status is still
approvable — badges advise, they never block.
"""

from __future__ import annotations

import json

import pytest

from draftforge import app
from draftforge.llm.client import LLMClient
from draftforge.models import Draft, MediaKind, MediaRef, Platform
from draftforge.store.db import Store

NOW = "2026-06-26T12:00:00Z"


@pytest.fixture
def store():
    return Store(":memory:")


def _seed(store, *, claim_status="clean", post="d1", batch="b1", **draft_overrides):
    """Persist one draft (with claim_flags) ready for the review handlers."""
    store.add_batch(batch, guidance_prompt="warm voice", url_set=[], batch_size=1, now=NOW)
    base = dict(
        id=post, platform=Platform.instagram, angle="relaxation",
        caption="Sink into stillness.", hashtags=["#float"],
    )
    base.update(draft_overrides)
    store.save_draft(Draft(**base), batch, now=NOW)
    store.set_claim_flags(
        post, batch, {"status": claim_status, "notes": ["a note"], "revised_text": None}
    )


# --- offline llm doubles (for regenerate) ---------------------------------------


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)

    def text(self, *, model, system, user, max_tokens):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(*responses):
    return LLMClient(FakeTransport(responses), model_fast="f", model_smart="s", sleep=lambda *_: None)


def _generate(n):
    return json.dumps(
        {"posts": [{"caption": f"caption {i}", "hashtags": [f"#t{i}"]} for i in range(n)]}
    )


def _claims_clean():
    return json.dumps(
        {"claims": [], "harmful": False, "harmful_reason": "", "softened_caption": None}
    )


@pytest.fixture
def grounded_base(tmp_path):
    voice = tmp_path / "prompts" / "voice_exemplars.md"
    voice.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n\n---\n\n".join(f"Post {i}: sink into stillness." for i in range(6))
    voice.write_text(f"## Facebook\n\n{blocks}\n", encoding="utf-8")
    corpus = tmp_path / "data" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "ep1.txt").write_text("Sam: we sell calm, not cures.", encoding="utf-8")
    reg = tmp_path / "data" / "claims_register.json"
    reg.write_text(json.dumps([
        {"claim_text": "magnesium is absorbed through the skin", "claim_type": "hard",
         "approved": True, "source_citation": "Waring 2006", "notes": ""}
    ]), encoding="utf-8")
    return tmp_path


# --- _claim_badge ---------------------------------------------------------------


@pytest.mark.parametrize(
    "status,tone",
    [
        ("clean", "clean"),
        ("advisory", "advisory"),
        ("softened", "softened"),
        ("flagged", "flagged"),
        ("needs_manual_review", "needs_review"),
    ],
)
def test_claim_badge_maps_status_to_tone(status, tone):
    label, t = app._claim_badge({"status": status, "notes": [], "revised_text": None})
    assert t == tone
    assert isinstance(label, str) and label  # non-empty label


def test_claim_badge_advisory_wording_is_the_signed_off_copy():
    label, tone = app._claim_badge({"status": "advisory", "notes": [], "revised_text": None})
    assert label == "ⓘ Health claim — your call"
    assert tone == "advisory"


def test_claim_badge_advisory_is_distinct_from_softened_and_flagged():
    adv = app._claim_badge({"status": "advisory"})
    soft = app._claim_badge({"status": "softened"})
    flag = app._claim_badge({"status": "flagged"})
    assert adv != soft
    assert adv != flag


def test_claim_badge_none_or_missing_is_clean():
    assert app._claim_badge(None)[1] == "clean"
    assert app._claim_badge({})[1] == "clean"


def test_claim_badge_unknown_status_fails_safe_to_needs_review():
    # An unexpected status must never read as benign; default to the safe tier.
    assert app._claim_badge({"status": "bananas"})[1] == "needs_review"


# --- review handlers ------------------------------------------------------------


def test_handle_edit_sets_text_and_status(store):
    _seed(store)
    d = app.handle_edit("d1", "b1", "a calmer rewrite", store=store)
    assert d.edited_text == "a calmer rewrite"
    assert d.status == "edited"


def test_handle_approve_sets_status_and_date(store):
    _seed(store)
    d = app.handle_approve("d1", "b1", store=store, scheduled_date="2026-07-01")
    assert d.status == "approved"
    assert d.scheduled_date == "2026-07-01"


@pytest.mark.parametrize("status", ["advisory", "softened", "flagged", "needs_manual_review"])
def test_handle_approve_is_non_blocking_for_every_claim_status(store, status):
    # The compliance invariant: badges ADVISE, they never block approval.
    _seed(store, claim_status=status)
    d = app.handle_approve("d1", "b1", store=store)
    assert d.status == "approved"


def test_handle_reject_reverts_to_draft(store):
    _seed(store)
    app.handle_approve("d1", "b1", store=store)
    d = app.handle_reject("d1", "b1", store=store)
    assert d.status == "draft"


def test_handle_set_media_swaps_then_removes(store):
    _seed(store)
    ref = MediaRef(kind=MediaKind.uploaded_image, ref="a.jpg")
    d = app.handle_set_media("d1", "b1", ref, store=store)
    assert d.media == ref
    # swap to a different one
    ref2 = MediaRef(kind=MediaKind.uploaded_video, ref="b.mp4")
    d = app.handle_set_media("d1", "b1", ref2, store=store)
    assert d.media == ref2
    # remove
    d = app.handle_set_media("d1", "b1", None, store=store)
    assert d.media is None


def test_handle_regenerate_writes_new_text_into_edited_text(grounded_base, store):
    _seed(store)
    llm = make_llm(_generate(1), _claims_clean())  # regenerate one + claims-check
    d = app.handle_regenerate("d1", "b1", llm=llm, store=store, base_dir=grounded_base)
    assert d.status == "edited"
    assert d.edited_text == "caption 0"  # the regenerated caption from _generate(1)


def test_handle_regenerate_missing_post_raises(grounded_base, store):
    store.add_batch("b1", guidance_prompt="g", url_set=[], batch_size=1, now=NOW)
    with pytest.raises(KeyError):
        app.handle_regenerate("ghost", "b1", llm=make_llm(), store=store, base_dir=grounded_base)


# --- review display helpers -----------------------------------------------------


def test_card_markdown_shows_badge_and_media(store):
    _seed(store, claim_status="advisory")
    store.set_media("d1", "b1", MediaRef(kind=MediaKind.uploaded_image, ref="/path/to/a.jpg"))
    md = app._card_markdown(store.get_draft("d1", "b1"), store.get_post_row("d1", "b1")["claim_flags"])
    assert "Health claim — your call" in md
    assert "a.jpg" in md
    assert "instagram" in md


def test_card_markdown_shows_shot_direction_when_no_media(store):
    _seed(store, image_direction="calm water, phone away")
    md = app._card_markdown(store.get_draft("d1", "b1"), {"status": "clean", "notes": []})
    assert "no media" in md
    assert "calm water" in md


def test_media_from_label_resolves_keep_remove_and_ref(store):
    _seed(store)
    store.set_media("d1", "b1", MediaRef(kind=MediaKind.uploaded_image, ref="/x/a.jpg"))
    assert app._media_from_label("(keep)", "b1", store) == "keep"
    assert app._media_from_label("(remove)", "b1", store) is None
    assert app._media_from_label("uploaded_image: a.jpg", "b1", store) == MediaRef(
        kind=MediaKind.uploaded_image, ref="/x/a.jpg"
    )
