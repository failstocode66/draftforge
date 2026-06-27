"""Tests for the voice-grounded generate stage.

Offline via a fake transport. We verify that (a) a batch of N generated posts
becomes N Drafts with platform set, status "draft", and deterministic ids;
(b) the smart model is used (D8: generate is fast=False); and (c) the injected
grounding (voice exemplars + guidance) actually appears in the system prompt,
so we know grounding is wired and not silently dropped.
"""

import json

import pytest

from draftforge.llm.client import LLMClient
from draftforge.models import Draft, ExtractedItem, Platform
from draftforge.stages.generate import generate_posts


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def text(self, *, model, system, user, max_tokens):
        self.calls.append(
            {"model": model, "system": system, "user": user, "max_tokens": max_tokens}
        )
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(transport):
    return LLMClient(transport, model_fast="fast", model_smart="smart", sleep=lambda *_: None)


def _batch(n):
    """Build a canned GeneratedBatch JSON string with n posts."""
    posts = [
        {
            "caption": f"caption {i}",
            "hashtags": [f"#tag{i}", "#float"],
            "image_direction": f"image {i}",
            "claims_used": [],
        }
        for i in range(n)
    ]
    return json.dumps({"posts": posts})


ITEM = ExtractedItem(hook="Float away stress", core_benefit="deep relaxation")
VOICE = "FB exemplar: Sink into stillness.\nIG exemplar: 60 minutes. Zero noise."
CORPUS = "Sam: we never make medical claims; we sell calm, not cures."
GUIDANCE = "Write in a warm, grounded, non-hypey voice."


def test_generate_returns_n_drafts_with_deterministic_ids():
    transport = FakeTransport([_batch(3)])
    client = make_client(transport)

    drafts = generate_posts(
        ITEM,
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        platform=Platform.instagram,
        n=3,
        llm=client,
        angle="educational",
    )

    assert len(drafts) == 3
    assert all(isinstance(d, Draft) for d in drafts)
    assert [d.id for d in drafts] == ["post-0", "post-1", "post-2"]
    assert all(d.platform == Platform.instagram for d in drafts)
    assert all(d.status == "draft" for d in drafts)
    assert drafts[0].caption == "caption 0"
    assert drafts[1].hashtags == ["#tag1", "#float"]
    assert drafts[2].image_direction == "image 2"


def test_generate_custom_id_prefix():
    transport = FakeTransport([_batch(2)])
    client = make_client(transport)

    drafts = generate_posts(
        ITEM,
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        platform=Platform.facebook,
        n=2,
        llm=client,
        angle="educational",
        id_prefix="fb",
    )

    assert [d.id for d in drafts] == ["fb-0", "fb-1"]
    assert all(d.platform == Platform.facebook for d in drafts)


def test_generate_uses_the_smart_model():
    transport = FakeTransport([_batch(1)])
    client = make_client(transport)

    generate_posts(
        ITEM,
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        platform=Platform.instagram,
        n=1,
        llm=client,
        angle="educational",
    )

    assert transport.calls[0]["model"] == "smart"


def test_generate_injects_voice_corpus_and_guidance_into_prompt():
    transport = FakeTransport([_batch(1)])
    client = make_client(transport)

    generate_posts(
        ITEM,
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        platform=Platform.instagram,
        n=1,
        llm=client,
        angle="educational",
    )

    system = transport.calls[0]["system"]
    # Grounding must actually reach the model, not be silently dropped.
    assert VOICE in system
    assert CORPUS in system
    assert GUIDANCE in system
    # The extracted item the post is built from is present too.
    assert ITEM.hook in system or ITEM.hook in transport.calls[0]["user"]
    # n is communicated to the model.
    assert "1" in system or "1" in transport.calls[0]["user"]


def test_generate_platform_conventions_differ_by_platform():
    fb = FakeTransport([_batch(1)])
    ig = FakeTransport([_batch(1)])

    generate_posts(ITEM, guidance=GUIDANCE, voice_exemplars=VOICE, corpus=CORPUS,
                   platform=Platform.facebook, n=1, llm=make_client(fb), angle="educational")
    generate_posts(ITEM, guidance=GUIDANCE, voice_exemplars=VOICE, corpus=CORPUS,
                   platform=Platform.instagram, n=1, llm=make_client(ig), angle="educational")

    # The platform name appears in each respective prompt.
    assert "facebook" in fb.calls[0]["system"].lower()
    assert "instagram" in ig.calls[0]["system"].lower()


def test_generate_threads_angle_onto_drafts():
    transport = FakeTransport([_batch(2)])
    client = make_client(transport)

    drafts = generate_posts(
        ITEM,
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        platform=Platform.instagram,
        n=2,
        llm=client,
        angle="myth_buster",
    )

    assert all(d.angle == "myth_buster" for d in drafts)


def test_generate_requires_angle():
    # angle is keyword-only and REQUIRED (no "other" default) so Draft.angle is
    # always the real classified angle, never a silent placeholder.
    transport = FakeTransport([_batch(1)])
    client = make_client(transport)

    with pytest.raises(TypeError):
        generate_posts(
            ITEM,
            guidance=GUIDANCE,
            voice_exemplars=VOICE,
            corpus=CORPUS,
            platform=Platform.instagram,
            n=1,
            llm=client,
        )


def test_generate_warns_when_model_returns_wrong_count(caplog):
    # Contract is best-effort: ask for n, but if the model returns a different
    # count, log a WARNING and proceed with what it returned (no hard failure).
    transport = FakeTransport([_batch(2)])  # model returns 2 ...
    client = make_client(transport)

    with caplog.at_level("WARNING"):
        drafts = generate_posts(
            ITEM,
            guidance=GUIDANCE,
            voice_exemplars=VOICE,
            corpus=CORPUS,
            platform=Platform.instagram,
            n=5,  # ... but we asked for 5
            llm=client,
            angle="educational",
        )

    assert len(drafts) == 2  # proceeds with the returned posts
    assert any(
        "5" in r.message and "2" in r.message and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_generate_no_warning_when_count_matches(caplog):
    transport = FakeTransport([_batch(3)])
    client = make_client(transport)

    with caplog.at_level("WARNING"):
        generate_posts(
            ITEM,
            guidance=GUIDANCE,
            voice_exemplars=VOICE,
            corpus=CORPUS,
            platform=Platform.instagram,
            n=3,
            llm=client,
            angle="educational",
        )

    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_generate_maps_claims_used_through():
    posts = {
        "posts": [
            {
                "caption": "We sell calm, not cures.",
                "hashtags": ["#float"],
                "image_direction": "calm water",
                "claims_used": ["clients report better sleep"],
            }
        ]
    }
    transport = FakeTransport([json.dumps(posts)])
    client = make_client(transport)

    drafts = generate_posts(
        ITEM,
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        platform=Platform.facebook,
        n=1,
        llm=client,
        angle="educational",
    )

    assert drafts[0].claims_used == ["clients report better sleep"]
