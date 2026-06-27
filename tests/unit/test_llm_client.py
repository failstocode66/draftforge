import pytest
from pydantic import BaseModel

from draftforge.llm.client import LLMClient, LLMError


class Out(BaseModel):
    label: str


class FakeTransport:
    """Test double for a transport.

    `responses` is a list whose items are either a `str` (returned from
    `.text()`) or an `Exception` instance (raised from `.text()`). Each call
    pops the next item and records the call.
    """

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
    return LLMClient(
        transport,
        model_fast="model-fast",
        model_smart="model-smart",
        sleep=lambda *_: None,
    )


def test_happy_path_returns_validated_model():
    transport = FakeTransport(['{"label": "ok"}'])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert isinstance(result, Out)
    assert result.label == "ok"
    assert len(transport.calls) == 1


def test_uses_smart_model_by_default_and_fast_when_requested():
    transport = FakeTransport(['{"label": "a"}', '{"label": "b"}'])
    client = make_client(transport)

    client.complete_json("sys", "usr", Out)
    client.complete_json("sys", "usr", Out, fast=True)

    assert transport.calls[0]["model"] == "model-smart"
    assert transport.calls[1]["model"] == "model-fast"


def test_extracts_json_embedded_in_prose():
    transport = FakeTransport(['Sure! Here you go:\n{"label": "ok"}\nThanks.'])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "ok"
    assert len(transport.calls) == 1


def test_bad_json_then_good_retries_once_and_succeeds():
    transport = FakeTransport(["not json at all", '{"label": "ok"}'])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "ok"
    assert len(transport.calls) == 2
    # The error from the first reply is fed back into the user prompt.
    assert "usr" in transport.calls[1]["user"]
    assert transport.calls[1]["user"] != transport.calls[0]["user"]


def test_two_bad_responses_raise_llmerror():
    transport = FakeTransport(["nope", "still nope"])
    client = make_client(transport)

    with pytest.raises(LLMError):
        client.complete_json("sys", "usr", Out)

    assert len(transport.calls) == 2


def test_validation_error_then_good_retries_once():
    # Valid JSON but wrong shape (missing required `label`) -> ValidationError.
    transport = FakeTransport(['{"wrong": "field"}', '{"label": "ok"}'])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "ok"
    assert len(transport.calls) == 2


def test_transport_exception_sleeps_and_retries():
    sleeps = []
    transport = FakeTransport([RuntimeError("rate limit"), '{"label": "ok"}'])
    client = LLMClient(
        transport,
        model_fast="model-fast",
        model_smart="model-smart",
        sleep=lambda s: sleeps.append(s),
    )

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "ok"
    assert len(transport.calls) == 2
    # Backoff of 2 ** attempt on the first (attempt 0) failure.
    assert sleeps == [1]


def test_two_transport_exceptions_raise_llmerror():
    transport = FakeTransport([RuntimeError("boom"), RuntimeError("boom again")])
    client = make_client(transport)

    with pytest.raises(LLMError):
        client.complete_json("sys", "usr", Out)

    assert len(transport.calls) == 2


# --- INT-1: multi-candidate JSON extraction -------------------------------------


def test_stray_brace_before_real_json_validates_without_burning_retry():
    # Realistic prose-wrapped output: a stray {curly} mention precedes the actual
    # JSON. The first balanced block ({curly}) is NOT valid for the schema, but
    # the extractor must keep scanning to the real object and validate it on the
    # SAME (first) transport call -- not waste the sole retry.
    reply = 'Here are the posts (in {curly} style): {"label": "ok"} -- enjoy!'
    transport = FakeTransport([reply])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "ok"
    assert len(transport.calls) == 1  # no retry burned


def test_fenced_json_block_validates():
    # A ```json fenced code block is the most common wrapper; it must validate.
    reply = '```json\n{"label": "fenced"}\n```'
    transport = FakeTransport([reply])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "fenced"
    assert len(transport.calls) == 1


def test_bare_fence_without_lang_validates():
    reply = '```\n{"label": "barefence"}\n```'
    transport = FakeTransport([reply])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "barefence"


def test_first_valid_candidate_is_returned_when_multiple_present():
    # Two balanced objects; the first that validates wins.
    reply = 'first {"label": "one"} then {"label": "two"}'
    transport = FakeTransport([reply])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "one"


def test_skips_invalid_object_to_reach_valid_one_same_call():
    # The first balanced object is valid JSON but the WRONG shape; the extractor
    # walks to the next candidate (right shape) within the same transport call.
    reply = 'note: {"unrelated": 1} actual: {"label": "ok"}'
    transport = FakeTransport([reply])
    client = make_client(transport)

    result = client.complete_json("sys", "usr", Out)

    assert result.label == "ok"
    assert len(transport.calls) == 1


def test_pure_prose_still_raises_after_retry():
    # No JSON at all in either reply -> the retry path is unchanged -> LLMError.
    transport = FakeTransport(["just some prose", "still only prose"])
    client = make_client(transport)

    with pytest.raises(LLMError):
        client.complete_json("sys", "usr", Out)

    assert len(transport.calls) == 2
