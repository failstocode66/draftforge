import hashlib
import typing

from draftforge.models import Source
from draftforge.ingest.normalize import to_source


def test_to_source_has_parameter_type_hints():
    # P1a polish: to_source carries parameter type hints (kind/uri/text at least).
    # get_type_hints resolves the stringified annotations (PEP 563) to objects;
    # we supply Source so the TYPE_CHECKING-only return forward-ref resolves.
    hints = typing.get_type_hints(to_source, localns={"Source": Source})
    for name in ("kind", "uri", "text"):
        assert hints[name] is str
    assert hints["return"] is Source


def test_produces_valid_source_with_injected_now_string():
    source = to_source(
        "url",
        "http://example.com/float",
        "some readable body text",
        title="Float Guide",
        now="2026-06-25T12:00:00Z",
    )

    assert isinstance(source, Source)
    assert source.type == "url"
    assert source.title == "Float Guide"
    assert source.text == "some readable body text"
    assert source.fetched_at == "2026-06-25T12:00:00Z"


def test_now_accepts_callable():
    source = to_source(
        "file",
        "/docs/notes.txt",
        "body",
        now=lambda: "2026-01-01T00:00:00Z",
    )
    assert source.fetched_at == "2026-01-01T00:00:00Z"


def test_source_id_is_deterministic_sha1_prefix_of_uri():
    uri = "http://example.com/float"
    expected = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]

    a = to_source("url", uri, "text one", now="2026-06-25T12:00:00Z")
    b = to_source("url", uri, "different text", now="2026-06-26T09:00:00Z")

    assert a.source_id == expected
    # Same URI -> same id regardless of body or timestamp.
    assert a.source_id == b.source_id


def test_different_uris_yield_different_ids():
    a = to_source("url", "http://a.example.com", "x", now="2026-06-25T12:00:00Z")
    b = to_source("url", "http://b.example.com", "x", now="2026-06-25T12:00:00Z")
    assert a.source_id != b.source_id


def test_title_defaults_to_none():
    source = to_source("file", "/docs/notes.txt", "body", now="2026-06-25T12:00:00Z")
    assert source.title is None


def test_round_trips_as_source_json():
    source = to_source(
        "url", "http://example.com", "body", title="T", now="2026-06-25T12:00:00Z"
    )
    restored = Source.model_validate_json(source.model_dump_json())
    assert restored == source
