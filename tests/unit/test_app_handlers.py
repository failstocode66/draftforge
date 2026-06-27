"""Unit tests for the Gradio app's Run-tab handler (Task 3.2).

The handler ``handle_run`` is gradio-free: it ingests the URL list, runs the
pipeline with an injected FakeLLM, pairs uploaded media, and persists the batch
into an in-memory Store — all offline. ``build_ui``/``launch`` import gradio
lazily, so importing this module never requires gradio or a key.
"""

from __future__ import annotations

import json

import pytest

from draftforge import app
from draftforge.llm.client import LLMClient
from draftforge.models import MediaKind
from draftforge.store.db import Store


# --- offline doubles (mirrors test_cli) -----------------------------------------


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)

    def text(self, *, model, system, user, max_tokens):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(*responses):
    return LLMClient(
        FakeTransport(responses), model_fast="fast", model_smart="smart",
        sleep=lambda *_: None,
    )


def _classify(angle="educational"):
    return json.dumps({"angle": angle})


def _extract():
    return json.dumps({"hook": "Float away stress", "core_benefit": "deep calm"})


def _generate(n):
    return json.dumps(
        {"posts": [{"caption": f"caption {i}", "hashtags": [f"#t{i}"]} for i in range(n)]}
    )


def _claims_clean():
    return json.dumps(
        {"claims": [], "harmful": False, "harmful_reason": "", "softened_caption": None}
    )


def _full_run_llm():
    # classify, extract, then (generate + claims) per platform for n=2.
    return make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),  # facebook
        _generate(1), _claims_clean(),  # instagram
    )


_ARTICLE_HTML = "<html><body><article>" + ("Floating is deeply calming. " * 20) + "</article></body></html>"


def _getter(html=_ARTICLE_HTML):
    class _Resp:
        status_code = 200
        text = html

    def _g(url, *a, **k):
        return _Resp()

    return _g


@pytest.fixture
def store():
    return Store(":memory:")


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


# --- handle_run -----------------------------------------------------------------


def test_handle_run_ingests_runs_and_persists(grounded_base, store):
    out = app.handle_run(
        ["https://example.com/a"], "warm voice", 2,
        llm=_full_run_llm(), store=store, base_dir=grounded_base,
        getter=_getter(), batch_id_factory=lambda: "batch-app-1",
    )
    assert out["batch_id"] == "batch-app-1"
    assert len(out["drafts"]) == 2
    # persisted + queryable for the review tab
    assert len(store.list_drafts("batch-app-1")) == 2


def test_handle_run_pairs_uploaded_media(grounded_base, store, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    out = app.handle_run(
        ["https://example.com/a"], "g", 2,
        llm=_full_run_llm(), store=store, base_dir=grounded_base,
        getter=_getter(), media_paths=[str(img)], batch_id_factory=lambda: "b",
    )
    paired = [d for d in out["drafts"] if d.media is not None]
    assert len(paired) == 1
    assert paired[0].media.kind == MediaKind.uploaded_image
    assert paired[0].media.ref.endswith("a.jpg")


def test_handle_run_appends_uploaded_transcript_to_corpus(grounded_base, store, monkeypatch):
    captured = {}

    def fake_run_batch(sources, **kw):
        captured.update(kw)
        from draftforge.pipeline import BatchResult
        return BatchResult(drafts=[], errors=[], claim_checks={})

    monkeypatch.setattr(app, "run_batch", fake_run_batch)
    app.handle_run(
        ["https://example.com/a"], "g", 2,
        llm=make_llm(), store=store, base_dir=grounded_base, getter=_getter(),
        transcript_text="A NEW EPISODE TRANSCRIPT", batch_id_factory=lambda: "b",
    )
    assert "A NEW EPISODE TRANSCRIPT" in captured["corpus"]


def test_handle_run_collects_failed_url_without_aborting(grounded_base, store):
    def bad_getter(url, *a, **k):
        raise TimeoutError("boom")

    out = app.handle_run(
        ["https://bad.example"], "g", 2,
        llm=make_llm(), store=store, base_dir=grounded_base,
        getter=bad_getter, batch_id_factory=lambda: "b",
    )
    assert out["drafts"] == []
    assert any("bad.example" in e for e in out["errors"])


def test_handle_run_skips_blank_urls(grounded_base, store):
    out = app.handle_run(
        ["", "   ", "https://example.com/a"], "g", 2,
        llm=_full_run_llm(), store=store, base_dir=grounded_base,
        getter=_getter(), batch_id_factory=lambda: "b",
    )
    assert len(out["drafts"]) == 2  # only the one real URL ingested


def test_build_ui_is_callable():
    # build_ui must exist and be importable WITHOUT gradio loaded at module top.
    assert callable(app.build_ui)
