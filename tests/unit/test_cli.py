"""Tests for the CLI entrypoint.

All offline: no Anthropic key, no network. The CLI core is a testable function
(:func:`run_cli`) that accepts an injected ``llm`` and ``getter`` so the whole
ingest -> pipeline -> render path can be exercised without a key or a socket.
``main()`` is only the thin wrapper that builds the real client and calls it.

We verify:
* arg parsing works with no key present (``--help``-grade testability);
* the D9 fail-loud seam: a missing/empty ``--voice-file`` or ``--corpus-dir``
  raises a clear, named error (no placeholder fallback);
* ingest from a directory and from a URL both reach the pipeline with the
  injected llm/getter, producing rendered drafts; and skipped sources are
  summarized.
"""

import json

import pytest

from draftforge import cli
from draftforge.ingest.media import MediaItem
from draftforge.llm.client import LLMClient
from draftforge.models import MediaKind, MediaRef, Platform
from draftforge.store.db import Store


# --- offline llm/getter doubles -------------------------------------------------


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def text(self, *, model, system, user, max_tokens):
        self.calls.append({"model": model})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(*responses):
    return LLMClient(
        FakeTransport(responses),
        model_fast="fast",
        model_smart="smart",
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
    """A canned ClaimAnalysis the claims gate resolves to ``clean`` (no claims)."""
    return json.dumps(
        {"claims": [], "harmful": False, "harmful_reason": "", "softened_caption": None}
    )


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def fake_getter_factory(html):
    def _getter(url, *args, **kwargs):
        return FakeResponse(html)

    return _getter


# --- fixtures -------------------------------------------------------------------


@pytest.fixture
def voice_file(tmp_path):
    p = tmp_path / "voice.md"
    p.write_text("FB exemplar: Sink into stillness.", encoding="utf-8")
    return p


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "positions.md").write_text("We sell calm, not cures.", encoding="utf-8")
    return d


@pytest.fixture
def input_dir(tmp_path):
    d = tmp_path / "input"
    d.mkdir()
    (d / "doc1.txt").write_text("Floating reduces stress for many people.", encoding="utf-8")
    return d


def _stage_register(base, *, entries=None):
    """Write a valid data/claims_register.json under ``base`` and return its path."""
    data = entries if entries is not None else [
        {
            "claim_text": "magnesium is absorbed through the skin",
            "claim_type": "hard",
            "approved": True,
            "source_citation": "Waring 2006",
            "notes": "",
        }
    ]
    d = base / "data"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "claims_register.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def grounded_base(tmp_path):
    """Stage ALL startup receivers (voice + corpus + claims register) under a base.

    Mirrors a real, fully-grounded deployment so a run reaches the pipeline via
    the default open-receiver registry (no override flags). Returns the base dir.
    """
    voice = tmp_path / "prompts" / "voice_exemplars.md"
    voice.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n\n---\n\n".join(f"Post {i}: sink into stillness." for i in range(6))
    voice.write_text(f"## Facebook\n\n{blocks}\n", encoding="utf-8")

    corpus = tmp_path / "data" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "ep1.txt").write_text("Sam: we sell calm, not cures.", encoding="utf-8")

    _stage_register(tmp_path)
    return tmp_path


# --- arg parsing ----------------------------------------------------------------


def test_import_and_parse_args_work_with_key_unset(monkeypatch):
    # The offline guarantee (TEST-2): importing the CLI module and parsing args
    # must NOT require ANTHROPIC_API_KEY or the network. The autouse conftest
    # guard already deletes the key; this asserts the import + parse path is
    # clean under that condition (the --help-grade testability contract).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    import importlib

    import draftforge.cli as cli_module

    importlib.reload(cli_module)
    args = cli_module.parse_args(["--input", "d", "--guidance", "g"])

    assert args.input == "d"
    assert args.guidance == "g"
    assert args.command is None


def test_parse_args_basic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = cli.parse_args(
        [
            "--input", "some/dir",
            "--guidance", "warm voice",
            "--n", "12",
            "--voice-file", "v.md",
            "--corpus-dir", "corpus",
        ]
    )
    assert args.input == "some/dir"
    assert args.guidance == "warm voice"
    assert args.n == 12
    assert args.voice_file == "v.md"
    assert args.corpus_dir == "corpus"


def test_parse_args_defaults_n():
    args = cli.parse_args(
        ["--input", "d", "--guidance", "g", "--voice-file", "v", "--corpus-dir", "c"]
    )
    assert isinstance(args.n, int)
    assert args.n > 0


def test_parse_args_voice_and_corpus_are_optional_overrides():
    # The registry receivers are the default source; the flags only override.
    args = cli.parse_args(["--input", "d", "--guidance", "g"])
    assert args.voice_file is None
    assert args.corpus_dir is None


def test_parse_args_rejects_n_below_one():
    # --n < 1 is rejected at parse time with a clear argparse error (exit 2).
    with pytest.raises(SystemExit) as ei:
        cli.parse_args(
            ["--input", "d", "--guidance", "g", "--n", "0"]
        )
    assert ei.value.code == 2


def test_parse_args_rejects_negative_n():
    with pytest.raises(SystemExit):
        cli.parse_args(["--input", "d", "--guidance", "g", "--n", "-3"])


# --- preflight subcommand -------------------------------------------------------


def test_preflight_subcommand_parses():
    args = cli.parse_args(["preflight"])
    assert args.command == "preflight"


def test_main_preflight_exits_nonzero_when_nothing_configured(tmp_path, capsys):
    # Point the receiver base at an empty tmp dir: every startup receiver missing.
    code = cli.main(["preflight", "--base-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code != 0
    assert "voice_exemplars" in out
    assert "transcript_corpus" in out
    assert "claims_register" in out


def test_main_preflight_exits_zero_when_grounding_present(tmp_path, capsys):
    # Stage a well-formed voice file + corpus; skip the P2 claims register.
    voice = tmp_path / "prompts" / "voice_exemplars.md"
    voice.parent.mkdir(parents=True, exist_ok=True)
    blocks = "\n\n---\n\n".join(f"Post {i}: sink into stillness." for i in range(6))
    voice.write_text(f"## Facebook\n\n{blocks}\n", encoding="utf-8")
    corpus = tmp_path / "data" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "ep1.txt").write_text("Sam: we sell calm, not cures.", encoding="utf-8")

    code = cli.main(["preflight", "--base-dir", str(tmp_path), "--skip", "claims_register"])
    assert code == 0


# --- D9 fail-loud ---------------------------------------------------------------


def test_missing_voice_file_raises_loud(tmp_path, corpus_dir, input_dir):
    missing = tmp_path / "nope.md"
    args = cli.parse_args(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--voice-file", str(missing),
            "--corpus-dir", str(corpus_dir),
        ]
    )
    with pytest.raises(cli.MissingInputError) as ei:
        cli.run_cli(args, llm=make_llm(), getter=fake_getter_factory("<html></html>"))
    assert "voice" in str(ei.value).lower()


def test_empty_voice_file_raises_loud(tmp_path, corpus_dir, input_dir):
    empty = tmp_path / "voice.md"
    empty.write_text("   \n", encoding="utf-8")  # whitespace only
    args = cli.parse_args(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--voice-file", str(empty),
            "--corpus-dir", str(corpus_dir),
        ]
    )
    with pytest.raises(cli.MissingInputError) as ei:
        cli.run_cli(args, llm=make_llm(), getter=fake_getter_factory("<html></html>"))
    assert "voice" in str(ei.value).lower()


def test_missing_corpus_dir_raises_loud(tmp_path, voice_file, input_dir):
    missing = tmp_path / "no_corpus"
    args = cli.parse_args(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--voice-file", str(voice_file),
            "--corpus-dir", str(missing),
        ]
    )
    with pytest.raises(cli.MissingInputError) as ei:
        cli.run_cli(args, llm=make_llm(), getter=fake_getter_factory("<html></html>"))
    assert "corpus" in str(ei.value).lower()


def test_empty_corpus_dir_raises_loud(tmp_path, voice_file, input_dir):
    empty = tmp_path / "empty_corpus"
    empty.mkdir()  # exists but no files
    args = cli.parse_args(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--voice-file", str(voice_file),
            "--corpus-dir", str(empty),
        ]
    )
    with pytest.raises(cli.MissingInputError) as ei:
        cli.run_cli(args, llm=make_llm(), getter=fake_getter_factory("<html></html>"))
    assert "corpus" in str(ei.value).lower()


def test_run_cli_without_flags_fails_loud_through_registry(tmp_path, input_dir):
    # No --voice-file/--corpus-dir: the DEFAULT is the registry receivers, and
    # their absence fails loud via MissingInputsError naming the real file path.
    from draftforge.inputs import MissingInputsError

    args = cli.parse_args(["--input", str(input_dir), "--guidance", "g"])
    with pytest.raises(MissingInputsError) as ei:
        cli.run_cli(
            args,
            llm=make_llm(),
            getter=fake_getter_factory("<html></html>"),
            base_dir=tmp_path,  # empty -> receivers unfilled
        )
    assert "voice_exemplars.md" in str(ei.value)


def test_run_cli_missing_claims_register_fails_loud(tmp_path, input_dir):
    # The claims register is a fail-loud open receiver (D9): voice + corpus are
    # present (via override flags), but with no data/claims_register.json the run
    # must raise MissingInputsError naming the register file — never silently skip
    # the claims-safety gate's authority.
    from draftforge.inputs import MissingInputsError

    voice = tmp_path / "voice.md"
    voice.write_text("FB exemplar: Sink into stillness.", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "p.md").write_text("We sell calm, not cures.", encoding="utf-8")

    args = cli.parse_args(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--voice-file", str(voice),
            "--corpus-dir", str(corpus),
        ]
    )
    with pytest.raises(MissingInputsError) as ei:
        cli.run_cli(
            args,
            llm=make_llm(),
            getter=fake_getter_factory("<html></html>"),
            base_dir=tmp_path,  # no data/claims_register.json staged here
        )
    assert "claims_register.json" in str(ei.value)


def test_run_cli_uses_registry_when_flags_omitted(grounded_base, input_dir, capsys):
    # Stage ALL receivers under base_dir; omit the override flags entirely.
    args = cli.parse_args(["--input", str(input_dir), "--guidance", "g", "--n", "2"])
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),  # facebook
        _generate(1), _claims_clean(),  # instagram
    )

    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=Store(":memory:"),
    )
    assert len(result.drafts) == 2


# --- ingest -> pipeline wiring (offline) ----------------------------------------


def test_run_cli_from_directory_produces_rendered_drafts(grounded_base, input_dir, capsys):
    # One file -> one source -> 2 platforms. batch_size n=2 -> 1 per platform.
    # Chain per platform: generate(1) + one claims_check for the produced draft.
    args = cli.parse_args(
        [
            "--input", str(input_dir),
            "--guidance", "warm voice",
            "--n", "2",
        ]
    )
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),  # facebook
        _generate(1), _claims_clean(),  # instagram
    )

    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=Store(":memory:"),  # hermetic: never touch the real data/store.db
    )

    assert len(result.drafts) == 2
    out = capsys.readouterr().out
    # Rendered drafts were printed (caption + a platform name appear).
    assert "caption 0" in out
    assert "facebook" in out.lower() or "instagram" in out.lower()


def test_run_cli_from_url_uses_injected_getter(grounded_base, capsys):
    html = "<html><body><article>" + ("Floating is deeply calming. " * 20) + "</article></body></html>"
    args = cli.parse_args(
        [
            "--input", "https://example.com/float",
            "--guidance", "warm voice",
            "--n", "2",
        ]
    )
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),  # facebook
        _generate(1), _claims_clean(),  # instagram
    )

    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory(html),
        base_dir=grounded_base,
        store=Store(":memory:"),
    )

    assert len(result.drafts) == 2  # no network: the injected getter was used


def test_run_cli_summarizes_skipped_sources(grounded_base, input_dir, capsys):
    # classify fails on both attempts -> the one source is skipped & summarized.
    args = cli.parse_args(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--n", "2",
        ]
    )
    llm = make_llm(RuntimeError("boom"), RuntimeError("boom"))

    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=Store(":memory:"),
    )

    assert result.drafts == []
    assert len(result.errors) == 1
    out = capsys.readouterr().out.lower()
    assert "skip" in out  # the summary mentions the skipped source


# --- main() exit-code contract (offline; drives the production wrapper) ----------


class _RaisingTransport:
    """Stand-in for AnthropicTransport whose every call fails (no network)."""

    def __init__(self, *args, **kwargs):  # accepts however main() constructs it
        pass

    def text(self, *, model, system, user, max_tokens):
        raise RuntimeError("transport unavailable (offline test double)")


def test_main_returns_2_when_api_key_unset(monkeypatch, capsys):
    # A run (not preflight) requires ANTHROPIC_API_KEY via Settings.load(); when
    # it is unset, main() fails loud with exit code 2 (not a crash).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    code = cli.main(["--input", "some/dir", "--guidance", "g"])

    assert code == 2
    err = capsys.readouterr().err.lower()
    assert "anthropic_api_key" in err


def test_main_returns_2_when_required_inputs_missing(
    monkeypatch, tmp_path, input_dir, capsys
):
    # Key IS set (so Settings.load succeeds and the real client is built), but an
    # overridden grounding input is missing -> MissingInputError -> exit code 2.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    # Don't let a real transport be constructed/used; the failure is upstream of
    # any LLM call, but stay defensive against accidental network construction.
    monkeypatch.setattr(
        "draftforge.llm.anthropic_transport.AnthropicTransport", _RaisingTransport
    )
    missing_voice = tmp_path / "nope_voice.md"

    code = cli.main(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--voice-file", str(missing_voice),  # missing -> fail loud
            "--corpus-dir", str(tmp_path),  # present dir (doesn't matter; voice fails first)
        ]
    )

    assert code == 2
    err = capsys.readouterr().err.lower()
    assert "error" in err
    assert "voice" in err


def test_main_returns_1_when_every_source_fails(
    monkeypatch, tmp_path, voice_file, corpus_dir, input_dir
):
    # Key set, grounding present, a real input source -- but the (injected) real
    # transport raises on every call, so every source is skipped and zero drafts
    # are produced. main() must exit 1 to pin the empty-batch contract for CI.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(
        "draftforge.llm.anthropic_transport.AnthropicTransport", _RaisingTransport
    )
    # main() builds the client without a sleep override, so its one retry would
    # use the real backoff (~1s). Swap the LLMClient main() constructs for one
    # that injects a no-op sleep, keeping the test fast and fully offline.
    real_llmclient = cli.LLMClient

    def _no_sleep_llmclient(transport, *, model_fast, model_smart):
        return real_llmclient(
            transport,
            model_fast=model_fast,
            model_smart=model_smart,
            sleep=lambda *_: None,
        )

    monkeypatch.setattr(cli, "LLMClient", _no_sleep_llmclient)
    # The claims register is a fail-loud receiver with no override flag; main()
    # resolves it from the real repo root, which has no committed register. This
    # test is about the empty-batch EXIT CODE, not register loading, so stub the
    # loader to return a valid register and let the run reach run_batch (where
    # every source then fails on the raising transport).
    monkeypatch.setattr(cli.inputs, "load_claims_register", lambda **_: [])

    code = cli.main(
        [
            "--input", str(input_dir),
            "--guidance", "g",
            "--n", "2",
            "--voice-file", str(voice_file),
            "--corpus-dir", str(corpus_dir),
            "--db", str(tmp_path / "store.db"),  # hermetic: own the db file
        ]
    )

    assert code == 1


# --- SPEC-3: the CLI persists the BatchResult into the store --------------------


def _claims_flagged(reason="asserts a disease cure that cannot be softened"):
    """A canned ClaimAnalysis the claims gate resolves to ``flagged`` (harmful)."""
    return json.dumps(
        {"claims": [], "harmful": True, "harmful_reason": reason, "softened_caption": None}
    )


def test_db_arg_defaults_and_parses():
    # --db has a sane gitignored default and is overridable.
    default_args = cli.parse_args(["--input", "d", "--guidance", "g"])
    assert default_args.db == "data/store.db"
    override = cli.parse_args(["--input", "d", "--guidance", "g", "--db", "x/y.db"])
    assert override.db == "x/y.db"


def test_run_cli_persists_drafts_and_claim_flags(grounded_base, input_dir, tmp_path):
    # A real on-disk --db: after the run, the store holds the batch, its source,
    # each draft, and each draft's claim-safety verdict in posts.claim_flags.
    db_path = str(tmp_path / "out" / "store.db")  # nested dir is auto-created
    args = cli.parse_args(
        ["--input", str(input_dir), "--guidance", "warm voice", "--n", "2", "--db", db_path]
    )
    # facebook draft -> flagged verdict; instagram draft -> clean verdict.
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_flagged(),  # facebook
        _generate(1), _claims_clean(),  # instagram
    )

    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        batch_id_factory=lambda: "batch-test-1",  # deterministic id (offline)
        now="2026-06-25T12:00:00+00:00",
    )

    assert len(result.drafts) == 2

    # Re-open the persisted file in a FRESH connection: the rows are durable.
    store = Store(db_path)
    try:
        assert store.get_batch("batch-test-1") is not None
        assert len(store.list_drafts("batch-test-1")) == 2
        assert len(store.list_sources("batch-test-1")) == 1

        # Every draft persisted with its claim_flags (the full ClaimCheck dict).
        for draft in result.drafts:
            row = store.get_post_row(draft.id, "batch-test-1")
            assert row is not None
            flags = row["claim_flags"]
            assert flags is not None
            # claim_flags round-trips the ClaimCheck.model_dump() shape.
            assert flags["status"] == result.claim_checks[draft.id].status
            assert "notes" in flags and "revised_text" in flags

        # The flagged draft's persisted verdict is exactly "flagged".
        fb = next(d for d in result.drafts if d.platform == Platform.facebook)
        assert (
            store.get_post_row(fb.id, "batch-test-1")["claim_flags"]["status"]
            == "flagged"
        )
    finally:
        store.close()


def test_run_cli_persists_into_injected_store(grounded_base, input_dir):
    # With an injected in-memory store, the same persistence happens against it.
    args = cli.parse_args(["--input", str(input_dir), "--guidance", "g", "--n", "2"])
    store = Store(":memory:")
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),  # facebook
        _generate(1), _claims_clean(),  # instagram
    )

    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=store,
        batch_id_factory=lambda: "batch-mem-1",
    )

    # The injected store was NOT closed by run_cli (the caller owns it), so we can
    # still query it here.
    assert len(store.list_drafts("batch-mem-1")) == 2
    for draft in result.drafts:
        assert (
            store.get_post_row(draft.id, "batch-mem-1")["claim_flags"]["status"]
            == "clean"
        )
    store.close()


# --- media pairing wiring (M4) --------------------------------------------------


def test_parse_args_media_dir_optional():
    args = cli.parse_args(["--input", "d", "--guidance", "g"])
    assert args.media_dir is None


def test_run_cli_pairs_injected_media_and_persists(grounded_base, input_dir):
    # Injected MediaItems pair onto the drafts (order strategy) and the paired
    # Draft.media is persisted, round-tripping through the store.
    args = cli.parse_args(["--input", str(input_dir), "--guidance", "g", "--n", "2"])
    store = Store(":memory:")
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),  # facebook
        _generate(1), _claims_clean(),  # instagram
    )
    media = [MediaItem(kind=MediaKind.uploaded_image, ref="a.jpg", filename="a.jpg", size_bytes=1)]

    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=store,
        media_items=media,
        batch_id_factory=lambda: "batch-media-1",
    )

    # Order strategy: exactly one upload -> the first draft is paired, rest None.
    assert sum(1 for d in result.drafts if d.media is not None) == 1
    assert result.drafts[0].media == MediaRef(kind=MediaKind.uploaded_image, ref="a.jpg")

    # Persisted media round-trips (match by id; list ordering is independent).
    by_id = {d.id: d for d in result.drafts}
    for stored in store.list_drafts("batch-media-1"):
        assert stored.media == by_id[stored.id].media
    store.close()


def test_run_cli_without_media_leaves_drafts_unpaired(grounded_base, input_dir):
    args = cli.parse_args(["--input", str(input_dir), "--guidance", "g", "--n", "2"])
    store = Store(":memory:")
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),
        _generate(1), _claims_clean(),
    )
    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=store,
        batch_id_factory=lambda: "batch-nomedia",
    )
    assert all(d.media is None for d in result.drafts)
    store.close()


def test_run_cli_loads_media_dir_and_skips_non_media(grounded_base, input_dir, tmp_path):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "a.jpg").write_bytes(b"x")
    (media_dir / "notes.txt").write_text("ignore me", encoding="utf-8")  # skipped
    args = cli.parse_args(
        ["--input", str(input_dir), "--guidance", "g", "--n", "2",
         "--media-dir", str(media_dir)]
    )
    store = Store(":memory:")
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),
        _generate(1), _claims_clean(),
    )
    result = cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=store,
        batch_id_factory=lambda: "batch-mediadir",
    )
    paired = [d for d in result.drafts if d.media is not None]
    assert len(paired) == 1  # only the .jpg was picked up
    assert paired[0].media.kind == MediaKind.uploaded_image
    assert paired[0].media.ref.endswith("a.jpg")
    store.close()


def test_run_cli_passes_loaded_register_to_run_batch(grounded_base, input_dir, monkeypatch):
    # The register loaded from the open receiver is the SAME object handed to
    # run_batch — proving the CLI actually wires the claims authority through.
    captured = {}
    real_run_batch = cli.run_batch

    def _spy_run_batch(sources, **kwargs):
        captured["register"] = kwargs["register"]
        return real_run_batch(sources, **kwargs)

    monkeypatch.setattr(cli, "run_batch", _spy_run_batch)

    args = cli.parse_args(["--input", str(input_dir), "--guidance", "g", "--n", "2"])
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_clean(),
        _generate(1), _claims_clean(),
    )

    cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=Store(":memory:"),
    )

    # grounded_base stages a register with one approved entry; it reached run_batch.
    from draftforge.models import RegisterEntry

    assert "register" in captured
    assert len(captured["register"]) == 1
    assert isinstance(captured["register"][0], RegisterEntry)
    assert captured["register"][0].claim_text == "magnesium is absorbed through the skin"


def test_run_cli_report_includes_claim_safety_summary(grounded_base, input_dir, capsys):
    # _report prints a claim-safety line counting each verdict (clean/softened/
    # flagged/needs_manual_review), so a reviewer sees how many need attention.
    args = cli.parse_args(["--input", str(input_dir), "--guidance", "g", "--n", "2"])
    llm = make_llm(
        _classify(), _extract(),
        _generate(1), _claims_flagged(),  # facebook -> flagged
        _generate(1), _claims_clean(),  # instagram -> clean
    )

    cli.run_cli(
        args,
        llm=llm,
        getter=fake_getter_factory("<html></html>"),
        base_dir=grounded_base,
        store=Store(":memory:"),
    )

    out = capsys.readouterr().out.lower()
    assert "claim safety" in out
    assert "clean=1" in out
    assert "flagged=1" in out


# --- P2-2: persisting a draft with NO claim verdict fails loud ------------------


def _bare_args(n=2, db="ignored.db"):
    import argparse

    return argparse.Namespace(guidance="g", n=n, db=db)


def test_persist_fails_loud_when_draft_has_no_claim_verdict():
    # A draft must NEVER be persisted without its claim-safety verdict (the gate
    # guarantees one per draft). _persist raises MissingClaimVerdictError rather
    # than silently writing a compliance hole.
    from draftforge.models import Draft, Source
    from draftforge.pipeline import BatchResult

    store = Store(":memory:")
    draft = Draft(
        id="d1", platform=Platform.facebook, angle="a", caption="c", hashtags=[]
    )
    source = Source(source_id="s1", type="url", text="t", fetched_at="2026-06-25T00:00:00Z")
    # BatchResult with the draft present but NO claim_checks entry for it.
    result = BatchResult(drafts=[draft], errors=[], claim_checks={})

    with pytest.raises(cli.MissingClaimVerdictError, match="no claim-safety verdict"):
        cli._persist(
            result,
            [source],
            store=store,
            args=_bare_args(),
            batch_id="b1",
            now="2026-06-25T00:00:00Z",
        )
    store.close()


def test_persist_missing_verdict_rolls_back_leaving_no_orphan_batch():
    # The P2-2 fail-loud is wrapped in the DI-1 transaction: the partial batch
    # (the batch row + source + draft written before the raise) is rolled back.
    from draftforge.models import Draft, Source
    from draftforge.pipeline import BatchResult

    store = Store(":memory:")
    draft = Draft(
        id="d1", platform=Platform.facebook, angle="a", caption="c", hashtags=[]
    )
    source = Source(source_id="s1", type="url", text="t", fetched_at="2026-06-25T00:00:00Z")
    result = BatchResult(drafts=[draft], errors=[], claim_checks={})

    with pytest.raises(cli.MissingClaimVerdictError):
        cli._persist(
            result,
            [source],
            store=store,
            args=_bare_args(),
            batch_id="b1",
            now="2026-06-25T00:00:00Z",
        )

    # No orphan/partial batch survives the rollback.
    assert store.get_batch("b1") is None
    assert store.list_sources("b1") == []
    assert store.list_drafts("b1") == []
    store.close()


# --- DI-1 / INT-P2-01: overlapping re-runs coexist (no IntegrityError) ----------


def test_run_cli_overlapping_reruns_coexist_as_two_batches(grounded_base, input_dir):
    # Re-running the SAME input (same source_id -> same draft ids) twice must NOT
    # crash on a primary-key collision. Each run is its own batch; the two
    # coexist as independent batch rows.
    store = Store(":memory:")

    def _run(batch_id):
        args = cli.parse_args(["--input", str(input_dir), "--guidance", "g", "--n", "2"])
        llm = make_llm(
            _classify(), _extract(),
            _generate(1), _claims_clean(),  # facebook
            _generate(1), _claims_clean(),  # instagram
        )
        return cli.run_cli(
            args,
            llm=llm,
            getter=fake_getter_factory("<html></html>"),
            base_dir=grounded_base,
            store=store,
            batch_id_factory=lambda: batch_id,
        )

    first = _run("batch-A")
    second = _run("batch-B")  # SAME source/draft ids — must not collide

    assert len(first.drafts) == 2
    assert len(second.drafts) == 2
    # The two overlapping runs produced the SAME draft ids (content-derived) ...
    assert {d.id for d in first.drafts} == {d.id for d in second.drafts}
    # ... yet coexist as two independent batches, each with its own posts.
    assert store.get_batch("batch-A") is not None
    assert store.get_batch("batch-B") is not None
    assert len(store.list_drafts("batch-A")) == 2
    assert len(store.list_drafts("batch-B")) == 2
    store.close()
