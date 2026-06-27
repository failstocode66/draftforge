"""Tests for the fail-loud open-receivers registry + preflight (Task 1.10 / D9).

Every real-data input the app needs is a *receiver* declared in
``REQUIRED_INPUTS``. A startup receiver is "open" (waiting to be plugged in)
until its real file/dir is present and valid; while open it reports MISSING
with a specific, actionable error naming the file and how to fill it. There are
NO silent placeholders — the loaders raise rather than fall back to empty/sample
data, and ``preflight`` enumerates EVERY unfilled receiver at once so a missing
slot is impossible to overlook.

All offline. Base paths are injected via ``base_dir`` so tests stage present /
absent states under ``tmp_path`` without touching the repo's real dirs.
"""

from __future__ import annotations

import pytest

from draftforge import inputs
from draftforge.inputs import (
    MissingInput,
    MissingInputsError,
    REQUIRED_INPUTS,
    check_inputs,
    load_corpus,
    load_voice_exemplars,
    require_inputs,
    run_preflight,
)


# --- helpers: stage well-formed receivers under a tmp base_dir ------------------


def _good_voice_text(n: int = 6) -> str:
    """A well-formed voice_exemplars file: >= n example post blocks.

    Format (documented in prompts/voice_exemplars.EXAMPLE.md): posts live under
    ``## Facebook`` / ``## Instagram`` headings and are separated by a ``---``
    line. We split the n blocks across the two headings.
    """
    half = max(1, n // 2)
    fb = "\n\n---\n\n".join(f"FB example post {i}: sink into stillness." for i in range(half))
    ig = "\n\n---\n\n".join(
        f"IG example post {i}: 60 minutes, zero noise." for i in range(n - half)
    )
    return f"## Facebook\n\n{fb}\n\n## Instagram\n\n{ig}\n"


def _stage_voice(base_dir, *, n: int = 6) -> None:
    p = base_dir / "prompts" / "voice_exemplars.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_good_voice_text(n), encoding="utf-8")


def _stage_corpus(base_dir, *, files: int = 1) -> None:
    d = base_dir / "data" / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(files):
        (d / f"transcript_{i}.txt").write_text(
            f"Sam on episode {i}: we sell calm, not cures. " * 5,
            encoding="utf-8",
        )


# --- registry shape -------------------------------------------------------------


def test_registry_declares_every_required_input():
    keys = {r.key for r in REQUIRED_INPUTS}
    assert keys == {
        "voice_exemplars",
        "transcript_corpus",
        "claims_register",
        "run_urls",
    }


def test_run_urls_is_a_runtime_receiver_not_a_startup_one():
    by_key = {r.key: r for r in REQUIRED_INPUTS}
    assert by_key["run_urls"].kind == "runtime"
    # The three grounding stores are checked at startup.
    assert by_key["voice_exemplars"].kind == "startup"
    assert by_key["transcript_corpus"].kind == "startup"
    assert by_key["claims_register"].kind == "startup"


def test_claims_register_marked_wired_in_p2():
    by_key = {r.key: r for r in REQUIRED_INPUTS}
    assert "P2" in by_key["claims_register"].description


# --- check_inputs: collected, never first-fail-only -----------------------------


def test_check_inputs_reports_every_missing_startup_input_when_nothing_configured(
    tmp_path,
):
    missing = check_inputs(base_dir=tmp_path)
    keys = {m.key for m in missing}
    # Every STARTUP receiver is reported missing at once (not just the first).
    assert keys == {"voice_exemplars", "transcript_corpus", "claims_register"}
    # run_urls is runtime — never a startup miss.
    assert "run_urls" not in keys
    assert all(isinstance(m, MissingInput) for m in missing)


def test_check_inputs_missing_entries_explain_how_to_fill(tmp_path):
    by_key = {m.key: m for m in check_inputs(base_dir=tmp_path)}

    voice = by_key["voice_exemplars"]
    assert "voice_exemplars.md" in voice.how_to_fill
    assert "voice_exemplars.EXAMPLE.md" in voice.how_to_fill

    corpus = by_key["transcript_corpus"]
    assert "data/corpus" in corpus.how_to_fill.replace("\\", "/")


def test_check_inputs_passes_when_voice_and_corpus_present(tmp_path):
    _stage_voice(tmp_path)
    _stage_corpus(tmp_path)

    missing = check_inputs(base_dir=tmp_path)
    keys = {m.key for m in missing}
    # voice + corpus now satisfied; only claims_register (P2 open receiver) remains.
    assert "voice_exemplars" not in keys
    assert "transcript_corpus" not in keys
    assert keys == {"claims_register"}


def test_check_inputs_voice_below_threshold_still_missing(tmp_path):
    _stage_voice(tmp_path, n=3)  # below the >= 6 minimum
    _stage_corpus(tmp_path)

    keys = {m.key for m in check_inputs(base_dir=tmp_path)}
    assert "voice_exemplars" in keys


def test_check_inputs_empty_voice_file_is_missing(tmp_path):
    p = tmp_path / "prompts" / "voice_exemplars.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("   \n", encoding="utf-8")
    _stage_corpus(tmp_path)

    keys = {m.key for m in check_inputs(base_dir=tmp_path)}
    assert "voice_exemplars" in keys


def test_check_inputs_corpus_dir_with_no_text_files_is_missing(tmp_path):
    _stage_voice(tmp_path)
    d = tmp_path / "data" / "corpus"
    d.mkdir(parents=True, exist_ok=True)
    (d / "notes.csv").write_text("not,a,transcript", encoding="utf-8")  # wrong ext

    keys = {m.key for m in check_inputs(base_dir=tmp_path)}
    assert "transcript_corpus" in keys


# --- require_inputs -------------------------------------------------------------


def test_require_inputs_raises_listing_every_missing(tmp_path):
    with pytest.raises(MissingInputsError) as ei:
        require_inputs(base_dir=tmp_path)
    msg = str(ei.value)
    # The single error enumerates EVERY unfilled startup receiver.
    assert "voice_exemplars" in msg
    assert "transcript_corpus" in msg
    assert "claims_register" in msg


def test_require_inputs_passes_when_all_startup_present(tmp_path):
    _stage_voice(tmp_path)
    _stage_corpus(tmp_path)
    # claims_register is an open receiver until P2; allow it to be skipped so an
    # all-grounding-present state can pass. (Same hook preflight uses.)
    require_inputs(base_dir=tmp_path, skip={"claims_register"})


# --- loaders fail loud via their receiver --------------------------------------


def test_load_voice_exemplars_missing_raises_specific_error(tmp_path):
    with pytest.raises(MissingInputsError) as ei:
        load_voice_exemplars(base_dir=tmp_path)
    assert "voice_exemplars.md" in str(ei.value)


def test_load_voice_exemplars_returns_text_when_present(tmp_path):
    _stage_voice(tmp_path)
    text = load_voice_exemplars(base_dir=tmp_path)
    assert "Facebook" in text
    assert "example post" in text


def test_load_corpus_missing_raises_specific_error(tmp_path):
    with pytest.raises(MissingInputsError) as ei:
        load_corpus(base_dir=tmp_path)
    assert "data/corpus" in str(ei.value).replace("\\", "/")


def test_load_corpus_concatenates_files_when_present(tmp_path):
    _stage_corpus(tmp_path, files=2)
    corpus = load_corpus(base_dir=tmp_path)
    assert "transcript_0.txt" in corpus
    assert "transcript_1.txt" in corpus
    assert "we sell calm, not cures" in corpus


# --- preflight ------------------------------------------------------------------


def test_preflight_exits_nonzero_and_lists_all_when_nothing_configured(tmp_path):
    out: list[str] = []
    code = run_preflight(base_dir=tmp_path, emit=out.append)
    text = "\n".join(out)

    assert code != 0
    # Every startup receiver appears with a missing marker + how-to-fill.
    assert "voice_exemplars" in text
    assert "transcript_corpus" in text
    assert "claims_register" in text
    assert "voice_exemplars.md" in text  # the how-to-fill detail is shown
    # The runtime receiver is reported informationally, not as a startup miss.
    assert "run_urls" in text
    assert "run time" in text.lower()


def test_preflight_exits_zero_when_all_startup_present(tmp_path):
    _stage_voice(tmp_path)
    _stage_corpus(tmp_path)
    # claims_register stays open until P2; preflight may treat it as informational
    # the same way require_inputs' skip does, so a fully-grounded run can pass.
    out: list[str] = []
    code = run_preflight(base_dir=tmp_path, emit=out.append, skip={"claims_register"})
    text = "\n".join(out)

    assert code == 0
    assert "voice_exemplars" in text  # still reported, now as present


def test_preflight_present_receivers_render_a_present_marker(tmp_path):
    _stage_voice(tmp_path)
    _stage_corpus(tmp_path)
    out: list[str] = []
    run_preflight(base_dir=tmp_path, emit=out.append, skip={"claims_register"})
    joined = "\n".join(out)
    # A present startup receiver shows the present marker, a missing one the miss.
    assert inputs.PRESENT_MARK in joined  # voice + corpus present


def test_preflight_output_is_ascii_encodable(tmp_path):
    # Regression: the real CLI prints to a Windows console (cp1252 by default),
    # so every emitted line must encode without a UnicodeEncodeError. Emoji marks
    # would crash the actual command even though list-capturing tests pass.
    out: list[str] = []
    run_preflight(base_dir=tmp_path, emit=out.append)  # nothing configured
    for line in out:
        line.encode("ascii")  # raises UnicodeEncodeError on any non-ASCII char


def test_missing_inputs_error_message_is_ascii_encodable(tmp_path):
    # The MissingInputsError message is printed to stderr by the CLI, so it too
    # must be pure ASCII (its description + how_to_fill text comes from the same
    # registry strings preflight emits).
    with pytest.raises(MissingInputsError) as ei:
        require_inputs(base_dir=tmp_path)
    str(ei.value).encode("ascii")  # raises if any non-ASCII char leaked in


def test_registry_strings_are_ascii(tmp_path):
    # Belt-and-suspenders: every emitted registry string (description + each
    # receiver's how_to_fill) is pure ASCII at the source, not just per-run output.
    for receiver in REQUIRED_INPUTS:
        receiver.description.encode("ascii")
        if receiver.validate is not None:
            how = receiver.validate(tmp_path)  # MISSING -> returns the how-to-fill
            if how is not None:
                how.encode("ascii")
