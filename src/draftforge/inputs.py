"""Fail-loud open-receivers registry + preflight (Task 1.10 / D9).

Tyler's requirement (D9): when we wire the client's REAL data there must be NO
placeholder left anywhere - only **fail-loud "open receivers."** Every real-data
input the app needs is declared here as a *receiver*. A receiver is "open" (the
slot is waiting to be plugged in) until its real file/dir is present and valid;
while open it reports MISSING with a **specific, actionable** error naming the
file and exactly how to fill it. There is no silent fallback to empty or sample
data - the loaders raise instead.

The whole point is that a missing slot is impossible to overlook:

* :func:`check_inputs` validates EVERY startup receiver and returns EVERY missing
  one (collected, never first-fail-only).
* :func:`require_inputs` raises one :class:`MissingInputsError` whose message
  lists every unfilled startup receiver at once.
* :func:`run_preflight` (the ``preflight`` CLI command) prints each receiver's
  status - present / missing + how-to-fill - and the runtime ones informationally,
  exiting non-zero if any startup receiver is missing.

**Receiver kinds.** A ``startup`` receiver is a persistent grounding store that
must exist before any run (voice exemplars, the transcript corpus, the approved-
claims register). A ``runtime`` receiver is supplied per run (the URL list); it
is reported by preflight as "provided at run time", not as a startup miss.

**No silent placeholders.** The only sample data in the repo is test fixtures
under ``tests/fixtures/`` and the obviously-fake ``*.EXAMPLE.md`` templates
(never read at runtime). Real runtime grounding lives in the gitignored ``data/``
and ``scratch/`` dirs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

# Status markers used by preflight output (exported for tests / callers).
# ASCII-only on purpose, and the WHOLE emitted surface is ASCII-only (markers,
# descriptions, how-to-fill text, error messages): the preflight command prints
# to the console, and a strict legacy encoding (cp437/cp1252) cannot encode
# non-ASCII characters such as emoji or em-dashes - emitting one would crash with
# UnicodeEncodeError. Plain ASCII is portable and can't fail; the invariant is
# enforced by test_preflight_output_is_ascii_encodable (asserts .encode("ascii")).
PRESENT_MARK = "[OK]"
MISSING_MARK = "[MISSING]"
RUNTIME_MARK = "[RUNTIME]"  # provided at run time

# Minimum number of example post blocks a valid voice_exemplars file must hold.
_MIN_VOICE_EXEMPLARS = 6

# File extensions that count as a transcript in the corpus dir.
_CORPUS_SUFFIXES = (".txt", ".md", ".pdf")

# Receiver paths, relative to the injectable base dir (the repo root in prod).
_VOICE_REL = ("prompts", "voice_exemplars.md")
_VOICE_EXAMPLE_REL = ("prompts", "voice_exemplars.EXAMPLE.md")
_CORPUS_REL = ("data", "corpus")
_CLAIMS_REGISTER_REL = ("data", "claims_register.json")
_CLAIMS_REGISTER_EXAMPLE_REL = ("data", "claims_register.EXAMPLE.json")


@dataclass(frozen=True)
class MissingInput:
    """One unfilled receiver: what it is and exactly how to fill it."""

    key: str
    description: str
    how_to_fill: str


class MissingInputsError(Exception):
    """Raised when one or more required inputs are missing.

    The message enumerates EVERY missing input (key + how-to-fill), so a missing
    slot can never hide behind a first-fail-only error.
    """

    def __init__(self, missing: list[MissingInput]) -> None:
        self.missing = list(missing)
        lines = ["Missing required input(s) - fill each before running:"]
        for m in self.missing:
            lines.append(f"  - [{m.key}] {m.description}")
            lines.append(f"      how to fill: {m.how_to_fill}")
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class Receiver:
    """A declared real-data input slot.

    Attributes:
        key: Stable identifier (also the dict key callers reference).
        kind: ``"startup"`` (must exist before any run) or ``"runtime"`` (supplied
            per run; reported informationally by preflight, never a startup miss).
        description: Human-readable description of what plugs in here.
        validate: For a startup receiver, ``validate(base_dir)`` returns ``None``
            when the receiver is satisfied, or a ``how_to_fill`` string when it is
            missing/invalid. ``None`` for runtime receivers.
    """

    key: str
    kind: str  # "startup" | "runtime"
    description: str
    validate: Callable[[Path], str | None] | None = None

    def check(self, base_dir: Path) -> MissingInput | None:
        """Return a :class:`MissingInput` if this startup receiver is unfilled."""
        if self.kind != "startup" or self.validate is None:
            return None
        how_to_fill = self.validate(base_dir)
        if how_to_fill is None:
            return None
        return MissingInput(
            key=self.key, description=self.description, how_to_fill=how_to_fill
        )


# --- per-receiver validators ----------------------------------------------------


def _count_voice_exemplars(text: str) -> int:
    """Count example post blocks in a voice_exemplars file.

    Format: posts live under ``## Facebook`` / ``## Instagram`` headings and are
    separated by a line containing only ``---``. A "post" is a maximal run of
    content lines; both a ``---`` separator line and a ``##`` heading line break
    one post from the next (so the first post under each heading is counted).
    """
    count = 0
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        is_separator = stripped == "---"
        is_heading = stripped.startswith("#")
        if not stripped or is_separator or is_heading:
            in_block = False
            continue
        if not in_block:
            count += 1
            in_block = True
    return count


def _validate_voice_exemplars(base_dir: Path) -> str | None:
    path = base_dir.joinpath(*_VOICE_REL)
    rel = "/".join(_VOICE_REL)
    rel_example = "/".join(_VOICE_EXAMPLE_REL)
    how = (
        f"create {rel} with the business's real past posts - at least "
        f"{_MIN_VOICE_EXEMPLARS} example post blocks under ## Facebook / "
        f"## Instagram headings, separated by '---' lines. See the committed "
        f"format template {rel_example} (copy it, then replace every "
        f"placeholder with real posts). The template is never read at runtime."
    )
    if not path.is_file():
        return how
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return f"{rel} is empty. {how}"
    if _count_voice_exemplars(text) < _MIN_VOICE_EXEMPLARS:
        return (
            f"{rel} has fewer than {_MIN_VOICE_EXEMPLARS} example post blocks. "
            f"{how}"
        )
    return None


def _validate_transcript_corpus(base_dir: Path) -> str | None:
    d = base_dir.joinpath(*_CORPUS_REL)
    rel = "/".join(_CORPUS_REL)
    how = (
        f"add at least one transcript ({', '.join(_CORPUS_SUFFIXES)}) with "
        f"extractable text to {rel}/. This is the business owner's own words "
        f"(e.g. podcast transcripts) - the on-brand, legally-defensible voice "
        f"corpus. The directory is gitignored; it holds real data, not samples."
    )
    if not d.is_dir():
        return how
    has_text_file = any(
        f.is_file() and f.suffix.lower() in _CORPUS_SUFFIXES for f in d.iterdir()
    )
    if not has_text_file:
        return f"{rel}/ has no {'/'.join(_CORPUS_SUFFIXES)} transcript. {how}"
    return None


def _validate_claims_register(base_dir: Path) -> str | None:
    """The approved-claims register the claims-safety gate (P2) checks against.

    Validates the real file's SHAPE (a non-empty JSON array of well-formed
    ``RegisterEntry`` rows), so a malformed register fails loud here at preflight
    rather than deep inside a run. Returns the how-to-fill string when the file is
    absent/empty/invalid, else ``None``.
    """
    path = base_dir.joinpath(*_CLAIMS_REGISTER_REL)
    rel = "/".join(_CLAIMS_REGISTER_REL)
    rel_example = "/".join(_CLAIMS_REGISTER_EXAMPLE_REL)
    how = (
        f"populate the approved-claims register at {rel} with at least one "
        f"approved claim entry (the claims-safety gate, P2, checks hard claims "
        f"against it). Each entry is "
        f'{{"claim_text", "claim_type": "soft"|"hard", "approved": true|false, '
        f'"source_citation", "notes"}}. See the committed format template '
        f"{rel_example} (copy it, replace every placeholder with real approved "
        f"claims). The template is never read at runtime; this dir is gitignored."
    )
    if not path.is_file():
        return how
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return f"{rel} is empty. {how}"
    import json

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return f"{rel} is not valid JSON. {how}"
    if not isinstance(data, list) or not data:
        return f"{rel} must be a non-empty JSON array of claim entries. {how}"
    # Imported here (not at module top) to keep this module's import cheap and
    # free of the models import until a register is actually validated/loaded.
    from pydantic import ValidationError

    from draftforge.models import RegisterEntry

    try:
        for row in data:
            RegisterEntry.model_validate(row)
    except ValidationError as exc:
        return f"{rel} has an invalid claim entry ({exc.error_count()} error(s)). {how}"
    return None


# --- the registry ---------------------------------------------------------------

REQUIRED_INPUTS: tuple[Receiver, ...] = (
    Receiver(
        key="voice_exemplars",
        kind="startup",
        description=(
            "Few-shot exemplars of the business's real past posts (the brand "
            "voice the generator imitates)."
        ),
        validate=_validate_voice_exemplars,
    ),
    Receiver(
        key="transcript_corpus",
        kind="startup",
        description=(
            "The transcript corpus - the owner's own words (e.g. podcast "
            "transcripts) grounding claims and stance."
        ),
        validate=_validate_transcript_corpus,
    ),
    Receiver(
        key="claims_register",
        kind="startup",
        description=(
            "The approved-claims register the claims-safety gate checks against "
            "(wired in P2)."
        ),
        validate=_validate_claims_register,
    ),
    Receiver(
        key="run_urls",
        kind="runtime",
        description=(
            "The per-run URL list (at least one source URL provided at run "
            "time, not a startup file)."
        ),
        validate=None,
    ),
)


def _repo_root() -> Path:
    """The repo root: src/draftforge/inputs.py -> ../../.. == repo root."""
    return Path(__file__).resolve().parents[2]


def _resolve_base(base_dir: Path | str | None) -> Path:
    return Path(base_dir) if base_dir is not None else _repo_root()


# --- public API -----------------------------------------------------------------


def check_inputs(
    *,
    base_dir: Path | str | None = None,
    skip: Iterable[str] = (),
) -> list[MissingInput]:
    """Validate ALL startup receivers and return EVERY missing one.

    Collected, never first-fail-only - so a missing slot cannot hide behind an
    earlier one. Runtime receivers (e.g. ``run_urls``) are never reported here.

    Args:
        base_dir: Root the receiver paths resolve under (default: repo root).
            Injected in tests to stage present/absent states under ``tmp_path``.
        skip: Receiver keys to exclude from the check (e.g. an open receiver not
            yet wired, like ``claims_register`` before P2).

    Returns:
        A list of :class:`MissingInput`, one per unfilled startup receiver.
    """
    base = _resolve_base(base_dir)
    skip_set = set(skip)
    missing: list[MissingInput] = []
    for receiver in REQUIRED_INPUTS:
        if receiver.key in skip_set:
            continue
        result = receiver.check(base)
        if result is not None:
            missing.append(result)
    return missing


def require_inputs(
    *,
    base_dir: Path | str | None = None,
    skip: Iterable[str] = (),
) -> None:
    """Raise :class:`MissingInputsError` if any startup receiver is missing.

    The single raised error lists EVERY unfilled receiver at once.
    """
    missing = check_inputs(base_dir=base_dir, skip=skip)
    if missing:
        raise MissingInputsError(missing)


def load_voice_exemplars(*, base_dir: Path | str | None = None) -> str:
    """Load the real voice-exemplars file, failing loud via its receiver.

    Raises :class:`MissingInputsError` (naming the file + how to fill it) if the
    receiver is unfilled - never a silent fallback to empty/sample data.
    """
    base = _resolve_base(base_dir)
    require_inputs(base_dir=base, skip=_all_keys_except("voice_exemplars"))
    return base.joinpath(*_VOICE_REL).read_text(encoding="utf-8")


def load_corpus(*, base_dir: Path | str | None = None) -> str:
    """Load + concatenate the real transcript corpus, failing loud via its receiver.

    Reads every ``.txt/.md/.pdf`` transcript in ``data/corpus/`` (via the shared
    document loader so PDFs extract correctly), each prefixed with its filename.
    Raises :class:`MissingInputsError` if the receiver is unfilled.
    """
    base = _resolve_base(base_dir)
    require_inputs(base_dir=base, skip=_all_keys_except("transcript_corpus"))

    # Imported here (not at module top) to keep importing this module free of the
    # PDF dependency until corpus loading is actually exercised.
    from draftforge.ingest.loader import EmptyDocumentError, load_document

    d = base.joinpath(*_CORPUS_REL)
    chunks: list[str] = []
    for file in sorted(d.iterdir()):
        if not file.is_file() or file.suffix.lower() not in _CORPUS_SUFFIXES:
            continue
        try:
            text = load_document(file)
        except EmptyDocumentError:
            continue
        chunks.append(f"# {file.name}\n{text}")
    return "\n\n".join(chunks).strip()


def load_claims_register(*, base_dir: Path | str | None = None) -> list:
    """Load the real approved-claims register, failing loud via its receiver.

    Reads ``data/claims_register.json`` and returns a list of validated
    :class:`~draftforge.models.RegisterEntry`. Raises :class:`MissingInputsError`
    (naming the file + how to fill it) if the receiver is unfilled or malformed -
    never a silent fallback to empty/sample data. The committed
    ``data/claims_register.EXAMPLE.json`` is a format template only and is NEVER
    read here.

    Returns:
        ``list[RegisterEntry]`` - the parsed register (guaranteed non-empty).
    """
    base = _resolve_base(base_dir)
    # The receiver's validate() already enforces present + non-empty + valid
    # shape, raising MissingInputsError with the specific how-to-fill if not.
    require_inputs(base_dir=base, skip=_all_keys_except("claims_register"))

    import json

    from draftforge.models import RegisterEntry

    text = base.joinpath(*_CLAIMS_REGISTER_REL).read_text(encoding="utf-8")
    data = json.loads(text)
    return [RegisterEntry.model_validate(row) for row in data]


def _all_keys_except(key: str) -> set[str]:
    return {r.key for r in REQUIRED_INPUTS if r.key != key}


def run_preflight(
    *,
    base_dir: Path | str | None = None,
    emit: Callable[[str], None] = print,
    skip: Iterable[str] = (),
) -> int:
    """Print every receiver's status; return a process exit code.

    Each startup receiver is reported present ([OK]) or missing ([MISSING] +
    how to fill). Runtime receivers are reported informationally ([RUNTIME]
    provided at run time).
    Skipped receivers are still listed (so nothing disappears) but do not affect
    the exit code.

    Returns:
        ``0`` if every (non-skipped) startup receiver is present, else ``1``.
    """
    base = _resolve_base(base_dir)
    skip_set = set(skip)

    emit("Preflight - open-receiver status (D9: no silent placeholders)")
    emit("")

    missing_count = 0
    for receiver in REQUIRED_INPUTS:
        if receiver.kind == "runtime":
            emit(f"{RUNTIME_MARK} {receiver.key} - provided at run time")
            emit(f"    {receiver.description}")
            continue

        problem = receiver.check(base)
        skipped = receiver.key in skip_set
        if problem is None:
            emit(f"{PRESENT_MARK} {receiver.key} - present")
        else:
            note = " (skipped - open until wired)" if skipped else ""
            emit(f"{MISSING_MARK} {receiver.key} - MISSING{note}")
            emit(f"    {receiver.description}")
            emit(f"    how to fill: {problem.how_to_fill}")
            if not skipped:
                missing_count += 1

    emit("")
    if missing_count:
        emit(
            f"{MISSING_MARK} {missing_count} startup receiver(s) unfilled. "
            "Fill each above, then re-run preflight."
        )
        return 1
    emit(f"{PRESENT_MARK} All startup receivers present. Ready to run.")
    return 0
