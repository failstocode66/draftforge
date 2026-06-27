"""Command-line entrypoint for the content pipeline.

Two commands:

    python -m draftforge.cli --input <dir-or-url> --guidance "..." --n 12
        [--voice-file <path>] [--corpus-dir <path>]

    python -m draftforge.cli preflight

The module is split so its core is testable WITHOUT an API key or a network:

* :func:`parse_args` - pure argparse; importable and callable with no key.
* :func:`run_cli` - the testable core. Takes an **injected** ``llm`` and HTTP
  ``getter``, ingests the input, runs :func:`~draftforge.pipeline.run_batch`,
  prints each rendered draft, summarizes any skipped sources, and returns the
  :class:`~draftforge.pipeline.BatchResult`.
* :func:`main` - the thin production wrapper. For ``preflight`` it just runs the
  open-receiver check (no key needed). For a run it builds a real
  :class:`~draftforge.llm.client.LLMClient` from :class:`~draftforge.config.Settings`
  + the real Anthropic transport, then calls :func:`run_cli`.

**D9 fail-loud seam.** The brand voice and business corpus are *open receivers*
(see :mod:`draftforge.inputs`). By default :func:`run_cli` loads them via the
registry, which raises :class:`~draftforge.inputs.MissingInputsError` (naming the
real file + how to fill it) when absent - there is *no* placeholder fallback.
``--voice-file`` / ``--corpus-dir`` remain as optional **overrides**; when given,
those explicit paths are loaded and an absence raises :class:`MissingInputError`.
``preflight`` enumerates EVERY unfilled receiver at once so a slot can't be missed.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

import requests

from draftforge import inputs
from draftforge.ingest.fetcher import fetch_url
from draftforge.ingest.loader import EmptyDocumentError, load_document
from draftforge.ingest.media import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MediaItem,
    load_media,
)
from draftforge.ingest.normalize import to_source
from draftforge.llm.client import LLMClient
from draftforge.models import Platform, Source
from draftforge.output.render import render_text
from draftforge.persistence import MissingClaimVerdictError, persist_batch
from draftforge.pipeline import BatchResult, run_batch
from draftforge.stages.pair_media import pair_media
from draftforge.store.db import Store

# Re-exported for backward compatibility: callers/tests reference
# ``cli.MissingClaimVerdictError``. The canonical definition lives in
# :mod:`draftforge.persistence` (shared by the CLI and the app).
__all__ = ["MissingClaimVerdictError"]

_DEFAULT_N = 12

# Default on-disk path for the persistent SQLite store (gitignored: data/* and
# *.db are both excluded). Injectable per run via --db.
_DEFAULT_DB = "data/store.db"

# File extensions the directory ingester will attempt to load.
_SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf")


def _default_batch_id() -> str:
    """Generate a fresh batch id. Injected in tests for offline determinism."""
    return f"batch-{uuid.uuid4().hex[:12]}"


class MissingInputError(Exception):
    """Raised when an explicitly-overridden voice/corpus path is missing/empty.

    This is the override path (``--voice-file`` / ``--corpus-dir``). When no
    override is given, the default registry loaders raise
    :class:`~draftforge.inputs.MissingInputsError` instead.
    """


def _positive_int(value: str) -> int:
    """argparse type for ``--n``: a strictly-positive integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Pure - no key, no network, no side effects."""
    parser = argparse.ArgumentParser(
        prog="draftforge",
        description=(
            "Turn a business's source material into review-gated FB/IG drafts."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    preflight = subparsers.add_parser(
        "preflight",
        help="Report every real-data open receiver's status (D9) and exit "
        "non-zero if any startup receiver is unfilled.",
    )
    preflight.add_argument(
        "--base-dir",
        default=None,
        help="Root the receiver paths resolve under (default: repo root). "
        "For tests / alternate deployments.",
    )
    preflight.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="KEY",
        help="Receiver key to treat as informational (e.g. an open receiver "
        "not yet wired, like claims_register before P2). Repeatable.",
    )

    parser.add_argument(
        "--input",
        help="A directory of source documents (.txt/.md/.pdf) OR a single URL.",
    )
    parser.add_argument(
        "--guidance",
        help="Run instruction prompt (tone, focus, do/don't).",
    )
    parser.add_argument(
        "--n",
        type=_positive_int,
        default=_DEFAULT_N,
        help=f"Total posts to produce per source (default {_DEFAULT_N}; >= 1).",
    )
    parser.add_argument(
        "--voice-file",
        default=None,
        help="Override path to a brand-voice exemplars file. Default: the "
        "voice_exemplars open receiver (prompts/voice_exemplars.md).",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Override directory of business corpus files. Default: the "
        "transcript_corpus open receiver (data/corpus/).",
    )
    parser.add_argument(
        "--media-dir",
        default=None,
        help="Optional directory of images/videos to pair to the generated "
        "posts (order-based, D10). Non-media files are ignored; video is "
        "upload-only. The Gradio app passes uploads directly instead.",
    )
    parser.add_argument(
        "--db",
        default=_DEFAULT_DB,
        help=f"Path to the SQLite store the run persists into (default "
        f"{_DEFAULT_DB}; gitignored).",
    )

    args = parser.parse_args(argv)

    # A run (no subcommand) requires --input and --guidance. (preflight needs
    # neither.) Enforced here rather than via required=True so the preflight
    # subcommand can be invoked without them.
    if args.command is None:
        missing = [
            name
            for name, val in (("--input", args.input), ("--guidance", args.guidance))
            if val is None
        ]
        if missing:
            parser.error(
                "the following arguments are required: " + ", ".join(missing)
            )

    return args


def run_cli(
    args: argparse.Namespace,
    *,
    llm: LLMClient,
    getter=requests.get,
    out=None,
    base_dir: Path | str | None = None,
    store: Store | None = None,
    media_items: list[MediaItem] | None = None,
    batch_id_factory: Callable[[], str] = _default_batch_id,
    now=None,
) -> BatchResult:
    """Run the pipeline for ``args`` using the injected ``llm`` (and ``getter``).

    After the batch runs, its result is persisted to a :class:`Store` (SPEC-3):
    one ``batches`` row, a ``sources`` row per source, a ``posts`` row per draft,
    and each draft's claim-safety verdict written to ``posts.claim_flags`` via
    :meth:`Store.set_claim_flags`. ``run_batch`` itself stays pure — it never
    touches the store; persistence lives here in the caller.

    Args:
        args: Parsed CLI arguments (see :func:`parse_args`).
        llm: The LLM client to drive the stages (injected for testability).
        getter: HTTP getter for URL ingest (default :func:`requests.get`).
            Injected in tests to avoid the network.
        out: Output stream (default ``sys.stdout``).
        base_dir: Root the default open receivers resolve under (default: repo
            root). Injected in tests; ignored when an override flag is given.
        store: The persistent store to write into. Default: a :class:`Store`
            opened at ``args.db``. Injected in tests (e.g. an in-memory store).
        media_items: Validated uploads to pair onto the drafts (D10). The app
            passes Gradio uploads here directly; when ``None``, the CLI loads
            them from ``--media-dir`` (or none). Pairing runs after the pure
            ``run_batch`` and before persistence.
        batch_id_factory: Zero-arg callable producing the batch id. Default is a
            uuid-based generator; injected in tests for offline determinism.
        now: Injectable timestamp (callable or ISO string) for the store's
            ``created_at`` stamps (default real UTC time).

    Returns:
        The :class:`~draftforge.pipeline.BatchResult` from the run.

    Raises:
        MissingInputError: if an explicitly-overridden voice/corpus path is
            missing or empty.
        draftforge.inputs.MissingInputsError: if a default open receiver is
            unfilled (the fail-loud registry path).
    """
    stream = out if out is not None else sys.stdout

    # D9: load grounding inputs, failing loud if absent/empty. The DEFAULT is the
    # open-receiver registry; the flags only override.
    if args.voice_file is not None:
        voice_exemplars = _load_voice_file(args.voice_file)
    else:
        voice_exemplars = inputs.load_voice_exemplars(base_dir=base_dir)

    if args.corpus_dir is not None:
        corpus = _load_corpus_dir(args.corpus_dir)
    else:
        corpus = inputs.load_corpus(base_dir=base_dir)

    # D9: the approved-claims register is an open receiver too. There is no
    # override flag and no placeholder fallback — load it fail-loud so a run can
    # never silently skip the claims-safety gate's authority.
    register = inputs.load_claims_register(base_dir=base_dir)

    sources = _ingest(args.input, getter=getter)

    result = run_batch(
        sources,
        guidance=args.guidance,
        voice_exemplars=voice_exemplars,
        corpus=corpus,
        batch_size=args.n,
        register=register,
        llm=llm,
    )

    # M4 / D10: pair uploaded media onto the drafts (order strategy) AFTER the
    # pure run_batch, before persistence — keeping run_batch store/media-free.
    # Media comes from an injected list (the app passes Gradio uploads directly)
    # or, for the CLI, from --media-dir. Pairing preserves draft ids, so the
    # claim_checks keyed by id stay valid.
    media = (
        media_items
        if media_items is not None
        else _ingest_media(getattr(args, "media_dir", None))
    )
    if media:
        result.drafts = pair_media(result.drafts, media)

    # SPEC-3: persist the (pure) BatchResult. Open the default file store only if
    # the caller did not inject one; ensure its parent dir exists first.
    owns_store = store is None
    if owns_store:
        _ensure_parent_dir(args.db)
        store = Store(args.db)
    try:
        _persist(
            result,
            sources,
            store=store,
            args=args,
            batch_id=batch_id_factory(),
            now=now,
        )
    finally:
        if owns_store:
            store.close()

    _report(result, stream=stream)
    return result


def _persist(
    result: BatchResult,
    sources: list[Source],
    *,
    store: Store,
    args: argparse.Namespace,
    batch_id: str,
    now,
) -> None:
    """Write one batch's sources, drafts, and claim flags into ``store`` (SPEC-3).

    The whole write — the batch row, every source, every draft, and every
    draft's claim-safety verdict — runs inside ONE :meth:`Store.transaction`
    (DI-1 / INT-P2-01), so a failure mid-persist rolls the lot back and never
    leaves an orphan/partial batch. Because each run is its own ``batch_id`` and
    the store keys sources/posts by ``(batch_id, ...)``, re-running an
    overlapping source coexists as a separate batch instead of colliding on a
    primary key.

    A draft with NO claim-safety verdict is a fail-loud error (P2-2): the gate
    guarantees a verdict per draft, so a missing one is exactly the silent
    compliance hole a safety control must never write. The shared
    :func:`~draftforge.persistence.persist_batch` raises
    :class:`MissingClaimVerdictError` and the surrounding transaction rolls back
    the partial batch.

    This is a thin adapter mapping the CLI's ``args`` onto the shared
    :func:`persist_batch` (used identically by the Gradio app).
    """
    persist_batch(
        store,
        result,
        sources,
        guidance=args.guidance,
        batch_size=args.n,
        batch_id=batch_id,
        now=now,
    )


def main(argv: list[str] | None = None) -> int:
    """Production entrypoint.

    ``preflight`` runs the open-receiver check (no key needed). A run builds the
    real :class:`~draftforge.llm.client.LLMClient`, which requires
    ``ANTHROPIC_API_KEY`` (via :meth:`Settings.load`) - but :func:`parse_args`,
    ``--help``, and ``preflight`` do not.
    """
    args = parse_args(argv)

    if args.command == "preflight":
        return inputs.run_preflight(
            base_dir=args.base_dir, emit=print, skip=set(args.skip)
        )

    # Imported here (not at module top) so importing this module - and thus
    # arg-parsing in tests - never requires a key or the anthropic SDK.
    from draftforge.config import Settings
    from draftforge.llm.anthropic_transport import AnthropicTransport

    try:
        settings = Settings.load()
    except KeyError:
        print(
            "error: ANTHROPIC_API_KEY is not set. Running the pipeline requires "
            "a key (set it in the environment or a local .env).",
            file=sys.stderr,
        )
        return 2

    llm = LLMClient(
        AnthropicTransport(),
        model_fast=settings.model_fast,
        model_smart=settings.model_smart,
    )

    try:
        result = run_cli(args, llm=llm)
    except (MissingInputError, inputs.MissingInputsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Non-zero exit if every source failed, so scripts/CI notice.
    return 0 if result.drafts else 1


# --- D9 input loaders -----------------------------------------------------------


def _load_voice_file(path: str) -> str:
    """Load the brand-voice exemplars file, failing loud if missing/empty."""
    p = Path(path)
    if not p.is_file():
        raise MissingInputError(
            f"--voice-file not found: {p}. A brand-voice exemplars file is "
            "required (no placeholder fallback)."
        )
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise MissingInputError(
            f"--voice-file is empty: {p}. Provide real brand-voice exemplars."
        )
    return text


def _load_corpus_dir(path: str) -> str:
    """Load + concatenate the corpus directory, failing loud if missing/empty."""
    d = Path(path)
    if not d.is_dir():
        raise MissingInputError(
            f"--corpus-dir not found (or not a directory): {d}. A business "
            "corpus directory is required (no placeholder fallback)."
        )

    chunks: list[str] = []
    for file in sorted(d.iterdir()):
        if not file.is_file() or file.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        try:
            text = load_document(file)
        except EmptyDocumentError:
            continue
        chunks.append(f"# {file.name}\n{text}")

    corpus = "\n\n".join(chunks).strip()
    if not corpus:
        raise MissingInputError(
            f"--corpus-dir has no usable content: {d}. Provide at least one "
            "non-empty .txt/.md/.pdf corpus file."
        )
    return corpus


# --- ingest ---------------------------------------------------------------------


def _ingest(input_arg: str, *, getter=requests.get) -> list[Source]:
    """Ingest the ``--input`` value: a URL becomes one source, a dir becomes many.

    A missing path that is not a URL raises :class:`MissingInputError`.
    """
    if _looks_like_url(input_arg):
        text = fetch_url(input_arg, getter=getter)
        return [to_source("url", input_arg, text)]

    directory = Path(input_arg)
    if not directory.is_dir():
        raise MissingInputError(
            f"--input is neither a URL nor an existing directory: {input_arg}"
        )

    sources: list[Source] = []
    for file in sorted(directory.iterdir()):
        if not file.is_file() or file.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        try:
            text = load_document(file)
        except EmptyDocumentError:
            continue
        sources.append(to_source("file", str(file), text, title=file.name))
    return sources


def _ingest_media(media_dir: str | None) -> list[MediaItem]:
    """Load media files from ``--media-dir`` (skipping non-media), or ``[]``.

    Mirrors the ``--input`` directory ingest: only files with a supported
    image/video extension are picked up, in sorted order so pairing is
    deterministic; any other file is ignored. A given-but-missing directory
    fails loud (a typo'd path should not silently produce no media).
    """
    if media_dir is None:
        return []
    d = Path(media_dir)
    if not d.is_dir():
        raise MissingInputError(
            f"--media-dir not found (or not a directory): {d}"
        )
    media_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    paths: list[str | Path] = [
        str(f)
        for f in sorted(d.iterdir())
        if f.is_file() and f.suffix.lower() in media_exts
    ]
    return load_media(paths)


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory for ``path`` if it does not exist (for --db)."""
    parent = Path(path).expanduser().parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


# --- reporting ------------------------------------------------------------------

# The four claim-safety statuses, in the order the summary reports them.
_CLAIM_STATUSES = ("clean", "softened", "flagged", "needs_manual_review")


def _report(result: BatchResult, *, stream) -> None:
    """Print each rendered draft, then summaries of skips and claim safety."""
    for draft in result.drafts:
        print(render_text(draft), file=stream)
        print(file=stream)

    n_drafts = len(result.drafts)
    print(
        f"Summary: produced {n_drafts} draft(s); "
        f"skipped {len(result.errors)} source(s).",
        file=stream,
    )
    for err in result.errors:
        print(f"  - SKIPPED {err}", file=stream)

    # Claim-safety summary: count each verdict so a reviewer immediately sees how
    # many drafts need attention (anything that is not "clean").
    counts = {status: 0 for status in _CLAIM_STATUSES}
    for check in result.claim_checks.values():
        counts[check.status] = counts.get(check.status, 0) + 1
    summary = ", ".join(f"{status}={counts.get(status, 0)}" for status in _CLAIM_STATUSES)
    print(f"Claim safety: {summary}.", file=stream)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
