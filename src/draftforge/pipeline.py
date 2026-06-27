"""Batch orchestration: wire the stages into a runnable pipeline.

:func:`run_batch` runs the full prompt chain for each :class:`Source`:

    classify -> extract -> generate (once per target platform)

and collects the resulting :class:`Draft` posts.

**Per-source failure isolation.** A single bad source must never sink the whole
batch. Each source is run inside a guard: if *any* stage raises for that source,
the error is captured as a :class:`SourceError` and the source is skipped, while
every other source still produces its drafts. The collected errors are returned
on the :class:`BatchResult` so callers (the CLI summary, the next unit's
receiver registry, the P3 review UI) can surface what was skipped and why —
nothing fails silently.

**Batch-size distribution.** ``batch_size`` is the total posts requested per
source. It is split as evenly as possible across the target platforms: each
platform gets ``batch_size // len(platforms)`` posts, and the remainder is handed
one-at-a-time to the leading platforms. So ``batch_size=12`` over two platforms
is 6 + 6; ``batch_size=5`` is 3 + 2. (Exactness is not the contract — sane,
documented coverage is.) A platform that would receive zero posts is skipped.

**Stable ids (INT-4).** Each draft id is ``f"{source.source_id}-{platform}-{i}"``
(the source's *stable* id — a content hash of its uri, not its list position —
the platform, and the per-cell post index), so ids are unique and reproducible
across runs regardless of how the input list is ordered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from draftforge.llm.client import LLMClient
from draftforge.models import ClaimCheck, Draft, ExtractedItem, Platform, RegisterEntry, Source
from draftforge.stages.claims import claims_check
from draftforge.stages.classify import classify_content
from draftforge.stages.extract import extract_marketing_data
from draftforge.stages.generate import generate_posts

logger = logging.getLogger(__name__)

_DEFAULT_PLATFORMS: tuple[Platform, ...] = (Platform.facebook, Platform.instagram)


@dataclass(frozen=True)
class SourceError:
    """A captured per-source failure, recorded instead of aborting the batch."""

    source_index: int
    source_id: str
    stage: str  # which stage raised: "classify" | "extract" | "generate"
    error: Exception

    def __str__(self) -> str:  # human-friendly one-liner for the CLI summary
        return (
            f"source #{self.source_index} ({self.source_id}) failed at "
            f"{self.stage}: {self.error}"
        )


@dataclass
class BatchResult:
    """Outcome of a batch run: the produced drafts, per-source errors, and the
    claims-safety verdict for each draft.

    ``claim_checks`` maps a ``draft.id`` to the :class:`~draftforge.models.ClaimCheck`
    the claims-safety gate (Task 2.2) returned for it. It is kept SEPARATE from
    ``draft.status`` on purpose: ``draft.status`` is the *review lifecycle*
    (``draft -> edited -> approved ...``) while the claim check is the
    *claim-safety* verdict (``clean | softened | flagged | needs_manual_review``).
    Overloading the lifecycle status with the safety status would conflate two
    independent axes; the caller (CLI persistence, the P3 review UI) reads both.
    """

    drafts: list[Draft] = field(default_factory=list)
    errors: list[SourceError] = field(default_factory=list)
    claim_checks: dict[str, ClaimCheck] = field(default_factory=dict)


def run_batch(
    sources: list[Source],
    *,
    guidance: str,
    voice_exemplars: str,
    corpus: str,
    batch_size: int,
    register: list[RegisterEntry],
    platforms: tuple[Platform, ...] = _DEFAULT_PLATFORMS,
    llm: LLMClient,
) -> BatchResult:
    """Run the full pipeline over ``sources`` and collect drafts + claim checks.

    For each source the chain is classify -> extract -> (per platform) generate,
    and **after each generated draft the claims-safety gate runs**
    (:func:`~draftforge.stages.claims.claims_check`), so every draft carries a
    claim-safety verdict before it reaches the human review queue.

    Args:
        sources: Normalized ingested sources to process.
        guidance: The run's instruction prompt (in-memory; D9).
        voice_exemplars: Few-shot brand-voice examples (in-memory; D9).
        corpus: Excerpts of the business's real positions/voice (in-memory; D9).
        batch_size: Total posts to produce per source, split across platforms.
        register: The approved-claims register (from ``load_claims_register``)
            the claims gate checks each draft's hard claims against.
        platforms: Target platforms (default Facebook + Instagram).
        llm: The schema-validating LLM client.

    Returns:
        A :class:`BatchResult` with all drafts, any per-source errors, and a
        ``claim_checks`` map (``draft.id -> ClaimCheck``). The function never
        raises on a per-source stage failure — it records and skips, so a bad
        source cannot abort the batch.
    """
    result = BatchResult()
    per_platform = _distribute(batch_size, len(platforms))

    for index, source in enumerate(sources):
        try:
            drafts, claim_checks = _process_source(
                source=source,
                index=index,
                guidance=guidance,
                voice_exemplars=voice_exemplars,
                corpus=corpus,
                register=register,
                platforms=platforms,
                per_platform=per_platform,
                llm=llm,
            )
        except _StageFailure as failure:
            logger.warning("skipping source: %s", failure.source_error)
            result.errors.append(failure.source_error)
            continue

        result.drafts.extend(drafts)
        result.claim_checks.update(claim_checks)

    return result


class _StageFailure(Exception):
    """Internal: carries the :class:`SourceError` up out of ``_process_source``."""

    def __init__(self, source_error: SourceError) -> None:
        super().__init__(str(source_error))
        self.source_error = source_error


def _process_source(
    *,
    source: Source,
    index: int,
    guidance: str,
    voice_exemplars: str,
    corpus: str,
    register: list[RegisterEntry],
    platforms: tuple[Platform, ...],
    per_platform: list[int],
    llm: LLMClient,
) -> tuple[list[Draft], dict[str, ClaimCheck]]:
    """Run the chain for one source, claims-gating each draft.

    Returns the source's drafts plus a ``{draft.id: ClaimCheck}`` map. Raises
    :class:`_StageFailure` on a classify/extract/generate error (the per-source
    guard then skips the whole source). The claims gate itself is fail-safe and
    never raises, so a draft always gets a verdict once it is generated.
    """
    # classify -> extract happen once per source (the angle/item is shared).
    try:
        classification = classify_content(source.text, llm)
    except Exception as exc:  # noqa: BLE001 — isolate every per-source failure
        raise _StageFailure(_err(index, source, "classify", exc)) from exc

    angle = classification.angle
    try:
        item = extract_marketing_data(source.text, angle, llm)
    except Exception as exc:  # noqa: BLE001
        raise _StageFailure(_err(index, source, "extract", exc)) from exc

    # generate runs once per platform, each with its share of batch_size; each
    # produced draft is then run through the claims-safety gate.
    drafts: list[Draft] = []
    claim_checks: dict[str, ClaimCheck] = {}
    for platform, n in zip(platforms, per_platform):
        if n <= 0:
            continue
        try:
            posts = generate_posts(
                item,
                guidance=guidance,
                voice_exemplars=voice_exemplars,
                corpus=corpus,
                platform=platform,
                n=n,
                llm=llm,
                id_prefix=f"{source.source_id}-{platform}",
                angle=angle,
            )
        except Exception as exc:  # noqa: BLE001
            raise _StageFailure(_err(index, source, "generate", exc)) from exc

        for post in posts:
            # DI-2: draft ids are unique by construction
            # (f"{source_id}-{platform}-{i}"), so a collision here is a broken
            # invariant — not a normal condition. Overwriting would silently drop
            # one draft's claim-safety verdict, exactly the kind of silent loss a
            # compliance control must never allow. Raise loudly instead.
            if post.id in claim_checks:
                raise ValueError(
                    f"duplicate draft id {post.id!r} within a single batch; "
                    "draft ids must be unique (overwriting would drop a "
                    "claim-safety verdict)"
                )
            # claims_check is fail-safe (never raises): on any internal error it
            # returns needs_manual_review rather than silently passing the draft.
            claim_checks[post.id] = claims_check(
                post, extracted_item=item, register=register, llm=llm
            )
        drafts.extend(posts)

    return drafts, claim_checks


def _err(index: int, source: Source, stage: str, exc: Exception) -> SourceError:
    return SourceError(
        source_index=index,
        source_id=source.source_id,
        stage=stage,
        error=exc,
    )


def _distribute(total: int, buckets: int) -> list[int]:
    """Split ``total`` across ``buckets`` as evenly as possible.

    The remainder is handed one-at-a-time to the leading buckets, so the result
    is front-loaded but differs by at most one between any two buckets, e.g.
    ``_distribute(5, 2) == [3, 2]`` and ``_distribute(4, 2) == [2, 2]``.
    """
    if buckets <= 0:
        return []
    base, remainder = divmod(max(total, 0), buckets)
    return [base + (1 if i < remainder else 0) for i in range(buckets)]
