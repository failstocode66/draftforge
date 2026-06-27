"""Persist a completed batch to the store — shared by the CLI and the Gradio app.

Extracted (3.2) so both entrypoints write a batch identically: one atomic
:meth:`~draftforge.store.db.Store.transaction` (DI-1) holding the batch row, every
source, every draft, and every draft's claim-safety verdict. A draft reaching
here with NO verdict is a fail-loud compliance error (P2-2) — the transaction
rolls back the partial batch rather than persist an unverified draft.
"""

from __future__ import annotations

from draftforge.models import Source
from draftforge.pipeline import BatchResult
from draftforge.store.db import Store


class MissingClaimVerdictError(Exception):
    """Raised when persistence is asked to write a draft with no claims verdict.

    The claims-safety gate guarantees a verdict per draft, so a draft reaching
    :func:`persist_batch` without a ``claim_checks`` entry is a broken invariant —
    a compliance control must fail loud here rather than silently persist an
    unverified draft. The surrounding transaction rolls back the partial batch.
    """


def persist_batch(
    store: Store,
    result: BatchResult,
    sources: list[Source],
    *,
    guidance: str,
    batch_size: int,
    batch_id: str,
    now=None,
) -> None:
    """Write one batch's sources, drafts, and claim flags into ``store``.

    The whole write runs inside ONE :meth:`Store.transaction` (DI-1), so a
    failure mid-persist rolls the lot back and never leaves an orphan/partial
    batch. Each draft's full :class:`~draftforge.models.ClaimCheck`
    (``status`` + ``notes`` + ``revised_text``) rides in the post's
    ``claim_flags`` JSON column for the review UI / audit; ``Draft.media`` (set by
    the pairing stage) is persisted by :meth:`Store.save_draft`.

    Raises:
        MissingClaimVerdictError: if any draft lacks a claim-safety verdict
            (P2-2). The transaction rolls back the partial batch.
    """
    with store.transaction():
        store.add_batch(
            batch_id,
            guidance_prompt=guidance,
            url_set=[s.title or s.source_id for s in sources],
            batch_size=batch_size,
            now=now,
        )
        for source in sources:
            store.save_source(source, batch_id)
        for draft in result.drafts:
            store.save_draft(draft, batch_id, now=now)
            check = result.claim_checks.get(draft.id)
            if check is None:
                raise MissingClaimVerdictError(
                    f"draft {draft.id!r} has no claim-safety verdict; refusing "
                    "to persist a draft without its compliance check (the gate "
                    "must produce a verdict for every draft)"
                )
            store.set_claim_flags(draft.id, batch_id, check.model_dump())
