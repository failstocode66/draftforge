"""Normalize raw ingested text into a :class:`~draftforge.models.Source`.

``to_source`` is the join point where loaded files and fetched URLs become the
typed payload the rest of the pipeline consumes. Two things are made
deterministic for testing and reproducibility:

* ``source_id`` is a stable 12-char prefix of the SHA-1 of the ``uri`` — the
  same URI always yields the same id, independent of body text or timestamp, so
  re-ingesting a source is idempotent.
* ``fetched_at`` comes from an injectable ``now`` (a callable, an ISO string, or
  ``None`` to use the real wall clock), so tests need not freeze time globally.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle (models <-> ingest) at runtime
    from draftforge.models import Source

_SOURCE_ID_LEN = 12


def to_source(
    kind: str,
    uri: str,
    text: str,
    *,
    title: str | None = None,
    now: str | Callable[[], str] | None = None,
) -> "Source":
    """Build a :class:`~draftforge.models.Source` from ingested text.

    Args:
        kind: Source type, ``"url"`` or ``"file"``.
        uri: The origin (URL or file path); hashed to derive ``source_id``.
        text: The readable body text.
        title: Optional human-readable title.
        now: Timestamp control — a zero-arg callable returning an ISO string, an
            ISO string, or ``None`` to use the current UTC time.

    Returns:
        A validated ``Source``.
    """
    # Imported lazily to avoid any import cycle between models and ingest.
    from draftforge.models import Source

    source_id = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:_SOURCE_ID_LEN]
    return Source(
        source_id=source_id,
        type=kind,
        title=title,
        text=text,
        fetched_at=_resolve_now(now),
    )


def _resolve_now(now: str | Callable[[], str] | None) -> str:
    """Resolve the injectable ``now`` into an ISO-8601 string."""
    if now is None:
        return datetime.now(timezone.utc).isoformat()
    if callable(now):
        return now()
    return now
