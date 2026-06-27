"""Media -> draft pairing stage (M3 / D10).

``pair_media`` assigns uploaded :class:`~draftforge.ingest.media.MediaItem`s to
generated :class:`~draftforge.models.Draft`s. It is **pure and deterministic** —
no I/O, no randomness — and returns NEW drafts (via ``model_copy``) rather than
mutating the inputs, so it is trivially unit-testable and safe to re-run.

The default ``order`` strategy assigns media to drafts positionally: the first N
drafts receive the N uploads in upload order; any remaining drafts keep
``media=None`` and rely on their ``image_direction`` shot suggestion; any media
beyond the draft count is dropped. The ``strategy`` argument is the seam for a
later, smarter ``image_direction``-aware match (and, eventually, M6 generation
filling the unpaired remainder).
"""

from __future__ import annotations

from draftforge.ingest.media import MediaItem
from draftforge.models import Draft, MediaRef

# The pairing strategies this stage understands. Kept as a set so an unknown
# strategy fails loud rather than silently no-op'ing.
_STRATEGIES: frozenset[str] = frozenset({"order"})


def _to_ref(item: MediaItem) -> MediaRef:
    """Project a validated upload into the draft-facing media reference."""
    return MediaRef(kind=item.kind, ref=item.ref)


def pair_media(
    drafts: list[Draft],
    media_items: list[MediaItem],
    *,
    strategy: str = "order",
) -> list[Draft]:
    """Pair ``media_items`` onto ``drafts`` and return new drafts.

    Args:
        drafts: the generated drafts, in display order.
        media_items: validated uploads (from :func:`draftforge.ingest.media.load_media`),
            in upload order. An empty list leaves every draft unpaired.
        strategy: pairing strategy. Only ``"order"`` is implemented (positional).

    Returns:
        A list the same length as ``drafts``: the first ``len(media_items)``
        drafts carry a :class:`~draftforge.models.MediaRef`; the rest keep
        ``media=None``. Inputs are never mutated.

    Raises:
        ValueError: if ``strategy`` is not a known strategy.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown pairing strategy {strategy!r}; known: {sorted(_STRATEGIES)}"
        )

    paired: list[Draft] = []
    for index, draft in enumerate(drafts):
        if index < len(media_items):
            paired.append(draft.model_copy(update={"media": _to_ref(media_items[index])}))
        else:
            paired.append(draft.model_copy())
    return paired
