"""Unit tests for the media -> draft pairing stage (M3).

``pair_media`` is pure and deterministic: it assigns uploaded media to drafts
(default ``order`` strategy = first N drafts get the N uploads) and returns NEW
drafts, never mutating the inputs. Drafts beyond the media count keep
``media=None`` and rely on their ``image_direction`` shot suggestion.
"""

from __future__ import annotations

import pytest

from draftforge.ingest.media import MediaItem
from draftforge.models import Draft, MediaKind, MediaRef, Platform
from draftforge.stages.pair_media import pair_media


def _draft(post_id: str) -> Draft:
    return Draft(
        id=post_id,
        platform=Platform.instagram,
        angle="relaxation",
        caption="Sink into stillness.",
        hashtags=[],
        image_direction="calm water",
    )


def _img(name: str) -> MediaItem:
    return MediaItem(
        kind=MediaKind.uploaded_image, ref=name, filename=name, size_bytes=1
    )


def _vid(name: str) -> MediaItem:
    return MediaItem(
        kind=MediaKind.uploaded_video, ref=name, filename=name, size_bytes=1
    )


def test_no_media_leaves_all_drafts_unpaired():
    drafts = [_draft("d1"), _draft("d2")]
    out = pair_media(drafts, [])
    assert len(out) == 2
    assert all(d.media is None for d in out)


def test_order_strategy_pairs_first_n_in_order():
    drafts = [_draft("d1"), _draft("d2"), _draft("d3")]
    media = [_img("a.jpg"), _vid("b.mp4")]
    out = pair_media(drafts, media)

    assert out[0].media == MediaRef(kind=MediaKind.uploaded_image, ref="a.jpg")
    assert out[1].media == MediaRef(kind=MediaKind.uploaded_video, ref="b.mp4")
    assert out[2].media is None  # third draft keeps its shot suggestion


def test_more_drafts_than_media_leaves_remainder_unpaired():
    drafts = [_draft(f"d{i}") for i in range(4)]
    out = pair_media(drafts, [_img("only.jpg")])
    assert out[0].media.ref == "only.jpg"
    assert all(d.media is None for d in out[1:])


def test_more_media_than_drafts_drops_extra_media():
    drafts = [_draft("d1")]
    out = pair_media(drafts, [_img("a.jpg"), _img("b.jpg"), _img("c.jpg")])
    assert len(out) == 1
    assert out[0].media.ref == "a.jpg"


def test_equal_counts_pair_one_to_one():
    drafts = [_draft("d1"), _draft("d2")]
    out = pair_media(drafts, [_img("a.jpg"), _vid("b.mp4")])
    assert [d.media.ref for d in out] == ["a.jpg", "b.mp4"]
    assert out[0].media.kind == MediaKind.uploaded_image
    assert out[1].media.kind == MediaKind.uploaded_video


def test_pairing_is_pure_inputs_not_mutated():
    drafts = [_draft("d1")]
    pair_media(drafts, [_img("a.jpg")])
    assert drafts[0].media is None  # original untouched


def test_image_direction_is_preserved_on_paired_draft():
    out = pair_media([_draft("d1")], [_img("a.jpg")])
    assert out[0].image_direction == "calm water"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        pair_media([_draft("d1")], [_img("a.jpg")], strategy="smart-match")
