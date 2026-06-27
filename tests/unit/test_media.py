"""Unit tests for the media upload ingest (M2).

``load_media`` turns a list of uploaded file paths into typed
:class:`~draftforge.ingest.media.MediaItem`s, detecting image vs. video by
extension and failing loud on an unsupported type, a missing file, or an
oversize file. Media is OPTIONAL — an empty upload is valid (every post then
keeps its ``image_direction`` shot suggestion).
"""

from __future__ import annotations

import pytest

from draftforge.ingest.media import (
    MediaItem,
    OversizeMediaError,
    UnsupportedMediaError,
    load_media,
)
from draftforge.models import MediaKind


def _write(path, data: bytes = b"x"):
    path.write_bytes(data)
    return str(path)


def test_empty_upload_is_valid_and_returns_empty_list():
    assert load_media([]) == []


def test_single_image_detected(tmp_path):
    p = _write(tmp_path / "calm.jpg")
    items = load_media([p])
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, MediaItem)
    assert item.kind == MediaKind.uploaded_image
    assert item.ref == p
    assert item.filename == "calm.jpg"
    assert item.size_bytes == 1


def test_single_video_detected(tmp_path):
    p = _write(tmp_path / "reel.mp4")
    items = load_media([p])
    assert items[0].kind == MediaKind.uploaded_video


@pytest.mark.parametrize("ext", [".png", ".jpeg", ".webp", ".gif"])
def test_image_extensions(tmp_path, ext):
    p = _write(tmp_path / f"img{ext}")
    assert load_media([p])[0].kind == MediaKind.uploaded_image


@pytest.mark.parametrize("ext", [".mov", ".webm", ".m4v"])
def test_video_extensions(tmp_path, ext):
    p = _write(tmp_path / f"vid{ext}")
    assert load_media([p])[0].kind == MediaKind.uploaded_video


def test_extension_case_insensitive(tmp_path):
    p = _write(tmp_path / "calm.JPG")
    assert load_media([p])[0].kind == MediaKind.uploaded_image


def test_order_is_preserved(tmp_path):
    a = _write(tmp_path / "a.jpg")
    b = _write(tmp_path / "b.mp4")
    c = _write(tmp_path / "c.png")
    refs = [i.ref for i in load_media([a, b, c])]
    assert refs == [a, b, c]


def test_unsupported_extension_fails_loud(tmp_path):
    p = _write(tmp_path / "notes.txt")
    with pytest.raises(UnsupportedMediaError):
        load_media([p])


def test_missing_file_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_media([str(tmp_path / "ghost.jpg")])


def test_oversize_image_fails_loud(tmp_path):
    p = _write(tmp_path / "big.jpg", data=b"x" * 50)
    with pytest.raises(OversizeMediaError):
        load_media([p], max_image_bytes=10)


def test_oversize_video_fails_loud(tmp_path):
    p = _write(tmp_path / "big.mp4", data=b"x" * 50)
    with pytest.raises(OversizeMediaError):
        load_media([p], max_video_bytes=10)


def test_image_under_video_cap_still_uses_image_cap(tmp_path):
    # A 50-byte image must fail under a 10-byte IMAGE cap even if the video cap is huge.
    p = _write(tmp_path / "big.jpg", data=b"x" * 50)
    with pytest.raises(OversizeMediaError):
        load_media([p], max_image_bytes=10, max_video_bytes=10_000)
