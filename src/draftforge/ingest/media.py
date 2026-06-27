"""Media upload ingest (M2 / D10).

``load_media`` turns a list of uploaded file paths (e.g. from the Gradio Run
tab) into typed :class:`MediaItem`s, detecting image vs. video by extension and
failing loud — never silently dropping a file — on:

* a missing file (:class:`FileNotFoundError`);
* an unsupported extension (:class:`UnsupportedMediaError`);
* a file over its per-kind size cap (:class:`OversizeMediaError`).

Media is OPTIONAL: an empty upload returns ``[]`` and is valid (every post then
keeps its ``image_direction`` as a shot suggestion). The returned list preserves
input order, which the default order-based pairing strategy (M3) relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from draftforge.models import MediaKind

# Supported upload extensions, mapped to their media kind. Video is upload-only
# (there is no generated_video); AI-generated images are an M6 concern, not an
# upload kind, so they never appear here.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".gif"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".webm", ".m4v"})

# Default per-kind size caps. Videos are legitimately far larger than images, so
# they get their own ceiling rather than a single shared cap.
DEFAULT_MAX_IMAGE_BYTES: int = 15 * 1024 * 1024  # 15 MB
DEFAULT_MAX_VIDEO_BYTES: int = 200 * 1024 * 1024  # 200 MB


class UnsupportedMediaError(ValueError):
    """Raised when an uploaded file's extension is not a supported image/video."""


class OversizeMediaError(ValueError):
    """Raised when an uploaded file exceeds its per-kind size cap."""


@dataclass(frozen=True)
class MediaItem:
    """A validated uploaded media file ready to be paired to a draft.

    ``ref`` is the file path/handle (the locator that becomes
    :class:`~draftforge.models.MediaRef.ref` in the pairing stage); ``kind`` is
    ``uploaded_image`` or ``uploaded_video``; ``filename`` and ``size_bytes`` are
    retained for display + auditing.
    """

    kind: MediaKind
    ref: str
    filename: str
    size_bytes: int


def _detect_kind(suffix: str) -> MediaKind:
    if suffix in IMAGE_EXTENSIONS:
        return MediaKind.uploaded_image
    if suffix in VIDEO_EXTENSIONS:
        return MediaKind.uploaded_video
    supported = sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    raise UnsupportedMediaError(
        f"unsupported media type {suffix!r}; supported: {supported}"
    )


def load_media(
    paths: list[str | Path],
    *,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
) -> list[MediaItem]:
    """Validate uploaded files and return them as ordered :class:`MediaItem`s.

    Args:
        paths: uploaded file paths, in the order the user supplied them. An empty
            list is valid and returns ``[]``.
        max_image_bytes: per-image size cap (default 15 MB).
        max_video_bytes: per-video size cap (default 200 MB).

    Raises:
        FileNotFoundError: if any path does not exist.
        UnsupportedMediaError: if any file's extension is not a supported type.
        OversizeMediaError: if any file exceeds its per-kind cap.
    """
    items: list[MediaItem] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"media file not found: {p}")

        kind = _detect_kind(p.suffix.lower())
        size = p.stat().st_size
        cap = (
            max_image_bytes
            if kind is MediaKind.uploaded_image
            else max_video_bytes
        )
        if size > cap:
            raise OversizeMediaError(
                f"media file {p.name!r} is {size} bytes, over the "
                f"{cap}-byte cap for {kind.value}"
            )

        items.append(
            MediaItem(kind=kind, ref=str(p), filename=p.name, size_bytes=size)
        )
    return items
