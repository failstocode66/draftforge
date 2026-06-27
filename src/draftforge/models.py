"""Domain models for the content pipeline.

These Pydantic v2 models are the typed payloads that flow between pipeline
stages: a :class:`Source` (normalized ingested material) is extracted into
:class:`ExtractedItem`s, which are generated into :class:`Draft`s, which the
claims-safety gate evaluates into :class:`ClaimCheck`s.

All models are plain data carriers — no behavior — so they JSON round-trip
cleanly (``model_dump_json`` / ``model_validate_json``) for storage and for
feeding the LLM client's schema-validated completion.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Platform(StrEnum):
    """Target social platform for a generated draft."""

    facebook = "facebook"
    instagram = "instagram"


class ClaimType(StrEnum):
    """Strength of a marketing claim.

    ``soft`` claims are subjective/experiential ("many people feel relaxed");
    ``hard`` claims assert objective effects ("reduces cortisol") and require
    a supporting source or softening by the claims-safety gate.
    """

    soft = "soft"
    hard = "hard"


class MediaKind(StrEnum):
    """Origin and type of media paired to a draft (D10).

    ``uploaded_image`` / ``uploaded_video`` are the studio's own assets supplied at
    run time; ``generated_image`` is the staged, opt-in AI-image fast-follow (M6)
    that fills a media-less draft from its ``image_direction``. Video is
    upload-only — there is deliberately no ``generated_video``.
    """

    uploaded_image = "uploaded_image"
    uploaded_video = "uploaded_video"
    generated_image = "generated_image"


class MediaRef(BaseModel):
    """A reference to the media paired to a :class:`Draft` (D10).

    ``ref`` is an opaque locator — an uploaded file's path/handle, or a
    generator reference. ``source_prompt`` records the ``image_direction`` a
    generated image was produced from; it is ``None`` for uploads (an upload has
    no generation prompt).
    """

    kind: MediaKind
    ref: str
    source_prompt: str | None = None


class Source(BaseModel):
    """Normalized ingested source material feeding the pipeline."""

    source_id: str
    type: str  # "url" | "file"
    title: str | None = None
    text: str
    fetched_at: str  # ISO 8601 timestamp


class ExtractedItem(BaseModel):
    """A marketing angle's worth of structured data pulled from a source."""

    hook: str
    core_benefit: str
    claim: str | None = None
    claim_type: ClaimType | None = None
    supporting_source: str | None = None
    audience: str | None = None
    suggested_cta: str | None = None


class Draft(BaseModel):
    """A generated, review-gated social post draft."""

    id: str
    platform: Platform
    angle: str
    caption: str
    hashtags: list[str]
    image_direction: str | None = None
    claims_used: list[str] = Field(default_factory=list)
    status: str = "draft"
    scheduled_date: str | None = None
    edited_text: str | None = None
    media: MediaRef | None = None


class RegisterEntry(BaseModel):
    """One approved-claims-register entry (a row of ``data/claims_register.json``).

    The register is the owner-editable source of truth the claims-safety gate
    checks hard claims against. An entry is "approved" only when ``approved`` is
    True; the gate's :func:`~draftforge.stages.claims_register.match_claim`
    considers approved entries exclusively, so an unapproved row never licenses a
    hard claim to go out as-is.
    """

    claim_text: str
    claim_type: ClaimType
    approved: bool
    source_citation: str = ""
    notes: str = ""


class ClaimCheck(BaseModel):
    """Result of the claims-safety gate evaluating a draft's claims."""

    status: str  # clean | softened | flagged | needs_manual_review
    notes: list[str] = Field(default_factory=list)
    revised_text: str | None = None
