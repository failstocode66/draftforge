import pytest
from pydantic import ValidationError

from draftforge.models import (
    ClaimCheck,
    ClaimType,
    Draft,
    ExtractedItem,
    MediaKind,
    MediaRef,
    Platform,
    Source,
)


def test_platform_and_claim_type_enum_values():
    assert Platform.facebook == "facebook"
    assert Platform.instagram == "instagram"
    assert ClaimType.soft == "soft"
    assert ClaimType.hard == "hard"


def test_source_construction_and_round_trip():
    source = Source(
        source_id="abc123",
        type="url",
        title="A Title",
        text="some body text",
        fetched_at="2026-06-25T12:00:00Z",
    )

    assert source.source_id == "abc123"
    assert source.type == "url"
    assert source.title == "A Title"

    restored = Source.model_validate_json(source.model_dump_json())
    assert restored == source


def test_source_title_optional():
    source = Source(
        source_id="abc123",
        type="file",
        text="body",
        fetched_at="2026-06-25T12:00:00Z",
    )
    assert source.title is None


def test_extracted_item_construction_and_round_trip():
    item = ExtractedItem(
        hook="Float away your stress",
        core_benefit="deep relaxation",
        claim="reduces cortisol",
        claim_type=ClaimType.hard,
        supporting_source="study-2020",
        audience="busy professionals",
        suggested_cta="Book your float today",
    )

    assert item.claim_type == ClaimType.hard
    restored = ExtractedItem.model_validate_json(item.model_dump_json())
    assert restored == item


def test_extracted_item_optionals_default_to_none():
    item = ExtractedItem(hook="h", core_benefit="b")

    assert item.claim is None
    assert item.claim_type is None
    assert item.supporting_source is None
    assert item.audience is None
    assert item.suggested_cta is None


def test_draft_construction_defaults_and_round_trip():
    draft = Draft(
        id="d1",
        platform=Platform.instagram,
        angle="relaxation",
        caption="Sink into stillness.",
        hashtags=["#floattherapy", "#wellness"],
    )

    assert draft.platform == Platform.instagram
    assert draft.image_direction is None
    assert draft.claims_used == []
    assert draft.status == "draft"
    assert draft.scheduled_date is None
    assert draft.edited_text is None

    restored = Draft.model_validate_json(draft.model_dump_json())
    assert restored == draft


def test_draft_full_construction_round_trip():
    draft = Draft(
        id="d2",
        platform=Platform.facebook,
        angle="recovery",
        caption="Recover faster.",
        hashtags=["#recovery"],
        image_direction="calm blue pool",
        claims_used=["reduces soreness"],
        status="approved",
        scheduled_date="2026-07-01",
        edited_text="Recover faster, naturally.",
    )

    restored = Draft.model_validate_json(draft.model_dump_json())
    assert restored == draft


def test_claim_check_construction_defaults_and_round_trip():
    check = ClaimCheck(status="clean")

    assert check.notes == []
    assert check.revised_text is None

    full = ClaimCheck(
        status="softened",
        notes=["softened a hard claim"],
        revised_text="may help many people relax",
    )
    restored = ClaimCheck.model_validate_json(full.model_dump_json())
    assert restored == full


def test_platform_enum_rejects_bad_value():
    with pytest.raises(ValidationError):
        Draft(
            id="x",
            platform="myspace",
            angle="a",
            caption="c",
            hashtags=[],
        )


def test_claim_type_enum_rejects_bad_value():
    with pytest.raises(ValidationError):
        ExtractedItem(hook="h", core_benefit="b", claim_type="medium")


# --------------------------------------------------------------------------- #
# Media (M1 / D10)
# --------------------------------------------------------------------------- #


def test_media_kind_enum_values():
    assert MediaKind.uploaded_image == "uploaded_image"
    assert MediaKind.uploaded_video == "uploaded_video"
    assert MediaKind.generated_image == "generated_image"


def test_media_ref_construction_defaults_and_round_trip():
    m = MediaRef(kind=MediaKind.uploaded_image, ref="img_03.jpg")
    assert m.kind == MediaKind.uploaded_image
    assert m.ref == "img_03.jpg"
    assert m.source_prompt is None  # uploads carry no generation prompt

    restored = MediaRef.model_validate_json(m.model_dump_json())
    assert restored == m


def test_media_ref_generated_carries_source_prompt():
    m = MediaRef(
        kind=MediaKind.generated_image,
        ref="gen://abc",
        source_prompt="calm water, phone in a drawer",
    )
    assert m.source_prompt == "calm water, phone in a drawer"
    assert MediaRef.model_validate_json(m.model_dump_json()) == m


def test_media_kind_enum_rejects_bad_value():
    with pytest.raises(ValidationError):
        MediaRef(kind="gif_meme", ref="x")


def test_draft_media_defaults_none():
    draft = Draft(
        id="d3",
        platform=Platform.instagram,
        angle="relaxation",
        caption="Sink into stillness.",
        hashtags=[],
    )
    assert draft.media is None


def test_draft_with_media_round_trips():
    draft = Draft(
        id="d4",
        platform=Platform.instagram,
        angle="relaxation",
        caption="Sink into stillness.",
        hashtags=["#float"],
        media=MediaRef(kind=MediaKind.uploaded_image, ref="img_03.jpg"),
    )
    assert draft.media.ref == "img_03.jpg"
    restored = Draft.model_validate_json(draft.model_dump_json())
    assert restored == draft
    assert restored.media == draft.media
