"""Unit tests for the SQLite persistent store.

The store is pure stdlib ``sqlite3`` — no network, no LLM — so it runs under the
suite-wide offline guard untouched. Every test uses an in-memory database
(``Store(path=":memory:")``) or a ``tmp_path`` file, and any ``created_at`` /
``approved_at`` stamp is injected via a fixed ``now`` so assertions are
deterministic.
"""

from __future__ import annotations

import sqlite3

import pytest

from draftforge.models import Draft, MediaKind, MediaRef, Platform, Source
from draftforge.store import Store

FIXED_NOW = "2026-06-25T12:00:00Z"
LATER_NOW = "2026-06-26T09:30:00Z"


@pytest.fixture
def store() -> Store:
    """A fresh in-memory store for each test."""
    return Store(path=":memory:")


def _sample_draft(post_id: str = "d1", **overrides) -> Draft:
    base = dict(
        id=post_id,
        platform=Platform.instagram,
        angle="relaxation",
        caption="Sink into stillness.",
        hashtags=["#floattherapy", "#wellness"],
        image_direction="calm blue pool",
        claims_used=["many people feel relaxed"],
    )
    base.update(overrides)
    return Draft(**base)


# --------------------------------------------------------------------------- #
# Schema creation
# --------------------------------------------------------------------------- #


def test_creates_all_tables(store: Store):
    names = {
        row[0]
        for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"batches", "sources", "posts", "url_store"} <= names


def test_table_creation_is_idempotent(tmp_path):
    db = tmp_path / "store.db"
    first = Store(path=str(db))
    first.add_batch(
        "b1", guidance_prompt="g", url_set=["u"], batch_size=3, now=FIXED_NOW
    )
    first.close()

    # Re-opening the same file must not wipe data or error on re-creation.
    second = Store(path=str(db))
    batches = second.conn.execute("SELECT id FROM batches").fetchall()
    assert [r[0] for r in batches] == ["b1"]
    second.close()


# --------------------------------------------------------------------------- #
# Draft round-trip
# --------------------------------------------------------------------------- #


def test_draft_round_trip_save_get_list(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    draft = _sample_draft()
    store.save_draft(draft, "b1", now=FIXED_NOW)

    got = store.get_draft("d1", "b1")
    assert got == draft  # full Pydantic equality, json columns intact

    listed = store.list_drafts("b1")
    assert listed == [draft]


def test_get_draft_missing_returns_none(store: Store):
    assert store.get_draft("nope", "b1") is None


def test_save_draft_records_injected_created_at(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft(), "b1", now=FIXED_NOW)
    row = store.get_post_row("d1", "b1")
    assert row["created_at"] == FIXED_NOW
    assert row["claim_flags"] is None  # not set until the P2b gate runs


def test_draft_media_round_trips_through_store(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    media = MediaRef(kind=MediaKind.uploaded_image, ref="img_03.jpg")
    draft = _sample_draft("dm", media=media)
    store.save_draft(draft, "b1", now=FIXED_NOW)

    got = store.get_draft("dm", "b1")
    assert got.media == media
    assert got == draft  # full equality with media intact


def test_draft_without_media_round_trips_none(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft("dn"), "b1", now=FIXED_NOW)
    assert store.get_draft("dn", "b1").media is None


def test_set_media_swaps_and_clears(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft("d1"), "b1", now=FIXED_NOW)
    assert store.get_draft("d1", "b1").media is None

    ref = MediaRef(kind=MediaKind.uploaded_image, ref="x.jpg")
    store.set_media("d1", "b1", ref)
    assert store.get_draft("d1", "b1").media == ref

    # swap to a different upload
    ref2 = MediaRef(kind=MediaKind.uploaded_video, ref="y.mp4")
    store.set_media("d1", "b1", ref2)
    assert store.get_draft("d1", "b1").media == ref2

    # remove
    store.set_media("d1", "b1", None)
    assert store.get_draft("d1", "b1").media is None


def test_set_media_missing_post_raises(store: Store):
    with pytest.raises(KeyError):
        store.set_media("nope", "b1", None)


def test_list_drafts_scoped_to_batch(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=2, now=FIXED_NOW
    )
    store.add_batch(
        "b2", guidance_prompt="g", url_set=[], batch_size=2, now=FIXED_NOW
    )
    store.save_draft(_sample_draft("d1"), "b1", now=FIXED_NOW)
    store.save_draft(_sample_draft("d2"), "b2", now=FIXED_NOW)

    assert [d.id for d in store.list_drafts("b1")] == ["d1"]
    assert [d.id for d in store.list_drafts("b2")] == ["d2"]


# --------------------------------------------------------------------------- #
# Status lifecycle
# --------------------------------------------------------------------------- #


def test_full_status_lifecycle(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft(), "b1", now=FIXED_NOW)

    # draft -> edited (carries the edited text)
    store.update_status("d1", "b1", "edited", edited_text="Sink deeper.", now=LATER_NOW)
    d = store.get_draft("d1", "b1")
    assert d.status == "edited"
    assert d.edited_text == "Sink deeper."

    # edited -> approved (stamps approved_at)
    store.update_status("d1", "b1", "approved", now=LATER_NOW)
    row = store.get_post_row("d1", "b1")
    assert row["status"] == "approved"
    assert row["approved_at"] == LATER_NOW

    # approved -> scheduled (carries the scheduled date)
    store.update_status(
        "d1", "b1", "scheduled", scheduled_date="2026-07-01", now=LATER_NOW
    )
    d = store.get_draft("d1", "b1")
    assert d.status == "scheduled"
    assert d.scheduled_date == "2026-07-01"

    # scheduled -> exported
    store.update_status("d1", "b1", "exported", now=LATER_NOW)
    assert store.get_draft("d1", "b1").status == "exported"


def test_update_status_default_now_does_not_clobber_approved_at(store: Store):
    """approved_at is only stamped on the approved transition, not every update."""
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft(), "b1", now=FIXED_NOW)
    store.update_status("d1", "b1", "approved", now=FIXED_NOW)
    store.update_status(
        "d1", "b1", "scheduled", scheduled_date="2026-07-01", now=LATER_NOW
    )
    # approved_at stays at the approval moment, not the later schedule moment.
    assert store.get_post_row("d1", "b1")["approved_at"] == FIXED_NOW


def test_update_status_explicit_fields_override(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft(), "b1", now=FIXED_NOW)
    store.update_status(
        "d1",
        "b1",
        "needs_manual_review",
        claim_flags=["unsupported hard claim"],
        now=FIXED_NOW,
    )
    row = store.get_post_row("d1", "b1")
    assert row["status"] == "needs_manual_review"
    assert row["claim_flags"] == ["unsupported hard claim"]


def test_update_status_unknown_raises(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft(), "b1", now=FIXED_NOW)
    with pytest.raises(ValueError, match="status"):
        store.update_status("d1", "b1", "published", now=FIXED_NOW)


def test_update_status_missing_post_raises(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    with pytest.raises(KeyError):
        store.update_status("ghost", "b1", "approved", now=FIXED_NOW)


def test_valid_statuses_constant():
    assert Store.VALID_STATUSES == {
        "draft",
        "edited",
        "approved",
        "scheduled",
        "exported",
        "needs_manual_review",
    }


# --------------------------------------------------------------------------- #
# Claim flags
# --------------------------------------------------------------------------- #


def test_set_claim_flags(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft(), "b1", now=FIXED_NOW)
    store.set_claim_flags("d1", "b1", ["hard claim without source"])
    assert (
        store.get_post_row("d1", "b1")["claim_flags"]
        == ["hard claim without source"]
    )


# --------------------------------------------------------------------------- #
# Batch + sources round-trip
# --------------------------------------------------------------------------- #


def test_add_batch_round_trip(store: Store):
    batch_id = store.add_batch(
        "b1",
        guidance_prompt="make it calm",
        url_set=["https://a.example", "https://b.example"],
        batch_size=4,
        now=FIXED_NOW,
    )
    assert batch_id == "b1"
    row = store.get_batch("b1")
    assert row["guidance_prompt"] == "make it calm"
    assert row["url_set"] == ["https://a.example", "https://b.example"]
    assert row["batch_size"] == 4
    assert row["created_at"] == FIXED_NOW


def test_source_round_trip(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    source = Source(
        source_id="s1",
        type="url",
        title="A Title",
        text="some body text",
        fetched_at=FIXED_NOW,
    )
    store.save_source(source, "b1")

    sources = store.list_sources("b1")
    assert sources == [source]


def test_source_optional_title(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    source = Source(source_id="s1", type="file", text="body", fetched_at=FIXED_NOW)
    store.save_source(source, "b1")
    assert store.list_sources("b1")[0].title is None


def test_batch_with_sources_and_posts_round_trip(store: Store):
    store.add_batch(
        "b1", guidance_prompt="g", url_set=["u"], batch_size=2, now=FIXED_NOW
    )
    store.save_source(
        Source(source_id="s1", type="url", text="t", fetched_at=FIXED_NOW), "b1"
    )
    store.save_draft(_sample_draft("d1"), "b1", now=FIXED_NOW)
    store.save_draft(_sample_draft("d2", caption="Float on."), "b1", now=FIXED_NOW)

    assert [s.source_id for s in store.list_sources("b1")] == ["s1"]
    assert {d.id for d in store.list_drafts("b1")} == {"d1", "d2"}


# --------------------------------------------------------------------------- #
# url_store
# --------------------------------------------------------------------------- #


def test_url_store_add_list_deactivate(store: Store):
    store.upsert_url("https://a.example", label="Homepage")
    store.upsert_url("https://b.example")

    active = store.list_urls(active=True)
    assert {u["url"] for u in active} == {"https://a.example", "https://b.example"}
    homepage = next(u for u in active if u["url"] == "https://a.example")
    assert homepage["label"] == "Homepage"
    assert homepage["active"] is True

    store.set_url_active("https://a.example", False)
    active_now = store.list_urls(active=True)
    assert {u["url"] for u in active_now} == {"https://b.example"}

    # list_urls(active=None) returns everything regardless of active flag.
    every = store.list_urls(active=None)
    assert {u["url"] for u in every} == {"https://a.example", "https://b.example"}


def test_upsert_url_updates_existing(store: Store):
    store.upsert_url("https://a.example", label="Old", active=True)
    store.upsert_url("https://a.example", label="New", active=False)
    rows = store.list_urls(active=None)
    assert len(rows) == 1
    assert rows[0]["label"] == "New"
    assert rows[0]["active"] is False


def test_set_url_active_missing_raises(store: Store):
    with pytest.raises(KeyError):
        store.set_url_active("https://missing.example", True)


# --------------------------------------------------------------------------- #
# now injection forms
# --------------------------------------------------------------------------- #


def test_now_accepts_callable(store: Store):
    store.add_batch(
        "b1",
        guidance_prompt="g",
        url_set=[],
        batch_size=1,
        now=lambda: FIXED_NOW,
    )
    assert store.get_batch("b1")["created_at"] == FIXED_NOW


def test_default_now_is_real_iso_timestamp(store: Store):
    # No now injected: a real ISO-8601 string is stamped (offline-safe, no socket).
    store.add_batch("b1", guidance_prompt="g", url_set=[], batch_size=1)
    created = store.get_batch("b1")["created_at"]
    # Parseable as an ISO-8601 instant.
    from datetime import datetime

    datetime.fromisoformat(created)


# --------------------------------------------------------------------------- #
# DI-1: composite (batch_id, ...) keys — overlapping re-runs coexist
# --------------------------------------------------------------------------- #


def test_same_source_id_in_two_batches_coexists(store: Store):
    # A content-derived source_id recurs when an overlapping URL is re-run. With
    # the composite (batch_id, source_id) key, the two are SEPARATE rows — no
    # IntegrityError, and each batch sees only its own source.
    store.add_batch("b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW)
    store.add_batch("b2", guidance_prompt="g", url_set=[], batch_size=1, now=LATER_NOW)
    src = Source(source_id="shared", type="url", text="t1", fetched_at=FIXED_NOW)
    src2 = Source(source_id="shared", type="url", text="t2", fetched_at=LATER_NOW)

    store.save_source(src, "b1")
    store.save_source(src2, "b2")  # SAME source_id, different batch -> no crash

    assert [s.text for s in store.list_sources("b1")] == ["t1"]
    assert [s.text for s in store.list_sources("b2")] == ["t2"]


def test_same_draft_id_in_two_batches_coexists(store: Store):
    # The same content-derived draft id in two batches is two independent posts
    # (composite (batch_id, id) key). Editing one never touches the other.
    store.add_batch("b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW)
    store.add_batch("b2", guidance_prompt="g", url_set=[], batch_size=1, now=LATER_NOW)
    store.save_draft(_sample_draft("dup", caption="first"), "b1", now=FIXED_NOW)
    store.save_draft(_sample_draft("dup", caption="second"), "b2", now=LATER_NOW)

    assert store.get_draft("dup", "b1").caption == "first"
    assert store.get_draft("dup", "b2").caption == "second"


def test_prior_batch_post_unchanged_after_overlapping_rerun(store: Store):
    # A reviewer approves a draft in batch b1. A later overlapping run (b2)
    # re-creates the SAME draft id as a fresh draft. b1's approved post must be
    # untouched — re-runs never clobber prior human edits (the DI-1 guarantee).
    store.add_batch("b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW)
    store.save_draft(_sample_draft("dup"), "b1", now=FIXED_NOW)
    store.update_status("dup", "b1", "edited", edited_text="human edit", now=FIXED_NOW)
    store.update_status("dup", "b1", "approved", now=FIXED_NOW)

    # Later overlapping run: same draft id lands in a NEW batch as a fresh draft.
    store.add_batch("b2", guidance_prompt="g", url_set=[], batch_size=1, now=LATER_NOW)
    store.save_draft(_sample_draft("dup", caption="regenerated"), "b2", now=LATER_NOW)

    prior = store.get_post_row("dup", "b1")
    assert prior["status"] == "approved"
    assert prior["edited_text"] == "human edit"
    assert prior["approved_at"] == FIXED_NOW
    # The new batch's post is an independent, un-approved draft.
    fresh = store.get_post_row("dup", "b2")
    assert fresh["status"] == "draft"
    assert fresh["caption"] == "regenerated"
    assert fresh["approved_at"] is None


# --------------------------------------------------------------------------- #
# DI-1: atomic persist via Store.transaction()
# --------------------------------------------------------------------------- #


def test_transaction_commits_all_writes_atomically(store: Store):
    with store.transaction():
        store.add_batch("b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW)
        store.save_source(
            Source(source_id="s1", type="url", text="t", fetched_at=FIXED_NOW), "b1"
        )
        store.save_draft(_sample_draft("d1"), "b1", now=FIXED_NOW)

    # After the block all writes are durable in one unit.
    assert store.get_batch("b1") is not None
    assert [s.source_id for s in store.list_sources("b1")] == ["s1"]
    assert [d.id for d in store.list_drafts("b1")] == ["d1"]


def test_transaction_rolls_back_on_error_leaving_no_orphan_batch(store: Store):
    # A failure mid-persist must roll the WHOLE block back: no batch row, no
    # source, no post survives. This is the orphan-partial-batch guard (DI-1).
    boom = RuntimeError("forced mid-persist failure")
    with pytest.raises(RuntimeError, match="forced mid-persist"):
        with store.transaction():
            store.add_batch(
                "b1", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
            )
            store.save_source(
                Source(source_id="s1", type="url", text="t", fetched_at=FIXED_NOW), "b1"
            )
            store.save_draft(_sample_draft("d1"), "b1", now=FIXED_NOW)
            raise boom  # something blows up before the block completes

    # NOTHING was committed — the batch row that add_batch wrote is gone.
    assert store.get_batch("b1") is None
    assert store.list_sources("b1") == []
    assert store.list_drafts("b1") == []
    # And the connection is usable again (rollback did not wedge it).
    store.add_batch("b2", guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW)
    assert store.get_batch("b2") is not None


def test_transaction_is_not_reentrant(store: Store):
    with pytest.raises(RuntimeError, match="re-entrant"):
        with store.transaction():
            with store.transaction():
                pass


# --------------------------------------------------------------------------- #
# DI-3: status-transition whitelist
# --------------------------------------------------------------------------- #


def _seed_draft(store: Store, batch="b1", post="d1") -> None:
    store.add_batch(
        batch, guidance_prompt="g", url_set=[], batch_size=1, now=FIXED_NOW
    )
    store.save_draft(_sample_draft(post), batch, now=FIXED_NOW)


@pytest.mark.parametrize(
    "path",
    [
        ["edited", "approved", "scheduled", "exported"],  # full happy path
        ["approved", "exported"],  # approved -> exported directly
        ["edited", "draft"],  # re-edit back to draft
        ["needs_manual_review", "approved"],  # gate routed, then approved
        ["approved", "edited"],  # reject an approval, re-edit
    ],
)
def test_legal_status_transitions(store: Store, path):
    _seed_draft(store)
    for target in path:
        store.update_status("d1", "b1", target, now=FIXED_NOW)
        assert store.get_draft("d1", "b1").status == target


def test_exported_is_terminal(store: Store):
    _seed_draft(store)
    store.update_status("d1", "b1", "approved", now=FIXED_NOW)
    store.update_status("d1", "b1", "exported", now=FIXED_NOW)
    # Nothing leaves exported — not even back to scheduled.
    with pytest.raises(ValueError, match="illegal status transition"):
        store.update_status("d1", "b1", "scheduled", now=FIXED_NOW)


def test_exported_unreachable_without_passing_approved(store: Store):
    # The human gate: a draft cannot jump straight to exported (or scheduled).
    _seed_draft(store)
    with pytest.raises(ValueError, match="illegal status transition"):
        store.update_status("d1", "b1", "exported", now=FIXED_NOW)


def test_scheduled_unreachable_from_draft(store: Store):
    _seed_draft(store)
    with pytest.raises(ValueError, match="illegal status transition"):
        store.update_status("d1", "b1", "scheduled", now=FIXED_NOW)


def test_no_op_same_status_is_allowed(store: Store):
    # status -> same status is always legal (idempotent side-field re-write).
    _seed_draft(store)
    store.update_status("d1", "b1", "approved", now=FIXED_NOW)
    store.update_status("d1", "b1", "approved", now=LATER_NOW)  # no-op, no raise
    assert store.get_draft("d1", "b1").status == "approved"


# --------------------------------------------------------------------------- #
# DI-4: stale approved_at is cleared on a backward move
# --------------------------------------------------------------------------- #


def test_approved_at_cleared_when_moving_back_to_edited(store: Store):
    _seed_draft(store)
    store.update_status("d1", "b1", "approved", now=FIXED_NOW)
    assert store.get_post_row("d1", "b1")["approved_at"] == FIXED_NOW

    # Move back out of the approved chain -> the stale stamp is nulled.
    store.update_status("d1", "b1", "edited", now=LATER_NOW)
    assert store.get_post_row("d1", "b1")["approved_at"] is None


def test_approved_at_cleared_from_scheduled_back_to_edited(store: Store):
    _seed_draft(store)
    store.update_status("d1", "b1", "approved", now=FIXED_NOW)
    store.update_status("d1", "b1", "scheduled", scheduled_date="2026-07-01", now=FIXED_NOW)
    assert store.get_post_row("d1", "b1")["approved_at"] == FIXED_NOW

    store.update_status("d1", "b1", "edited", now=LATER_NOW)
    assert store.get_post_row("d1", "b1")["approved_at"] is None


def test_explicit_approved_at_not_cleared_on_backward_move(store: Store):
    # If the caller passes an explicit approved_at, it is honored even on a move
    # out of the approved chain (the auto-clear only fires when not overridden).
    _seed_draft(store)
    store.update_status("d1", "b1", "approved", now=FIXED_NOW)
    store.update_status("d1", "b1", "edited", approved_at=LATER_NOW, now=LATER_NOW)
    assert store.get_post_row("d1", "b1")["approved_at"] == LATER_NOW


# --------------------------------------------------------------------------- #
# DI-5 / P2-1: claim_flags round-trips the real ClaimCheck.model_dump() dict
# --------------------------------------------------------------------------- #


def test_set_claim_flags_round_trips_claimcheck_dict(store: Store):
    # The ONLY real caller stores a ClaimCheck.model_dump() dict, not a list. The
    # store must persist and return that exact dict shape (status/notes/revised).
    from draftforge.models import ClaimCheck

    _seed_draft(store)
    check = ClaimCheck(
        status="softened",
        notes=["softened a hard claim", "see register"],
        revised_text="Floating may help you relax.",
    )
    store.set_claim_flags("d1", "b1", check.model_dump())

    flags = store.get_post_row("d1", "b1")["claim_flags"]
    assert flags == check.model_dump()
    assert flags["status"] == "softened"
    assert flags["notes"] == ["softened a hard claim", "see register"]
    assert flags["revised_text"] == "Floating may help you relax."


def test_update_status_round_trips_claimcheck_dict(store: Store):
    # update_status(claim_flags=...) accepts the same dict shape end-to-end.
    from draftforge.models import ClaimCheck

    _seed_draft(store)
    check = ClaimCheck(status="flagged", notes=["disease claim"], revised_text=None)
    store.update_status(
        "d1", "b1", "needs_manual_review", claim_flags=check.model_dump(), now=FIXED_NOW
    )
    assert store.get_post_row("d1", "b1")["claim_flags"] == check.model_dump()
