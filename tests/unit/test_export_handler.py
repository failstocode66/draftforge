"""Unit tests for the Calendar/Export handlers + auth (Task 3.4).

``handle_schedule`` assigns dates to the batch's APPROVED drafts (moving them to
``scheduled``); ``handle_export`` writes the approved/scheduled drafts to a file
in ``exports/`` and returns its path. Only signed-off (approved/scheduled/
exported) drafts reach the calendar — drafts/edited are excluded. ``_auth``
builds the Gradio login tuple from ``APP_PASSWORD``.
"""

from __future__ import annotations

import csv
import io
import os

import pytest

from draftforge import app
from draftforge.models import Draft, Platform
from draftforge.store.db import Store

NOW = "2026-06-26T12:00:00Z"


@pytest.fixture
def store():
    return Store(":memory:")


def _seed_approved(store, n=2, batch="b1"):
    """Seed ``n`` APPROVED drafts + one unapproved (draft) that must be excluded."""
    store.add_batch(batch, guidance_prompt="g", url_set=[], batch_size=n, now=NOW)
    for i in range(n):
        store.save_draft(
            Draft(id=f"d{i}", platform=Platform.instagram, angle="relaxation",
                  caption=f"Caption {i}", hashtags=["#f"], status="approved"),
            batch, now=NOW,
        )
    store.save_draft(
        Draft(id="draftonly", platform=Platform.facebook, angle="a",
              caption="not approved", hashtags=[], status="draft"),
        batch, now=NOW,
    )


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# --- handle_schedule ------------------------------------------------------------


def test_handle_schedule_assigns_dates_and_moves_to_scheduled(store):
    _seed_approved(store, 3)
    out = app.handle_schedule("b1", store=store, start_date="2026-07-01", per_week=3, now=NOW)
    assert [d.scheduled_date for d in out] == ["2026-07-01", "2026-07-03", "2026-07-05"]
    assert all(d.status == "scheduled" for d in out)
    # the unapproved draft is untouched
    assert store.get_draft("draftonly", "b1").status == "draft"
    assert store.get_draft("draftonly", "b1").scheduled_date is None


# --- handle_export --------------------------------------------------------------


def test_handle_export_writes_csv_with_only_signed_off_drafts(store, tmp_path):
    _seed_approved(store, 2)
    app.handle_schedule("b1", store=store, start_date="2026-07-01", now=NOW)
    path = app.handle_export("b1", "csv", store=store, exports_dir=str(tmp_path / "exports"))
    assert os.path.isfile(path)
    rows = list(csv.DictReader(io.StringIO(_read(path))))
    assert len(rows) == 2  # draft-only excluded
    assert rows[0]["date"] == "2026-07-01"


def test_handle_export_each_format_writes_nonempty_file(store, tmp_path):
    _seed_approved(store, 1)
    app.handle_schedule("b1", store=store, start_date="2026-07-01", now=NOW)
    for fmt in ("md", "csv", "ics"):
        p = app.handle_export("b1", fmt, store=store, exports_dir=str(tmp_path))
        assert p.endswith(f".{fmt}")
        assert os.path.getsize(p) > 0


def test_handle_export_excludes_unapproved(store, tmp_path):
    _seed_approved(store, 1)  # 1 approved + 1 draft-only
    p = app.handle_export("b1", "csv", store=store, exports_dir=str(tmp_path))
    rows = list(csv.DictReader(io.StringIO(_read(p))))
    assert len(rows) == 1


def test_handle_export_bad_format_raises(store, tmp_path):
    _seed_approved(store, 1)
    with pytest.raises(ValueError):
        app.handle_export("b1", "pdf", store=store, exports_dir=str(tmp_path))


# --- auth -----------------------------------------------------------------------


def test_auth_builds_login_tuple_from_password():
    assert app._auth("s3cret") == ("draftforge", "s3cret")
