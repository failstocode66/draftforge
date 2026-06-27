"""Unit tests for content-calendar date assignment + export (Task 3.1).

``assign_dates`` spaces approved drafts across the calendar deterministically
(``per_week`` posts per 7-day window, default 3 ≈ every ~2.3 days, matching
the studio's 1–3-day burst cadence — the owner re-dates bursts in review). ``export_calendar``
renders the scheduled drafts to ``md`` / ``csv`` / ``ics``.
"""

from __future__ import annotations

import csv
import datetime
import io

import pytest

from draftforge.models import Draft, MediaKind, MediaRef, Platform
from draftforge.output.calendar import assign_dates, export_calendar


def _draft(post_id: str, **overrides) -> Draft:
    base = dict(
        id=post_id,
        platform=Platform.instagram,
        angle="relaxation",
        caption="Sink into stillness.",
        hashtags=["#float"],
    )
    base.update(overrides)
    return Draft(**base)


def _dates(drafts):
    return [d.scheduled_date for d in drafts]


# --- assign_dates ---------------------------------------------------------------


def test_assign_dates_default_per_week_spacing():
    drafts = [_draft(f"d{i}") for i in range(7)]
    out = assign_dates(drafts, start_date="2026-07-01")
    # per_week=3 -> offsets floor(i*7/3): 0,2,4,7,9,11,14
    assert _dates(out) == [
        "2026-07-01", "2026-07-03", "2026-07-05", "2026-07-08",
        "2026-07-10", "2026-07-12", "2026-07-15",
    ]


def test_assign_dates_one_per_week():
    out = assign_dates([_draft(f"d{i}") for i in range(3)], start_date="2026-07-01", per_week=1)
    assert _dates(out) == ["2026-07-01", "2026-07-08", "2026-07-15"]


def test_assign_dates_daily_when_per_week_seven():
    out = assign_dates([_draft(f"d{i}") for i in range(3)], start_date="2026-07-01", per_week=7)
    assert _dates(out) == ["2026-07-01", "2026-07-02", "2026-07-03"]


def test_assign_dates_accepts_date_object():
    out = assign_dates([_draft("d0")], start_date=datetime.date(2026, 7, 1))
    assert out[0].scheduled_date == "2026-07-01"


def test_assign_dates_is_pure_inputs_not_mutated():
    drafts = [_draft("d0")]
    assign_dates(drafts, start_date="2026-07-01")
    assert drafts[0].scheduled_date is None


def test_assign_dates_rejects_non_positive_per_week():
    with pytest.raises(ValueError):
        assign_dates([_draft("d0")], start_date="2026-07-01", per_week=0)


def test_assign_dates_empty_list():
    assert assign_dates([], start_date="2026-07-01") == []


# --- export_calendar ------------------------------------------------------------


def _scheduled():
    return assign_dates(
        [
            _draft("d0", caption="Float on."),
            _draft("d1", platform=Platform.facebook, caption="Recover faster."),
        ],
        start_date="2026-07-01",
    )


def test_export_markdown_has_header_and_rows():
    md = export_calendar(_scheduled(), "md")
    assert "Date" in md and "Platform" in md and "Caption" in md
    assert "2026-07-01" in md
    assert "Float on." in md
    assert "facebook" in md


def test_export_markdown_sanitizes_pipes_and_newlines():
    drafts = assign_dates([_draft("d0", caption="a | b\nc")], start_date="2026-07-01")
    md = export_calendar(drafts, "md")
    # A raw pipe/newline in a caption must not break the table row.
    row_lines = [ln for ln in md.splitlines() if "2026-07-01" in ln]
    assert len(row_lines) == 1
    assert "a" in row_lines[0] and "b" in row_lines[0] and "c" in row_lines[0]


def test_export_csv_parses_with_header_and_one_row_per_draft():
    out = export_calendar(_scheduled(), "csv")
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 2
    assert {"date", "platform", "angle", "caption", "hashtags", "media"} <= set(rows[0])
    assert rows[0]["date"] == "2026-07-01"
    assert rows[0]["caption"] == "Float on."


def test_export_csv_includes_media_ref():
    drafts = assign_dates(
        [_draft("d0", media=MediaRef(kind=MediaKind.uploaded_image, ref="a.jpg"))],
        start_date="2026-07-01",
    )
    rows = list(csv.DictReader(io.StringIO(export_calendar(drafts, "csv"))))
    assert rows[0]["media"] == "a.jpg"


def test_export_ics_is_a_valid_vcalendar_with_one_event_per_draft():
    ics = export_calendar(_scheduled(), "ics")
    assert ics.startswith("BEGIN:VCALENDAR")
    assert ics.strip().endswith("END:VCALENDAR")
    assert ics.count("BEGIN:VEVENT") == 2
    assert "DTSTART;VALUE=DATE:20260701" in ics
    assert "SUMMARY:" in ics


def test_export_ics_escapes_description():
    drafts = assign_dates([_draft("d0", caption="rest, relax; breathe")], start_date="2026-07-01")
    ics = export_calendar(drafts, "ics")
    # ICS requires comma/semicolon escaping in TEXT values.
    assert "rest\\, relax\\; breathe" in ics


def test_export_rejects_unknown_format():
    with pytest.raises(ValueError):
        export_calendar(_scheduled(), "pdf")
