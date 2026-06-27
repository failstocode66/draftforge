"""Content-calendar date assignment + export (Task 3.1).

``assign_dates`` spaces drafts across the calendar deterministically: ``per_week``
posts per 7-day window (default 3 ≈ every ~2.3 days, matching the studio's 1–3-day
burst cadence — bursts are re-dated in the review UI). It is pure (returns new
drafts via ``model_copy``, never mutating inputs).

``export_calendar`` renders the scheduled drafts to one of three formats — the
human-readable ``md`` table, a spreadsheet-friendly ``csv``, and an ``ics``
iCalendar feed that drops straight into Google/Apple Calendar.
"""

from __future__ import annotations

import csv
import datetime
import io

from draftforge.models import Draft

_FORMATS: frozenset[str] = frozenset({"md", "csv", "ics"})


def _resolve_start(start_date: str | datetime.date) -> datetime.date:
    if isinstance(start_date, datetime.date):
        return start_date
    return datetime.date.fromisoformat(start_date)


def assign_dates(
    drafts: list[Draft],
    *,
    start_date: str | datetime.date,
    per_week: int = 3,
) -> list[Draft]:
    """Assign each draft a ``scheduled_date``, spaced ``per_week`` per 7 days.

    Draft ``i`` is dated ``start_date + floor(i * 7 / per_week)`` days, so the
    posts fan out evenly (e.g. ``per_week=3`` → offsets 0, 2, 4, 7, 9, 11, 14…).
    Returns NEW drafts; inputs are never mutated.

    Args:
        drafts: the drafts to schedule, in calendar order.
        start_date: the first post's date (an ISO ``YYYY-MM-DD`` string or a
            :class:`datetime.date`).
        per_week: posts per 7-day window (default 3). Must be >= 1.

    Raises:
        ValueError: if ``per_week`` < 1.
    """
    if per_week < 1:
        raise ValueError(f"per_week must be >= 1, got {per_week}")

    base = _resolve_start(start_date)
    scheduled: list[Draft] = []
    for index, draft in enumerate(drafts):
        offset_days = (index * 7) // per_week
        day = base + datetime.timedelta(days=offset_days)
        scheduled.append(draft.model_copy(update={"scheduled_date": day.isoformat()}))
    return scheduled


def export_calendar(drafts: list[Draft], fmt: str) -> str:
    """Render scheduled ``drafts`` to ``md`` | ``csv`` | ``ics``.

    Raises:
        ValueError: if ``fmt`` is not a supported format.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; supported: {sorted(_FORMATS)}")
    if fmt == "md":
        return _to_markdown(drafts)
    if fmt == "csv":
        return _to_csv(drafts)
    return _to_ics(drafts)


def _cell(text: str | None) -> str:
    """Flatten a value for a single-line Markdown table cell."""
    return (text or "").replace("\\", "/").replace("|", "\\|").replace("\n", " ").strip()


def _to_markdown(drafts: list[Draft]) -> str:
    lines = [
        "# Content Calendar",
        "",
        "| Date | Platform | Angle | Caption |",
        "| --- | --- | --- | --- |",
    ]
    for d in drafts:
        lines.append(
            f"| {_cell(d.scheduled_date)} | {_cell(str(d.platform))} "
            f"| {_cell(d.angle)} | {_cell(d.caption)} |"
        )
    return "\n".join(lines) + "\n"


def _to_csv(drafts: list[Draft]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["date", "platform", "angle", "caption", "hashtags", "status", "media"]
    )
    for d in drafts:
        writer.writerow(
            [
                d.scheduled_date or "",
                str(d.platform),
                d.angle,
                d.caption,
                " ".join(d.hashtags),
                d.status,
                d.media.ref if d.media is not None else "",
            ]
        )
    return buf.getvalue()


def _ics_escape(text: str) -> str:
    """Escape a value for an iCalendar TEXT field (RFC 5545)."""
    return (
        text.replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("\n", "\\n")
    )


def _to_ics(drafts: list[Draft]) -> str:
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//DraftForge//Content Calendar//EN",
    ]
    for d in drafts:
        if not d.scheduled_date:
            # An all-day VEVENT needs a date; an undated draft has no calendar
            # slot yet, so skip it rather than emit an invalid event.
            continue
        date_compact = d.scheduled_date.replace("-", "")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{d.id}@draftforge-content-studio",
            f"DTSTART;VALUE=DATE:{date_compact}",
            f"SUMMARY:{_ics_escape(f'{d.platform} — {d.angle}')}",
            f"DESCRIPTION:{_ics_escape(d.caption)}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\n".join(lines) + "\n"
