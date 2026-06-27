"""The SQLite persistent store backing the review-gated draft lifecycle.

A :class:`Store` owns a single ``sqlite3`` connection (in-memory for tests, a
file in production) and four tables:

* ``batches`` — one row per pipeline run (the guidance prompt, the URL set it
  ran against, the requested batch size).
* ``sources`` — the normalized ingested material for a batch.
* ``posts`` — the generated :class:`~draftforge.models.Draft`s and their status
  lifecycle (``draft -> edited -> approved -> scheduled -> exported``, plus the
  ``needs_manual_review`` side state the claims gate uses).
* ``url_store`` — the standing set of source URLs the UI offers, each with an
  ``active`` flag so a URL can be retired without losing its label.

The store is pure stdlib: no network, no LLM, no ORM. List-shaped columns
(``hashtags``, ``claims_used``, ``url_set``, ``claim_flags``) are JSON-encoded
text. Every method that stamps a timestamp (``created_at`` / ``approved_at``)
takes an injectable ``now`` — a callable or a literal ISO-8601 string —
defaulting to real UTC time in production so tests can pin it and assert exact
values.

The claims *register* (the audit log of which claims were used) lives in
``data/claims_register.json`` by deliberate decision and is **not** a table
here; ``posts.claim_flags`` is the only claims metadata the store persists, set
by the P2b gate via :meth:`Store.set_claim_flags` / :meth:`Store.update_status`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TypedDict

from draftforge.models import Draft, MediaRef, Platform, Source

# The closed set of statuses a post may hold. ``draft -> edited -> approved ->
# scheduled -> exported`` is the happy-path lifecycle; ``needs_manual_review`` is
# the side state the claims-safety gate routes a post into when it can neither
# pass nor auto-soften a claim. ``update_status`` rejects anything outside this
# set so a typo can never silently persist an unrenderable status.
VALID_STATUSES: frozenset[str] = frozenset(
    {"draft", "edited", "approved", "scheduled", "exported", "needs_manual_review"}
)

# The legal status-transition map (DI-3). Each key is a *current* status; its
# value is the set of statuses a post in that status may move to. This enforces
# the human review gate: ``exported`` is only reachable through ``approved`` (the
# human sign-off), never directly from ``draft``/``edited``. The lifecycle is:
#
#     draft -> edited -> approved -> scheduled -> exported
#
# with these additional rules:
#   * ``needs_manual_review`` is reachable from any non-terminal status (the
#     claims gate can route a post there at any point before export);
#   * a post can be re-edited — moved back to ``edited`` or ``draft`` — from any
#     non-terminal status (e.g. a reviewer rejects an approval and re-edits);
#   * ``exported`` is TERMINAL: nothing leaves it (an exported post is published
#     downstream and must not silently change state);
#   * a no-op (status -> same status) is always allowed so an idempotent re-write
#     of side fields never trips the gate.
#
# The approved -> {scheduled, exported} edges (and scheduled -> exported) are the
# only paths into the post-approval chain, so ``exported`` cannot be reached
# without passing ``approved``.
_NON_TERMINAL = frozenset(
    {"draft", "edited", "approved", "scheduled", "needs_manual_review"}
)
STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"edited", "approved", "needs_manual_review"}),
    "edited": frozenset({"draft", "edited", "approved", "needs_manual_review"}),
    "approved": frozenset(
        {"draft", "edited", "scheduled", "exported", "needs_manual_review"}
    ),
    "scheduled": frozenset(
        {"draft", "edited", "approved", "exported", "needs_manual_review"}
    ),
    "needs_manual_review": frozenset({"draft", "edited", "approved"}),
    "exported": frozenset(),  # terminal: no outgoing transitions
}

# The statuses that constitute the "approved chain" — once a post is in any of
# these it has a non-null ``approved_at``. Moving OUT of this chain to a status
# not in it must clear the stale stamp (DI-4).
_APPROVED_CHAIN = frozenset({"approved", "scheduled", "exported"})


class ClaimFlags(TypedDict):
    """The real shape persisted into ``posts.claim_flags`` (DI-5 / P2-1).

    This mirrors :meth:`draftforge.models.ClaimCheck.model_dump` — the only thing
    the production caller (``cli._persist``) ever stores — so the annotation no
    longer lies about being a ``list[str]``.
    """

    status: str  # clean | softened | flagged | needs_manual_review
    notes: list[str]
    revised_text: str | None


# The accepted persisted shapes for a post's claim flags. The production caller
# always writes a :class:`ClaimFlags` dict (``ClaimCheck.model_dump()``); the
# bare ``list[str]`` form is retained for legacy/manual callers and tests.
ClaimFlagsArg = ClaimFlags | dict[str, object] | list[str]

NowArg = Callable[[], str] | str | None


def _resolve_now(now: NowArg) -> str:
    """Resolve an injected ``now`` to an ISO-8601 string.

    Accepts a literal ISO string, a zero-arg callable returning one, or ``None``
    (real UTC time). This keeps every stamping method deterministic under test
    while defaulting to wall-clock time in production.
    """
    if now is None:
        return datetime.now(timezone.utc).isoformat()
    if callable(now):
        return now()
    return now


def _dumps(value) -> str:
    """JSON-encode a list/dict column with stable, ASCII-safe output."""
    return json.dumps(value, ensure_ascii=False)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id              TEXT PRIMARY KEY,
    guidance_prompt TEXT,
    url_set         TEXT,
    batch_size      INTEGER,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    source_id  TEXT,
    batch_id   TEXT,
    type       TEXT,
    uri        TEXT,
    title      TEXT,
    text       TEXT,
    fetched_at TEXT,
    PRIMARY KEY (batch_id, source_id),
    FOREIGN KEY (batch_id) REFERENCES batches (id)
);

CREATE TABLE IF NOT EXISTS posts (
    id              TEXT,
    batch_id        TEXT,
    platform        TEXT,
    angle           TEXT,
    caption         TEXT,
    hashtags        TEXT,
    image_direction TEXT,
    claims_used     TEXT,
    claim_flags     TEXT,
    media           TEXT,
    status          TEXT,
    scheduled_date  TEXT,
    edited_text     TEXT,
    approved_at     TEXT,
    created_at      TEXT,
    PRIMARY KEY (batch_id, id),
    FOREIGN KEY (batch_id) REFERENCES batches (id)
);

CREATE TABLE IF NOT EXISTS url_store (
    url    TEXT PRIMARY KEY,
    label  TEXT,
    active INTEGER
);
"""


class Store:
    """A SQLite-backed persistent store for batches, sources, drafts, and URLs.

    Args:
        path: SQLite database path. ``":memory:"`` (the default) gives an
            ephemeral in-memory database — used throughout the unit suite.
    """

    VALID_STATUSES: frozenset[str] = VALID_STATUSES

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # When True, the per-method ``_commit`` calls no-op so a multi-write
        # :meth:`transaction` stays atomic (one BEGIN ... COMMIT/ROLLBACK).
        self._in_transaction = False
        self._create_tables()

    def _create_tables(self) -> None:
        """Create all four tables if they do not already exist (idempotent)."""
        self.conn.executescript(_SCHEMA)
        # Idempotent column migration: a database file created before the
        # ``media`` column existed (CREATE TABLE IF NOT EXISTS won't add it) gains
        # it here, so an older dev store keeps working without a manual rebuild.
        self._ensure_column("posts", "media", "TEXT")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, decl: str) -> None:
        """Add ``column`` to ``table`` if it is not already present (idempotent)."""
        existing = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _commit(self) -> None:
        """Commit, unless inside a :meth:`transaction` block.

        Every mutator self-commits via this helper. While a :meth:`transaction`
        is open, these calls are suppressed so the whole block commits (or rolls
        back) as one unit — the atomic-persist requirement (DI-1).
        """
        if not self._in_transaction:
            self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[Store]:
        """Run a block of store mutations as ONE atomic transaction (DI-1).

        Inside the block the per-method self-commits are suppressed, so the
        batch + every source + every post + every claim-flag write commit
        together on a clean exit. If ANY statement raises, the whole block is
        rolled back, leaving NO partial/orphan rows behind.

        Re-entrancy is not supported (the pipeline persists one batch at a time);
        a nested call raises :class:`RuntimeError`.

        Example::

            with store.transaction():
                store.add_batch(...)
                for s in sources:
                    store.save_source(s, batch_id)
                ...
        """
        if self._in_transaction:
            raise RuntimeError("Store.transaction() is not re-entrant")
        # Flush any pending autocommitted state, then open an explicit BEGIN so
        # the rollback below can unwind every write made inside the block.
        self.conn.commit()
        self.conn.execute("BEGIN")
        self._in_transaction = True
        try:
            yield self
        except BaseException:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()
        finally:
            self._in_transaction = False

    def close(self) -> None:
        """Close the underlying connection."""
        self.conn.close()

    # ------------------------------------------------------------------ #
    # batches
    # ------------------------------------------------------------------ #

    def add_batch(
        self,
        batch_id: str,
        *,
        guidance_prompt: str,
        url_set: list[str],
        batch_size: int,
        now: NowArg = None,
    ) -> str:
        """Insert a batch row and return its id.

        ``url_set`` is JSON-encoded; ``created_at`` is stamped from ``now``.
        """
        self.conn.execute(
            "INSERT INTO batches (id, guidance_prompt, url_set, batch_size, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (batch_id, guidance_prompt, _dumps(url_set), batch_size, _resolve_now(now)),
        )
        self._commit()
        return batch_id

    def get_batch(self, batch_id: str) -> dict | None:
        """Return a batch row as a dict (``url_set`` decoded), or ``None``."""
        row = self.conn.execute(
            "SELECT * FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["url_set"] = json.loads(data["url_set"]) if data["url_set"] else []
        return data

    # ------------------------------------------------------------------ #
    # sources
    # ------------------------------------------------------------------ #

    def save_source(self, source: Source, batch_id: str) -> None:
        """Persist a :class:`Source` under a batch.

        The model has no ``uri`` field of its own (its locator lives in
        ``source_id`` / ``title``), so ``uri`` is stored as ``NULL`` for now;
        the column exists for the ingest layer to populate later.
        """
        self.conn.execute(
            "INSERT INTO sources (source_id, batch_id, type, uri, title, text, "
            "fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source.source_id,
                batch_id,
                source.type,
                None,
                source.title,
                source.text,
                source.fetched_at,
            ),
        )
        self._commit()

    def list_sources(self, batch_id: str) -> list[Source]:
        """Return all sources for a batch as :class:`Source` models."""
        rows = self.conn.execute(
            "SELECT * FROM sources WHERE batch_id = ? ORDER BY source_id",
            (batch_id,),
        ).fetchall()
        return [
            Source(
                source_id=row["source_id"],
                type=row["type"],
                title=row["title"],
                text=row["text"],
                fetched_at=row["fetched_at"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # posts / drafts
    # ------------------------------------------------------------------ #

    def save_draft(self, draft: Draft, batch_id: str, *, now: NowArg = None) -> None:
        """Persist a :class:`Draft` under a batch.

        ``hashtags`` and ``claims_used`` are JSON-encoded; ``status`` defaults to
        the draft's own status; ``created_at`` is stamped from ``now``.
        ``media`` is JSON-encoded from the draft's :class:`~draftforge.models.MediaRef`
        (``NULL`` when the draft has no paired media). ``claim_flags`` and
        ``approved_at`` start ``NULL`` — they are populated later by the claims
        gate and the approve transition respectively.
        """
        self.conn.execute(
            "INSERT INTO posts (id, batch_id, platform, angle, caption, hashtags, "
            "image_direction, claims_used, claim_flags, media, status, "
            "scheduled_date, edited_text, approved_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                draft.id,
                batch_id,
                str(draft.platform),
                draft.angle,
                draft.caption,
                _dumps(draft.hashtags),
                draft.image_direction,
                _dumps(draft.claims_used),
                None,
                _dumps(draft.media.model_dump(mode="json"))
                if draft.media is not None
                else None,
                draft.status,
                draft.scheduled_date,
                draft.edited_text,
                None,
                _resolve_now(now),
            ),
        )
        self._commit()

    def get_post_row(self, post_id: str, batch_id: str) -> dict | None:
        """Return the full post row as a dict, including store-only metadata.

        A post is identified by ``(batch_id, post_id)`` — the composite key
        (DI-1). The same content-derived draft id can recur in a *different*
        batch (a re-run of an overlapping source), so the batch must be supplied
        to disambiguate; a bare ``post_id`` would be ambiguous across runs.

        Unlike :meth:`get_draft`, this exposes ``claim_flags``, ``approved_at``,
        ``created_at``, and ``batch_id`` — fields that are not on the
        :class:`Draft` model. JSON columns are decoded.
        """
        row = self.conn.execute(
            "SELECT * FROM posts WHERE batch_id = ? AND id = ?", (batch_id, post_id)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["hashtags"] = json.loads(data["hashtags"]) if data["hashtags"] else []
        data["claims_used"] = (
            json.loads(data["claims_used"]) if data["claims_used"] else []
        )
        data["claim_flags"] = (
            json.loads(data["claim_flags"]) if data["claim_flags"] is not None else None
        )
        data["media"] = (
            json.loads(data["media"]) if data["media"] is not None else None
        )
        return data

    def get_draft(self, post_id: str, batch_id: str) -> Draft | None:
        """Reconstruct a :class:`Draft` from its post row, or ``None``.

        Scoped to ``(batch_id, post_id)`` — the composite key (DI-1) — so an
        overlapping draft id in another batch is a separate row.

        ``claim_flags`` / ``approved_at`` are store-only metadata and are *not*
        part of the returned model; use :meth:`get_post_row` for those.
        """
        data = self.get_post_row(post_id, batch_id)
        if data is None:
            return None
        return Draft(
            id=data["id"],
            platform=Platform(data["platform"]),
            angle=data["angle"],
            caption=data["caption"],
            hashtags=data["hashtags"],
            image_direction=data["image_direction"],
            claims_used=data["claims_used"],
            status=data["status"],
            scheduled_date=data["scheduled_date"],
            edited_text=data["edited_text"],
            media=data["media"],
        )

    def list_drafts(self, batch_id: str) -> list[Draft]:
        """Return all drafts for a batch as :class:`Draft` models."""
        ids = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM posts WHERE batch_id = ? ORDER BY created_at, id",
                (batch_id,),
            ).fetchall()
        ]
        return [self.get_draft(pid, batch_id) for pid in ids]  # type: ignore[misc]

    def update_status(
        self,
        post_id: str,
        batch_id: str,
        status: str,
        *,
        edited_text: str | None = None,
        scheduled_date: str | None = None,
        claim_flags: ClaimFlagsArg | None = None,
        approved_at: str | None = None,
        now: NowArg = None,
    ) -> None:
        """Move a post to a new status, updating any provided side fields.

        The post is identified by the composite ``(batch_id, post_id)`` key
        (DI-1). The new ``status`` must be a *legal* transition from the post's
        CURRENT status per :data:`STATUS_TRANSITIONS` (DI-3) — this enforces the
        human review gate (``exported`` is only reachable through ``approved``).

        Args:
            post_id: The post's draft id (unique within its batch).
            batch_id: The batch the post belongs to (composite-key half).
            status: The target lifecycle status (must be a legal transition).
            edited_text: New edited body, if changing it.
            scheduled_date: New scheduled date, if changing it.
            claim_flags: The claim-safety verdict to persist. The production
                caller writes a :class:`ClaimFlags` dict
                (``ClaimCheck.model_dump()``); a bare ``list[str]`` is also
                accepted (legacy/manual). JSON-encoded into ``posts.claim_flags``.
            approved_at: Explicit approval timestamp override.
            now: Injectable timestamp for the auto-stamp (callable or ISO string).

        Raises:
            ValueError: if ``status`` is not in :data:`VALID_STATUSES`, or the
                transition from the current status is illegal
                (:data:`STATUS_TRANSITIONS`).
            KeyError: if no post ``(batch_id, post_id)`` exists.

        ``approved_at`` is stamped automatically on the transition *to*
        ``approved`` (from the resolved ``now``) when the caller does not pass an
        explicit value, and is otherwise left untouched — so a later
        ``scheduled`` / ``exported`` update never clobbers the original approval
        time. Conversely, moving a post BACK out of the approved chain
        (``approved``/``scheduled``/``exported`` -> a non-approved status) clears
        the now-stale ``approved_at`` (DI-4), unless the caller passes an explicit
        value. Only the side fields the caller actually provides are written;
        omitted ones keep their stored value.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"unknown status {status!r}; must be one of "
                f"{sorted(VALID_STATUSES)}"
            )

        # Read the current status first so we can validate the transition and
        # decide whether a stale approved_at must be cleared.
        current_row = self.conn.execute(
            "SELECT status, approved_at FROM posts WHERE batch_id = ? AND id = ?",
            (batch_id, post_id),
        ).fetchone()
        if current_row is None:
            raise KeyError(f"no post with id {post_id!r} in batch {batch_id!r}")
        current_status = current_row["status"]

        # DI-3: enforce the legal lifecycle. A no-op (same -> same) always passes.
        if status != current_status and status not in STATUS_TRANSITIONS.get(
            current_status, frozenset()
        ):
            allowed = sorted(STATUS_TRANSITIONS.get(current_status, frozenset()))
            raise ValueError(
                f"illegal status transition {current_status!r} -> {status!r}; "
                f"{current_status!r} may only move to {allowed}"
            )

        assignments = ["status = ?"]
        params: list = [status]

        if edited_text is not None:
            assignments.append("edited_text = ?")
            params.append(edited_text)
        if scheduled_date is not None:
            assignments.append("scheduled_date = ?")
            params.append(scheduled_date)
        if claim_flags is not None:
            assignments.append("claim_flags = ?")
            params.append(_dumps(claim_flags))

        # Stamp approved_at on the approve transition (unless caller overrides).
        if approved_at is None and status == "approved":
            approved_at = _resolve_now(now)
        if approved_at is not None:
            assignments.append("approved_at = ?")
            params.append(approved_at)
        elif (
            current_status in _APPROVED_CHAIN
            and status not in _APPROVED_CHAIN
            and current_row["approved_at"] is not None
        ):
            # DI-4: the post left the approved chain — null the stale stamp so a
            # later reader never sees an approval time on a non-approved post.
            assignments.append("approved_at = NULL")

        params.extend([batch_id, post_id])
        cursor = self.conn.execute(
            f"UPDATE posts SET {', '.join(assignments)} "
            "WHERE batch_id = ? AND id = ?",
            params,
        )
        if cursor.rowcount == 0:  # pragma: no cover — existence checked above
            raise KeyError(f"no post with id {post_id!r} in batch {batch_id!r}")
        self._commit()

    def set_claim_flags(self, post_id: str, batch_id: str, flags: ClaimFlagsArg) -> None:
        """Set a post's claim flags (convenience for the P2b claims gate).

        Scoped to the composite ``(batch_id, post_id)`` key (DI-1).

        Args:
            post_id: The post's draft id.
            batch_id: The batch the post belongs to.
            flags: The claim-safety verdict — a :class:`ClaimFlags` dict
                (``ClaimCheck.model_dump()``, what the production caller writes)
                or a bare ``list[str]`` (legacy/manual). JSON-encoded into
                ``posts.claim_flags``.

        Raises:
            KeyError: if no post ``(batch_id, post_id)`` exists.
        """
        cursor = self.conn.execute(
            "UPDATE posts SET claim_flags = ? WHERE batch_id = ? AND id = ?",
            (_dumps(flags), batch_id, post_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"no post with id {post_id!r} in batch {batch_id!r}")
        self._commit()

    def set_media(self, post_id: str, batch_id: str, media: MediaRef | None) -> None:
        """Set or clear a post's paired media (the review-queue swap/remove, M5).

        Scoped to the composite ``(batch_id, post_id)`` key. ``media`` is a
        :class:`~draftforge.models.MediaRef` (JSON-encoded into ``posts.media``) or
        ``None`` to clear it.

        Raises:
            KeyError: if no post ``(batch_id, post_id)`` exists.
        """
        payload = _dumps(media.model_dump(mode="json")) if media is not None else None
        cursor = self.conn.execute(
            "UPDATE posts SET media = ? WHERE batch_id = ? AND id = ?",
            (payload, batch_id, post_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"no post with id {post_id!r} in batch {batch_id!r}")
        self._commit()

    # ------------------------------------------------------------------ #
    # url_store
    # ------------------------------------------------------------------ #

    def upsert_url(
        self, url: str, *, label: str | None = None, active: bool = True
    ) -> None:
        """Insert or replace a URL in the standing URL store."""
        self.conn.execute(
            "INSERT INTO url_store (url, label, active) VALUES (?, ?, ?) "
            "ON CONFLICT(url) DO UPDATE SET label = excluded.label, "
            "active = excluded.active",
            (url, label, 1 if active else 0),
        )
        self._commit()

    def list_urls(self, active: bool | None = True) -> list[dict]:
        """List stored URLs.

        Args:
            active: ``True`` (default) returns only active URLs, ``False`` only
                inactive ones, ``None`` returns every URL regardless of flag.

        Each row is a dict with ``url``, ``label``, and a boolean ``active``.
        """
        if active is None:
            rows = self.conn.execute(
                "SELECT url, label, active FROM url_store ORDER BY url"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT url, label, active FROM url_store WHERE active = ? "
                "ORDER BY url",
                (1 if active else 0,),
            ).fetchall()
        return [
            {"url": r["url"], "label": r["label"], "active": bool(r["active"])}
            for r in rows
        ]

    def set_url_active(self, url: str, active: bool) -> None:
        """Toggle a URL's active flag.

        Raises:
            KeyError: if no such URL exists.
        """
        cursor = self.conn.execute(
            "UPDATE url_store SET active = ? WHERE url = ?",
            (1 if active else 0, url),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"no url {url!r}")
        self._commit()
