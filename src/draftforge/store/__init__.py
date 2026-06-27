"""Persistent store package.

Re-exports the :class:`~draftforge.store.db.Store` SQLite store and its
:data:`~draftforge.store.db.VALID_STATUSES` set — the foundation the claims gate
(P2b) and pipeline wiring (P2c) build on for tracking drafts through their
status lifecycle.
"""

from __future__ import annotations

from draftforge.store.db import (
    STATUS_TRANSITIONS,
    VALID_STATUSES,
    ClaimFlags,
    Store,
)

__all__ = ["Store", "VALID_STATUSES", "STATUS_TRANSITIONS", "ClaimFlags"]
