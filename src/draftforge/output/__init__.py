"""Output rendering for review-gated drafts.

Two pure renderers turn a :class:`~draftforge.models.Draft` into the two shapes
the rest of the system needs:

* :func:`~draftforge.output.render.render_text` — a human-readable block (via a
  committed Jinja2 template) for terminal/preview display.
* :func:`~draftforge.output.render.render_json` — the model's loss-free JSON for
  storage and round-tripping.

Neither touches the LLM or the network, so they are trivially testable.
"""

from __future__ import annotations

from draftforge.output.render import render_json, render_text

__all__ = ["render_json", "render_text"]
