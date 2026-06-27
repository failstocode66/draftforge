"""Render a :class:`~draftforge.models.Draft` to text and to JSON.

The text renderer uses a committed Jinja2 template
(``templates/post.txt.j2``) loaded from disk at call time, mirroring the
prompt-file pattern the stages use: a template edit takes effect without a code
change. The JSON renderer is just the model's own ``model_dump_json`` so output
round-trips back to an equal ``Draft`` with no loss.

Both are pure and offline. The text renderer takes an injectable ``now`` (and
``model_label``) so its footer line is deterministic in tests — no wall-clock or
build-version leak into the rendered string.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from draftforge.models import Draft

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_TEXT_TEMPLATE = "post.txt.j2"

# Placeholder shown on the footer line when the caller does not name the model
# that produced the draft (the pipeline can thread the real id through later).
_DEFAULT_MODEL_LABEL = "unspecified"


@lru_cache(maxsize=1)
def _env() -> Environment:
    """Build (once) the Jinja2 environment rooted at the templates dir."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),  # plain text, no HTML
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_text(
    draft: Draft,
    *,
    now: str | None = None,
    model_label: str = _DEFAULT_MODEL_LABEL,
) -> str:
    """Render ``draft`` as a human-readable text block.

    Args:
        draft: The draft to render.
        now: Footer timestamp — an ISO-8601 string, or ``None`` to use the
            current UTC time. Inject a fixed value for deterministic output.
        model_label: Footer label naming the model/run that produced the draft.

    Returns:
        The rendered text.
    """
    template = _env().get_template(_TEXT_TEMPLATE)
    return template.render(
        draft=draft,
        generated_at=_resolve_now(now),
        model_label=model_label,
    )


def render_json(draft: Draft) -> str:
    """Render ``draft`` as JSON that round-trips back to an equal ``Draft``."""
    return draft.model_dump_json(indent=2)


def _resolve_now(now: str | None) -> str:
    """Resolve the injectable footer timestamp into an ISO-8601 string."""
    if now is None:
        return datetime.now(timezone.utc).isoformat()
    return now
