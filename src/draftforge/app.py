"""Gradio app — handlers + UI for DraftForge.

The **handlers** (:func:`handle_run`, plus the review/export handlers added in
3.3/3.4) are gradio-free and unit-tested with a ``FakeLLM`` + in-memory
:class:`~draftforge.store.db.Store`. :func:`build_ui` / :func:`launch` import
gradio **lazily** (the same pattern :mod:`draftforge.cli` uses for the Anthropic
SDK) so importing this module — and running the handler tests — never requires
gradio, an API key, or the network.

UI is kept thin: every button callback is a one-line adapter onto a tested
handler. The three tabs (Run / Review / Calendar) share one :class:`Store` and a
``current batch`` state for the app's lifetime.
"""

from __future__ import annotations

import os
import uuid

import requests

from draftforge import inputs
from draftforge.ingest.fetcher import FetchError, fetch_url
from draftforge.ingest.media import load_media
from draftforge.ingest.normalize import to_source
from draftforge.llm.client import LLMClient
from draftforge.models import Draft, ExtractedItem, MediaRef
from draftforge.output.calendar import assign_dates, export_calendar
from draftforge.persistence import persist_batch
from draftforge.pipeline import run_batch
from draftforge.stages.claims import claims_check
from draftforge.stages.generate import generate_posts
from draftforge.stages.pair_media import pair_media
from draftforge.store.db import Store

_DEFAULT_DB = "data/store.db"


def _default_batch_id() -> str:
    """Generate a fresh batch id (injected in tests for offline determinism)."""
    return f"batch-{uuid.uuid4().hex[:12]}"


# --- handlers (gradio-free, unit-tested) ---------------------------------------


def _ingest_urls(urls, *, getter=requests.get):
    """Fetch each non-blank URL into a Source, collecting per-URL failures.

    A failed fetch never aborts the run — its error is collected and the other
    URLs proceed (per-item isolation), mirroring ``run_batch``'s per-source
    isolation. Blank/whitespace entries (from a multiline textbox) are skipped.
    """
    sources = []
    errors = []
    for raw in urls or []:
        url = (raw or "").strip()
        if not url:
            continue
        try:
            text = fetch_url(url, getter=getter)
        except FetchError as exc:
            errors.append(str(exc))
            continue
        sources.append(to_source("url", url, text))
    return sources, errors


def handle_run(
    urls,
    guidance,
    n,
    *,
    llm: LLMClient,
    store: Store,
    media_paths=None,
    transcript_text=None,
    base_dir=None,
    getter=requests.get,
    batch_id_factory=_default_batch_id,
    now=None,
) -> dict:
    """Run the pipeline for the Run-tab inputs and persist the batch.

    Loads the D9 grounding open receivers (voice exemplars, corpus, claims
    register) fail-loud; an uploaded transcript, if any, is appended to the
    corpus for THIS run only (it does not mutate ``data/corpus/``). Ingests the
    URL list (skipping blanks, collecting failures), runs the pure pipeline,
    pairs uploaded media (order strategy, D10), and persists the batch.

    Returns:
        ``{"batch_id", "drafts", "errors", "result"}`` — ``errors`` merges
        per-source pipeline errors with per-URL ingest failures (both as strings)
        for display in the UI.
    """
    voice = inputs.load_voice_exemplars(base_dir=base_dir)
    corpus = inputs.load_corpus(base_dir=base_dir)
    if transcript_text and transcript_text.strip():
        corpus = f"{corpus}\n\n# (uploaded transcript)\n{transcript_text.strip()}"
    register = inputs.load_claims_register(base_dir=base_dir)

    sources, ingest_errors = _ingest_urls(urls, getter=getter)

    result = run_batch(
        sources,
        guidance=guidance,
        voice_exemplars=voice,
        corpus=corpus,
        batch_size=n,
        register=register,
        llm=llm,
    )

    media = load_media([str(p) for p in (media_paths or [])])
    if media:
        result.drafts = pair_media(result.drafts, media)

    batch_id = batch_id_factory()
    persist_batch(
        store, result, sources,
        guidance=guidance, batch_size=n, batch_id=batch_id, now=now,
    )

    return {
        "batch_id": batch_id,
        "drafts": result.drafts,
        "errors": [str(e) for e in result.errors] + ingest_errors,
        "result": result,
    }


def _run_summary(out: dict) -> str:
    """Render a Run result as Markdown for the Run tab's status box."""
    lines = [
        f"**Batch `{out['batch_id']}`** — produced **{len(out['drafts'])}** draft(s).",
    ]
    if out["errors"]:
        lines.append("")
        lines.append(f"Skipped {len(out['errors'])} source(s):")
        lines += [f"- {e}" for e in out["errors"]]
    lines.append("")
    lines.append("Open the **Review** tab to edit, approve, and pair media.")
    return "\n".join(lines)


# --- review queue: claim badge + handlers (3.3 / M5) ----------------------------

# Improvement #1 (approved 2026-06-26): each draft renders a claim badge derived
# from its persisted claim_flags.status. The ``advisory`` lane is GENTLE and
# explicitly NON-blocking — the softest tier above ``clean``. Nothing here blocks
# approval; badges only convey severity + affordance. Wording for ``advisory`` is
# the copy Tyler signed off ("ⓘ Health claim — your call").
_CLAIM_BADGES: dict[str, tuple[str, str]] = {
    "clean": ("✓ clear", "clean"),
    "advisory": ("ⓘ Health claim — your call", "advisory"),
    "softened": ("✎ Suggested hedge", "softened"),
    "flagged": ("⚠ Review — higher-liability claim", "flagged"),
    "needs_manual_review": ("● Needs your review", "needs_review"),
}
# Fail-safe: an unknown/unexpected status must never read as benign — default to
# the strongest tier so a reviewer always looks.
_SAFE_BADGE: tuple[str, str] = _CLAIM_BADGES["needs_manual_review"]


def _claim_badge(claim_flags) -> tuple[str, str]:
    """Map a draft's persisted ``claim_flags`` to a ``(label, tone)`` badge.

    ``tone`` is a semantic class the UI maps to colour:
    ``clean`` < ``advisory`` < ``softened`` < ``flagged`` < ``needs_review``.
    ``advisory`` is the gentle, non-blocking tier. ``None``/empty flags read as
    ``clean``; an unknown status fails safe to ``needs_review``.
    """
    if not claim_flags or not isinstance(claim_flags, dict):
        return _CLAIM_BADGES["clean"]
    status = claim_flags.get("status", "clean")
    return _CLAIM_BADGES.get(status, _SAFE_BADGE)


def handle_edit(post_id: str, batch_id: str, new_text: str, *, store: Store) -> Draft:
    """Save an editor's rewrite into ``edited_text`` and move the draft to ``edited``."""
    store.update_status(post_id, batch_id, "edited", edited_text=new_text)
    return store.get_draft(post_id, batch_id)


def handle_approve(
    post_id: str, batch_id: str, *, store: Store, scheduled_date: str | None = None
) -> Draft:
    """Approve a draft (the human review gate), optionally assigning a date.

    Approval is NON-blocking with respect to claim status — a draft with any
    badge (incl. ``flagged`` / ``needs_manual_review``) is still approvable; the
    badge advises, the reviewer decides.
    """
    store.update_status(post_id, batch_id, "approved", scheduled_date=scheduled_date)
    return store.get_draft(post_id, batch_id)


def handle_reject(post_id: str, batch_id: str, *, store: Store) -> Draft:
    """Reject a draft — send it back to the ``draft`` (unreviewed) state."""
    store.update_status(post_id, batch_id, "draft")
    return store.get_draft(post_id, batch_id)


def handle_set_media(
    post_id: str, batch_id: str, media: MediaRef | None, *, store: Store
) -> Draft:
    """Swap (a new :class:`MediaRef`) or remove (``None``) a post's paired media."""
    store.set_media(post_id, batch_id, media)
    return store.get_draft(post_id, batch_id)


def handle_regenerate(
    post_id: str, batch_id: str, *, llm: LLMClient, store: Store, base_dir=None
) -> Draft:
    """Regenerate a draft's text and stage it into ``edited_text``.

    The store does not persist the original :class:`ExtractedItem`, so this seeds
    a fresh take from the draft's own angle + current text + the run's guidance
    and grounding (a "give me another version" affordance, not a re-run from the
    source article). The new caption lands in ``edited_text`` (status → ``edited``)
    and is re-checked by the claims gate; the verdict replaces ``claim_flags``.

    Raises:
        KeyError: if no post ``(batch_id, post_id)`` exists.
    """
    draft = store.get_draft(post_id, batch_id)
    if draft is None:
        raise KeyError(f"no post with id {post_id!r} in batch {batch_id!r}")

    batch = store.get_batch(batch_id)
    guidance = (batch or {}).get("guidance_prompt") or ""
    voice = inputs.load_voice_exemplars(base_dir=base_dir)
    corpus = inputs.load_corpus(base_dir=base_dir)
    register = inputs.load_claims_register(base_dir=base_dir)

    item = ExtractedItem(
        hook=(draft.edited_text or draft.caption)[:200],
        core_benefit=draft.angle,
        claim=(draft.claims_used[0] if draft.claims_used else None),
    )
    regenerated = generate_posts(
        item,
        guidance=guidance,
        voice_exemplars=voice,
        corpus=corpus,
        platform=draft.platform,
        n=1,
        llm=llm,
        angle=draft.angle,
        id_prefix=draft.id,
    )[0]
    check = claims_check(regenerated, extracted_item=item, register=register, llm=llm)
    store.update_status(
        post_id, batch_id, "edited",
        edited_text=regenerated.caption, claim_flags=check.model_dump(),
    )
    return store.get_draft(post_id, batch_id)


# --- review-tab display helpers (pure-ish; drive the Stacked card) --------------


def _card_markdown(draft: Draft, claim_flags) -> str:
    """Render a draft as the Stacked review card's header (badge → meta → media).

    The editable caption lives in a separate Textbox; this is everything above
    it: the claim badge (Improvement #1), platform/date/status, the media line
    (paired media, or the ``image_direction`` shot suggestion when none), and the
    gate's note.
    """
    label, tone = _claim_badge(claim_flags)
    date = draft.scheduled_date or "unscheduled"
    if draft.media is not None:
        media_line = (
            f"📎 **media:** {draft.media.kind.value} — {os.path.basename(draft.media.ref)}"
        )
    else:
        media_line = f"🖼 **no media** — direction: {draft.image_direction or '(none)'}"
    notes = "; ".join((claim_flags or {}).get("notes", []) or []) if isinstance(claim_flags, dict) else ""
    parts = [
        f"### {draft.platform} · {date} · status `{draft.status}`",
        f"**{label}**  _(claim tier: {tone})_",
        media_line,
    ]
    if notes:
        parts.append(f"> {notes}")
    return "\n\n".join(parts)


def _review_choices(batch_id, store: Store) -> list[str]:
    """Draft ids in a batch, for the review-tab selector."""
    if not batch_id:
        return []
    return [d.id for d in store.list_drafts(batch_id)]


def _batch_media(batch_id, store: Store) -> dict[str, MediaRef]:
    """The distinct media already paired across a batch (the swap pool).

    The review tab lets you re-assign or remove media that the run uploaded;
    options are keyed by a human label (``"uploaded_image: a.jpg"``).
    """
    refs: dict[str, MediaRef] = {}
    if batch_id:
        for d in store.list_drafts(batch_id):
            if d.media is not None:
                refs[f"{d.media.kind.value}: {os.path.basename(d.media.ref)}"] = d.media
    return refs


def _media_from_label(label, batch_id, store: Store):
    """Resolve a swap-dropdown label to a MediaRef, ``None`` (remove), or "keep"."""
    if label in (None, "", "(keep)"):
        return "keep"
    if label == "(remove)":
        return None
    return _batch_media(batch_id, store).get(label, "keep")


# --- calendar / export handlers (3.4) ------------------------------------------

# Only signed-off drafts reach the content calendar: approval is the human gate
# (D2), so draft/edited posts are excluded from scheduling + export.
_EXPORTABLE = ("approved", "scheduled", "exported")


def handle_schedule(
    batch_id: str, *, store: Store, start_date, per_week: int = 3, now=None
) -> list[Draft]:
    """Assign calendar dates to the batch's APPROVED drafts and schedule them.

    Spaces the approved (and already-scheduled) drafts via
    :func:`~draftforge.output.calendar.assign_dates` (``per_week`` per 7 days),
    persists each new ``scheduled_date`` and moves the draft to ``scheduled``.
    Unapproved drafts are left untouched. Returns the scheduled drafts.
    """
    approved = [
        d for d in store.list_drafts(batch_id) if d.status in ("approved", "scheduled")
    ]
    scheduled = assign_dates(approved, start_date=start_date, per_week=per_week)
    for d in scheduled:
        store.update_status(
            d.id, batch_id, "scheduled", scheduled_date=d.scheduled_date, now=now
        )
    return [store.get_draft(d.id, batch_id) for d in scheduled]


def handle_export(
    batch_id: str, fmt: str, *, store: Store, exports_dir: str = "exports"
) -> str:
    """Export the batch's signed-off drafts to ``exports/<batch_id>.<fmt>``.

    Renders the approved/scheduled/exported drafts via
    :func:`~draftforge.output.calendar.export_calendar` (which validates ``fmt``),
    writes the file, and returns its path. Draft/edited posts are excluded — only
    reviewer-approved content reaches the calendar.

    Raises:
        ValueError: if ``fmt`` is not ``md`` | ``csv`` | ``ics``.
    """
    drafts = [d for d in store.list_drafts(batch_id) if d.status in _EXPORTABLE]
    content = export_calendar(drafts, fmt)  # validates fmt before any file I/O
    os.makedirs(exports_dir, exist_ok=True)
    path = os.path.join(exports_dir, f"{batch_id}.{fmt}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def _auth(app_password: str) -> tuple[str, str]:
    """Build the Gradio basic-auth login tuple (username ``draftforge``)."""
    return ("draftforge", app_password)


# --- production wiring (lazy) ---------------------------------------------------


def _production_llm() -> LLMClient:
    """Build the real Anthropic-backed client from settings (lazy import)."""
    from draftforge.config import Settings
    from draftforge.llm.anthropic_transport import AnthropicTransport

    settings = Settings.load()
    return LLMClient(
        AnthropicTransport(),
        model_fast=settings.model_fast,
        model_smart=settings.model_smart,
    )


# --- UI (gradio imported lazily) ------------------------------------------------


def build_ui(*, db_path: str = _DEFAULT_DB, base_dir=None, llm: LLMClient | None = None):
    """Construct the Gradio Blocks UI (Run / Review / Calendar tabs).

    Gradio is imported here, not at module top, so importing this module is
    gradio-free. The store is opened once and shared across tabs for the app's
    lifetime. ``llm`` may be injected (tests); otherwise the real client is built
    lazily on first Run (so constructing the UI needs no API key).
    """
    import gradio as gr

    store = Store(db_path)

    def _on_run(urls_text, guidance, n, media_files):
        urls = (urls_text or "").splitlines()
        media_paths = [
            getattr(f, "name", f) for f in (media_files or [])
        ]
        client = llm if llm is not None else _production_llm()
        out = handle_run(
            urls, guidance, int(n),
            llm=client, store=store, media_paths=media_paths, base_dir=base_dir,
        )
        return _run_summary(out), out["batch_id"]

    with gr.Blocks(title="DraftForge") as demo:
        gr.Markdown("# DraftForge")
        current_batch = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Run"):
                gr.Markdown(
                    "Add source URLs (one per line), a guidance prompt, batch "
                    "size, and optional media to pair to the posts."
                )
                urls_in = gr.Textbox(
                    label="Source URLs (one per line)", lines=4,
                    placeholder="https://example.com/article",
                )
                guidance_in = gr.Textbox(
                    label="Guidance prompt",
                    placeholder="e.g. lean into spring stress-relief",
                )
                n_in = gr.Slider(
                    label="Batch size", minimum=1, maximum=30, value=12, step=1,
                )
                media_in = gr.File(
                    label="Media to pair (images / video, optional)",
                    file_count="multiple",
                    file_types=["image", "video"],
                )
                run_btn = gr.Button("Generate drafts", variant="primary")
                run_status = gr.Markdown()
                run_btn.click(
                    _on_run,
                    inputs=[urls_in, guidance_in, n_in, media_in],
                    outputs=[run_status, current_batch],
                )

            with gr.Tab("Review"):
                gr.Markdown(
                    "Review the current batch one **Stacked card** at a time: the "
                    "claim badge advises (it never blocks approval), edit the "
                    "caption, swap/remove its media, then approve."
                )
                load_btn = gr.Button("Load / refresh current batch")
                draft_pick = gr.Dropdown(label="Draft", choices=[], interactive=True)
                # --- Stacked card: badge → caption → media → actions ---
                card_md = gr.Markdown()
                caption_box = gr.Textbox(label="Caption (edit me)", lines=6)
                with gr.Row():
                    media_pick = gr.Dropdown(
                        label="Media", choices=["(keep)"], value="(keep)"
                    )
                    apply_media_btn = gr.Button("Apply media (swap / remove)")
                date_box = gr.Textbox(label="Scheduled date (YYYY-MM-DD, optional)")
                with gr.Row():
                    save_btn = gr.Button("Save edit")
                    approve_btn = gr.Button("Approve", variant="primary")
                    regen_btn = gr.Button("Regenerate")
                    reject_btn = gr.Button("Reject")
                review_status = gr.Markdown()

                def _resolve_llm():
                    return llm if llm is not None else _production_llm()

                def _card(post_id, batch_id):
                    row = store.get_post_row(post_id, batch_id)
                    draft = store.get_draft(post_id, batch_id)
                    flags = row["claim_flags"] if row else None
                    return (
                        _card_markdown(draft, flags),
                        draft.edited_text or draft.caption,
                        draft.scheduled_date or "",
                    )

                def _refresh(batch_id):
                    ids = _review_choices(batch_id, store)
                    media_opts = ["(keep)", "(remove)"] + list(
                        _batch_media(batch_id, store)
                    )
                    return (
                        gr.update(choices=ids, value=(ids[0] if ids else None)),
                        gr.update(choices=media_opts, value="(keep)"),
                    )

                def _show(post_id, batch_id):
                    if not post_id or not batch_id:
                        return "", "", ""
                    return _card(post_id, batch_id)

                def _after(post_id, batch_id, msg):
                    return (*_card(post_id, batch_id), msg)

                def _save(post_id, batch_id, text):
                    handle_edit(post_id, batch_id, text, store=store)
                    return _after(post_id, batch_id, "Saved edit.")

                def _approve(post_id, batch_id, date):
                    handle_approve(
                        post_id, batch_id, store=store, scheduled_date=(date or None)
                    )
                    return _after(post_id, batch_id, "Approved.")

                def _reject(post_id, batch_id):
                    handle_reject(post_id, batch_id, store=store)
                    return _after(post_id, batch_id, "Rejected — back to draft.")

                def _regen(post_id, batch_id):
                    handle_regenerate(
                        post_id, batch_id, llm=_resolve_llm(), store=store, base_dir=base_dir
                    )
                    return _after(post_id, batch_id, "Regenerated into the caption.")

                def _apply_media(post_id, batch_id, choice):
                    ref = _media_from_label(choice, batch_id, store)
                    if ref == "keep":
                        return _after(post_id, batch_id, "No media change.")
                    handle_set_media(post_id, batch_id, ref, store=store)
                    return _after(post_id, batch_id, "Media updated.")

                _card_out = [card_md, caption_box, date_box]
                _card_out_msg = [card_md, caption_box, date_box, review_status]
                load_btn.click(_refresh, [current_batch], [draft_pick, media_pick])
                draft_pick.change(_show, [draft_pick, current_batch], _card_out)
                save_btn.click(_save, [draft_pick, current_batch, caption_box], _card_out_msg)
                approve_btn.click(_approve, [draft_pick, current_batch, date_box], _card_out_msg)
                reject_btn.click(_reject, [draft_pick, current_batch], _card_out_msg)
                regen_btn.click(_regen, [draft_pick, current_batch], _card_out_msg)
                apply_media_btn.click(
                    _apply_media, [draft_pick, current_batch, media_pick], _card_out_msg
                )

            with gr.Tab("Calendar"):
                gr.Markdown(
                    "Assign dates to your **approved** posts, then export the "
                    "content calendar (Markdown / CSV / iCalendar)."
                )
                with gr.Row():
                    start_in = gr.Textbox(label="Start date (YYYY-MM-DD)")
                    perweek_in = gr.Slider(
                        label="Posts per week", minimum=1, maximum=7, value=3, step=1
                    )
                schedule_btn = gr.Button("Assign dates")
                fmt_in = gr.Radio(
                    label="Export format", choices=["md", "csv", "ics"], value="md"
                )
                export_btn = gr.Button("Export calendar", variant="primary")
                cal_status = gr.Markdown()
                cal_file = gr.File(label="Download")

                def _schedule(batch_id, start, perweek):
                    if not batch_id:
                        return "Run and approve a batch first."
                    if not (start or "").strip():
                        return "Enter a start date (YYYY-MM-DD)."
                    out = handle_schedule(
                        batch_id, store=store, start_date=start.strip(),
                        per_week=int(perweek),
                    )
                    return f"Scheduled {len(out)} approved post(s) from {start.strip()}."

                def _export(batch_id, fmt):
                    if not batch_id:
                        return "Run and approve a batch first.", None
                    path = handle_export(batch_id, fmt, store=store)
                    return f"Exported `{path}`.", path

                schedule_btn.click(
                    _schedule, [current_batch, start_in, perweek_in], [cal_status]
                )
                export_btn.click(
                    _export, [current_batch, fmt_in], [cal_status, cal_file]
                )

    return demo


def launch(*, base_dir=None, **kwargs):  # pragma: no cover - thin production wrapper
    """Launch the Gradio app, gated by ``APP_PASSWORD`` (username ``draftforge``)."""
    from draftforge.config import Settings

    settings = Settings.load()
    build_ui(base_dir=base_dir).launch(auth=_auth(settings.app_password), **kwargs)


if __name__ == "__main__":  # pragma: no cover
    launch()
