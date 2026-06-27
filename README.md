---
title: DraftForge
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# DraftForge

An AI content pipeline that turns a wellness business's source material into a batch of
**review-gated, brand-voiced, claims-safe Facebook & Instagram draft posts**, dated across a
content calendar and exported for scheduling. It **stops at an approved draft** — a human
reviews and approves every post before it leaves the tool (no auto-publishing).

Built as **Portfolio Project #2** of a prompt-engineering retraining track, and shipped as a
**live tool** for a real wellness studio — an independent float-tank / sensory-deprivation
business. (Client details are kept private; the examples below use a fictional *Stillwater
Float Co.* in their place.)

> **Why it exists.** A small wellness business posts constantly across FB + IG. This reclaims
> the owner's content-creation time *without* dropping cadence — while keeping a hard
> owner-review gate and a **claims-safety guardrail** that flags un-substantiated health
> claims (a real advertising-standards exposure for wellness copy).

---

## What it demonstrates (portfolio lens)

File handling · API integration · **prompt chaining** · structured-output validation ·
**content moderation (a claims-safety gate)** · a web frontend — plus production rigor:
schema-validated structured outputs, cost-aware model routing, baseline-comparison prompt
docs, a fail-loud "open receivers" input model, and **419 offline unit tests** (the whole
chain is testable without an API key via an injected `FakeLLM`).

---

## Architecture

```
                   ┌──────────────────────── per-run inputs ───────────────────────┐
                   │  URLs · guidance prompt · batch size · media uploads · (corpus) │
                   └───────────────────────────────┬───────────────────────────────┘
                                                    ▼
  ingest ──▶ classify ──▶ extract ──▶ generate ──▶ claims-safety gate ──▶ output
 (url/file)   (angle)   (per-angle)  (voice +      (soften / advise /     (text + JSON,
                                      corpus +       flag; never blocks      calendar
                                      exemplars)     approval)               md/csv/ics)
                                                    │
                            media upload ──▶ pair_media (D10) ──▶ Draft.media
                                                    │
                                                    ▼
                         SQLite store: draft → edited → approved → scheduled → exported
                                                    │
                                                    ▼
                        Gradio UI (auth-gated):  Run · Review · Calendar
```

- **One retrying, schema-validating LLM client** wraps every model call (retry on bad
  JSON / rate-limit; Pydantic-validated outputs) and is dependency-injected, so the whole
  pipeline runs in tests against canned responses.
- **Cost-aware routing:** Haiku for classify/extract, Opus for generate + claims-check.
- **`run_batch` is pure** (no store, no I/O); persistence + media pairing happen in the
  caller, so the core stays trivially testable.

---

## The claims-safety gate (the differentiator)

Wellness copy routinely asserts health effects ("relieves chronic pain", "releases dopamine
and endorphins") without citations. The gate classifies each claim and routes it — **relaxed,
hedge-aware, and advisory** (this is a wellness tool, not a medical one), and **it never
blocks approval**; the reviewer always decides:

| Tier | Meaning | UI badge |
|------|---------|----------|
| `clean` | no health claim | ✓ clear |
| `advisory` | a hedged physiological claim — noted, untouched | ⓘ Health claim — your call |
| `softened` | assertive phrasing softened to "may help…" | ✎ Suggested hedge |
| `flagged` | a higher-liability uncited hard claim | ⚠ Review |
| `needs_manual_review` | the gate errored — **fail-safe**, never auto-passed | ● Needs your review |

Hard claims are checked against an approved-claims register (`data/claims_register.json`).

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # (Scripts/ on Windows; bin/ on POSIX)
cp .env.example .env                                # then fill in the values
```

`.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
MODEL_FAST=claude-haiku-4-5-20251001
MODEL_SMART=claude-opus-4-8
APP_PASSWORD=change-me           # gates the Gradio app (username: draftforge)
```

### Real-data "open receivers" (fail-loud, no silent placeholders)

The pipeline refuses to run on placeholder data. Fill these real inputs (kept gitignored),
or `preflight` will tell you exactly what's missing:

- `prompts/voice_exemplars.md` — ≥6 real past posts across FB + IG (the brand-voice few-shot).
- `data/corpus/` — the business's voice corpus (e.g. cleaned podcast transcripts).
- `data/claims_register.json` — the approved-claims register.

```bash
python -m draftforge.cli preflight          # lists every unfilled receiver and exits non-zero
```

---

## Usage

### CLI

```bash
python -m draftforge.cli \
  --input <dir-of-docs | url> \
  --guidance "lean into spring stress-relief" \
  --n 12 \
  --media-dir ./media        # optional: images/videos paired to posts (D10)
```

### Web app (Gradio, auth-gated)

```bash
python app.py                  # http://localhost:7860 — login: draftforge / $APP_PASSWORD
```

- **Run** — source URLs + guidance + batch size + media upload → generate drafts.
- **Review** — one Stacked card at a time: the claim badge (advisory, non-blocking), edit
  the caption, swap/remove its media, approve (or regenerate / reject).
- **Calendar** — assign dates across the approved posts, export **Markdown / CSV / iCalendar**.

---

## Supported inputs & post angles

- **Documents:** `.txt`, `.md`, `.pdf`. **URLs:** readable-article extraction (trafilatura).
- **Media:** images (`.jpg/.jpeg/.png/.webp/.gif`) + video (`.mp4/.mov/.webm/.m4v`),
  upload-only; AI image generation is a designed-for, opt-in, default-OFF future stage.
- **Angles:** educational/science · personal-story · benefit-spotlight · myth-buster ·
  offer/promo · other.

---

## Testing

```bash
pytest -q                       # 419 offline unit tests (no key, no network)
pytest -q tests/smoke -m smoke  # live API smoke tests (opt-in; needs ANTHROPIC_API_KEY)
```

Every stage is unit-tested against an injected `FakeLLM`; an offline-guard conftest keeps the
suite network-free.

---

## Project status & roadmap

**v1 (current):** ingest → … → claims gate → review → approved draft → calendar export.
Phases P0–P3 + media (Phase M) are **build-complete with a green offline test suite**.
Deploy-hardened: URL fetching has an **SSRF guard** (refuses internal/metadata/private
addresses + non-http(s) schemes, re-checks redirects) and scraped source text is
**delimited as untrusted data** against indirect prompt injection.

**Known limitations / next:**

- **Live end-to-end smoke is pending** real grounding data (the open receivers above) + a key —
  the app *correctly* fails loud until they're filled.
- **Transcript cleaner** (the `data/corpus/` builder) is **mechanical** by design (it preserves
  the author's voice rather than LLM-rewriting it); on very noisy auto-transcripts it leaves
  some rough edges — fine for voice grounding, not publication-grade prose.
- **v2:** direct publishing to Meta (Graph API), scheduled scrape watchers, AI image-gen.

---

## Stack

Python · `anthropic` · PyPDF2 · trafilatura + requests · Jinja2 · Gradio · SQLite · pytest.

The architecture and key design decisions are summarized above; the full design + phased
implementation notes are kept in a private working repo.
