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
**review-gated, brand-voiced, claims-checked Facebook & Instagram draft posts**, dated across a
content calendar and exported for scheduling. It **stops at an approved draft** — a human
reviews and approves every post before it leaves the tool (no auto-publishing).

DraftForge is a portfolio project. The examples below use a fictional *Stillwater Float Co.*;
no real client data is included in this repository.

> **Why it exists.** A small wellness business posts constantly across FB + IG. This reclaims
> the owner's content-creation time *without* dropping cadence — while keeping a hard
> owner-review gate and an **advisory claims check** that flags un-substantiated health
> claims for the reviewer (a real advertising-standards exposure for wellness copy).

---

## What it demonstrates (portfolio lens)

File handling · API integration · **prompt chaining** · structured-output validation ·
**content moderation (an advisory claims check)** · a web frontend — plus production rigor:
schema-validated structured outputs, cost-aware model routing, baseline-comparison prompt
docs, a fail-loud "open receivers" input model, and an **offline unit-test suite** (the whole
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
- **`run_batch` is dependency-injected and separate from persistence:** it never touches the
  store (persistence + media pairing happen in the caller), and it makes model calls only
  through the LLM client it is given — live API calls with the real transport, canned
  responses in tests.

---

## The claims check (advisory — the differentiator)

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
APP_PASSWORD=                    # REQUIRED for the web app (username: draftforge): your own value
# DRAFTFORGE_HOST=127.0.0.1      # optional web-app bind; 0.0.0.0 only if other hosts must connect
```

There is **no default password**: the web app refuses to start if `APP_PASSWORD` is unset,
blank, whitespace-only, or a documented example value (such as the old `change-me`). That is a
deny-list, not a strength check — choosing a long, unique password is your responsibility.

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
python app.py                  # http://127.0.0.1:7860 — login: draftforge / $APP_PASSWORD
python app.py --host 0.0.0.0   # opt in: reachable from other hosts (still login-gated)
python app.py --share          # opt in: temporary public gradio.live link (still login-gated)
```

`python -m draftforge.app` is the same launcher. Before building the UI, opening the store or
listening, it validates `APP_PASSWORD` (see Setup) and requires `ANTHROPIC_API_KEY`. It binds
`127.0.0.1` with no share link unless you opt in with `--host` / `DRAFTFORGE_HOST` or `--share`;
Gradio's own `GRADIO_SERVER_NAME` / `GRADIO_SHARE` variables are not used. A hosted container
(e.g. a Hugging Face Space, which runs `app.py` as its app file) needs `APP_PASSWORD` set as a
secret and `DRAFTFORGE_HOST=0.0.0.0` set as a variable; that hosted path is not exercised by this
repository's tests. Importing `app.py` builds nothing (there is no module-level `demo`), so a
discovery-style launcher cannot serve the UI without the login.

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
pytest -q                       # offline unit tests (no key, no network)
pytest -q tests/smoke -m smoke  # live API smoke tests (opt-in; needs ANTHROPIC_API_KEY)
```

Every stage is unit-tested against an injected `FakeLLM`; an offline-guard conftest keeps the
suite network-free.

---

## Project status & roadmap

**v1 (current):** ingest → … → claims check → review → approved draft → calendar export.
Phases P0–P3 + media (Phase M) are implemented and covered by the offline unit suite. This
repository does not demonstrate a live deployment or a live end-to-end run.

**Security model (what the code enforces):**

- **Login is mandatory** for the web app: no default password; unset / blank / example values
  are refused before anything starts (see Setup).
- **Loopback by default:** `127.0.0.1`, share off; external reach is an explicit opt-in.
- **URL fetching (SSRF guard):** http/https on ports 80/443 only, no embedded credentials.
  Every hop — including each redirect target — is validated *before* it is contacted: every
  resolved address must be public (loopback, private, link-local incl. cloud metadata, shared,
  reserved, multicast and unspecified are refused, also when embedded in IPv6). Redirects are
  followed manually (relative `Location`s resolved, at most 5, loops refused). Every A and AAAA
  answer is checked — one non-public answer rejects the host — and the connection is made only
  to those validated numeric addresses — no second lookup — while HTTPS still verifies the
  certificate for the original host name. Each request has one 30 s deadline covering connect,
  TLS, headers, chunk framing and body, so a slow-drip server cannot hold a fetch open.
- Scraped source text is **delimited as untrusted data** in prompts — a mitigation for indirect
  prompt injection, not a guarantee.

**Known limitations / next:**

- **Live end-to-end smoke is pending** real grounding data (the open receivers above) + a key —
  the app *correctly* fails loud until they're filled.
- **Security residuals:** the Gradio login is the only access control, so any external bind
  needs a strong password and, outside a trusted network, HTTPS in front. The URL fetcher
  refuses a host whose DNS answers mix public and non-public addresses, ignores proxy settings
  (an egress proxy is unsupported), refuses compressed or >10 MiB responses, and applies its
  30 s deadline per request (a redirect chain is at most 6 requests). A running service's
  configuration is outside what this repository can show.
- **Transcript cleaner** (the `data/corpus/` builder) is **mechanical** by design (it preserves
  the author's voice rather than LLM-rewriting it); on very noisy auto-transcripts it leaves
  some rough edges — fine for voice grounding, not publication-grade prose.
- **v2:** direct publishing to Meta (Graph API), scheduled scrape watchers, AI image-gen.

---

## Stack

Python · `anthropic` · PyPDF2 · trafilatura + requests · Jinja2 · Gradio · SQLite · pytest.

The architecture and key design decisions are summarized above; the full design + phased
implementation notes are kept in a private working repo.
