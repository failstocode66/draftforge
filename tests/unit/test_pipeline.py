"""Tests for the batch pipeline orchestration.

Offline via a fake transport that hands back canned JSON in the per-source chain
order: classify (1) -> extract (1) -> then, PER platform, generate (1 batch)
followed by one claims_check PER produced draft (Task 2.4 wires the gate in).

We verify:
* a clean run produces drafts for every (source x platform) cell with stable
  ids that encode the source_id + platform (INT-4);
* the claims gate runs for every draft (``claim_checks`` has an entry per draft),
  and an uncited hard claim routes that draft's check to softened/flagged/review;
* per-source failure isolation: if a source's *classify* raises, that source is
  skipped (its error recorded and observable) while the other source still
  produces drafts — the batch never aborts wholesale;
* the batch_size is distributed across cells (sane coverage, not exactness).
"""

import json

import pytest

from draftforge.llm.client import LLMClient
from draftforge.models import ClaimType, Draft, Platform, RegisterEntry, Source
from draftforge.pipeline import BatchResult, _distribute, run_batch


class FakeTransport:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def text(self, *, model, system, user, max_tokens):
        self.calls.append({"model": model, "system": system, "user": user})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_llm(*responses):
    return LLMClient(
        FakeTransport(responses),
        model_fast="fast",
        model_smart="smart",
        sleep=lambda *_: None,
    )


def _classify(angle="educational"):
    return json.dumps({"angle": angle})


def _extract():
    return json.dumps({"hook": "Float away stress", "core_benefit": "deep calm"})


def _generate(n):
    posts = [
        {
            "caption": f"caption {i}",
            "hashtags": [f"#tag{i}"],
            "image_direction": None,
            "claims_used": [],
        }
        for i in range(n)
    ]
    return json.dumps({"posts": posts})


def _claims_clean():
    """A canned ClaimAnalysis the claims gate resolves to ``clean`` (no claims)."""
    return json.dumps(
        {"claims": [], "harmful": False, "harmful_reason": "", "softened_caption": None}
    )


def _claims_flagged(reason="disease cure claim"):
    """A canned ClaimAnalysis the claims gate resolves to ``flagged`` (harmful)."""
    return json.dumps(
        {
            "claims": [],
            "harmful": True,
            "harmful_reason": reason,
            "softened_caption": None,
        }
    )


def _platform_chain(n):
    """The canned responses for one platform: generate(n) + one clean claims/draft."""
    return [_generate(n), *[_claims_clean() for _ in range(n)]]


def _source(idx):
    return Source(
        source_id=f"s{idx}",
        type="file",
        title=f"doc {idx}",
        text=f"body text {idx}",
        fetched_at="2026-06-25T12:00:00Z",
    )


GUIDANCE = "Warm, grounded, non-hypey."
VOICE = "Exemplar: Sink into stillness."
CORPUS = "Sam: we sell calm, not cures."


def test_run_batch_clean_run_covers_every_source_x_platform_cell():
    # One source, two platforms, batch_size 4 -> 2 per platform.
    # Chain per source: classify, extract, then PER platform [generate(2) +
    # one claims_check per produced draft].
    llm = make_llm(
        _classify(),  # classify (once per source)
        _extract(),  # extract (once per source)
        *_platform_chain(2),  # generate facebook (2) + 2 clean claims checks
        *_platform_chain(2),  # generate instagram (2) + 2 clean claims checks
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=4,
        register=[],
        llm=llm,
    )

    assert isinstance(result, BatchResult)
    assert result.errors == []
    platforms = {d.platform for d in result.drafts}
    assert platforms == {Platform.facebook, Platform.instagram}
    assert len(result.drafts) == 4
    assert all(isinstance(d, Draft) for d in result.drafts)
    # The claims gate ran for every draft: one verdict per draft id.
    assert set(result.claim_checks) == {d.id for d in result.drafts}
    assert all(c.status == "clean" for c in result.claim_checks.values())


def test_run_batch_ids_encode_source_id_and_platform():
    # INT-4: ids are built from the source's STABLE source_id (not its list
    # position), so they are reproducible across runs regardless of input order.
    llm = make_llm(
        _classify(), _extract(), *_platform_chain(1), *_platform_chain(1)
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=llm,
    )

    ids = sorted(d.id for d in result.drafts)
    # Ids carry the stable source_id ("s0") and the platform so they are unique
    # and deterministic across the whole batch.
    assert any(i.startswith("s0-") and "facebook" in i for i in ids)
    assert any(i.startswith("s0-") and "instagram" in i for i in ids)
    assert len(set(ids)) == len(ids)  # all unique


def test_run_batch_ids_are_stable_across_input_ordering():
    # INT-4: the same source produces the same draft ids regardless of where it
    # sits in the input list — ids are derived from source_id, not list index.
    src = _source(7)
    decoy = _source(99)

    # Run 1: src is the ONLY source (would be index 0).
    first = run_batch(
        [src],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=make_llm(
            _classify(), _extract(), *_platform_chain(1), *_platform_chain(1)
        ),
    )

    # Run 2: src sits at index 1, behind a decoy that runs its own full chain.
    second = run_batch(
        [decoy, src],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=make_llm(
            # decoy chain
            _classify(), _extract(), *_platform_chain(1), *_platform_chain(1),
            # src chain
            _classify(), _extract(), *_platform_chain(1), *_platform_chain(1),
        ),
    )

    src_ids_run1 = sorted(d.id for d in first.drafts)
    src_ids_run2 = sorted(d.id for d in second.drafts if d.id.startswith("s7-"))
    # The decoy at index 0 did NOT shift src's ids; they are identical across runs.
    assert src_ids_run1 == src_ids_run2
    assert all(d.id.startswith("s7-") for d in first.drafts)


def test_run_batch_distributes_batch_size_across_cells():
    # batch_size 6, one source, two platforms -> 3 per platform.
    llm = make_llm(
        _classify(), _extract(), *_platform_chain(3), *_platform_chain(3)
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=6,
        register=[],
        llm=llm,
    )

    fb = [d for d in result.drafts if d.platform == Platform.facebook]
    ig = [d for d in result.drafts if d.platform == Platform.instagram]
    assert len(fb) == 3
    assert len(ig) == 3


def test_run_batch_isolates_per_source_failure_and_continues():
    # Two sources. Source 0's classify raises -> skipped + recorded.
    # Source 1 runs its full chain and produces drafts.
    # NB: LLMClient retries a transport error once (MAX_ATTEMPTS=2), so source
    # 0's classify must raise on BOTH attempts; the client then wraps it in an
    # LLMError. We assert on the wrapped message.
    llm = make_llm(
        RuntimeError("classify blew up for source 0"),  # source 0 classify attempt 1
        RuntimeError("classify blew up for source 0"),  # source 0 classify attempt 2
        _classify(),  # source 1 classify
        _extract(),  # source 1 extract
        *_platform_chain(1),  # source 1 facebook + claims check
        *_platform_chain(1),  # source 1 instagram + claims check
    )

    result = run_batch(
        [_source(0), _source(1)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=llm,
    )

    # The batch did NOT abort: source 1 still produced drafts.
    assert len(result.drafts) == 2
    # Per-source isolation holds for claim_checks too: only the surviving source's
    # drafts have verdicts, and the skipped source contributed none.
    assert set(result.claim_checks) == {d.id for d in result.drafts}
    assert all(d.id.startswith("s1-") for d in result.drafts)
    assert all(d.platform in (Platform.facebook, Platform.instagram) for d in result.drafts)

    # The failure is observable: one recorded error naming source 0.
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.source_id == "s0"
    assert err.stage == "classify"
    # The original transport error is preserved in the wrapped LLMError chain.
    assert "classify blew up" in str(err.error)


def test_run_batch_records_error_when_extract_fails():
    # classify succeeds, but extract's LLM call fails on both attempts ->
    # whole source skipped and recorded at the "extract" stage, batch continues.
    # Single source here, so drafts end up empty but the error is captured
    # (never raised out of run_batch).
    llm = make_llm(
        _classify("educational"),  # classify ok
        RuntimeError("extract transport down"),  # extract attempt 1
        RuntimeError("extract transport down"),  # extract attempt 2
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=llm,
    )

    assert result.drafts == []
    assert len(result.errors) == 1
    assert result.errors[0].source_id == "s0"
    assert result.errors[0].stage == "extract"
    assert result.claim_checks == {}


def test_source_error_error_field_typed_as_exception():
    # SourceError.error is an Exception (recoverable), not a BaseException —
    # the batch isolates Exceptions, but must never swallow KeyboardInterrupt /
    # SystemExit, so the type narrows to Exception.
    import typing

    from draftforge.pipeline import SourceError

    hints = typing.get_type_hints(SourceError)
    assert hints["error"] is Exception


def test_run_batch_empty_sources_returns_empty_result():
    llm = make_llm()  # no responses needed
    result = run_batch(
        [],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=4,
        register=[],
        llm=llm,
    )
    assert result.drafts == []
    assert result.errors == []
    assert result.claim_checks == {}


def test_run_batch_threads_classified_angle_onto_drafts():
    # The angle classify returns must reach Draft.angle (not a placeholder) so
    # downstream (review UI, calendar, claims gate) sees the real angle.
    llm = make_llm(
        _classify("myth_buster"),  # classify -> myth_buster
        _extract(),
        *_platform_chain(1),  # facebook + claims check
        *_platform_chain(1),  # instagram + claims check
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=llm,
    )

    assert result.errors == []
    assert len(result.drafts) == 2
    assert all(d.angle == "myth_buster" for d in result.drafts)


# --- Task 2.4: the claims-safety gate runs in the pipeline ----------------------


def test_run_batch_runs_claims_gate_for_every_draft():
    # Every produced draft gets a claim_checks entry keyed by its id; the verdict
    # is carried SEPARATELY from draft.status (lifecycle), which stays "draft".
    llm = make_llm(
        _classify(), _extract(), *_platform_chain(1), *_platform_chain(1)
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=llm,
    )

    assert len(result.drafts) == 2
    # One verdict per draft, keyed by draft.id.
    assert set(result.claim_checks) == {d.id for d in result.drafts}
    # The claim-safety status rides in claim_checks, NOT on draft.status.
    assert all(d.status == "draft" for d in result.drafts)
    assert all(c.status == "clean" for c in result.claim_checks.values())


def test_run_batch_flags_draft_with_an_unsafe_claim():
    # A draft whose claims-gate response trips the harmful pass routes that
    # draft's check to a non-clean status (flagged), while clean drafts stay clean
    # — the verdict is per-draft.
    llm = make_llm(
        _classify(),
        _extract(),
        # facebook: generate one draft, then a FLAGGED claims response for it
        _generate(1),
        _claims_flagged("asserts a disease cure that cannot be softened"),
        # instagram: generate one draft, then a CLEAN claims response
        _generate(1),
        _claims_clean(),
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        llm=llm,
    )

    assert len(result.drafts) == 2
    statuses = {c.status for c in result.claim_checks.values()}
    # One draft flagged, one clean — the gate did not blanket-pass.
    assert statuses == {"flagged", "clean"}
    fb = next(d for d in result.drafts if d.platform == Platform.facebook)
    assert result.claim_checks[fb.id].status == "flagged"


def test_run_batch_threads_register_into_the_claims_gate():
    # The register passed to run_batch reaches the gate: an APPROVED entry for the
    # caption's hard claim makes the gate resolve it clean (citation surfaced).
    register = [
        RegisterEntry(
            claim_text="lowers blood pressure",
            claim_type=ClaimType.hard,
            approved=True,
            source_citation="Smith 2020",
        )
    ]
    # The generated caption asserts the hard claim; the claims-gate response
    # inventories it as a hard claim. With the approved register entry, the gate
    # matches it and stays clean.
    hard_caption_gen = json.dumps(
        {
            "posts": [
                {
                    "caption": "Floating lowers blood pressure.",
                    "hashtags": ["#float"],
                    "image_direction": None,
                    "claims_used": [],
                }
            ]
        }
    )
    hard_claim_analysis = json.dumps(
        {
            "claims": [
                {
                    "text": "lowers blood pressure",
                    "claim_type": "hard",
                    "assertive": True,
                    "is_disease_treatment": False,
                }
            ],
            "harmful": False,
            "harmful_reason": "",
            "softened_caption": None,
        }
    )
    llm = make_llm(
        _classify(),
        _extract(),
        hard_caption_gen,  # facebook generate
        hard_claim_analysis,  # facebook claims check
    )

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=1,  # facebook only ([1, 0] distribution)
        register=register,
        platforms=(Platform.facebook,),
        llm=llm,
    )

    assert len(result.drafts) == 1
    check = result.claim_checks[result.drafts[0].id]
    # The approved register entry licensed the hard claim -> clean, with citation.
    assert check.status == "clean"
    assert any("Smith 2020" in n for n in check.notes)


def test_run_batch_respects_single_platform():
    llm = make_llm(_classify(), _extract(), *_platform_chain(2))

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=2,
        register=[],
        platforms=(Platform.facebook,),
        llm=llm,
    )

    assert len(result.drafts) == 2
    assert all(d.platform == Platform.facebook for d in result.drafts)


# --- batch-size distribution semantics ------------------------------------------


@pytest.mark.parametrize(
    "total, buckets, expected",
    [
        (1, 2, [1, 0]),  # remainder front-loads the first bucket; second gets 0
        (5, 2, [3, 2]),  # 2 each + 1 remainder to the leading bucket
        (0, 2, [0, 0]),  # nothing to give -> every bucket gets 0
        (3, 0, []),       # no buckets -> empty distribution
        (-3, 2, [0, 0]),  # negative total is clamped to 0 (max(total, 0))
    ],
)
def test_distribute_matches_implemented_semantics(total, buckets, expected):
    assert _distribute(total, buckets) == expected


def test_run_batch_batch_size_one_produces_only_facebook_drafts():
    # batch_size=1 over (facebook, instagram) distributes to [1, 0], so only the
    # leading platform (Facebook) is generated; Instagram's 0 share is skipped.
    # Hence only ONE generate call is made (facebook) + its one claims check.
    llm = make_llm(_classify(), _extract(), *_platform_chain(1))

    result = run_batch(
        [_source(0)],
        guidance=GUIDANCE,
        voice_exemplars=VOICE,
        corpus=CORPUS,
        batch_size=1,
        register=[],
        llm=llm,
    )

    assert result.errors == []
    assert len(result.drafts) == 1
    assert all(d.platform == Platform.facebook for d in result.drafts)


# --- DI-2: a duplicate draft id within a batch fails loud -----------------------


def test_run_batch_raises_on_duplicate_draft_id(monkeypatch):
    # Draft ids are unique by construction, so this is unreachable in practice —
    # but it is a safety control: if two drafts ever shared an id, the
    # claim_checks dict assignment would SILENTLY drop one draft's safety verdict.
    # The pipeline must raise loudly instead of overwriting. We force a collision
    # by stubbing generate_posts to return two drafts with the SAME id.
    import draftforge.pipeline as pipeline

    def _colliding_generate(*args, **kwargs):
        return [
            Draft(
                id="s0-facebook-0",  # SAME id for both -> collision
                platform=Platform.facebook,
                angle="educational",
                caption=f"caption {i}",
                hashtags=[],
            )
            for i in range(2)
        ]

    monkeypatch.setattr(pipeline, "generate_posts", _colliding_generate)

    # classify + extract still run (one each); generate is stubbed; the claims
    # gate would run but the duplicate is caught before the second check.
    llm = make_llm(_classify(), _extract(), _claims_clean())

    with pytest.raises(ValueError, match="duplicate draft id"):
        run_batch(
            [_source(0)],
            guidance=GUIDANCE,
            voice_exemplars=VOICE,
            corpus=CORPUS,
            batch_size=1,
            register=[],
            platforms=(Platform.facebook,),
            llm=llm,
        )
