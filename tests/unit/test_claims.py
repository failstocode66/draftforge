"""Tests for the claims-safety gate (Task 2.2) - the project's differentiator.

Fully offline: a fake transport pops a canned ``ClaimAnalysis`` JSON, so the LLM
"finding" of claims is deterministic and we test the gate's *policy* (register
match, soft/hard resolution, softening, the INT-5 cross-check, and - critically -
the fail-safe behavior when the LLM raises).

The gate must be FAIL-SAFE: it never silently passes a draft it could not check.
"""

from __future__ import annotations

import json

import pytest

from draftforge.llm.client import LLMClient
from draftforge.models import ClaimCheck, ClaimType, Draft, ExtractedItem, Platform, RegisterEntry
from draftforge.stages.claims import claims_check


class FakeTransport:
    """Pops canned response strings (or raises a canned Exception); records calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def text(self, *, model, system, user, max_tokens):
        self.calls.append(
            {"model": model, "system": system, "user": user, "max_tokens": max_tokens}
        )
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


def _analysis(claims, *, harmful=False, harmful_reason="", softened_caption=None):
    """Build a canned ClaimAnalysis JSON string."""
    return json.dumps(
        {
            "claims": claims,
            "harmful": harmful,
            "harmful_reason": harmful_reason,
            "softened_caption": softened_caption,
        }
    )


def _claim(text, *, claim_type="soft", assertive=False, is_disease_treatment=False):
    return {
        "text": text,
        "claim_type": claim_type,
        "assertive": assertive,
        "is_disease_treatment": is_disease_treatment,
    }


def _draft(caption, *, claims_used=None, hashtags=None, image_direction=None):
    return Draft(
        id="post-0",
        platform=Platform.instagram,
        angle="benefit_spotlight",
        caption=caption,
        hashtags=hashtags if hashtags is not None else ["#float"],
        image_direction=image_direction,
        claims_used=claims_used or [],
    )


# A soft-only extracted item (no hard claim) is the baseline source for most tests.
SOFT_ITEM = ExtractedItem(
    hook="Sink into stillness",
    core_benefit="deep relaxation",
    claim="many people report feeling relaxed",
    claim_type=ClaimType.soft,
)


def _register(*entries) -> list[RegisterEntry]:
    return list(entries)


def _entry(claim_text, *, claim_type="hard", approved=True, source_citation="Doe 2019", notes=""):
    return RegisterEntry(
        claim_text=claim_text,
        claim_type=claim_type,
        approved=approved,
        source_citation=source_citation,
        notes=notes,
    )


# --- clean: only soft claims ----------------------------------------------------


def test_clean_when_only_soft_claims():
    draft = _draft("Many people report feeling deeply relaxed after a float.")
    llm = make_llm(
        _analysis([_claim("feeling deeply relaxed", claim_type="soft")])
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert isinstance(result, ClaimCheck)
    assert result.status == "clean"
    assert result.revised_text is None


def test_clean_uses_smart_model():
    draft = _draft("A calm, quiet hour just for you.")
    transport = FakeTransport([_analysis([])])
    llm = LLMClient(transport, model_fast="fast", model_smart="smart", sleep=lambda *_: None)

    claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert transport.calls[0]["model"] == "smart"


def test_caption_is_sent_to_the_model():
    draft = _draft("Floating lowers blood pressure.")
    transport = FakeTransport([_analysis([])])
    llm = LLMClient(transport, model_fast="fast", model_smart="smart", sleep=lambda *_: None)

    claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    sent = transport.calls[0]["user"] + transport.calls[0]["system"]
    assert "Floating lowers blood pressure." in sent


# --- hard claim NOT in register -> softened -------------------------------------


def test_hard_claim_absent_from_register_is_softened():
    draft = _draft("Floating lowers your blood pressure.")
    llm = make_llm(
        _analysis(
            [_claim("lowers your blood pressure", claim_type="hard", assertive=True)],
            softened_caption="Floating may help support healthy blood pressure; many people report feeling calmer.",
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "softened"
    # The softened caption hedges the assertive medical phrasing.
    assert result.revised_text is not None
    assert result.revised_text != draft.caption
    lowered = result.revised_text.lower()
    assert "may" in lowered or "many people report" in lowered
    # A note names the claim and suggests hedging the flat assertion.
    joined = " ".join(result.notes).lower()
    assert "blood pressure" in joined
    assert "hedg" in joined or "flat medical fact" in joined or "may help" in joined


def test_softened_falls_back_to_hedged_text_when_model_gives_no_rewrite():
    # If the model returns no softened_caption, the gate still must NOT pass the
    # assertive caption as-is: it produces a hedged revised_text deterministically.
    draft = _draft("Floating reduces anxiety.")
    llm = make_llm(
        _analysis(
            [_claim("reduces anxiety", claim_type="hard", assertive=True)],
            softened_caption=None,
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "softened"
    assert result.revised_text is not None
    assert result.revised_text != draft.caption


# --- CS-5: a softened rewrite must not retain the original assertion -------------
# The model's "softened" rewrite is preferred for voice, but it must not slip the
# original assertive claim through unchanged. If the normalized assertive claim is
# still a substring of the rewrite, reject it and fall through to the deterministic
# hedge - the gate must never depend on the model to actually hedge.


def test_softened_rewrite_retaining_original_assertion_uses_hedge():
    draft = _draft("Floating lowers your blood pressure.")
    llm = make_llm(
        _analysis(
            [_claim("lowers your blood pressure", claim_type="hard", assertive=True)],
            # A "rewrite" that bolts on a hedge but KEEPS the original assertion.
            softened_caption=(
                "Floating lowers your blood pressure, and many people feel calmer."
            ),
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "softened"
    assert result.revised_text is not None
    # The model's rewrite still contained the assertion verbatim, so the gate must
    # NOT use it - it falls through to the deterministic hedge.
    assert result.revised_text != (
        "Floating lowers your blood pressure, and many people feel calmer."
    )
    # The deterministic hedge marker proves the fallback was taken.
    assert "[Needs review" in result.revised_text


def test_softened_rewrite_that_actually_hedges_is_kept():
    # Regression: a genuine rewrite that DROPS the assertive phrasing is kept.
    draft = _draft("Floating lowers your blood pressure.")
    rewrite = "Many people report feeling calmer and more relaxed after a float."
    llm = make_llm(
        _analysis(
            [_claim("lowers your blood pressure", claim_type="hard", assertive=True)],
            softened_caption=rewrite,
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "softened"
    assert result.revised_text == rewrite


# --- hard claim present + approved in register -> clean --------------------------


def test_hard_claim_approved_in_register_is_clean():
    draft = _draft("Magnesium is absorbed through the skin during your float.")
    register = _register(
        _entry(
            "magnesium is absorbed through the skin",
            claim_type="hard",
            source_citation="Waring 2006, magnesium uptake study",
        )
    )
    llm = make_llm(
        _analysis(
            [_claim("magnesium is absorbed through the skin", claim_type="hard", assertive=True)]
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=register, llm=llm)

    assert result.status == "clean"
    assert result.revised_text is None
    # The supporting citation is surfaced in the notes.
    assert any("Waring 2006" in n for n in result.notes)


def test_hard_claim_in_register_but_unapproved_is_softened():
    draft = _draft("Floating lowers your blood pressure.")
    register = _register(
        _entry("lowers your blood pressure", claim_type="hard", approved=False)
    )
    llm = make_llm(
        _analysis(
            [_claim("lowers your blood pressure", claim_type="hard", assertive=True)],
            softened_caption="Floating may help you feel calmer.",
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=register, llm=llm)

    # An unapproved register row must NOT license the claim.
    assert result.status == "softened"


# --- disease/treatment cure claim -> cannot safely soften -----------------------


def test_disease_treatment_claim_escalates():
    draft = _draft("Floating cures depression.")
    llm = make_llm(
        _analysis(
            [
                _claim(
                    "cures depression",
                    claim_type="hard",
                    assertive=True,
                    is_disease_treatment=True,
                )
            ]
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status in {"flagged", "needs_manual_review"}
    joined = " ".join(result.notes).lower()
    assert "depression" in joined or "treat" in joined or "cure" in joined


# --- harmful content pass -------------------------------------------------------


def test_harmful_content_is_flagged():
    draft = _draft("Stop taking your prescribed medication and just float instead.")
    llm = make_llm(
        _analysis(
            [],
            harmful=True,
            harmful_reason="advises stopping prescribed medication",
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status in {"flagged", "needs_manual_review"}
    assert any("medication" in n.lower() or "harmful" in n.lower() for n in result.notes)


# --- FAIL-SAFE: the llm raising -> needs_manual_review ---------------------------


def test_fail_safe_when_llm_raises():
    draft = _draft("Floating lowers your blood pressure.")
    # The transport raises on BOTH attempts so the client exhausts retries and the
    # gate's wrapper catches the resulting LLMError.
    llm = make_llm(RuntimeError("boom"), RuntimeError("boom again"))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "needs_manual_review"
    assert result.revised_text is None
    # Never a silent pass: the note explains the gate could not check the draft.
    assert any(
        "could not" in n.lower() or "manual" in n.lower() or "error" in n.lower()
        for n in result.notes
    )


def test_fail_safe_on_malformed_model_output():
    # Model returns garbage that never validates -> LLMError -> fail-safe.
    draft = _draft("Floating reduces anxiety.")
    llm = make_llm("not json at all", "still not json")

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "needs_manual_review"


# --- INT-5 cross-check ----------------------------------------------------------


def test_int5_generator_invented_hard_claim_escalates():
    # The extracted item carries only a SOFT claim, but the generator's
    # claims_used self-reports a HARD medical claim that was never extracted ->
    # generator-invented hard claim -> needs_manual_review.
    draft = _draft(
        "Floating is a calm hour.",
        claims_used=["floating lowers blood pressure"],
    )
    llm = make_llm(_analysis([_claim("a calm hour", claim_type="soft")]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "needs_manual_review"
    joined = " ".join(result.notes).lower()
    assert "blood pressure" in joined
    assert "extract" in joined or "invent" in joined or "introduc" in joined


def test_int5_hard_claims_used_with_extracted_claim_none_escalates():
    # P2-4: the extracted item carries NO claim at all (claim=None). A hard claim in
    # claims_used cannot have come from the (absent) source, so it must escalate -
    # and the match against the empty/None extracted claim must NOT false-positive.
    item = ExtractedItem(
        hook="Sink into stillness",
        core_benefit="deep relaxation",
        claim=None,
        claim_type=None,
    )
    draft = _draft(
        "Floating is a calm hour.",
        claims_used=["floating lowers blood pressure"],
    )
    llm = make_llm(_analysis([_claim("a calm hour", claim_type="soft")]))

    result = claims_check(draft, extracted_item=item, register=[], llm=llm)

    assert result.status == "needs_manual_review"
    joined = " ".join(result.notes).lower()
    assert "blood pressure" in joined


def test_int5_claims_used_matching_extracted_soft_claim_is_fine():
    # claims_used echoing the extracted soft claim is NOT an INT-5 escalation.
    draft = _draft(
        "Many people report feeling relaxed.",
        claims_used=["many people report feeling relaxed"],
    )
    llm = make_llm(_analysis([_claim("feeling relaxed", claim_type="soft")]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "clean"


def test_int5_extracted_hard_claim_in_claims_used_is_not_invented():
    # When the extracted item itself carries the hard claim, claims_used repeating
    # it is faithful (not invented). With it approved in the register -> clean.
    item = ExtractedItem(
        hook="Magnesium soak",
        core_benefit="recovery",
        claim="magnesium is absorbed through the skin",
        claim_type=ClaimType.hard,
    )
    draft = _draft(
        "Magnesium is absorbed through the skin during your float.",
        claims_used=["magnesium is absorbed through the skin"],
    )
    register = _register(
        _entry("magnesium is absorbed through the skin", claim_type="hard",
               source_citation="Waring 2006")
    )
    llm = make_llm(
        _analysis(
            [_claim("magnesium is absorbed through the skin", claim_type="hard")]
        )
    )

    result = claims_check(draft, extracted_item=item, register=register, llm=llm)

    assert result.status == "clean"


# --- bypass resistance: deterministic full-caption hard-term rescan -------------
# The safety guarantee must NOT lean on the LLM's COMPLETENESS. A fooled/failing
# model can OMIT a hard claim from its inventory entirely (return claims:[] or a
# soft-only list) for a caption that plainly contains a hard medical term. The
# per-claim loop only runs on what the model reported, so without a rescan such a
# caption would exit `clean`. The deterministic rescan closes that one surviving
# path to `clean` against a misbehaving model -> needs_manual_review.


def test_omitted_hard_claim_empty_list_escalates_not_clean():
    # The caption plainly contains a hard medical term, but the model returns an
    # EMPTY claim list (omission / bypass). This must NOT pass as clean.
    draft = _draft("Floating lowers your blood pressure.")
    llm = make_llm(_analysis([]))  # model inventoried nothing

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    # Relaxed policy: an omitted FLAT (non-disease, non-quantified) assertion -> softened
    # (suggest a hedge). It must NOT pass as clean - bypass-resistance is preserved.
    assert result.status == "softened"
    joined = " ".join(result.notes).lower()
    # The note explains the analysis failed to inventory a hard term in the caption.
    assert "inventor" in joined or "wasn't" in joined or "hedg" in joined


def test_omitted_hard_claim_soft_only_list_escalates():
    # The model lists only an unrelated SOFT claim while the caption contains a
    # hard term it never inventoried -> still escalate.
    draft = _draft("Our tanks are spacious, and floating reduces anxiety.")
    llm = make_llm(_analysis([_claim("our tanks are spacious", claim_type="soft")]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    # Relaxed: the omitted flat "reduces anxiety" conjunct -> softened (not clean).
    assert result.status == "softened"


def test_rescan_does_not_false_positive_on_benign_caption():
    # Regression: a caption with NO hard medical term + an empty/soft claim list
    # must still be clean. The rescan must not fire on benign copy.
    draft = _draft("A calm, quiet hour just for you. Sink into stillness.")
    llm = make_llm(_analysis([]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "clean"


def test_rescan_does_not_override_a_handled_hard_claim():
    # When the model DID surface the hard claim (and the gate softened it), the
    # rescan must not also fire -> status stays `softened`, not escalated.
    draft = _draft("Floating lowers your blood pressure.")
    llm = make_llm(
        _analysis(
            [_claim("lowers your blood pressure", claim_type="hard", assertive=True)],
            softened_caption="Floating may help you feel calmer.",
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "softened"


# --- CS-1: per-clause accounting (a SECOND omitted hard claim slips the rescan) --
# The old rescan was a single boolean: it fired only when is_hard_medical(caption)
# AND no analyzed claim was hard. If the model inventories ONE hard claim but OMITS
# a SECOND, `any_hard_inventoried` was True, the rescan was suppressed, and the
# omitted disease-cure clause was silently downgraded to `softened`, un-named, and
# could survive verbatim in revised_text. The backstop must be per-clause and
# independent of the model's COMPLETENESS.


def test_second_omitted_hard_claim_escalates_not_softened():
    # Two hard clauses; the model reports ONLY the blood-pressure one. The
    # depression cure clause is never inventoried -> must escalate to manual
    # review (NOT be silently softened), be NAMED, and not survive in revised_text.
    draft = _draft("Floating lowers blood pressure. Floating cures depression.")
    llm = make_llm(
        _analysis(
            [_claim("lowers blood pressure", claim_type="hard", assertive=True)],
            softened_caption=(
                "Many people report feeling calmer after a float. "
                "Floating cures depression."
            ),
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "needs_manual_review"
    joined = " ".join(result.notes).lower()
    assert "depression" in joined or "cures depression" in joined
    # The uncovered disease-cure clause must NOT survive verbatim in any revised
    # text the gate would hand back.
    if result.revised_text is not None:
        assert "cures depression" not in result.revised_text.lower()


def test_per_clause_backstop_passes_when_every_hard_clause_is_inventoried():
    # Regression: when the model DOES inventory every hard clause (and the gate
    # softens them), the per-clause backstop must not fire -> stays `softened`.
    draft = _draft("Floating lowers blood pressure. Floating reduces anxiety.")
    llm = make_llm(
        _analysis(
            [
                _claim("lowers blood pressure", claim_type="hard", assertive=True),
                _claim("reduces anxiety", claim_type="hard", assertive=True),
            ],
            softened_caption=(
                "Many people report feeling calmer and less anxious after a float."
            ),
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "softened"


def test_per_clause_backstop_clean_when_inventoried_hard_clause_is_approved():
    # A hard clause covered by an APPROVED register entry must keep the draft clean
    # even though the per-clause scan sees the hard term (the clause IS inventoried
    # and approved). Guards against the backstop double-counting approved claims.
    draft = _draft("Magnesium is absorbed through the skin during your float.")
    register = _register(
        _entry("magnesium is absorbed through the skin", claim_type="hard",
               source_citation="Waring 2006")
    )
    llm = make_llm(
        _analysis(
            [_claim("magnesium is absorbed through the skin", claim_type="hard")]
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=register, llm=llm)

    assert result.status == "clean"


# --- CS-3 part 2: the backstop scans hashtags + image_direction, not just caption
# Hard claims smuggled into hashtags ("#LowersBloodPressure") or image_direction
# reach `clean` because the gate only inspected draft.caption.


def test_hard_claim_only_in_hashtag_escalates_not_clean():
    # Caption is benign; the hard claim hides in a hashtag. Must NOT be clean.
    draft = _draft(
        "A calm, quiet hour just for you.",
        hashtags=["#float", "#LowersBloodPressure"],
    )
    llm = make_llm(_analysis([_claim("a calm hour", claim_type="soft")]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    # Relaxed: a flat hard claim in a hashtag -> softened (not blocked, not clean).
    assert result.status == "softened"
    joined = " ".join(result.notes).lower()
    assert "hashtag" in joined or "blood pressure" in joined


def test_hard_claim_only_in_image_direction_escalates_not_clean():
    # Caption + hashtags are benign; the hard claim hides in image_direction.
    draft = _draft(
        "Sink into stillness.",
        hashtags=["#float"],
        image_direction="Overlay text: floating cures insomnia.",
    )
    llm = make_llm(_analysis([_claim("sink into stillness", claim_type="soft")]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "needs_manual_review"
    joined = " ".join(result.notes).lower()
    assert "image" in joined or "insomnia" in joined or "direction" in joined


def test_benign_hashtags_and_image_direction_stay_clean():
    # Regression: benign hashtags/image_direction must not trip the backstop.
    draft = _draft(
        "A calm, quiet hour just for you.",
        hashtags=["#float", "#relax", "#selfcare"],
        image_direction="Soft warm lighting over a calm float tank.",
    )
    llm = make_llm(_analysis([_claim("a calm hour", claim_type="soft")]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "clean"


# --- CS-7: compound-clause coverage ---------------------------------------------
# A sentence-clause can carry MULTIPLE hard assertions joined by a conjunction.
# The per-clause backstop matched coverage against the WHOLE clause, so when the
# model inventoried ONE conjunct ("reduces cortisol") its substring "covered" the
# whole "reduces cortisol and lowers your blood pressure" clause - leaving the
# un-inventoried "lowers your blood pressure" conjunct softened but un-hedged in
# revised_text. Each hard SUB-assertion must be individually covered.


def test_compound_clause_uninventoried_conjunct_escalates():
    # "reduces cortisol AND lowers your blood pressure"; model reports only the
    # cortisol conjunct -> the BP conjunct is un-inventoried -> escalate (not just
    # softened with the assertion surviving).
    draft = _draft("Floating reduces cortisol and lowers your blood pressure.")
    llm = make_llm(
        _analysis(
            [_claim("reduces cortisol", claim_type="hard", assertive=True)],
            softened_caption=(
                "Floating may help you relax and lowers your blood pressure."
            ),
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    # Relaxed: the uninventoried flat BP conjunct -> softened, named in notes (not clean).
    assert result.status == "softened"
    joined = " ".join(result.notes).lower()
    assert "blood pressure" in joined


def test_compound_clause_both_conjuncts_inventoried_does_not_over_escalate():
    # When BOTH hard conjuncts are inventoried (and here approved in the register),
    # the per-sub-clause backstop must NOT over-escalate -> stays clean.
    draft = _draft("Floating reduces cortisol and lowers your blood pressure.")
    register = _register(
        _entry("reduces cortisol", claim_type="hard", source_citation="A 2019"),
        _entry("lowers your blood pressure", claim_type="hard",
               source_citation="B 2020"),
    )
    llm = make_llm(
        _analysis(
            [
                _claim("reduces cortisol", claim_type="hard"),
                _claim("lowers your blood pressure", claim_type="hard"),
            ]
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=register, llm=llm)

    assert result.status == "clean"


def test_compound_clause_both_conjuncts_inventoried_softens_without_over_escalation():
    # Both hard conjuncts inventoried but UNapproved -> softened (not escalated).
    draft = _draft("Floating reduces cortisol and lowers your blood pressure.")
    llm = make_llm(
        _analysis(
            [
                _claim("reduces cortisol", claim_type="hard", assertive=True),
                _claim("lowers your blood pressure", claim_type="hard", assertive=True),
            ],
            softened_caption=(
                "Many people report feeling calmer and more relaxed after a float."
            ),
        )
    )

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "softened"


def test_benign_compound_clause_stays_clean():
    # "relax and unwind and let go" - no hard sub-assertion -> must stay clean,
    # no false hard escalation from the conjunction split.
    draft = _draft("Relax and unwind and let go.")
    llm = make_llm(_analysis([_claim("relax and unwind", claim_type="soft")]))

    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)

    assert result.status == "clean"


# --- relaxed, hedge-aware, ADVISORY policy (2026-06-26 owner decision) ----------
# Holistic-wellness domain + review-gated tool -> the gate ADVISES, not blocks.
# Permissive by default; a hedged physiological claim is `advisory` (noted, never
# blocked, text untouched); only disease/quantified/flat-assertion/harmful escalate.


def test_owner_example_1_inner_calm_is_clean():
    draft = _draft("A float can help you feel a sense of inner calm.")
    llm = make_llm(_analysis([_claim("feel a sense of inner calm", claim_type="soft")]))
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "clean"


def test_owner_example_2_hedged_physiological_is_advisory():
    # The key case: a HEDGED physiological claim is advisory - not blocked, text
    # untouched, but NOT certified clean (a hedge is not a substantiation shield).
    draft = _draft("A float may help to reduce inflammation and chronic pain.")
    llm = make_llm(
        _analysis(
            [_claim("may help to reduce inflammation and chronic pain", claim_type="hard")]
        )
    )
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "advisory"
    assert result.revised_text is None  # text is untouched; nothing is rewritten
    joined = " ".join(result.notes).lower()
    assert "not blocked" in joined or "heads up" in joined


def test_owner_example_3_reset_your_mind_is_clean():
    draft = _draft("A float can be a good way to reset your mind.")
    llm = make_llm(_analysis([_claim("a good way to reset your mind", claim_type="soft")]))
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "clean"


def test_owner_example_4_relieve_pressure_is_clean():
    draft = _draft("A float can help relieve pressure both physical and mental.")
    llm = make_llm(
        _analysis([_claim("relieve pressure both physical and mental", claim_type="soft")])
    )
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "clean"


def test_hedged_physiological_text_is_untouched_advisory():
    # A hedged BP claim -> advisory; the tool does NOT rewrite it (revised_text None).
    draft = _draft("Floating can help lower your blood pressure.")
    llm = make_llm(
        _analysis([_claim("can help lower your blood pressure", claim_type="hard")])
    )
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "advisory"
    assert result.revised_text is None


def test_hedged_disease_claim_still_flagged():
    # A HEDGED disease-cure claim still escalates - the hedge does not relax the
    # disease category (highest liability).
    draft = _draft("A float may cure your depression.")
    llm = make_llm(
        _analysis([_claim("may cure your depression", claim_type="hard")])
    )
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status in {"flagged", "needs_manual_review"}
    assert any("depression" in n.lower() or "disease" in n.lower() or "cure" in n.lower()
               for n in result.notes)


def test_quantified_medical_claim_is_flagged_even_if_hedged():
    # A quantified medical outcome needs a citation regardless of hedging.
    draft = _draft("Floating may reduce inflammation by 40%.")
    llm = make_llm(
        _analysis([_claim("may reduce inflammation by 40%", claim_type="hard")])
    )
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status in {"flagged", "needs_manual_review"}
    assert any("quantif" in n.lower() or "citation" in n.lower() or "number" in n.lower()
               for n in result.notes)


def test_flat_assertion_still_softened():
    # An UNHEDGED definitive medical assertion is still softened (suggest a hedge).
    draft = _draft("Floating lowers your blood pressure.")
    llm = make_llm(
        _analysis(
            [_claim("lowers your blood pressure", claim_type="hard")],
            softened_caption="Floating may help you feel calmer.",
        )
    )
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "softened"


def test_omitted_disease_clause_still_needs_manual_review():
    # Bypass-resistance for the RISKY set is retained: an omitted DISEASE-cure clause
    # still escalates to manual review even under the relaxed policy.
    draft = _draft("Relax in our calm space. Floating cures insomnia.")
    llm = make_llm(_analysis([_claim("relax in our calm space", claim_type="soft")]))
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "needs_manual_review"


def test_omitted_hedged_clause_only_advisory_no_over_escalation():
    # A compound caption whose omitted conjunct is HEDGED-physiological -> at most
    # advisory (never needs_manual_review); no over-firing on benign hedged copy.
    draft = _draft("Our rooms are private and floating may help ease your anxiety.")
    llm = make_llm(_analysis([_claim("our rooms are private", claim_type="soft")]))
    result = claims_check(draft, extracted_item=SOFT_ITEM, register=[], llm=llm)
    assert result.status == "advisory"


# --- prompt wiring --------------------------------------------------------------


def test_prompt_has_baseline_comparison_section():
    from draftforge.stages import load_prompt

    prompt = load_prompt("claims.md")
    assert "Baseline Comparison" in prompt
