"""Claims-safety gate (Task 2.2) - the project's differentiator.

This stage sits between generation and the human review queue. For one
:class:`~draftforge.models.Draft` it decides whether the draft's health/wellness
claims may be published as written, softens uncited assertive medical phrasing,
flags claims it cannot safely soften, and - critically - is **fail-safe**: it
NEVER silently passes a draft it could not check.

Division of labor (deliberate): the **LLM observes**, the **Python decides.**

* The LLM (smart model, ``fast=False``) does one structured-analysis call from
  ``prompts/claims.md``: it inventories each claim in ``draft.caption``,
  classifies each soft/hard, notes whether it is stated assertively and whether it
  is a disease-treatment claim, runs a generic harmful-content pass, and supplies a
  hedged rewrite of the whole caption.
* The **deterministic policy lives here in Python** (so a prompt edit can't loosen
  it): the register match (:func:`~draftforge.stages.claims_register.match_claim`,
  approved-only), the soft/hard resolution (backstopped by the
  :func:`~draftforge.stages.claims_register.is_hard_medical` heuristic), the INT-5
  ``claims_used`` cross-check, and the status decision.

Status (severity order, highest wins): ``needs_manual_review`` > ``flagged`` >
``softened`` > ``advisory`` > ``clean``.

Policy (relaxed, hedge-aware, ADVISORY - 2026-06-26 owner decision). This is a
holistic-wellness business, not a medical service, and the tool is review-gated
(nothing auto-publishes), so the gate advises rather than blocks. Permissive by
default; only the genuinely risky categories escalate:
  * soft / pure-experiential claim -> ``clean``.
  * hard claim matched to an APPROVED register entry -> ``clean`` (citation noted).
  * disease cure/treat/prevent claim (even hedged) -> ``flagged`` (highest liability).
  * quantified medical outcome ("by 40%") not approved -> ``flagged`` (needs a citation).
  * HEDGED physiological claim ("may help reduce inflammation") -> ``advisory``: NOT
    blocked, text untouched, NOT certified clean (a hedge is not a substantiation
    shield) - a light note for the reviewer.
  * flat, unhedged, definitive medical assertion -> ``softened`` (a hedged
    ``revised_text`` is proposed; not blocked).
  * harmful/inappropriate content -> ``flagged``.
  * INT-5: a HARD claim in ``draft.claims_used`` the source ``extracted_item`` did
    not carry (generator-invented) -> ``needs_manual_review``.
  * deterministic backstop: a hard clause the model OMITTED -> its risk-appropriate
    lane (disease/quantified -> ``needs_manual_review``; hedged -> ``advisory``;
    flat -> ``softened``).
  * any error / the LLM raising -> ``needs_manual_review`` (fail-safe).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from draftforge.llm.client import LLMClient
from draftforge.models import ClaimCheck, Draft, ExtractedItem, RegisterEntry
from draftforge.stages import load_prompt
import re

from draftforge.stages.claims_register import (
    _normalize,
    is_hard_medical,
    is_hedged,
    match_claim,
)

logger = logging.getLogger(__name__)

_PROMPT_FILE = "claims.md"

# Status constants + severity ordering. The final status is the most severe one
# any check raised; ``clean`` is the floor.
#
# Relaxed wellness-domain policy (2026-06-26, owner decision): this is a holistic-
# wellness business (not a medical service) and the tool is review-gated (nothing
# auto-publishes), so the gate is an ADVISOR, not a blocker. Permissive by default;
# only the genuinely risky categories escalate. The ``advisory`` lane is the key
# addition - a hedged physiological claim ("may help reduce inflammation") is NOT
# blocked and its text is untouched, but the tool does not certify it ``clean``
# either (a hedge is not a substantiation shield); it notes it for the reviewer.
_CLEAN = "clean"
_ADVISORY = "advisory"
_SOFTENED = "softened"
_FLAGGED = "flagged"
_NEEDS_REVIEW = "needs_manual_review"
_SEVERITY = {_CLEAN: 0, _ADVISORY: 1, _SOFTENED: 2, _FLAGGED: 3, _NEEDS_REVIEW: 4}

# A number/percent attached to a medical effect is a quantified claim (needs a
# citation regardless of hedging).
_NUMBER_RE = re.compile(r"\d")


def _is_quantified_medical(text: str) -> bool:
    """True if a number/percent is attached to a medical-effect claim.

    A quantified medical outcome ("reduces inflammation by 40%", "98% effective")
    is a specific, substantiation-needing assertion regardless of hedging - it
    needs a citation, so it is flagged unless register-approved.
    """
    return bool(_NUMBER_RE.search(text)) and is_hard_medical(text)


class AnalyzedClaim(BaseModel):
    """One claim the model found in the caption (its observation, not a verdict)."""

    text: str
    claim_type: str  # "soft" | "hard" (validated leniently; Python re-checks)
    assertive: bool = False
    is_disease_treatment: bool = False


class ClaimAnalysis(BaseModel):
    """The model's structured analysis of one caption (schema for the LLM call)."""

    claims: list[AnalyzedClaim] = Field(default_factory=list)
    harmful: bool = False
    harmful_reason: str = ""
    softened_caption: str | None = None


def claims_check(
    draft: Draft,
    *,
    extracted_item: ExtractedItem,
    register: list[RegisterEntry],
    llm: LLMClient,
) -> ClaimCheck:
    """Evaluate ``draft`` against the approved-claims register. Fail-safe.

    Args:
        draft: The generated draft to gate (its ``caption`` is analyzed; its
            ``claims_used`` self-report is cross-checked).
        extracted_item: The authoritative source the draft was generated from;
            used for the INT-5 cross-check (its ``claim``/``claim_type`` is the
            ground truth a generator-invented hard claim is measured against).
        register: The approved-claims register (from ``load_claims_register``).
        llm: The schema-validating LLM client (routed to the smart model).

    Returns:
        A :class:`~draftforge.models.ClaimCheck`. On ANY error (the LLM raising,
        malformed output, an unexpected bug) it returns
        ``status="needs_manual_review"`` with an explanatory note and no
        ``revised_text`` - it never silently passes an unchecked draft.
    """
    try:
        return _claims_check_inner(
            draft, extracted_item=extracted_item, register=register, llm=llm
        )
    except Exception as exc:  # FAIL-SAFE: never silently pass an unchecked draft.
        logger.warning(
            "claims_check: could not check draft %s (%s: %s); routing to manual review.",
            draft.id,
            type(exc).__name__,
            exc,
        )
        return ClaimCheck(
            status=_NEEDS_REVIEW,
            notes=[
                "The claims-safety gate could not check this draft "
                f"({type(exc).__name__}). Manual review required before publishing "
                "- the draft has NOT been cleared."
            ],
            revised_text=None,
        )


def _claims_check_inner(
    draft: Draft,
    *,
    extracted_item: ExtractedItem,
    register: list[RegisterEntry],
    llm: LLMClient,
) -> ClaimCheck:
    """The real policy. Wrapped by :func:`claims_check` for fail-safety."""
    system = load_prompt(_PROMPT_FILE)
    analysis = llm.complete_json(
        system, draft.caption, ClaimAnalysis, fast=False, max_tokens=2000
    )

    notes: list[str] = []
    status = _CLEAN
    # The assertive hard-claim texts we softened; the rewrite must not retain them
    # (CS-5). Collected during the per-claim loop, checked at revised_text assembly.
    softened_claims: list[str] = []

    # --- generic harmful-content pass -------------------------------------------
    if analysis.harmful:
        reason = analysis.harmful_reason.strip() or "flagged as harmful/inappropriate"
        notes.append(f"Harmful/inappropriate content: {reason}. Manual review required.")
        status = _escalate(status, _FLAGGED)

    # --- per-claim resolution (relaxed, hedge-aware, advisory) ------------------
    # Wellness domain + review-gated tool -> the gate ADVISES, it does not block.
    # Permissive by default; only the genuinely risky categories escalate, and even
    # then advisorily. Precedence: approved -> disease -> quantified -> hedged ->
    # flat assertion. (A hedged DISEASE/quantified claim still escalates - the hedge
    # only relaxes a plain physiological claim.)
    for claim in analysis.claims:
        # Hard if EITHER the model says hard OR the deterministic heuristic flags
        # it (recall-biased backstop: the safety floor never depends on the model).
        hard = claim.claim_type == "hard" or is_hard_medical(claim.text)
        if not hard:
            continue  # pure experiential / non-health copy -> clean.

        approved = match_claim(claim.text, register)
        if approved is not None:
            # Register-approved -> clean; surface the source.
            citation = approved.source_citation.strip()
            if citation:
                notes.append(f"Claim '{claim.text}' is approved (source: {citation}).")
            else:
                notes.append(f"Claim '{claim.text}' is approved in the register.")
            continue

        # 1) Disease cure/treat/prevent (even hedged) -> the highest-liability class.
        disease = claim.is_disease_treatment or _looks_like_disease_treatment(claim.text)
        if disease:
            notes.append(
                f"'{claim.text}' reads as a disease cure/treatment/prevention claim "
                "- the highest-liability category for an unlicensed wellness service. "
                "Consider rephrasing or citing a source. Flagged for your review."
            )
            status = _escalate(status, _FLAGGED)
            continue

        # 2) Quantified medical outcome -> needs a citation.
        if _is_quantified_medical(claim.text):
            notes.append(
                f"'{claim.text}' makes a specific, quantified health claim. A number "
                "like this needs a citation to be defensible. Flagged for your review "
                "(add a sourced, approved register entry to publish it as stated)."
            )
            status = _escalate(status, _FLAGGED)
            continue

        # 3) Hedged physiological / health claim -> ADVISORY: not blocked, text
        #    untouched, NOT certified clean (a hedge is not a substantiation shield).
        if is_hedged(claim.text):
            notes.append(
                f"Heads up: '{claim.text}' makes a wellness/health claim. It's hedged "
                "and not blocked - just make sure you're comfortable with the basis "
                "for it. (Add a sourced register entry if you'd like it cited.)"
            )
            status = _escalate(status, _ADVISORY)
            continue

        # 4) Flat, unhedged, definitive medical assertion -> suggest a hedge (not
        #    blocked; a softened version is offered below for one-click use).
        notes.append(
            f"'{claim.text}' is stated as a flat medical fact. Suggest hedging it "
            "(e.g. 'may help...'); a softened version is proposed below. Not blocked."
        )
        softened_claims.append(claim.text)
        status = _escalate(status, _SOFTENED)

    # --- INT-5 cross-check: did the generator INVENT a hard claim? --------------
    int5_status, int5_notes = _int5_cross_check(draft, extracted_item)
    notes.extend(int5_notes)
    status = _escalate(status, int5_status)

    # --- bypass resistance: deterministic per-CLAUSE hard-term rescan -----------
    # The per-claim loop above only acts on claims the MODEL reported, so it
    # catches the model MISLABELING a listed claim - but not the model OMITTING a
    # hard claim from its inventory entirely. The safety guarantee must not lean on
    # the LLM's COMPLETENESS, so we deterministically rescan every clause of the
    # draft's published surfaces (caption + hashtags + image_direction, CS-3) and,
    # for each clause that reads as a hard medical claim, check whether any
    # INVENTORIED hard claim covers it (CS-1). A single boolean ("did the model
    # report ANY hard claim?") was not enough: a model that reports ONE hard claim
    # but omits a SECOND would suppress the backstop and let the omitted clause be
    # silently softened. Per-clause accounting makes the backstop independent of
    # the model's completeness for EVERY hard clause. Any uncovered hard clause ->
    # needs_manual_review (we could not parse/cover the claim, so we cannot soften
    # it reliably) and the clause + where it was found is named.
    inventoried_hard_norms = [
        _normalize(c.text)
        for c in analysis.claims
        if c.claim_type == "hard" or is_hard_medical(c.text)
    ]
    inventoried_hard_norms = [n for n in inventoried_hard_norms if n]

    uncovered = _uncovered_hard_clauses(draft, inventoried_hard_norms)
    for where, clause in uncovered:
        # The model OMITTED this hard clause from its inventory. Assign the lane its
        # risk warrants: disease/quantified keep the strong safety net; a merely
        # hedged omitted claim is only advisory (no over-firing on benign copy).
        if _looks_like_disease_treatment(clause) or _is_quantified_medical(clause):
            notes.append(
                f"A disease/quantified health claim in the {where} ('{clause}') was "
                "not inventoried by the analysis and could not be parsed; flagged for "
                "manual review - the draft has NOT been cleared."
            )
            status = _escalate(status, _NEEDS_REVIEW)
        elif is_hedged(clause):
            notes.append(
                f"Heads up: a hedged health claim in the {where} ('{clause}') wasn't "
                "fully parsed by the analysis - not blocked; just have a look."
            )
            status = _escalate(status, _ADVISORY)
        else:
            notes.append(
                f"A flat medical claim in the {where} ('{clause}') wasn't inventoried "
                "by the analysis; suggest hedging it (a softened version is below)."
            )
            status = _escalate(status, _SOFTENED)

    # --- assemble revised_text only when we actually softened -------------------
    revised_text = None
    if status == _SOFTENED:
        revised_text = _resolve_softened_caption(
            draft.caption, analysis.softened_caption, softened_claims
        )

    return ClaimCheck(status=status, notes=notes, revised_text=revised_text)


def _split_clauses(text: str) -> list[str]:
    """Split free text into clauses on sentence-ending punctuation."""
    return [c.strip() for c in re.split(r"[.!?]+", text) if c.strip()]


# Coordinating conjunctions + list separators a single sentence-clause can chain
# multiple independent assertions on. The per-clause coverage check splits on
# these so each hard SUB-assertion must be covered individually (CS-7).
_SUBCLAUSE_SPLIT = re.compile(r"\b(?:and|but|or|nor)\b|[;,]", re.IGNORECASE)


def _split_subclauses(clause: str) -> list[str]:
    """Split one sentence-clause into sub-clauses on conjunctions / list separators.

    "reduces cortisol and lowers your blood pressure" -> ["reduces cortisol",
    "lowers your blood pressure"], so a hard conjunct the model never inventoried
    cannot hide behind an inventoried sibling that "covers" the whole clause.
    """
    return [p.strip() for p in _SUBCLAUSE_SPLIT.split(clause) if p.strip()]


def _scan_surfaces(draft: Draft) -> list[tuple[str, str]]:
    """Yield (where, clause) for every clause across the draft's published text.

    The deterministic backstop must inspect EVERY surface a claim can be published
    on, not just the caption (CS-3): hashtags ("#LowersBloodPressure") and
    image_direction ("Overlay text: floating cures insomnia") reach the audience
    too. Each hashtag is its own clause; caption + image_direction are split on
    sentence punctuation.
    """
    surfaces: list[tuple[str, str]] = []
    for clause in _split_clauses(draft.caption):
        surfaces.append(("caption", clause))
    for tag in draft.hashtags:
        tag = tag.strip()
        if tag:
            surfaces.append(("hashtags", tag))
    if draft.image_direction:
        for clause in _split_clauses(draft.image_direction):
            surfaces.append(("image_direction", clause))
    return surfaces


def _uncovered_hard_clauses(
    draft: Draft, inventoried_hard_norms: list[str]
) -> list[tuple[str, str]]:
    """Return (where, sub-clause) for each hard sub-assertion NOT covered by an
    inventoried claim.

    Each sentence-clause is split further on coordinating conjunctions / list
    separators (CS-7) so every hard SUB-assertion is checked individually - a hard
    conjunct the model never inventoried ("...and lowers your blood pressure")
    cannot hide behind an inventoried sibling that substring-"covers" the whole
    clause. A sub-clause is "covered" only when some INVENTORIED HARD claim
    accounts for its hard content (normalized substring match in EITHER direction);
    a soft inventoried claim never licenses a hard sub-clause.
    """
    uncovered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for where, clause in _scan_surfaces(draft):
        if not is_hard_medical(clause):
            continue
        for sub in _split_subclauses(clause):
            if not is_hard_medical(sub):
                continue
            sub_norm = _normalize(sub)
            covered = any(
                hard_norm in sub_norm or sub_norm in hard_norm
                for hard_norm in inventoried_hard_norms
            )
            if not covered and (where, sub) not in seen:
                seen.add((where, sub))
                uncovered.append((where, sub))
    return uncovered


def _escalate(current: str, candidate: str) -> str:
    """Return whichever status is more severe."""
    return candidate if _SEVERITY[candidate] > _SEVERITY[current] else current


def _looks_like_disease_treatment(text: str) -> bool:
    """Heuristic backstop for cure/treat/heal/prevent-a-condition phrasing.

    Mirrors the model's ``is_disease_treatment`` so the gate never depends solely
    on the model to catch the least-defensible claim class.
    """
    lowered = f" {text.lower()} "
    return any(
        f" {verb} " in lowered
        for verb in ("cure", "cures", "cured", "treat", "treats", "treated",
                     "heal", "heals", "healed", "prevent", "prevents", "prevented")
    )


def _int5_cross_check(
    draft: Draft, extracted_item: ExtractedItem
) -> tuple[str, list[str]]:
    """Flag a HARD claim in ``draft.claims_used`` the source item never carried.

    ``Draft.claims_used`` is the generator's *unvalidated free-text self-report*,
    decoupled from the extracted material. If it introduces a hard medical claim
    that the authoritative :class:`ExtractedItem` did not carry, the generator
    invented a medical claim out of thin air -> escalate to manual review (the
    register check on the caption is not enough; the self-report itself is suspect).
    """
    extracted_claim = (extracted_item.claim or "")
    notes: list[str] = []
    status = _CLEAN
    for used in draft.claims_used:
        if not is_hard_medical(used):
            continue
        # The extracted item carried this hard claim -> faithful, not invented.
        if match_claim(used, [_as_approved_entry(extracted_claim)]) is not None:
            continue
        notes.append(
            f"INT-5: the generator's claims_used reports a hard medical claim "
            f"'{used}' that the extracted source material did not contain "
            "(possibly introduced/invented). Manual review required."
        )
        status = _escalate(status, _NEEDS_REVIEW)
    return status, notes


def _as_approved_entry(claim_text: str) -> RegisterEntry:
    """Wrap a claim string as a throwaway APPROVED entry so we can reuse match_claim.

    Used by the INT-5 check to ask "does the extracted claim match this used claim?"
    with the same normalized fuzzy matching the register uses - a single matching
    implementation, no parallel string compare.
    """
    return RegisterEntry(
        claim_text=claim_text, claim_type="hard", approved=True
    )


# Deterministic hedged-rewrite fallback. Used only when we must soften but the
# model supplied no usable rewrite - we still must NOT pass the assertive caption.
_HEDGE_PREFIX = (
    "[Needs review - softened] Many people report that "
)


def _resolve_softened_caption(
    original: str,
    model_rewrite: str | None,
    softened_claims: list[str] | None = None,
) -> str:
    """Pick the softened caption: the model's hedged rewrite, or a safe fallback.

    The model's rewrite is preferred (it preserves voice). But the gate must never
    depend on the model to avoid passing an assertive caption, so if the model
    returned nothing usable, returned the caption unchanged, OR (CS-5) returned a
    "rewrite" that STILL contains one of the assertive claims we were softening,
    we apply a deterministic hedge - producing a revised_text that is clearly
    different from the original and visibly flagged for the human reviewer.
    """
    if model_rewrite is not None:
        candidate = model_rewrite.strip()
        if (
            candidate
            and candidate != original.strip()
            and not _retains_assertion(candidate, softened_claims)
        ):
            return candidate
    # Fallback: deterministically hedge. Lowercase the first letter of the original
    # so the prefix reads naturally, and mark it for review.
    body = original.strip()
    if body:
        body = body[0].lower() + body[1:]
    return f"{_HEDGE_PREFIX}{body}"


def _retains_assertion(candidate: str, softened_claims: list[str] | None) -> bool:
    """True if ``candidate`` still contains a softened assertion verbatim (CS-5).

    Normalized substring check: a rewrite that bolts a hedge onto the original
    assertion ("Floating lowers your blood pressure, and many people feel calmer")
    still publishes the un-hedged claim, so it must be rejected in favor of the
    deterministic hedge.
    """
    if not softened_claims:
        return False
    candidate_norm = _normalize(candidate)
    if not candidate_norm:
        return False
    for claim in softened_claims:
        claim_norm = _normalize(claim)
        if claim_norm and claim_norm in candidate_norm:
            return True
    return False
