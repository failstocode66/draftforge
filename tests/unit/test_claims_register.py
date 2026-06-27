"""Tests for the claims-register loader + pure matching logic (Task 2.1).

Two concerns, both fully offline:

* :func:`load_claims_register` — the fail-loud loader for the real, gitignored
  ``data/claims_register.json`` receiver. Like the other loaders it raises a
  specific :class:`MissingInputsError` (naming the file + how to fill it) rather
  than ever falling back to empty/sample data.
* The pure matching helpers in ``draftforge.stages.claims_register``:
  :func:`is_hard_medical` (heuristic classifier) and :func:`match_claim`
  (normalized fuzzy match against APPROVED register entries only).

Base paths are injected via ``base_dir`` so the loader tests stage files under
``tmp_path`` without touching the repo's real ``data/`` dir.
"""

from __future__ import annotations

import json

import pytest

from draftforge.inputs import MissingInputsError, load_claims_register
from draftforge.models import RegisterEntry
from draftforge.stages.claims_register import is_hard_medical, is_hedged, match_claim


# --- fixtures / helpers ---------------------------------------------------------


def _entry(
    claim_text: str,
    *,
    claim_type: str = "hard",
    approved: bool = True,
    source_citation: str = "Smith et al. 2018, J. Float Sci.",
    notes: str = "reviewed",
) -> dict:
    return {
        "claim_text": claim_text,
        "claim_type": claim_type,
        "approved": approved,
        "source_citation": source_citation,
        "notes": notes,
    }


def _write_register(base_dir, entries: list[dict]) -> None:
    d = base_dir / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "claims_register.json").write_text(json.dumps(entries), encoding="utf-8")


# --- load_claims_register: fail-loud receiver -----------------------------------


def test_load_register_missing_raises_specific_error(tmp_path):
    with pytest.raises(MissingInputsError) as ei:
        load_claims_register(base_dir=tmp_path)
    msg = str(ei.value)
    assert "claims_register" in msg
    assert "claims_register.json" in msg


def test_load_register_empty_file_raises(tmp_path):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "claims_register.json").write_text("   \n", encoding="utf-8")
    with pytest.raises(MissingInputsError):
        load_claims_register(base_dir=tmp_path)


def test_load_register_invalid_json_raises(tmp_path):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "claims_register.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(MissingInputsError):
        load_claims_register(base_dir=tmp_path)


def test_load_register_empty_array_raises(tmp_path):
    _write_register(tmp_path, [])
    with pytest.raises(MissingInputsError):
        load_claims_register(base_dir=tmp_path)


def test_load_register_returns_typed_entries(tmp_path):
    _write_register(
        tmp_path,
        [
            _entry("magnesium is absorbed through the skin"),
            _entry(
                "many people report feeling relaxed",
                claim_type="soft",
                source_citation="",
            ),
        ],
    )
    register = load_claims_register(base_dir=tmp_path)
    assert all(isinstance(e, RegisterEntry) for e in register)
    assert len(register) == 2
    assert register[0].claim_text == "magnesium is absorbed through the skin"
    assert register[0].claim_type == "hard"
    assert register[0].approved is True
    assert register[1].claim_type == "soft"


def test_load_register_rejects_bad_entry_shape(tmp_path):
    # An entry missing required keys must fail loud, not silently load partial.
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "claims_register.json").write_text(
        json.dumps([{"claim_text": "only text"}]), encoding="utf-8"
    )
    with pytest.raises(MissingInputsError):
        load_claims_register(base_dir=tmp_path)


def test_load_register_rejects_bad_claim_type(tmp_path):
    _write_register(tmp_path, [_entry("x", claim_type="medium")])
    with pytest.raises(MissingInputsError):
        load_claims_register(base_dir=tmp_path)


# --- is_hard_medical heuristic --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Floating lowers your blood pressure",
        "It reduces anxiety and depression",
        "alleviates chronic pain",
        "boosts magnesium absorption through the skin",
        "lowers cortisol levels",
        "This cures insomnia",
        "treats fibromyalgia",
        "heals your nervous system",
        "reduces inflammation in the body",
        "lowers your heart rate and stress hormones",
    ],
)
def test_is_hard_medical_flags_hard_claims(text):
    assert is_hard_medical(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Many people report feeling deeply relaxed",
        "Sink into stillness for 60 minutes",
        "A calm, quiet hour just for you",
        "Our float tanks are spacious and clean",
        "Book your first float this week",
        "Floating feels like a reset for your mind",
    ],
)
def test_is_hard_medical_passes_soft_claims(text):
    assert is_hard_medical(text) is False


def test_is_hard_medical_is_case_and_punctuation_insensitive():
    assert is_hard_medical("LOWERS BLOOD PRESSURE!!!") is True
    assert is_hard_medical("Reduces   Anxiety.") is True


# --- CS-2: synonym-verb recall gap ----------------------------------------------
# is_hard_medical is the deterministic backstop for BOTH per-claim resolution AND
# the rescan. It only fired when the effect verb was in the original _EFFECT_VERBS
# set (lower/reduce/boost/...). Elimination / normalization / calming synonyms
# slipped past, so a model that mislabels one as soft yields `clean` with NO
# backstop. Each of these pairs a known medical SUBJECT with a synonym effect verb
# and MUST read as hard.


@pytest.mark.parametrize(
    "text",
    [
        "drops your cortisol",
        "soothes chronic pain",
        "eliminates insomnia",
        "banishes anxiety",
        "eases fibromyalgia",
        "melts away migraines",
        "kills the inflammation",
        "resets your nervous system",
        "blood pressure normalizes",
        "rebalances your hormones",
        "detoxes your body",
    ],
)
def test_is_hard_medical_flags_synonym_effect_verbs(text):
    assert is_hard_medical(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Benign experiential copy: calming/melting verbs WITHOUT a medical subject
        # must still be soft (the heuristic gates on a _MEDICAL_TERMS subject).
        "melts away your stress",
        "calms your mind",
        "soothes your busy day",
        "resets your routine",
        "eases you into the evening",
    ],
)
def test_is_hard_medical_synonym_verbs_without_medical_subject_stay_soft(text):
    assert is_hard_medical(text) is False


# --- CS-3 part 1: _normalize splits camelCase/PascalCase ------------------------
# Hard claims published in hashtags (#LowersBloodPressure) reach `clean` because
# the run-together token never matches an effect verb or medical subject. Splitting
# camelCase before lowercasing turns the hashtag token into real words so
# is_hard_medical (and match_claim) catch it.


def test_normalize_splits_camelcase():
    from draftforge.stages.claims_register import _normalize

    assert _normalize("#LowersBloodPressure") == "lowers blood pressure"
    assert _normalize("ReducesAnxiety") == "reduces anxiety"


def test_normalize_leaves_plain_words_unchanged():
    from draftforge.stages.claims_register import _normalize

    # A digit->upper boundary still splits (vitamin-style tokens are rare in copy);
    # ordinary lowercased prose is unaffected.
    assert _normalize("lowers blood pressure") == "lowers blood pressure"
    assert _normalize("Lowers Blood Pressure!") == "lowers blood pressure"


def test_is_hard_medical_catches_camelcase_hashtag_token():
    assert is_hard_medical("#LowersBloodPressure") is True
    assert is_hard_medical("#ReducesAnxiety") is True


# --- match_claim: normalized fuzzy match against APPROVED entries ----------------


def _register(entries: list[dict]) -> list[RegisterEntry]:
    return [RegisterEntry.model_validate(e) for e in entries]


def test_match_claim_exact_match_returns_entry():
    reg = _register([_entry("magnesium is absorbed through the skin")])
    hit = match_claim("magnesium is absorbed through the skin", reg)
    assert hit is not None
    assert hit.claim_text == "magnesium is absorbed through the skin"


def test_match_claim_is_case_and_punctuation_insensitive():
    reg = _register([_entry("Magnesium is absorbed through the skin.")])
    hit = match_claim("magnesium IS absorbed through the skin!", reg)
    assert hit is not None


def test_match_claim_whitespace_normalized():
    reg = _register([_entry("lowers   blood    pressure")])
    hit = match_claim("lowers blood pressure", reg)
    assert hit is not None


def test_match_claim_fuzzy_near_match():
    # Minor wording differences still match (normalized fuzzy, not exact only).
    reg = _register([_entry("floating reduces cortisol levels")])
    hit = match_claim("floating reduces cortisol level", reg)
    assert hit is not None


def test_match_claim_no_match_returns_none():
    reg = _register([_entry("magnesium is absorbed through the skin")])
    assert match_claim("floating lowers blood pressure", reg) is None


def test_match_claim_ignores_unapproved_entries():
    # An entry present but approved=False must NOT count as a register match.
    reg = _register([_entry("lowers blood pressure", approved=False)])
    assert match_claim("lowers blood pressure", reg) is None


def test_match_claim_only_returns_approved_even_if_unapproved_dup_exists():
    reg = _register(
        [
            _entry("lowers blood pressure", approved=False, source_citation="bad"),
            _entry("lowers blood pressure", approved=True, source_citation="good"),
        ]
    )
    hit = match_claim("lowers blood pressure", reg)
    assert hit is not None
    assert hit.approved is True
    assert hit.source_citation == "good"


def test_match_claim_empty_register_returns_none():
    assert match_claim("anything", []) is None


# --- CS-4: polarity / antonym guard ---------------------------------------------
# SequenceMatcher >= 0.9 is char-overlap, which is high between a claim and its
# NEGATION or ANTONYM (adding "not" or swapping lower<->raise changes few chars).
# So "does not lower blood pressure" / "raises blood pressure" fuzzy-match an
# approved "lowers blood pressure" and the unapproved (often OPPOSITE) claim gets
# licensed as approved with a false citation. The guard rejects any sub-1.0 match
# whose negation tokens differ, or whose word-set straddles a known antonym pair.


def test_match_claim_rejects_negation_of_approved_claim():
    # "does not lower ..." must NOT match approved "... lowers ...".
    reg = _register(
        [_entry("floating lowers your resting blood pressure over time")]
    )
    assert (
        match_claim(
            "floating does not lower your resting blood pressure over time", reg
        )
        is None
    )


def test_match_claim_rejects_antonym_of_approved_claim():
    # "raises ..." (antonym of "lowers") must NOT match approved "lowers ...".
    reg = _register(
        [_entry("floating lowers your resting blood pressure over time")]
    )
    assert (
        match_claim("floating raises your resting blood pressure over time", reg)
        is None
    )


def test_match_claim_rejects_negated_absorption_claim():
    reg = _register([_entry("magnesium is absorbed through the skin")])
    assert match_claim("magnesium is not absorbed through the skin", reg) is None


def test_match_claim_rejects_reduce_vs_increase_antonym():
    reg = _register([_entry("reduces inflammation in your joints and tissues")])
    assert (
        match_claim("increases inflammation in your joints and tissues", reg)
        is None
    )


def test_match_claim_short_negation_and_antonym_are_none():
    # The spec's canonical short cases. (These already score below 0.9, but the
    # guard must hold regardless of threshold - pin them.)
    reg = _register([_entry("lowers blood pressure")])
    assert match_claim("does not lower blood pressure", reg) is None
    assert match_claim("raises blood pressure", reg) is None


def test_match_claim_genuine_paraphrase_above_threshold_still_matches():
    # Same polarity, no antonym swap, just "does"/"will" wording drift (~0.92).
    # The guard must NOT reject this - a real near-paraphrase must still match.
    reg = _register([_entry("this will lower your blood pressure significantly")])
    hit = match_claim("this does lower your blood pressure significantly", reg)
    assert hit is not None


# --- P2-3: pin the 0.9 threshold above the distinct-subject collision -----------


def test_match_claim_distinct_medical_subject_below_threshold_is_none():
    # "lowers blood pressure" vs "lowers blood sugar" score ~0.82 < 0.9: they are
    # DISTINCT claims and must not collide. Pins the threshold's lower edge.
    reg = _register([_entry("lowers blood sugar")])
    assert match_claim("lowers blood pressure", reg) is None


def test_match_claim_genuine_pair_just_above_threshold_matches():
    # A genuine same-claim pair scoring just above 0.9 (trailing-plural drift)
    # MUST match. Pins the threshold's upper edge.
    reg = _register([_entry("floating reduces cortisol levels")])
    assert match_claim("floating reduces cortisol level", reg) is not None


# --- P2-7: bare medical subject with NO effect/treatment verb -------------------
# DECISION: a bare medical-subject mention with no asserted effect verb is SOFT
# (is_hard_medical returns False, leaving it to the LLM). Rationale: is_hard_medical
# backstops every clause, so firing on every benign mention ("monitor your blood
# pressure at home", "talk to your doctor about anxiety") would over-fire on
# legitimate non-claim copy; the control still leans hard for any ASSERTED effect
# (the lower/reduce/... + synonym families). A bare subject carries no health
# CLAIM, so it stays soft. Pinned both ways below.


@pytest.mark.parametrize(
    "text",
    [
        "your blood pressure",
        "monitor your blood pressure at home",
        "talk to your doctor about anxiety",
        "a conversation about chronic pain",
    ],
)
def test_is_hard_medical_bare_subject_without_verb_is_soft(text):
    assert is_hard_medical(text) is False


@pytest.mark.parametrize(
    "text",
    [
        # The SAME subjects WITH an asserted effect verb are hard (the line we draw).
        "lowers your blood pressure",
        "eases your anxiety",
        "soothes chronic pain",
    ],
)
def test_is_hard_medical_subject_with_effect_verb_is_hard(text):
    assert is_hard_medical(text) is True


# --- is_hedged: hedge detector (relaxed wellness policy) ------------------------
# A hedged physiological claim routes to the non-blocking `advisory` lane instead
# of being softened. Deliberately generous (over-detecting a hedge is the safe,
# permissive direction in a wellness domain).


@pytest.mark.parametrize(
    "text",
    [
        "may help reduce inflammation",
        "can help lower your blood pressure",
        "might ease your anxiety",
        "could support better sleep",
        "designed to help you relax",
        "a good way to reset your mind",
        "many people report feeling calmer",
        "floating tends to reduce stress",
    ],
)
def test_is_hedged_true_for_hedged_phrasing(text):
    assert is_hedged(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "lowers your blood pressure",
        "reduces inflammation",
        "cures depression",
        "eliminates anxiety",
        "",
    ],
)
def test_is_hedged_false_for_flat_assertions(text):
    assert is_hedged(text) is False


# --- CS-6: numeric / dosage collision -------------------------------------------
# An unapproved claim differing only in a NUMBER (a different magnitude or dosage)
# fuzzy-matches an approved entry (>0.9 char overlap; the digits are a tiny part of
# the string) and is licensed with the approved citation -> reaches `clean` with a
# false, magnitude-wrong citation. The guard rejects any sub-1.0 match whose
# numeric-token multiset differs from the candidate's.


def test_match_claim_rejects_different_magnitude_number():
    # "by 50 points" must NOT match approved "by 5 points" (ratio ~0.985).
    reg = _register([_entry("lowers blood pressure by 5 points")])
    assert match_claim("lowers blood pressure by 50 points", reg) is None


def test_match_claim_rejects_different_dosage_number():
    reg = _register([_entry("take 1 capsule daily")])
    assert match_claim("take 3 capsules daily", reg) is None


def test_match_claim_identical_numbers_still_match():
    # Same numbers + trailing punctuation drift -> a genuine match must survive.
    reg = _register([_entry("reduces cortisol by 20 percent")])
    assert match_claim("reduces cortisol by 20 percent.", reg) is not None


def test_match_claim_nonnumeric_paraphrase_above_threshold_still_matches():
    # No numbers on either side -> the numeric guard is a no-op; paraphrase matches.
    reg = _register([_entry("floating reduces cortisol levels")])
    assert match_claim("floating reduces cortisol level", reg) is not None


def test_match_claim_twice_vs_once_daily_is_none():
    # "twice daily" vs "once daily" (2 vs 1 frequency) must NOT match. (These score
    # ~0.84 < 0.9 so the threshold already separates them; pinned regardless.)
    reg = _register([_entry("take once daily")])
    assert match_claim("take twice daily", reg) is None
