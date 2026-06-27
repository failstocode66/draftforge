"""Claims register: pure heuristics + matching (Task 2.1).

Two pure, fully-unit-tested functions the deterministic half of the
claims-safety gate (``stages.claims``) leans on. Keeping them here — free of any
LLM or I/O — means the gate's *policy* (what counts as a hard medical claim, and
whether a claim is approved) is testable in isolation and cannot drift with a
prompt edit:

* :func:`is_hard_medical` — a conservative heuristic that flags assertive medical
  / physiological claims (blood pressure, chronic pain, anxiety, cortisol,
  magnesium absorption, "cures/treats/heals", "lowers/reduces <condition>", ...).
  It is intentionally *recall-biased*: the gate is fail-safe, so a false positive
  merely softens an already-fine line, while a false negative would let an uncited
  medical claim slip out. The LLM does the nuanced classification; this is the
  deterministic backstop.
* :func:`match_claim` — normalized, case/punctuation/whitespace-insensitive fuzzy
  match of a claim against the **approved** register entries only. An unapproved
  entry never licenses a hard claim to go out as-is.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from draftforge.models import RegisterEntry

# --- is_hard_medical heuristic --------------------------------------------------

# Medical / physiological subjects. The presence of any of these terms in a claim
# is a strong signal it is a HARD (objective, substantiation-needing) claim rather
# than a soft experiential one. Seeded from the float-therapy ad-standards risk
# surface (typical wellness blood-pressure / chronic-pain / anxiety / magnesium claims).
_MEDICAL_TERMS: frozenset[str] = frozenset(
    {
        "blood pressure",
        "chronic pain",
        "pain",
        "anxiety",
        "depression",
        "magnesium",
        "cortisol",
        "inflammation",
        "insomnia",
        "fibromyalgia",
        "blood sugar",
        "heart rate",
        "heart disease",
        "immune",
        "nervous system",
        "stress hormone",
        "stress hormones",
        "muscle recovery",
        "arthritis",
        "migraine",
        "ptsd",
        "adhd",
        "hormone",
        "metabolism",
        "circulation",
        "detox",
    }
)

# Verbs that assert a clinical/therapeutic EFFECT. "cures/treats/heals/prevents"
# are themselves disease-claim verbs (hard on their own); the "lowers/reduces/..."
# family is hard when paired with a medical subject (handled below).
_TREATMENT_VERBS: frozenset[str] = frozenset(
    {"cure", "cures", "cured", "treat", "treats", "treated", "heal", "heals",
     "healed", "prevent", "prevents", "prevented", "remedy", "remedies"}
)

# Effect verbs that are only HARD in combination with a medical subject (e.g.
# "lowers blood pressure" is hard, but "lowers the lights" is not). The set is
# recall-biased: it covers the original lower/reduce/boost family PLUS the
# elimination / normalization / calming synonym set (CS-2), because a model that
# mislabels one of these as "soft" must still hit this deterministic backstop. The
# _MEDICAL_TERMS subject gate (in is_hard_medical) keeps these from over-firing on
# benign experiential copy like "melts away your stress" / "calms your mind".
_EFFECT_VERBS: frozenset[str] = frozenset(
    {
        # original effect family
        "lower", "lowers", "lowered", "reduce", "reduces", "reduced", "boost",
        "boosts", "boosted", "increase", "increases", "increased", "improve",
        "improves", "improved", "relieve", "relieves", "relieved", "alleviate",
        "alleviates", "alleviated", "regulate", "regulates", "regulated",
        "balance", "balances", "balanced", "absorb", "absorbs", "absorbed",
        "absorption",
        # CS-2: elimination / removal synonyms
        "drop", "drops", "dropped", "eliminate", "eliminates", "eliminated",
        "banish", "banishes", "banished", "kill", "kills", "killed",
        "end", "ends", "ended", "stop", "stops", "stopped",
        "dissolve", "dissolves", "dissolved", "melt", "melts", "melted",
        "flush", "flushes", "flushed", "detox", "detoxes", "detoxed",
        "detoxify", "detoxifies", "detoxified",
        # CS-2: calming / soothing synonyms
        "soothe", "soothes", "soothed", "ease", "eases", "eased",
        "calm", "calms", "calmed",
        # CS-2: normalization / restoration / repair synonyms
        "normalize", "normalizes", "normalized", "reset", "resets",
        "rebalance", "rebalances", "rebalanced", "restore", "restores",
        "restored", "fix", "fixes", "fixed", "heal", "heals", "healed",
    }
)


def _normalize(text: str) -> str:
    """Split camelCase, lowercase, strip punctuation to spaces, collapse whitespace.

    The single normalization used by BOTH the heuristic and the register match, so
    "Lowers Blood Pressure!" and "lowers  blood   pressure" reduce identically.

    CS-3: camelCase/PascalCase boundaries are split into words *before* lowercasing
    so a run-together hashtag token like ``#LowersBloodPressure`` normalizes to
    ``lowers blood pressure`` and is therefore caught by both is_hard_medical and
    match_claim (otherwise the hard claim hides as one unsplittable token).
    """
    # Insert a space at every lower/digit -> upper boundary (camelCase / PascalCase)
    # while the casing is still intact, THEN lowercase.
    decameled = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    lowered = decameled.lower()
    # Replace any run of non-alphanumeric chars with a single space.
    spaced = re.sub(r"[^a-z0-9]+", " ", lowered)
    return spaced.strip()


def is_hard_medical(claim_text: str) -> bool:
    """Return True if ``claim_text`` reads as an assertive HARD medical claim.

    Recall-biased on purpose (see module docstring): flags a claim when it either
    uses a disease/treatment verb ("cures", "treats", "heals", "prevents") OR
    pairs an effect verb ("lowers/reduces/boosts/alleviates/absorbs/...") with a
    known medical subject ("blood pressure", "anxiety", "cortisol", "magnesium",
    ...). Pure soft/experiential copy ("sink into stillness", "feels like a
    reset") returns False.
    """
    norm = _normalize(claim_text)
    if not norm:
        return False
    words = set(norm.split())

    # Disease/treatment verbs are hard on their own.
    if words & _TREATMENT_VERBS:
        return True

    has_medical_subject = any(term in norm for term in _MEDICAL_TERMS)
    if not has_medical_subject:
        return False

    # A medical subject paired with an effect verb (or "absorption") is a hard
    # physiological claim. A bare mention of a medical word without an asserted
    # effect (rare in marketing copy) is left to the LLM.
    if words & _EFFECT_VERBS:
        return True

    return False


# --- is_hedged: hedge / "may/can help" detector ---------------------------------

# Hedge markers. GENEROUS on purpose: in a holistic-wellness (non-medical) domain a
# hedged physiological claim ("may help reduce inflammation") is acceptable advisory
# copy - the hedge IS the compliance posture - so over-detecting a hedge errs toward
# the permissive/advisory lane (the safe direction) rather than softening legitimate
# copy. Only consulted for claims is_hard_medical already flagged.
_HEDGE_WORDS: frozenset[str] = frozenset(
    {"may", "might", "can", "could", "help", "helps", "helping", "helped",
     "tends", "tend", "support", "supports", "promote", "promotes",
     "encourage", "encourages", "associated", "often", "some", "many",
     "report", "reports", "reported", "find", "finds", "designed", "way",
     "perhaps", "possibly", "potentially", "generally", "typically"}
)
_HEDGE_PHRASES: tuple[str, ...] = (
    "designed to", "a way to", "good way", "great way", "many people",
    "some people", "can be", "is a way", "tends to", "may help", "can help",
    "associated with",
)


def is_hedged(claim_text: str) -> bool:
    """Return True if ``claim_text`` is HEDGED ("may/can help", "designed to", ...).

    In this holistic-wellness domain a hedged physiological claim is acceptable
    advisory copy, so the gate routes a hedged hard claim to the non-blocking
    ``advisory`` lane rather than softening it. Deliberately generous - over-
    detecting a hedge is the safe/permissive direction. Only consulted for claims
    ``is_hard_medical`` already flags (so it never makes benign copy "hard").
    """
    norm = _normalize(claim_text)
    if not norm:
        return False
    if any(phrase in norm for phrase in _HEDGE_PHRASES):
        return True
    return bool(set(norm.split()) & _HEDGE_WORDS)


# --- match_claim: normalized fuzzy match against APPROVED entries ----------------

# Similarity at/above which two normalized claims are treated as "the same claim".
# High enough that distinct claims ("lowers blood pressure" vs "lowers cortisol")
# do not collide, low enough to absorb trivial wording drift (a trailing plural,
# a dropped article).
_MATCH_THRESHOLD = 0.9

# CS-4: char-overlap similarity is high between a claim and its NEGATION ("not"
# adds ~3 chars) or its ANTONYM ("lower" -> "raise" swaps a few chars), so a
# fuzzy match can license the OPPOSITE of an approved claim with a false citation.
# Before accepting any sub-1.0 match we require the two claims to agree on polarity.
_NEGATIONS: frozenset[str] = frozenset(
    {"not", "no", "never", "without", "cannot", "cant", "doesnt", "dont",
     "wont", "isnt", "arent"}
)

# Directional antonym pairs whose presence on opposite sides flips the claim's
# meaning. Stored unordered (membership is checked symmetrically below).
_ANTONYM_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("lower", "raise"), ("lowers", "raises"), ("lowered", "raised"),
        ("reduce", "increase"), ("reduces", "increases"),
        ("reduced", "increased"),
        ("reduce", "boost"), ("reduces", "boosts"),
        ("decrease", "increase"), ("decreases", "increases"),
        ("drop", "raise"), ("drops", "raises"),
        ("lower", "boost"), ("lowers", "boosts"),
    )
)


def match_claim(
    claim_text: str, register: list[RegisterEntry]
) -> RegisterEntry | None:
    """Return the APPROVED register entry matching ``claim_text``, or ``None``.

    Matching is normalized (case/punctuation/whitespace-insensitive) and fuzzy
    (absorbs trivial wording drift via a similarity threshold). Only entries with
    ``approved is True`` are considered — an unapproved row, even an exact textual
    duplicate, never counts as a match — so a hard claim can only be licensed by a
    genuinely approved citation.

    On multiple approved candidates above threshold, the highest-similarity entry
    is returned.
    """
    target = _normalize(claim_text)
    if not target:
        return None
    target_words = set(target.split())

    best: RegisterEntry | None = None
    best_score = 0.0
    for entry in register:
        if not entry.approved:
            continue
        candidate = _normalize(entry.claim_text)
        if not candidate:
            continue
        if candidate == target:
            score = 1.0
        else:
            score = SequenceMatcher(None, target, candidate).ratio()
        if score < _MATCH_THRESHOLD or score <= best_score:
            continue
        if score < 1.0:
            # CS-4: a fuzzy match must not straddle a polarity flip - a negation
            # difference or a known antonym swap means the OPPOSITE claim.
            if _polarity_differs(target_words, set(candidate.split())):
                continue
            # CS-6: a fuzzy match must not differ only in a NUMBER - a different
            # magnitude/dosage ("by 50 points" vs "by 5 points") is a DIFFERENT,
            # un-substantiated claim that must not borrow the approved citation.
            if _numbers_differ(target, candidate):
                continue
        best = entry
        best_score = score
    return best


def _numbers_differ(text_a: str, text_b: str) -> bool:
    """True if the two normalized strings carry different multisets of numbers.

    Char-overlap stays high when a claim differs only in a magnitude or dosage
    ("lowers blood pressure by 50 points" vs "... by 5 points", ratio ~0.985), but
    the numbers ARE the claim - a different number is a different, un-substantiated
    assertion. Compared as a multiset (Counter) so a repeated number must repeat.
    """
    return Counter(re.findall(r"\d+", text_a)) != Counter(re.findall(r"\d+", text_b))


def _polarity_differs(words_a: set[str], words_b: set[str]) -> bool:
    """True if two normalized word-sets disagree in polarity (negation or antonym).

    Two ways the meaning can flip even at high char-overlap (CS-4):
    * one side carries a negation token the other lacks (symmetric difference of
      the negation tokens is non-empty), or
    * the symmetric difference of the word-sets straddles a known antonym pair
      ("lowers" on one side, "raises" on the other).
    """
    # Negation mismatch: the two claims disagree about a negation word.
    if (words_a & _NEGATIONS) ^ (words_b & _NEGATIONS):
        return True
    # Antonym straddle: a removed word and an added word form an antonym pair.
    only_a = words_a - words_b
    only_b = words_b - words_a
    for wa in only_a:
        for wb in only_b:
            if frozenset((wa, wb)) in _ANTONYM_PAIRS:
                return True
    return False
