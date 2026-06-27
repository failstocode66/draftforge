"""Unit tests for the Medium transcript-cleanup pipeline (format-lock §2.2).

Mechanical + dedup, words kept verbatim otherwise (NOT an LLM rewrite — that
would smooth away the distinctive voice). Stages: structural strip → optional
part segmentation at a configurable transition marker → near-duplicate dedup of
the multiple-takes → light disfluency cleanup → proper-noun glossary. Signature
phrases (e.g. "small still voice", "stay still") must survive.
"""

from __future__ import annotations

import re

from draftforge.ingest.transcript_clean import (
    CleanedTranscript,
    _apply_glossary,
    _dedup_sentences,
    _drop_filler,
    _rejoin_fragments,
    _segment_parts,
    _split_sentences,
    _strip_structural,
    clean_transcript,
)


# --- structural strip -----------------------------------------------------------


def test_strip_structural_removes_timestamps_and_separators():
    raw = "00:53\n\nMy name is Sam.\n\n  \n\n01:37\n\nIt gives us the chance.\n"
    out = _strip_structural(raw)
    assert "00:53" not in out
    assert "01:37" not in out
    assert "My name is Sam." in out
    assert "It gives us the chance." in out


# --- dedup (the core stage) -----------------------------------------------------


def test_dedup_collapses_consecutive_near_duplicate_takes():
    sentences = [
        "Well, we focus on the float aspect of DraftForge.",
        "Well, we focus on the float aspect of DraftForge.",
        "While we focus on the floating aspect of DraftForge.",
        "And listen for that small still voice inside of me.",
    ]
    out = _dedup_sentences(sentences)
    # the three near-identical takes collapse to one; the distinct line stays
    assert len(out) == 2
    assert any("small still voice" in s for s in out)


def test_dedup_keeps_the_most_complete_take():
    sentences = [
        "Floating became a responsibility.",
        "Floating became a real responsibility for me.",
    ]
    out = _dedup_sentences(sentences)
    assert out == ["Floating became a real responsibility for me."]


def test_dedup_preserves_genuinely_distinct_sentences():
    sentences = ["The tank is quiet.", "I love the coast.", "Stay still."]
    assert _dedup_sentences(sentences) == sentences


# --- disfluency -----------------------------------------------------------------


def test_drop_filler_removes_uh_um_hmm():
    out = _drop_filler(["Uh, it was always a question.", "Um.", "Hmm, taking it in."])
    joined = " ".join(out)
    # fillers gone (whole-word; case-insensitive check avoids matching e.g. "summer")
    assert not any(re.search(rf"\b{f}\b", joined, re.I) for f in ("uh", "um", "hmm"))
    # content preserved (the cleaner re-capitalizes a sentence after stripping a
    # leading filler, so match the case-insensitive content)
    assert "was always a question" in joined
    assert "taking it in" in joined.lower()


# --- glossary -------------------------------------------------------------------


def test_rejoin_fragments_merges_spurious_midclause_periods():
    # "Presence. And. Awareness." -> one clause; a dangling-word period is merged.
    out = _rejoin_fragments(["Presence.", "And.", "Awareness."])
    assert out == ["Presence and awareness."]


def test_rejoin_fragments_merges_when_previous_ends_dangling():
    out = _rejoin_fragments(["The one that.", "Allows me to connect."])
    assert out == ["The one that allows me to connect."]


def test_rejoin_fragments_leaves_complete_sentences_alone():
    sents = ["I love the tank.", "Stay still."]
    assert _rejoin_fragments(sents) == sents


def test_apply_glossary_fixes_proper_nouns():
    text = "I posted it on you tube over wi fi."
    out = _apply_glossary(text)
    assert "YouTube" in out
    assert "Wi-Fi" in out
    assert "you tube" not in out


# --- segmentation ---------------------------------------------------------------


def test_segment_parts_splits_at_a_supplied_marker():
    text = (
        "Welcome to the show. We have community events and news to share. "
        "And now, let's go deeper. Presence has been the lesson."
    )
    p1, p2 = _segment_parts(text, marker=r"let's go deeper")
    assert "community events and news" in p1
    assert "Presence has been the lesson" in p2
    assert "let's go deeper" in p2


def test_segment_parts_marker_absent_is_all_part1():
    text = "Welcome to the show. We have a community event this week."
    p1, p2 = _segment_parts(text, marker=r"let's go deeper")
    assert "community event" in p1
    assert p2 == ""


def test_segment_parts_no_marker_is_single_corpus():
    # The generic default: no show-specific marker -> the whole transcript is the
    # primary (part2) corpus, so the cleaner is reusable for any podcast.
    text = "Just one segment of reflection here."
    p1, p2 = _segment_parts(text, marker=None)
    assert p1 == ""
    assert "one segment of reflection" in p2


# --- end to end -----------------------------------------------------------------


def test_clean_transcript_end_to_end_preserves_voice():
    raw = (
        "00:57\n\nWelcome, this is the show.\n\n  \n\n"
        "04:05\n\nFollow us at Stillwater Float. Follow us at Stillwater Float.\n\n  \n\n"
        "09:31\n\nAnd now, let's go deeper. Uh, listen for that "
        "small still voice. I watched a you tube video.\n\n  \n\n"
        "18:22\n\nThanks for being with me. Stay still.\n"
    )
    cleaned = clean_transcript(raw, transition_marker=r"let's go deeper")
    assert isinstance(cleaned, CleanedTranscript)
    # timestamps gone, filler gone, glossary applied, voice intact
    assert "00:57" not in cleaned.part2
    assert "Uh," not in cleaned.part2
    assert "YouTube" in cleaned.part2
    assert "small still voice" in cleaned.part2
    assert "Stay still." in cleaned.part2
    # duplicate "Follow us" take in part 1 collapsed
    assert cleaned.part1.count("Follow us at Stillwater Float") == 1


def test_split_sentences_basic():
    assert _split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
