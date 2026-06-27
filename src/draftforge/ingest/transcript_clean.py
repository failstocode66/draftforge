"""Medium transcript-cleanup pipeline (format-lock §2.2).

Turns a wellness studio's raw auto-generated podcast transcripts into
a clean voice corpus, **mechanically** — the words stay verbatim; this is NOT an
LLM rewrite (that would smooth away the distinctive cadence we're capturing).

Stages (deterministic):

1. **Structural strip** — drop ``MM:SS`` timestamps and the Google-Docs blank /
   backslash separator noise.
2. **Part segmentation (optional)** — when a ``transition_marker`` regex is
   supplied, split the episode at that cue into Part 1 (the pre-transition
   segment, e.g. a news/promo intro) and Part 2 (the post-transition reflective
   segment — usually the higher-value voice corpus). A marker-less transcript
   stays a single corpus, so the cleaner is reusable for any show.
3. **Near-duplicate dedup (the core)** — collapse runs of near-identical
   sentences (the multiple recording takes), keeping the most-complete one.
4. **Light disfluency cleanup** — drop ``uh`` / ``um`` / ``hmm`` fillers.
5. **Proper-noun glossary** — fix the recurring mis-transcriptions.

The output is a :class:`CleanedTranscript` (``part2`` is the primary corpus;
``part1`` is the separate promo set). Signature voice — "small still voice",
"the tank", "Stay still." — survives untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Recurring proper-noun mis-transcriptions → corrections (case-insensitive).
# EXAMPLES only — the default is deliberately generic; replace with the proper
# nouns your show's transcriber actually mishears (it's a configurable default).
DEFAULT_GLOSSARY: dict[str, str] = {
    r"\byou ?tube\b": "YouTube",
    r"\bwi ?fi\b": "Wi-Fi",
}

_TIMESTAMP_LINE = re.compile(r"^\s*\d{1,2}:\d{2}\s*$")
_FILLER = re.compile(r"\b(uh|um+|hmm+)\b[\s,]*", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class CleanedTranscript:
    """A cleaned episode, split into its two parts.

    ``part2`` (the post-transition / reflective segment) is the PRIMARY voice
    corpus; ``part1`` (the pre-transition / promo segment) is a separate labelled
    set used only when a run's guidance calls for a CTA/promo angle. With no
    ``transition_marker`` the whole transcript is returned as ``part2``.
    """

    part1: str
    part2: str


def _strip_structural(raw: str) -> str:
    """Drop timestamp lines and the Doc's blank/backslash separator noise."""
    kept = []
    for line in raw.splitlines():
        if _TIMESTAMP_LINE.match(line):
            continue
        stripped = line.strip().strip("\\").strip()
        if stripped:
            kept.append(stripped)
    return " ".join(kept)


def _segment_parts(text: str, *, marker: str | None) -> tuple[str, str]:
    """Split ``text`` into (Part 1, Part 2) at a configurable transition marker.

    ``marker`` is a case-insensitive regex naming the host's transition into the
    reflective segment — supply the distinctive phrase your show uses. With no
    marker the whole transcript is the corpus (``("", text)``); with a marker that
    is absent from this episode it is treated as a single pre-transition segment
    (``(text, "")`` — e.g. a newsletter-only episode).
    """
    if not marker:
        return "", text.strip()
    match = re.search(marker, text, re.I)
    if match is None:
        return text.strip(), ""
    return text[: match.start()].strip(), text[match.start() :].strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation."""
    if not text.strip():
        return []
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _normalize(sentence: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for fuzzy comparison."""
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()


def _dedup_sentences(
    sentences: list[str], *, threshold: float = 0.72, window: int = 6
) -> list[str]:
    """Collapse runs of near-duplicate takes, keeping the most-complete one.

    Each sentence is compared (normalized fuzzy ratio) to the last ``window``
    kept sentences; a match above ``threshold`` is a re-take — the longer of the
    two is kept in place. This dissolves the 3–6× repeated takes the raw
    transcript captures while leaving genuinely distinct sentences alone.
    """
    kept: list[str] = []
    for sentence in sentences:
        norm = _normalize(sentence)
        if not norm:
            continue
        dup_index = None
        for j in range(max(0, len(kept) - window), len(kept)):
            if SequenceMatcher(None, _normalize(kept[j]), norm).ratio() >= threshold:
                dup_index = j
                break
        if dup_index is None:
            kept.append(sentence)
        elif len(sentence) > len(kept[dup_index]):
            kept[dup_index] = sentence  # prefer the more-complete take
    return kept


def _drop_filler(sentences: list[str]) -> list[str]:
    """Remove ``uh`` / ``um`` / ``hmm`` fillers; drop now-empty sentences."""
    cleaned = []
    for sentence in sentences:
        out = _FILLER.sub("", sentence)
        out = re.sub(r"\s{2,}", " ", out).strip()
        # tidy a leading orphaned comma left by a removed sentence-initial filler
        out = re.sub(r"^[,\s]+", "", out)
        if out and _normalize(out):
            cleaned.append(out[0].upper() + out[1:] if out else out)
    return cleaned


# Words that cannot legitimately END a sentence — a period after one is a
# spurious mid-clause break the auto-transcriber inserted, so the next fragment
# is rejoined. Likewise a fragment STARTING with a connector is a continuation.
_DANGLING_TAIL = frozenset(
    {
        "and", "but", "or", "so", "the", "a", "an", "to", "of", "that", "this",
        "been", "is", "was", "were", "are", "my", "your", "our", "for", "on",
        "in", "with", "into", "at", "as", "it", "i", "we", "he", "she", "they",
        "because", "which", "what", "when", "where", "who", "just", "very",
        "more", "about",
    }
)
_CONTINUATION_HEAD = frozenset({"and", "but", "or", "so", "which", "that"})


def _rejoin_fragments(sentences: list[str]) -> list[str]:
    """Rejoin fragments split by spurious mid-clause periods (spec §2.2 stage 4).

    The auto-transcriber sprinkles periods inside clauses, leaving fragments like
    ``"Presence. And. Awareness."`` A fragment is merged into the previous
    sentence when the previous one ends on a word that cannot end a sentence
    (``"the one that."``) OR the fragment itself opens with a connector
    (``"And."``). Genuinely complete sentences are left untouched.
    """
    out: list[str] = []
    for sentence in sentences:
        words = _normalize(sentence).split()
        head = words[0] if words else ""
        if out:
            prev_words = _normalize(out[-1]).split()
            prev_tail = prev_words[-1] if prev_words else ""
            if prev_tail in _DANGLING_TAIL or head in _CONTINUATION_HEAD:
                merged_tail = sentence[0].lower() + sentence[1:] if sentence else sentence
                out[-1] = out[-1].rstrip(".!? ").rstrip() + " " + merged_tail
                continue
        out.append(sentence)
    return out


def _apply_glossary(text: str, glossary: dict[str, str] | None = None) -> str:
    """Apply the proper-noun corrections (case-insensitive)."""
    glossary = DEFAULT_GLOSSARY if glossary is None else glossary
    for pattern, replacement in glossary.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def _clean_segment(segment: str, glossary: dict[str, str] | None) -> str:
    """Run dedup → filler → glossary over one segment, returning clean prose."""
    if not segment.strip():
        return ""
    sentences = _split_sentences(segment)
    sentences = _drop_filler(sentences)
    sentences = _rejoin_fragments(sentences)
    sentences = _dedup_sentences(sentences)
    return _apply_glossary(" ".join(sentences), glossary).strip()


def clean_transcript(
    raw: str,
    *,
    transition_marker: str | None = None,
    glossary: dict[str, str] | None = None,
) -> CleanedTranscript:
    """Clean a raw episode transcript into a :class:`CleanedTranscript`.

    Args:
        raw: the raw auto-generated transcript text.
        transition_marker: optional case-insensitive regex naming the show's
            transition into its reflective segment (e.g. a distinctive phrase the
            host says). When omitted, the whole transcript is returned as the
            single ``part2`` corpus (no show-specific segmentation).
        glossary: optional override of the proper-noun corrections.
    """
    text = _strip_structural(raw)
    part1_raw, part2_raw = _segment_parts(text, marker=transition_marker)
    return CleanedTranscript(
        part1=_clean_segment(part1_raw, glossary),
        part2=_clean_segment(part2_raw, glossary),
    )
