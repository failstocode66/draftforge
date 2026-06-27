"""Multi-format document loader.

``load_document`` reads a ``.txt``, ``.md``, or ``.pdf`` file and returns its
text. It fails loud rather than silently returning placeholder data:

* a missing file raises :class:`FileNotFoundError`;
* a file with no extractable text (whitespace only) raises
  :class:`EmptyDocumentError`.

Text longer than ``max_chars`` is truncated. Because callers sometimes need to
know that truncation happened, the truncation is *observable* via
:func:`load_document_meta`, which returns a :class:`LoadedDocument` carrying the
text, the original length, and a ``truncated`` flag. ``load_document`` is a thin
str-returning facade over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyPDF2 import PdfReader


class EmptyDocumentError(ValueError):
    """Raised when a document yields no extractable (non-whitespace) text."""


@dataclass(frozen=True)
class LoadedDocument:
    """Result of loading a document, with truncation observable."""

    text: str
    original_length: int
    truncated: bool


def load_document_meta(
    path: str | Path, *, max_chars: int = 200_000
) -> LoadedDocument:
    """Load ``path`` and return a :class:`LoadedDocument`.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        EmptyDocumentError: if the document has no non-whitespace text.
        ValueError: if the file extension is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"document not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        raw = _read_pdf(p)
    elif suffix in (".txt", ".md"):
        raw = p.read_text(encoding="utf-8")
    else:
        raise ValueError(f"unsupported document type: {suffix!r} ({p})")

    if not raw.strip():
        raise EmptyDocumentError(f"no extractable text in document: {p}")

    original_length = len(raw)
    truncated = original_length > max_chars
    text = raw[:max_chars] if truncated else raw
    return LoadedDocument(
        text=text, original_length=original_length, truncated=truncated
    )


def load_document(path: str | Path, *, max_chars: int = 200_000) -> str:
    """Load ``path`` and return its (possibly truncated) text.

    Thin facade over :func:`load_document_meta`; use that function when you need
    to know whether truncation occurred. Raises the same exceptions.
    """
    return load_document_meta(path, max_chars=max_chars).text


def _read_pdf(p: Path) -> str:
    """Extract and join page text from a PDF via PyPDF2."""
    reader = PdfReader(str(p))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
